import json
import logging
import os
import re
import sqlite3
import traceback
from contextlib import closing

import dspy
import requests
import telebot  # pip install pytelegrambotapi
import whisper  # pip install -U openai-whisper (exige o ffmpeg: no Windows, choco install ffmpeg)

from server import SCHEMA

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LM_MODEL = os.getenv("LM_MODEL", "openai/gemma-4-E2B-it-IQ4_XS")
LM_API_BASE = os.getenv("LM_API_BASE", "http://localhost:1337/v1")
TELEGRAM_MAX_CHARS = 4000  # o limite do Telegram é 4096
TRUNCATED_NOTICE = "… (resultado cortado)"
FALLBACK_REPLY = "Desculpe, não consegui processar sua mensagem."


def configure_lm() -> None:
    """Configura o LLM local que a DSPy usa para gerar o SQL."""
    dspy.configure(lm=dspy.LM(LM_MODEL, api_base=LM_API_BASE, api_key="not-needed"))


class TextToSQL(dspy.Signature):
    """Generate SQL from natural language."""

    dbschema = dspy.InputField(desc="Databases schema")
    question = dspy.InputField(desc="Natural language question")

    sql_query = dspy.OutputField(desc="Valid SQL query")


class SelectOnlyError(Exception):
    """SQL recusado pela checagem de "apenas SELECT" (requisito 3)."""


class ReliableSQLGenerator(dspy.Module):
    """Gera o SQL e só o devolve se for um SELECT válido para o esquema."""

    def __init__(self, schema: str, predictor=None):
        super().__init__()
        self.schema = schema
        self.generate_sql = predictor if predictor is not None else dspy.RLM(TextToSQL, max_iters=5, max_llm_calls=10)

    def forward(self, question: str) -> dspy.Prediction:
        sql = None
        try:  # requisito 2: try/catch no próprio gerador
            pred = self.generate_sql(dbschema=self.schema, question=question)
            sql = self.clean_sql(pred.sql_query)
            self.check_select_only(sql)  # requisito 3: antes de qualquer execução
            self.validate_in_memory(sql)  # requisito 2: SQLite em memória
        except SelectOnlyError as exc:
            return self._failure(str(exc), sql)
        except (sqlite3.Error, sqlite3.Warning) as exc:
            return self._failure(f"O SQL gerado é inválido: {exc}", sql)
        except Exception:
            logger.exception("Falha ao gerar o SQL")
            return self._failure("Não consegui gerar o SQL agora (falha no modelo de linguagem).", sql)
        return dspy.Prediction(sql_query=sql, error=None)

    @staticmethod
    def clean_sql(raw) -> str:
        """Tira os espaços, a cerca de markdown (```sql ... ```) e os ';' do final do SQL do LLM."""
        sql = (raw or "").strip()
        sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```$", "", sql.strip()).strip()
        while sql.endswith(";"):
            sql = sql[:-1].rstrip()
        return sql

    @staticmethod
    def check_select_only(sql: str) -> None:
        """Requisito 3: levanta SelectOnlyError se o SQL não for um único comando SELECT."""
        if not sql:
            raise SelectOnlyError("O modelo não gerou nenhum SQL.")
        if not re.match(r"select\b", sql, re.IGNORECASE):
            # WITH ... SELECT também cai aqui, de propósito: no SQLite o WITH abre DELETE e UPDATE
            raise SelectOnlyError("Por segurança, só consultas SELECT podem ser executadas.")
        if ";" in sql:  # o ';' do final já foi tirado, então sobrar um é mais de um comando
            raise SelectOnlyError("Só é permitido um comando SQL por vez.")

    def validate_in_memory(self, sql: str) -> None:
        """Requisito 2: roda o SQL num SQLite em memória, com o mesmo esquema, para pegar as exceções."""
        # um banco novo por chamada: é barato (as tabelas estão vazias) e seguro com threads
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.executescript(self.schema)
            conn.execute(sql).fetchall()

    @staticmethod
    def _failure(msg: str, sql: str | None = None) -> dspy.Prediction:
        """Loga a recusa e devolve uma predição sem SQL, só com o erro."""
        if sql:
            logger.warning("%s SQL recusado: %s", msg, sql)
        else:
            logger.warning("%s", msg)
        return dspy.Prediction(sql_query=None, error=msg)


class StockApiClient:
    """Cliente HTTP da API do estoque: o único caminho do bot até o banco."""

    def __init__(self, base_url: str = API_URL, timeout: float = 10):
        self.base_url = base_url
        self.timeout = timeout

    def query(self, sql: str) -> dict:
        """Chama GET /consulta e devolve {"colunas": [...], "linhas": [[...], ...]}."""
        response = requests.get(f"{self.base_url}/consulta", params={"sql": sql}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class WhisperTranscriber:
    """Transcreve áudio em texto com o Whisper, carregando o modelo só na primeira vez."""

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio: str) -> str:
        """Transcreve o áudio (caminho ou URL do arquivo) e devolve o texto."""
        if self._model is None:
            self._model = whisper.load_model(self.model_name)
        return self._model.transcribe(audio)["text"].strip()


class Text2SQLAgent:
    """Da pergunta em linguagem natural à resposta em texto, sem saber nada de Telegram."""

    def __init__(self, generator: ReliableSQLGenerator, api: StockApiClient):
        self.generator = generator
        self.api = api

    def answer(self, question: str) -> str:
        """Gera o SQL, consulta o estoque pela API e devolve o texto da resposta."""
        pred = self.generator(question=question)  # __call__, nunca forward: a DSPy depende dele
        if pred.error:
            return f"Não consegui responder: {pred.error}"
        logger.info("SQL aceito: %s", pred.sql_query)

        try:
            result = self.api.query(pred.sql_query)
        except requests.RequestException as exc:
            # atenção: bool(Response) é False em erro HTTP, então compara com None
            if exc.response is not None and exc.response.status_code == 400:
                return f"A consulta falhou no banco: {self._error_detail(exc.response)}"
            logger.exception("Falha ao consultar a API do estoque")
            return "Não consegui consultar o estoque agora. Tente de novo em instantes."

        rows = result["linhas"]
        if not rows:
            return "Nenhum resultado encontrado."
        text = json.dumps(rows, ensure_ascii=False)
        if len(text) > TELEGRAM_MAX_CHARS:
            text = text[: TELEGRAM_MAX_CHARS - len(TRUNCATED_NOTICE)] + TRUNCATED_NOTICE
        return text

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        """Pega a mensagem de erro do SQLite que a API devolve em {"detail": "..."}."""
        try:
            return str(response.json()["detail"])
        except (ValueError, KeyError, TypeError):
            return response.text


class TelegramBot:
    """Liga o Telegram ao agente: recebe texto e áudio e responde com o que o agente devolver."""

    def __init__(self, token: str, agent: Text2SQLAgent, transcriber: WhisperTranscriber):
        self.bot = telebot.TeleBot(token)
        self.agent = agent
        self.transcriber = transcriber
        self.bot.register_message_handler(self.on_text, content_types=["text"])
        self.bot.register_message_handler(self.on_voice, content_types=["voice"])

    def on_text(self, message) -> None:
        """Responde a uma mensagem de texto."""
        try:
            self.bot.reply_to(message, self.agent.answer(message.text))
        except Exception:
            self._log_failure("Falha ao processar a mensagem de texto")
            self.bot.reply_to(message, FALLBACK_REPLY)

    def on_voice(self, message) -> None:
        """Responde a uma mensagem de voz: transcreve o áudio e trata o texto como pergunta."""
        try:
            url = self.bot.get_file_url(message.voice.file_id)  # a URL contém o token: nunca vai para o log
            text = self.transcriber.transcribe(url)
            logger.info("Áudio transcrito: %s", text)
            self.bot.reply_to(message, self.agent.answer(text))
        except Exception:
            self._log_failure("Falha ao processar a mensagem de voz")
            self.bot.reply_to(message, FALLBACK_REPLY)

    def _log_failure(self, context: str) -> None:
        """Loga a exceção em curso sem o token, que aparece na URL do áudio e nas da API do Telegram."""
        logger.error("%s\n%s", context, traceback.format_exc().replace(self.bot.token, "***"))

    def run(self) -> None:
        """Fica escutando o Telegram; o infinity_polling não para no primeiro erro de rede."""
        self.bot.infinity_polling()

# GEPA

TRAINSET = [
    ("Quantos produtos existem?", "SELECT COUNT(*) FROM produto"),
    ("Qual o preço do sabonete?", "SELECT preco FROM produto WHERE nome = 'sabonete'"),
    ("Quais produtos custam mais de 10 reais?", "SELECT nome, preco FROM produto WHERE preco > 10"),
    (
        "Quais são os produtos do departamento de bebidas?",
        "SELECT p.nome FROM produto p JOIN departamento d ON d.id = p.departamento_id WHERE d.nome = 'bebidas'",
    ),
    (
        "Quantos lotes de leite há no estoque?",
        "SELECT COUNT(*) FROM estoque e JOIN produto p ON p.id = e.produto_id WHERE p.nome = 'leite'",
    ),
    (
        "Qual produto tem a maior quantidade em estoque?",
        "SELECT p.nome, e.quantidade FROM estoque e JOIN produto p ON p.id = e.produto_id"
        " ORDER BY e.quantidade DESC LIMIT 1",
    ),
    (
        "Quais produtos vencem antes de 2027-01-01?",
        "SELECT p.nome, e.data_vencimento FROM estoque e JOIN produto p ON p.id = e.produto_id"
        " WHERE e.data_vencimento < '2027-01-01'",
    ),
    ("Qual a soma das unidades em estoque?", "SELECT SUM(quantidade) FROM estoque"),
    ("Qual o preço médio dos produtos?", "SELECT AVG(preco) FROM produto"),
    (
        "Quantos produtos cada departamento tem?",
        "SELECT d.nome, COUNT(p.id) FROM departamento d LEFT JOIN produto p ON p.departamento_id = d.id GROUP BY d.id",
    ),
]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.error:
        return dspy.Prediction(score=0.0, feedback=pred.error)
    esperado = " ".join(gold.sql_query.split()).lower()
    gerado = " ".join(pred.sql_query.split()).lower()
    if esperado == gerado:
        return dspy.Prediction(score=1.0, feedback="O SQL gerado é o esperado.")
    return dspy.Prediction(score=0.0, feedback=f"Esperado: {gold.sql_query}. Gerado: {pred.sql_query}.")


def optimize(generator: ReliableSQLGenerator) -> ReliableSQLGenerator:
    trainset = [
        dspy.Example(question=question, sql_query=sql).with_inputs("question")
        for question, sql in TRAINSET
    ]
    optimizer = dspy.GEPA(metric=metric, max_full_evals=2, reflection_lm=dspy.settings.lm)
    return optimizer.compile(generator, trainset=trainset)

# GEPA

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Defina a variável de ambiente TELEGRAM_BOT_TOKEN antes de iniciar o bot.")
    configure_lm()
    generator = optimize(ReliableSQLGenerator(SCHEMA))
    TelegramBot(token, Text2SQLAgent(generator, StockApiClient()), WhisperTranscriber()).run()


if __name__ == "__main__":
    main()
