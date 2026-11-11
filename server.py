import logging
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "lojas.db")

HOST = "127.0.0.1"
PORT = 8000

# Esquema do estoque: a única fonte da DDL. Cria o banco real, vai para o LLM como `dbschema`
# (por isso os comentários ficam) e monta o SQLite em memória que o gerador usa para validar o SQL.
SCHEMA = """\
CREATE TABLE IF NOT EXISTS departamento (
    id   INTEGER PRIMARY KEY,
    nome TEXT    NOT NULL UNIQUE  -- higiene, bebidas, laticínios, limpeza, mercearia ou padaria
);

CREATE TABLE IF NOT EXISTS produto (
    id              INTEGER PRIMARY KEY,
    nome            TEXT    NOT NULL UNIQUE,                        -- em minúsculas
    departamento_id INTEGER NOT NULL REFERENCES departamento (id),
    preco           REAL    NOT NULL                                -- preço unitário, em reais
);

-- cada linha de estoque é um lote de um produto
CREATE TABLE IF NOT EXISTS estoque (
    id              INTEGER PRIMARY KEY,
    produto_id      INTEGER NOT NULL REFERENCES produto (id),
    quantidade      INTEGER NOT NULL,                               -- unidades do lote
    data_fabricacao TEXT    NOT NULL,                               -- AAAA-MM-DD
    data_vencimento TEXT    NOT NULL                                -- AAAA-MM-DD
);
"""

# Dados iniciais, com os id explícitos para serem sempre os mesmos (a avaliação depende disso).
# (id, nome)
SEED_DEPARTAMENTO = [
    (1, "higiene"),
    (2, "bebidas"),
    (3, "laticínios"),
    (4, "limpeza"),
    (5, "mercearia"),
    (6, "padaria"),
]

# (id, nome, departamento_id, preco)
SEED_PRODUTO = [
    (1, "sabonete", 1, 2.49),
    (2, "shampoo", 1, 14.90),
    (3, "creme dental", 1, 5.99),
    (4, "desodorante", 1, 12.50),
    (5, "água", 2, 1.99),
    (6, "coca", 2, 8.99),
    (7, "suco de laranja", 2, 7.49),
    (8, "cerveja", 2, 4.29),
    (9, "leite", 3, 4.79),
    (10, "iogurte", 3, 3.49),
    (11, "queijo", 3, 39.90),
    (12, "manteiga", 3, 11.90),
    (13, "detergente", 4, 2.79),
    (14, "sabão em pó", 4, 19.90),
    (15, "água sanitária", 4, 5.49),
    (16, "arroz", 5, 27.90),
    (17, "feijão", 5, 8.49),
    (18, "café", 5, 18.90),
    (19, "macarrão", 5, 4.59),
    (20, "pão de forma", 6, 9.99),
    (21, "bolo", 6, 15.00),
]

# (id, produto_id, quantidade, data_fabricacao, data_vencimento); o nome do produto é só conferência
SEED_ESTOQUE = [
    (1, 1, 120, "2026-03-10", "2028-03-10"),  # sabonete
    (2, 2, 45, "2026-05-02", "2028-05-02"),  # shampoo
    (3, 3, 80, "2026-01-20", "2027-07-20"),  # creme dental
    (4, 4, 30, "2025-10-15", "2027-10-15"),  # desodorante
    (5, 5, 200, "2026-09-01", "2027-09-01"),  # água
    (6, 6, 150, "2026-08-20", "2027-02-20"),  # coca
    (7, 7, 40, "2026-10-05", "2026-12-05"),  # suco de laranja
    (8, 8, 96, "2026-07-01", "2027-01-01"),  # cerveja
    (9, 9, 60, "2026-10-20", "2027-01-20"),  # leite
    (10, 9, 25, "2026-08-15", "2026-11-15"),  # leite
    (11, 10, 35, "2026-10-25", "2026-11-25"),  # iogurte
    (12, 10, 10, "2026-09-20", "2026-10-20"),  # iogurte
    (13, 11, 18, "2026-10-01", "2026-12-01"),  # queijo
    (14, 12, 22, "2026-07-10", "2026-11-10"),  # manteiga
    (15, 13, 70, "2026-02-14", "2028-02-14"),  # detergente
    (16, 14, 33, "2026-04-01", "2028-04-01"),  # sabão em pó
    (17, 15, 50, "2026-06-01", "2027-06-01"),  # água sanitária
    (18, 16, 90, "2026-05-15", "2027-05-15"),  # arroz
    (19, 17, 75, "2026-06-20", "2027-06-20"),  # feijão
    (20, 18, 55, "2026-04-10", "2027-04-10"),  # café
    (21, 19, 65, "2026-03-01", "2027-09-01"),  # macarrão
    (22, 20, 20, "2026-11-05", "2026-11-19"),  # pão de forma
    (23, 21, 8, "2026-11-08", "2026-11-14"),  # bolo
]


class StockDatabase:
    """Dono do arquivo SQLite do estoque: cria, popula e consulta (somente leitura)."""

    def __init__(self, path: str = DB_PATH):
        self.path = os.path.abspath(path)

    def initialize(self) -> None:
        """Cria a pasta, o arquivo e as tabelas e, se o banco estiver vazio, insere os dados iniciais."""
        # o git não versiona pasta vazia: no clone, a database/ ainda não existe
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")  # o SQLite vem com as chaves estrangeiras desligadas
            conn.executescript(SCHEMA)
            if conn.execute("SELECT COUNT(*) FROM departamento").fetchone()[0] == 0:
                conn.executemany("INSERT INTO departamento (id, nome) VALUES (?, ?)", SEED_DEPARTAMENTO)
                conn.executemany(
                    "INSERT INTO produto (id, nome, departamento_id, preco) VALUES (?, ?, ?, ?)", SEED_PRODUTO
                )
                conn.executemany(
                    "INSERT INTO estoque (id, produto_id, quantidade, data_fabricacao, data_vencimento)"
                    " VALUES (?, ?, ?, ?, ?)",
                    SEED_ESTOQUE,
                )
                conn.commit()  # um commit só no fim: ou entram as três tabelas, ou nenhuma
            totals = [
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("departamento", "produto", "estoque")
            ]
        logger.info("Banco do estoque pronto em %s: %d departamentos, %d produtos, %d lotes", self.path, *totals)

    def query(self, sql: str) -> dict:
        """Executa o SQL numa conexão nova e somente leitura. Erros do SQLite sobem para quem chamou."""
        # uma conexão por chamada: o FastAPI roda as rotas `def` em threads e a conexão não pode ser compartilhada
        # as_uri() porque um caminho do Windows (com \, espaço ou acento) não funciona cru numa URI
        uri = Path(self.path).as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            cursor = conn.execute(sql)
            return {
                "colunas": [column[0] for column in (cursor.description or [])],
                "linhas": [list(row) for row in cursor.fetchall()],
            }


def create_app(database: StockDatabase) -> FastAPI:
    """Monta a API do estoque, que expõe uma única rota: GET /consulta."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()  # roda quando o servidor sobe, com `python server.py` ou `uvicorn server:app`
        yield

    app = FastAPI(lifespan=lifespan)

    # `def` e não `async def`: o sqlite3 bloqueia, então o FastAPI roda a rota numa thread
    @app.get("/consulta")
    def consulta(sql: str = Query(..., min_length=1, description="Comando SELECT a executar no estoque")):
        try:
            return database.query(sql)
        except (sqlite3.Error, sqlite3.Warning) as exc:  # Warning: "mais de um comando" no Python até o 3.11
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app(StockDatabase())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
