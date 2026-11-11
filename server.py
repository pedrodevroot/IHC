import sqlite3
from fastapi import FastAPI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
db_path = os.path.join(BASE_DIR, "lojas.db")

def create_db():
  conn = sqlite3.connect(db_path)
  c = conn.cursor()

  # Create tables
  c.execute("""CREATE TABLE IF NOT EXISTS produtos (
                nome TEXT, 
                departamento TEXT
            )""")

  c.executemany("INSERT INTO produtos VALUES (?, ?)", [
    ("sabonete", "higiene"),
    ("agua", "bebidas"),
    ("coca", "bebidas"),
  ])

  conn.commit()
  conn.close()

create_db()

conn = sqlite3.connect(db_path)
results = conn.execute("SELECT * from produtos").fetchall()
print(results)