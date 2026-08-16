"""
Text-to-SQL tool: turns a natural-language question into a SQL query
against the "accidents" table, with safety guardrails, and executes it.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.rag_chain import get_llm

BASE_DIR = Path(__file__).resolve().parents[2]
SCHEMA_PATH = BASE_DIR / "src" / "sql" / "schema.sql"

load_dotenv()

# Only SELECT queries are ever allowed to run - these keywords are rejected
# outright, regardless of what the LLM generates.
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "TRUNCATE", "GRANT", "REVOKE",
]

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found. Check your .env file.")
        _engine = create_engine(database_url)
    return _engine


SQL_PROMPT = """You are a SQL expert. Given the table schema below, write a \
single PostgreSQL SELECT query that answers the question. Only output the \
raw SQL query, nothing else - no explanation, no markdown code fences.

Guidelines:
- For "deadliest"/"worst"/"most severe" questions, ORDER BY fatal_injury_count \
DESC (or the relevant severity column) and LIMIT the results.
- Always include identifying columns in the SELECT (e.g. ntsb_no, event_date, \
city, state, make, model, fatal_injury_count), so the result is readable on \
its own without needing to look up the report separately.
- The "make" and "model" columns are not normalized (e.g. a Learjet may be \
stored as 'LEARJET', 'LEARJET INC 45', or 'GATES LEARJET CORP 55'). Always \
match them with ILIKE '%keyword%' instead of an exact equality check.

Schema:
{schema}

Question: {question}
SQL query:"""


def generate_sql(question: str) -> str:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    llm = get_llm()
    prompt = SQL_PROMPT.format(schema=schema, question=question)
    response = llm.invoke(prompt)

    # Strip markdown fences in case the model adds them anyway
    query = response.content.strip()
    query = re.sub(r"^```sql\s*|```$", "", query, flags=re.IGNORECASE).strip()
    return query


def validate_sql(query: str) -> None:
    """Raises an error if the query isn't a safe, single read-only SELECT."""
    stripped = query.strip().rstrip(";")

    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed.")

    upper_query = stripped.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_query):
            raise ValueError(f"Forbidden keyword detected: {keyword}")


def run_sql(query: str) -> list[dict]:
    validate_sql(query)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = [dict(row._mapping) for row in result]
    return rows


def answer_with_sql(question: str) -> dict:
    """Full pipeline: generate SQL from the question, validate it, run it."""
    query = generate_sql(question)
    rows = run_sql(query)
    return {"question": question, "sql_query": query, "rows": rows}


if __name__ == "__main__":
    result = answer_with_sql("How many accidents happened in Alaska?")
    print("SQL:", result["sql_query"])
    print("Rows:", result["rows"])
