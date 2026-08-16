"""
Hybrid agent: routes a question to SQL, RAG, or both, and returns a
unified answer. This is the main entry point used by the dashboard.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent.router import route_question
from agent.sql_tool import answer_with_sql
from rag.build_vector_store import build_metadata_prefix
from rag.rag_chain import answer_question, generate_answer, retrieve


def _humanize_label(key: str) -> str:
    """Turns a column name like 'ntsb_no' into a readable label 'NTSB No'."""
    special = {"ntsb_no": "NTSB No", "sql": "SQL"}
    if key in special:
        return special[key]
    return key.replace("_", " ").title()


def _humanize_value(value) -> str:
    """Strips the redundant '00:00:00' from date-only values."""
    text = str(value)
    if text.endswith(" 00:00:00"):
        return text[: -len(" 00:00:00")]
    return text


def format_sql_result(sql_result: dict) -> str:
    rows = sql_result["rows"]
    if not rows:
        return "The query returned no results."

    if len(rows) == 1 and len(rows[0]) == 1:
        # Single aggregate value (e.g. COUNT(*)) - just return it directly
        return str(list(rows[0].values())[0])

    if len(rows) == 1:
        # Single detailed record - a clean bullet list reads much better
        # than one long comma-separated line.
        row = rows[0]
        return "\n".join(f"- **{_humanize_label(k)}:** {_humanize_value(v)}" for k, v in row.items())

    # Multiple rows - render as a markdown table
    preview = rows[:10]
    columns = list(preview[0].keys())
    header = "| " + " | ".join(_humanize_label(c) for c in columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body_lines = [
        "| " + " | ".join(_humanize_value(row.get(c, "")) for c in columns) + " |"
        for row in preview
    ]
    suffix = f"\n\n_...and {len(rows) - 10} more rows_" if len(rows) > 10 else ""
    return "\n".join([header, separator, *body_lines]) + suffix


def ask(question: str, top_k: int = 5) -> dict:
    """Routes the question and returns a dict with the route taken,
    the final answer text, and the raw SQL/RAG details for transparency."""
    route = route_question(question)

    if route == "SQL":
        sql_result = answer_with_sql(question)
        return {
            "route": "SQL",
            "answer": format_sql_result(sql_result),
            "sql_query": sql_result["sql_query"],
            "sql_rows": sql_result["rows"],
        }

    if route == "RAG":
        rag_result = answer_question(question, top_k=top_k)
        return {
            "route": "RAG",
            "answer": rag_result["answer"],
            "sources": [c["doc_id"] for c in rag_result["chunks"]],
        }

    # BOTH: run SQL first, then use what SQL found to build a retrieval
    # query - in the exact same "Location: ... Aircraft: ... Date: ..."
    # format used as the metadata prefix when the chunks were embedded
    # (see build_vector_store.py), which matches far better than mixing
    # the question sentence with the facts.
    sql_result = answer_with_sql(question)
    rag_query = question
    if sql_result["rows"]:
        row = sql_result["rows"][0]
        prefix = build_metadata_prefix(row)
        if prefix:
            rag_query = prefix

    rag_chunks = retrieve(rag_query, top_k=max(top_k, 10))
    rag_answer = generate_answer(question, rag_chunks)  # original question, for natural phrasing
    combined = (
        f"**📊 Key facts**\n\n{format_sql_result(sql_result)}\n\n"
        f"---\n\n"
        f"**📝 What likely happened**\n\n{rag_answer}"
    )
    return {
        "route": "BOTH",
        "answer": combined,
        "sql_query": sql_result["sql_query"],
        "sql_rows": sql_result["rows"],
        "sources": [c["doc_id"] for c in rag_chunks],
    }


if __name__ == "__main__":
    questions = [
        "How many accidents involved a Cessna aircraft?",
        "Why did the Learjet crash at Teterboro?",
    ]
    for q in questions:
        result = ask(q)
        print(f"\nQ: {q}")
        print(f"Route: {result['route']}")
        print(f"A: {result['answer']}")
