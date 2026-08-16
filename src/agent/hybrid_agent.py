"""
Hybrid agent: routes a question to SQL, RAG, or both, and returns a
unified answer. This is the main entry point used by the dashboard.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent.router import route_question
from agent.sql_tool import answer_with_sql
from rag.rag_chain import answer_question


def format_sql_result(sql_result: dict) -> str:
    rows = sql_result["rows"]
    if not rows:
        return "The query returned no results."
    if len(rows) == 1 and len(rows[0]) == 1:
        # Single aggregate value (e.g. COUNT(*)) - just return it directly
        return str(list(rows[0].values())[0])
    preview = rows[:10]
    lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in preview]
    suffix = f"\n... and {len(rows) - 10} more rows" if len(rows) > 10 else ""
    return "\n".join(lines) + suffix


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

    # BOTH
    sql_result = answer_with_sql(question)
    rag_result = answer_question(question, top_k=top_k)
    combined = (
        f"Statistic: {format_sql_result(sql_result)}\n\n"
        f"Context: {rag_result['answer']}"
    )
    return {
        "route": "BOTH",
        "answer": combined,
        "sql_query": sql_result["sql_query"],
        "sql_rows": sql_result["rows"],
        "sources": [c["doc_id"] for c in rag_result["chunks"]],
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
