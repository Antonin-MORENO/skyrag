"""
Classifies a natural-language question to decide which backend should
answer it: the SQL database (counts, filters, aggregates over structured
fields) or the RAG (narrative, "why/how" questions about specific reports).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.rag_chain import get_llm

ROUTER_PROMPT = """You are a router for an aviation accident assistant. \
Decide how to answer the user's question, based on two available tools:

- SQL: a structured database of accidents (dates, locations, aircraft make/\
model, injury counts, weather, phase of flight...). Use this for counts, \
statistics, filters, rankings, or "how many/which/list" questions.
- RAG: a search engine over the full narrative text of accident reports \
(causes, findings, what happened). Use this for "why/what caused/what \
happened" questions about specific events.
- BOTH: use this only if the question clearly needs a statistic AND an \
explanation (e.g. "what's the deadliest Cessna accident and why did it happen").

Respond with exactly one word: SQL, RAG, or BOTH.

Question: {question}
Answer:"""


def route_question(question: str) -> str:
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    route = response.content.strip().upper()

    if route not in ("SQL", "RAG", "BOTH"):
        # Default to RAG if the model didn't answer cleanly - it's the
        # safer fallback (a narrative answer is rarely actively wrong).
        return "RAG"
    return route


if __name__ == "__main__":
    test_questions = [
        "How many fatal accidents involved a Cessna?",
        "Why did the Learjet crash at Teterboro?",
        "What's the deadliest accident in Alaska and what caused it?",
    ]
    for q in test_questions:
        print(f"{route_question(q):6} <- {q}")
