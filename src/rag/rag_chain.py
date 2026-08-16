"""
Core RAG chain: retrieves relevant chunks from Chroma, then generates an
answer with a Groq-hosted LLM. Reused by the evaluation pipeline, the
hybrid agent, and later the Streamlit dashboard.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_COLLECTION_NAME = "accident_reports"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"

load_dotenv()

_embedding_model = None
_qdrant_client = None
_llm = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise ValueError("QDRANT_URL / QDRANT_API_KEY not found. Check your .env file.")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        _llm = ChatGroq(model=LLM_MODEL_NAME, api_key=api_key, temperature=0)
    return _llm


def _invoke_llm_with_retry(prompt: str, max_retries: int = 5) -> str:
    """Calls the LLM with retry/backoff on Groq's free-tier rate limit.
    Shared by both the query-rewriting and answer-generation steps, since
    each question now makes two LLM calls and either can hit the limit."""
    llm = get_llm()
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except RateLimitError:
            wait_seconds = 20 * (attempt + 1)
            print(f"  Rate limit hit, waiting {wait_seconds}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
    raise RuntimeError("Groq rate limit still exceeded after all retries.")


REWRITE_PROMPT = """Rewrite the question below into a short, keyword-dense \
search query for a semantic search engine over aviation accident reports. \
Include any aircraft type, location, and event type mentioned. Output ONLY \
the rewritten query, nothing else.

Question: {question}
Search query:"""


def rewrite_query_for_search(question: str) -> str:
    """Turns a natural-language question into a keyword-rich search query.
    The embedding model used here is lightweight and was found to retrieve
    noticeably better on keyword-dense queries than on short natural
    questions, so this rewriting step runs before every retrieval."""
    prompt = REWRITE_PROMPT.format(question=question)
    return _invoke_llm_with_retry(prompt).strip()


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Returns the top_k most relevant chunks for the query, with metadata.
    Uses Qdrant Cloud, with a higher search_ef than the default for better
    recall (see migrate_to_qdrant.py for the collection's HNSW config)."""
    model = get_embedding_model()
    client = get_qdrant_client()

    query_embedding = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        search_params={"hnsw_ef": 256},
    ).points

    chunks = []
    for point in results:
        chunks.append({
            "text": point.payload["text"],
            "doc_id": point.payload["doc_id"],
            "distance": 1 - point.score,  # Qdrant returns cosine similarity, not distance
        })
    return chunks


PROMPT_TEMPLATE = """You are an aviation safety analyst assistant. Answer the \
question using ONLY the context below, extracted from NTSB accident reports. \
If the context doesn't contain the answer, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


MAX_CHARS_PER_CHUNK_IN_PROMPT = 600


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generates an answer from the question and retrieved chunks.
    Each chunk's text is truncated before being sent to the LLM to stay
    well under Groq's free-tier tokens/minute limit - the full,
    untruncated chunks are still what gets recorded for RAGAS evaluation,
    only the generation prompt itself is shortened."""
    context = "\n\n---\n\n".join(
        c["text"][:MAX_CHARS_PER_CHUNK_IN_PROMPT] for c in chunks
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return _invoke_llm_with_retry(prompt)


def answer_question(question: str, top_k: int = 5) -> dict:
    """Full pipeline: rewrite the question into a search query, retrieve
    chunks with it, then generate an answer using the original question."""
    search_query = rewrite_query_for_search(question)
    chunks = retrieve(search_query, top_k=top_k)
    answer = generate_answer(question, chunks)
    return {
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "chunks": chunks,
    }


if __name__ == "__main__":
    result = answer_question("What caused the fatal Learjet accident at Teterboro?")
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:", [c["doc_id"] for c in result["chunks"]])
