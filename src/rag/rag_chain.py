"""
Core RAG chain: retrieves relevant chunks from Chroma, then generates an
answer with a Groq-hosted LLM. Reused by the evaluation pipeline, the
hybrid agent, and later the Streamlit dashboard.
"""

import os
import time
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "accident_reports"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"

load_dotenv()

_embedding_model = None
_llm = None

# In-memory cache of the full corpus: exact search, no ANN approximation.
# Chroma's default HNSW index was found to have poor recall at small top_k
# on this corpus (verified: the true nearest match was regularly missing
# from results even at top_k=20), so we bypass it and search the stored
# embeddings directly with numpy instead.
_corpus_embeddings = None
_corpus_documents = None
_corpus_metadatas = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _load_corpus() -> None:
    """Loads every embedding/document/metadata from Chroma into memory once.
    Paginated in batches, since fetching everything in a single call hits
    SQLite's bound-variable limit on a corpus this size."""
    global _corpus_embeddings, _corpus_documents, _corpus_metadatas

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    batch_size = 5000
    total = collection.count()

    all_embeddings, all_documents, all_metadatas = [], [], []
    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        all_embeddings.extend(batch["embeddings"])
        all_documents.extend(batch["documents"])
        all_metadatas.extend(batch["metadatas"])

    _corpus_embeddings = np.array(all_embeddings, dtype=np.float32)
    _corpus_documents = all_documents
    _corpus_metadatas = all_metadatas


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        _llm = ChatGroq(model=LLM_MODEL_NAME, api_key=api_key, temperature=0)
    return _llm


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Returns the top_k most relevant chunks for the query, with metadata.
    Uses exact cosine similarity over the full corpus (see module docstring
    on why this replaces Chroma's approximate HNSW search)."""
    if _corpus_embeddings is None:
        _load_corpus()

    model = get_embedding_model()
    query_embedding = model.encode([query])[0].astype(np.float32)

    # Cosine similarity = dot product of L2-normalized vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    corpus_norm = _corpus_embeddings / np.linalg.norm(
        _corpus_embeddings, axis=1, keepdims=True
    )
    similarities = corpus_norm @ query_norm

    top_indices = np.argsort(-similarities)[:top_k]

    chunks = []
    for i in top_indices:
        chunks.append({
            "text": _corpus_documents[i],
            "doc_id": _corpus_metadatas[i]["doc_id"],
            "distance": float(1 - similarities[i]),  # cosine distance, for consistency with before
        })
    return chunks


PROMPT_TEMPLATE = """You are an aviation safety analyst assistant. Answer the \
question using ONLY the context below, extracted from NTSB accident reports. \
If the context doesn't contain the answer, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict], max_retries: int = 5) -> str:
    """Generates an answer from the question and retrieved chunks.
    Retries with backoff if Groq's free-tier rate limit (tokens/minute)
    is hit, since evaluation runs many requests in a short span."""
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = get_llm()
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except RateLimitError:
            wait_seconds = 15 * (attempt + 1)
            print(f"  Rate limit hit, waiting {wait_seconds}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
    raise RuntimeError("Groq rate limit still exceeded after all retries.")


def answer_question(question: str, top_k: int = 5) -> dict:
    """Full pipeline: retrieve chunks, then generate an answer from them."""
    chunks = retrieve(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return {"question": question, "answer": answer, "chunks": chunks}


if __name__ == "__main__":
    result = answer_question("What caused the fatal Learjet accident at Teterboro?")
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:", [c["doc_id"] for c in result["chunks"]])
