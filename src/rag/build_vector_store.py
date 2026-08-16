"""
Builds the RAG vector store:
  1. Loads narratives.jsonl
  2. Splits each report into overlapping text chunks
  3. Embeds the chunks locally with sentence-transformers
  4. Stores everything in a persistent local Chroma collection

This is a one-time (or "rebuild when data changes") indexing script.
Run it once before using the retriever.
"""

import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
NARRATIVES_PATH = BASE_DIR / "data" / "processed" / "narratives.jsonl"
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "accident_reports"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 64
DB_WRITE_BATCH_SIZE = 500


def load_narratives() -> list[dict]:
    print("Loading narratives.jsonl...")
    docs = []
    with open(NARRATIVES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"{len(docs)} source documents loaded.")
    return docs


def build_metadata_prefix(metadata: dict) -> str:
    """
    Turns a document's metadata into a short text prefix, so that
    location/aircraft/date information is embedded alongside the narrative
    and can be matched by semantic search (it otherwise never appears in
    the report text itself).
    """
    parts = []
    city, state = metadata.get("city"), metadata.get("state")
    if city or state:
        parts.append(f"Location: {city or '?'}, {state or '?'}.")
    make, model = metadata.get("make"), metadata.get("model")
    if make or model:
        parts.append(f"Aircraft: {make or '?'} {model or '?'}.")
    if metadata.get("event_date"):
        parts.append(f"Date: {metadata['event_date']}.")
    if metadata.get("broad_phase_of_flight"):
        parts.append(f"Phase of flight: {metadata['broad_phase_of_flight']}.")
    return " ".join(parts)


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Splits each document's text into overlapping chunks. Each chunk is
    prefixed with a short summary of its metadata (location, aircraft,
    date, phase of flight), since that information otherwise only lives
    in the metadata dict and is invisible to semantic search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        prefix = build_metadata_prefix(doc["metadata"])
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunk_text = f"{prefix}\n\n{piece}" if prefix else piece
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk{i}",
                "doc_id": doc["doc_id"],
                "text": chunk_text,
                "metadata": doc["metadata"],
            })

    print(f"{len(chunks)} chunks created from {len(docs)} documents "
          f"(avg {len(chunks) / len(docs):.1f} chunks/doc).")
    return chunks


def build_vector_store(chunks: list[dict]) -> None:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Start fresh each time this script runs, to avoid duplicate/stale chunks
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(COLLECTION_NAME)

    print("Embedding and storing chunks...")
    for i in tqdm(range(0, len(chunks), DB_WRITE_BATCH_SIZE)):
        batch = chunks[i:i + DB_WRITE_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = model.encode(
            texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False
        ).tolist()

        # Chroma metadata values must be str/int/float/bool, not dicts,
        # so we flatten each chunk's nested metadata dict.
        metadatas = [
            {"doc_id": c["doc_id"], **c["metadata"]} for c in batch
        ]

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"Vector store built: {collection.count()} chunks stored at {CHROMA_DIR}")


if __name__ == "__main__":
    narratives = load_narratives()
    chunks = chunk_documents(narratives)
    build_vector_store(chunks)
