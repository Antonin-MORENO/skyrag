"""
Migrates all chunks (embeddings + text + metadata) from the local Chroma
vector store to a Qdrant Cloud collection. This is a one-time transfer -
no re-embedding needed, since we reuse the vectors already computed.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
CHROMA_COLLECTION_NAME = "accident_reports"
QDRANT_COLLECTION_NAME = "accident_reports"

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size
CHROMA_BATCH_SIZE = 5000  # SQLite bound-variable limit workaround (see rag_chain.py)
QDRANT_UPLOAD_BATCH_SIZE = 256

load_dotenv()


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("QDRANT_URL / QDRANT_API_KEY not found. Check your .env file.")
    return QdrantClient(url=url, api_key=api_key)


def load_all_chunks_from_chroma() -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    total = collection.count()
    print(f"Loading {total} chunks from Chroma...")

    all_chunks = []
    for offset in tqdm(range(0, total, CHROMA_BATCH_SIZE)):
        batch = collection.get(
            limit=CHROMA_BATCH_SIZE,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        for chunk_id, embedding, document, metadata in zip(
            batch["ids"], batch["embeddings"], batch["documents"], batch["metadatas"]
        ):
            all_chunks.append({
                "id": chunk_id,
                "embedding": embedding,
                "text": document,
                "metadata": metadata,
            })
    return all_chunks


def create_qdrant_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists, recreating it...")
        client.delete_collection(QDRANT_COLLECTION_NAME)

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        # Higher-quality HNSW construction than Chroma's defaults, since a
        # poorly tuned index was found to hurt recall at small top_k earlier.
        hnsw_config={"m": 32, "ef_construct": 200},
    )
    print(f"Created Qdrant collection '{QDRANT_COLLECTION_NAME}'.")


def upload_chunks(client: QdrantClient, chunks: list[dict]) -> None:
    print(f"Uploading {len(chunks)} chunks to Qdrant...")
    for i in tqdm(range(0, len(chunks), QDRANT_UPLOAD_BATCH_SIZE)):
        batch = chunks[i:i + QDRANT_UPLOAD_BATCH_SIZE]
        points = [
            PointStruct(
                id=i + j,  # Qdrant point IDs must be int or UUID, not the original chunk_id string
                vector=c["embedding"],
                payload={
                    "chunk_id": c["id"],
                    "text": c["text"],
                    "doc_id": c["metadata"]["doc_id"],
                    **c["metadata"],
                },
            )
            for j, c in enumerate(batch)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


if __name__ == "__main__":
    chunks = load_all_chunks_from_chroma()

    qdrant = get_qdrant_client()
    create_qdrant_collection(qdrant)
    upload_chunks(qdrant, chunks)

    count = qdrant.count(QDRANT_COLLECTION_NAME).count
    print(f"\nMigration complete: {count} points in Qdrant collection '{QDRANT_COLLECTION_NAME}'.")
