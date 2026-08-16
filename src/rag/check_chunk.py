"""
Diagnostic script: checks whether a specific doc_id's chunks exist in the
Chroma collection, and what their actual distance is to a given query
(even if they didn't make the top-k cut).
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "accident_reports"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TARGET_DOC_ID = "CEN17MA183"
QUERY = "Learjet accident Teterboro unstabilized approach"


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Total chunks in collection: {collection.count()}")

    # 1. Check the target doc's chunks actually exist, and inspect their text
    matches = collection.get(where={"doc_id": TARGET_DOC_ID})
    print(f"\nChunks found for doc_id='{TARGET_DOC_ID}': {len(matches['ids'])}")
    for cid, doc in zip(matches["ids"], matches["documents"]):
        print(f"\n  [{cid}]")
        print(f"  {doc[:250]}")

    if not matches["ids"]:
        print("\n⚠️ This doc_id has NO chunks in the collection at all — "
              "it may be missing from the rebuild.")
        return

    # 2. Embed the query and directly compute distance to this doc's chunks
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([QUERY]).tolist()

    # Ask Chroma for a large top_k (capped, SQLite has a variable-count limit)
    n_results = min(collection.count(), 3000)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    all_ids = results["ids"][0]
    all_distances = results["distances"][0]

    for cid in matches["ids"]:
        if cid in all_ids:
            rank = all_ids.index(cid) + 1
            distance = all_distances[all_ids.index(cid)]
            print(f"\n  {cid}: rank={rank}/{n_results}, distance={distance:.4f}")
        else:
            print(f"\n  {cid}: not found within top {n_results} results "
                  f"(out of {collection.count()} total chunks)")


if __name__ == "__main__":
    main()
