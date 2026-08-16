"""
Quick manual test of the RAG retriever: embeds a query and returns the
most similar chunks, using the exact (brute-force) search from rag_chain.
Used to sanity-check retrieval quality before running the RAGAS pipeline.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from rag_chain import retrieve


def search(query: str, top_k: int = 5):
    chunks = retrieve(query, top_k=top_k)

    print(f"\nQuery: {query}\n{'-' * 60}")
    for i, c in enumerate(chunks):
        print(f"\n[{i + 1}] doc_id={c['doc_id']} | distance={c['distance']:.4f}")
        print(c["text"][:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    search("engine failure during takeoff")
    search("pilot lost control after bird strike")
    search("Learjet accident Teterboro unstabilized approach", top_k=20)
