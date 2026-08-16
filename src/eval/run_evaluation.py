"""
Evaluates the RAG pipeline against the golden dataset using RAGAS.

For each question:
  1. Retrieve chunks + generate an answer (via rag_chain)
  2. Score the result with RAGAS metrics (faithfulness, answer relevancy,
     context precision, context recall) against the reference answer

Results are saved per-run (with a timestamp) so scores can be compared
over time as the pipeline changes (chunk size, prompt, model, etc.).
"""

import json
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- Workaround for a known ragas bug ---
# ragas unconditionally imports ChatVertexAI from an old langchain_community
# path that was removed in recent langchain_community versions. We never use
# VertexAI (we use Groq), so we register a harmless stub module to satisfy
# that import instead of downgrading the whole langchain ecosystem.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - placeholder, never instantiated
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("VertexAI is not used in this project.")

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# --- end workaround ---

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.rag_chain import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, answer_question

BASE_DIR = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = BASE_DIR / "data" / "eval" / "golden_dataset.json"
RUNS_DIR = BASE_DIR / "data" / "eval" / "runs"

load_dotenv()


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_on_golden_dataset(golden: list[dict]) -> list[dict]:
    """Runs the RAG pipeline on every golden question, collecting everything
    RAGAS needs: the question, the generated answer, the retrieved contexts,
    and the reference (ground truth) answer."""
    records = []
    for item in golden:
        print(f"  - {item['id']}: {item['question'][:60]}...")
        result = answer_question(item["question"], top_k=5)
        time.sleep(5)  # preventive throttling to stay under Groq's free-tier TPM limit
        records.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["chunks"]],
            "ground_truth": item["reference_answer"],
            "source_doc_id": item["source_doc_id"],
            "retrieved_doc_ids": [c["doc_id"] for c in result["chunks"]],
        })
    return records


def run_ragas_evaluation(records: list[dict]):
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=LLM_MODEL_NAME, api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )

    # Groq's API rejects requests with n > 1, so strictness=1 disables
    # RAGAS's default multi-sample self-consistency check for answer_relevancy.
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    # Groq's free tier has a strict rate limit, so we run few requests at a
    # time and allow a generous timeout instead of RAGAS's default high concurrency.
    run_config = RunConfig(max_workers=2, timeout=180)

    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=run_config,
    )
    return result


def check_retrieval_hit(records: list[dict]) -> float:
    """Simple retrieval-only sanity check: fraction of questions where the
    known source document was actually retrieved (independent of RAGAS)."""
    hits = sum(
        1 for r in records if r["source_doc_id"] in r["retrieved_doc_ids"]
    )
    return hits / len(records)


def save_run(records: list[dict], ragas_result, retrieval_hit_rate: float) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = RUNS_DIR / f"run_{timestamp}.json"

    scores = ragas_result.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()

    output = {
        "run_id": timestamp,
        "date": timestamp,
        "n_questions": len(records),
        "retrieval_hit_rate": retrieval_hit_rate,
        "ragas_scores": scores,
        "per_question_details": records,
    }

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nRun saved: {run_path}")
    return run_path


if __name__ == "__main__":
    print("Loading golden dataset...")
    golden = load_golden_dataset()

    print(f"Running RAG pipeline on {len(golden)} questions...")
    records = run_rag_on_golden_dataset(golden)

    retrieval_hit_rate = check_retrieval_hit(records)
    print(f"\nRetrieval hit rate (source doc found in top-5): {retrieval_hit_rate:.2%}")

    print("\nRunning RAGAS evaluation (this calls the LLM multiple times)...")
    ragas_result = run_ragas_evaluation(records)

    print("\nRAGAS scores:")
    print(ragas_result)

    save_run(records, ragas_result, retrieval_hit_rate)
