"""
Loads the history of evaluation runs (data/eval/runs/*.json) into a
DataFrame, for the dashboard's Evaluation tab to plot trends over time.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = BASE_DIR / "data" / "eval" / "runs"


def load_run_history() -> pd.DataFrame:
    """Returns one row per evaluation run, sorted chronologically."""
    rows = []
    for run_path in sorted(RUNS_DIR.glob("run_*.json")):
        with open(run_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        row = {
            "run_id": data["run_id"],
            "n_questions": data["n_questions"],
            "retrieval_hit_rate": data["retrieval_hit_rate"],
            **data["ragas_scores"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("run_id").reset_index(drop=True)
    return df
