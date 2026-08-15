"""
Cleans the raw NTSB CSV and splits it into two output streams:
  1. accidents_structured.csv  -> for the SQL database (tabular data)
  2. narratives.jsonl          -> for the RAG (report text, one document per line)
"""

import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "ntsb_final_reports_2016_2023.csv"
STRUCTURED_OUT = BASE_DIR / "data" / "processed" / "accidents_structured.csv"
NARRATIVES_OUT = BASE_DIR / "data" / "processed" / "narratives.jsonl"

# Long-text columns used to build the RAG documents.
# Every other column in the CSV automatically goes into the structured (SQL) stream.
TEXT_COLUMNS = ["ProbableCause", "Findings", "rep_text"]

# Technical/irrelevant columns excluded from both streams
DROP_COLUMNS = ["rep_num_jsg"]

# Column used as the unique record identifier
ID_COLUMN = "NtsbNo"


def load_raw_data() -> pd.DataFrame:
    print("Loading raw CSV...")
    # The NTSB file is semicolon-separated
    df = pd.read_csv(RAW_PATH, sep=";", low_memory=False)
    print(f"{len(df)} rows loaded, {len(df.columns)} columns.")
    print(f"Detected columns: {list(df.columns)}")
    return df


def clean_structured(df: pd.DataFrame) -> pd.DataFrame:
    # Structured columns = all CSV columns, minus the long-text and technical columns
    available_cols = [
        c for c in df.columns
        if c not in TEXT_COLUMNS and c not in DROP_COLUMNS
    ]
    print(f"Structured columns kept ({len(available_cols)}): {available_cols}")

    structured = df[available_cols].copy()

    # Normalize missing values
    structured = structured.replace(["NA", "N/A", "", " "], pd.NA)

    # Parse the date
    if "EventDate" in structured.columns:
        structured["EventDate"] = pd.to_datetime(
            structured["EventDate"], errors="coerce"
        ).dt.date

    # Injury counters: NaN -> 0, cast to int
    injury_cols = [
        "FatalInjuryCount", "SeriousInjuryCount",
        "MinorInjuryCount", "OnGroundInjuryCount",
    ]
    for col in injury_cols:
        if col in structured.columns:
            structured[col] = pd.to_numeric(
                structured[col], errors="coerce"
            ).fillna(0).astype(int)

    # Drop rows without a usable identifier
    id_col = ID_COLUMN
    structured = structured.dropna(subset=[id_col])
    structured = structured.drop_duplicates(subset=[id_col])

    print(f"{len(structured)} rows kept after structured cleaning.")
    return structured


def build_narratives(df: pd.DataFrame) -> list[dict]:
    
    """
    Builds one RAG document per row by concatenating ProbableCause, Findings,
    and rep_text (skips rows with less than 50 chars of combined text).
    Each document also stores metadata (date, location, aircraft, fatalities,
    report URL) alongside the text, for filtering/display later on.s
    """

    id_col = ID_COLUMN
    docs = []

    for _, row in df.iterrows():
        doc_id = row.get(id_col)
        if pd.isna(doc_id):
            continue

        # Assemble the narrative text from the available text columns
        parts = []
        for col in TEXT_COLUMNS:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    parts.append(str(val).strip())

        full_text = "\n\n".join(parts)
        if not full_text or len(full_text) < 50:
            continue  # skip near-empty reports, not useful for the RAG

        docs.append({
            "doc_id": str(doc_id),
            "text": full_text,
            "metadata": {
                "event_date": str(row.get("EventDate", "")),
                "city": row.get("City", ""),
                "state": row.get("State", ""),
                "make": row.get("Make", ""),
                "model": row.get("Model", ""),
                "broad_phase_of_flight": row.get("BroadPhaseofFlight", ""),
                "fatal_injury_count": int(row.get("FatalInjuryCount", 0))
                    if pd.notna(row.get("FatalInjuryCount", 0)) else 0,
                "report_url": row.get("ReportUrl", ""),
            },
        })

    print(f"{len(docs)} narrative documents built for the RAG.")
    return docs


def save_structured(df: pd.DataFrame) -> None:
    STRUCTURED_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STRUCTURED_OUT, index=False)
    print(f"Saved: {STRUCTURED_OUT}")


def save_narratives(docs: list[dict]) -> None:
    NARRATIVES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(NARRATIVES_OUT, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Saved: {NARRATIVES_OUT}")


if __name__ == "__main__":
    raw_df = load_raw_data()
    structured_df = clean_structured(raw_df)
    narratives = build_narratives(raw_df)

    save_structured(structured_df)
    save_narratives(narratives)

    print("\nCleaning complete.")