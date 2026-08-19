# ✈️ SkyRAG — Aviation Safety Assistant

A hybrid RAG and SQL assistant over 7,462 NTSB aviation accident reports (2016 to 2023), with a rigorous, testable evaluation protocol and a Streamlit dashboard.

Ask it "why did this accident happen?" and it searches accident narratives semantically. Ask it "how many accidents involved a Cessna?" and it queries a structured database. Ask something that needs both, and it combines them.

## Why this project

Most "RAG demo" projects wire an LLM to some PDFs and stop there. This one instead treats the RAG as a system to be measured, debugged, and improved, with a golden dataset, automated evaluation (RAGAS), and a documented history of what broke, why, and how it was fixed.

## Features

1. **Hybrid agent**: an LLM router classifies each question and dispatches it to a SQL tool, a RAG pipeline, or both.
2. **RAG pipeline**: chunking, embeddings (`all-MiniLM-L6-v2`), a Qdrant Cloud vector store, query rewriting, then generation with Groq.
3. **Text to SQL** with safety guardrails (read only, a single `SELECT`, forbidden keyword filtering) over a Postgres database hosted on Supabase.
4. **Evaluation protocol**: a 20 question golden dataset, scored with RAGAS (faithfulness, answer relevancy, context precision and recall), plus a custom retrieval hit rate metric. Every run is timestamped and saved for regression tracking.
5. **Streamlit dashboard**: a chat interface, accident analytics, and evaluation history, all in one aero themed app.

## Architecture

```
                    ┌─────────────────┐
   User question ──▶│   Router (LLM)  │
                    └────────┬────────┘
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
            ┌────────┐  ┌────────┐  ┌────────┐
            │  SQL   │  │  RAG   │  │  BOTH  │
            └───┬────┘  └───┬────┘  └───┬────┘
                ▼            ▼            ▼
          ┌───────────┐ ┌──────────┐ ┌─────────────┐
          │ Supabase  │ │ Qdrant   │ │ SQL + RAG   │
          │ (Postgres)│ │ Cloud    │ │ combined    │
          └───────────┘ └──────────┘ └─────────────┘
                              │
                              ▼
                      Groq (gpt-oss-20b)
                    query rewrite + answer
```

The app is deployed on Streamlit Community Cloud, which has no access to a local filesystem. Because of that, the vector store was migrated from a local Chroma index to Qdrant Cloud, and API keys are managed through Streamlit's secrets manager instead of a local `.env` file.

## Evaluation results

Four evaluation runs were made on the 20 question golden dataset as the RAG pipeline was iteratively debugged and fixed. Retrieval hit rate is always computed on the full 20 questions, at no LLM cost. RAGAS scores are computed on a sample of questions, limited by Groq's free tier daily token quota.

| Metric | Run 1 (baseline) | Run 2 (metadata + exact search) | Run 3 (query rewriting)* | Run 4 (model swap)** |
|---|---|---|---|---|
| Retrieval hit rate | 10% | 40% | 45% | 45% |
| Context recall | 22.3% | 51.1% | n/a | 28.6% |
| Context precision | 50.3% | 49.2% | n/a | 22.0% |
| Faithfulness | 53.1% | 59.2% | n/a | 68.2% |
| Answer relevancy | 66.0% | 70.8% | 70.4% | 79.8% |

\* Run 3's RAGAS judging mostly failed on Groq's daily quota. Only hit rate and answer relevancy are reliable for that run.

\*\* Groq deprecated the model used in runs 1 to 3 mid project. Run 4 uses its replacement (`openai/gpt-oss-20b`), which has a lower daily quota. Its context precision and recall come from a smaller, partial sample and are not a clean comparison to runs 1 and 2.

### What each fix addressed

1. **Baseline to Run 2**: chunks had no location, aircraft, or date in their text, so metadata was prefixed onto each chunk before embedding. This run also found that Chroma's default HNSW index was missing the true best match even at top 20 results. It was fixed by switching to exact search, later replaced by a properly tuned Qdrant index.
2. **Run 2 to Run 3**: short natural questions retrieved worse than keyword dense ones. Fixed with an LLM query rewriting step before retrieval.
3. **Run 3 to Run 4**: Groq deprecated the model in use. The project switched to its replacement and reduced the RAGAS sample size to fit its lower daily quota.

## Known limitations

1. The source dataset has no reports for 2020 and 2021, with no explanation given by its authors. This is likely an artifact of how the dataset was collected, not an absence of real accidents.
2. Comparative RAG queries (such as "what's the deadliest X and why") are not fully solved. The SQL half reliably finds the right record, but building a retrieval query from that record doesn't always surface the matching report text. This was documented rather than over engineered around, after several attempted fixes: query enrichment, metadata formatted queries, and a wider top_k.
3. The embedding model (`all-MiniLM-L6-v2`) is small and fast but limited in nuance. A larger model would likely improve retrieval further.

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Groq (`openai/gpt-oss-20b`) |
| Embeddings | sentence transformers (`all-MiniLM-L6-v2`) |
| Vector store | Qdrant Cloud |
| Relational DB | Supabase (Postgres) |
| Evaluation | RAGAS |
| Orchestration | LangChain |
| Dashboard | Streamlit and Plotly, deployed on Streamlit Community Cloud |

## Project structure

```
skyrag/
├── data/
│   ├── raw/                 (downloaded source CSV, gitignored)
│   ├── processed/           (cleaned data, narratives.jsonl, gitignored)
│   └── eval/
│       ├── golden_dataset.json
│       └── runs/            (timestamped evaluation results)
├── src/
│   ├── ingestion/           (download and clean/split the NTSB dataset)
│   ├── sql/                 (schema, import script, analytics queries)
│   ├── rag/                 (chunking, vector store, retriever)
│   ├── agent/                (router, SQL tool, hybrid agent)
│   └── eval/                 (RAGAS evaluation pipeline)
├── dashboard/
│   └── app.py                (Streamlit app)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

Create a `.env` file with:
```
DATABASE_URL=postgresql://...        # Supabase connection string
GROQ_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

Then, in order:
```bash
python src/ingestion/download_data.py
python src/ingestion/clean_split_data.py
python src/sql/import_data.py
python src/rag/build_vector_store.py      # builds a local Chroma index
python src/rag/migrate_to_qdrant.py       # migrates it to Qdrant Cloud
python src/eval/run_evaluation.py         # optional: run the evaluation suite
streamlit run dashboard/app.py
```

## Data source

US NTSB Aviation Accident and Incident Final Reports Dataset (2016 to 2023), by Embry Riddle Aeronautical University et al., available on Zenodo (https://zenodo.org/records/17096333) under the CC BY 4.0 license.

## Live demo

Deployed on Streamlit Community Cloud: *https://skyrag.streamlit.app/*
