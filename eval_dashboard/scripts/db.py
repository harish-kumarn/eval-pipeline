"""
db_loader.py
------------
Loads a RAG-metrics CSV into a SQLite database.

CSV schema
----------
  Scalar   : question, type, reformulated_question, top_k
  List/TEXT: vector_search, bm25, before_rr, after_rr, retrieved
             (comma-separated doc-IDs / scores, stored as TEXT)
  Float    : mrr, recall, precision, ndcg, faithfulness,
             context_precision, answer_correctness

Usage:
    python db_loader.py \
        --csv   metrics.csv \
        --db    rag_metrics.db \
        --table run_20240601_hybrid \
        [--hybrid-search] \
        [--query-reformulation] \
        [--reranking]

The script will:
  1. Validate and ingest the CSV into <table_name>.
  2. Upsert a summary row in the `index` table with averages and flags.
"""

import argparse
import sqlite3
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Column contract ────────────────────────────────────────────────────────────

REQUIRED_CSV_COLUMNS: list[str] = [
    "timestamp",
    "query",
    "reformulated_query",
    "vector_chunk_ids",
    "bm25_chunk_ids",
    "pre_rerank_chunk_ids",
    "post_rerank_chunk_ids",
    "final_retrieved_chunks",
    "generated_answer",
    "qid",
    "type",
    "top_k",
    "actual_chunk",
    "mrr",
    "recall",
    "precision",
    "ndcg",
    "faithfulness",
    "context_precision",
    "answer_correctness",
]

# Columns that hold comma-separated lists → stored as TEXT verbatim
LIST_COLUMNS: list[str] = ["vector_chunk_ids", "bm25_chunk_ids", "pre_rerank_chunk_ids", "post_rerank_chunk_ids", "final_retrieved_chunks"]

METRIC_COLUMNS: list[str] = [
    "mrr",
    "recall",
    "precision",
    "ndcg",
    "faithfulness",
    "context_precision",
    "answer_correctness",
]

# ── DDL ────────────────────────────────────────────────────────────────────────

METRICS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS "{table}" (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             TEXT,
    qid                   TEXT,
    query                 TEXT,
    type                  TEXT,
    reformulated_query    TEXT,
    top_k                 INTEGER,
    vector_chunk_ids      TEXT,
    bm25_chunk_ids        TEXT,
    pre_rerank_chunk_ids  TEXT,
    post_rerank_chunk_ids TEXT,
    final_retrieved_chunks TEXT,
    actual_chunk          TEXT,
    generated_answer      TEXT,
    mrr                   REAL,
    recall                REAL,
    precision             REAL,
    ndcg                  REAL,
    faithfulness          REAL,
    context_precision     REAL,
    answer_correctness    REAL
);
"""

INDEX_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS "index" (
    table_name                  TEXT    PRIMARY KEY,
    timestamp                   TEXT    NOT NULL,
    hybrid_search               INTEGER NOT NULL DEFAULT 0,   -- 0/1 bool
    query_reformulation         INTEGER NOT NULL DEFAULT 0,
    reranking                   INTEGER NOT NULL DEFAULT 0,
    mrr_average                 REAL,
    recall_average              REAL,
    precision_average           REAL,
    ndcg_average                REAL,
    faithfulness_average        REAL,
    context_precision_average   REAL,
    answer_correctness_average  REAL
);
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and verify all required columns are present."""
    log.info("Reading CSV: %s", csv_path)
    try:
        df = pd.read_csv(csv_path, dtype=str)  # read everything as str first
    except Exception as exc:
        log.error("Failed to read CSV: %s", exc)
        sys.exit(1)

    missing = set(REQUIRED_CSV_COLUMNS) - set(df.columns)
    if missing:
        log.error("CSV is missing required columns: %s", sorted(missing))
        sys.exit(1)

    # Keep only the columns we care about
    df = df[[c for c in REQUIRED_CSV_COLUMNS if c in df.columns]].copy()

    # top_k → nullable integer
    df["top_k"] = pd.to_numeric(df["top_k"], errors="coerce").astype("Int64")

    # List columns (vector_search, bm25, before_rr, after_rr, retrieved) → TEXT
    # Strip whitespace but preserve the comma-separated string as-is
    for col in LIST_COLUMNS:
        df[col] = df[col].str.strip()

    # Metric columns → float; unparseable values become NaN / NULL
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Loaded %d rows, %d columns from CSV", len(df), len(df.columns))
    return df


def compute_averages(df: pd.DataFrame) -> dict[str, float | None]:
    """Return mean of each metric column (None if all NaN)."""
    avgs: dict[str, float | None] = {}
    for col in METRIC_COLUMNS:
        mean_val = df[col].mean()
        avgs[f"{col}_average"] = None if pd.isna(mean_val) else round(float(mean_val), 6)
    return avgs


def ingest(
    csv_path: Path,
    db_path: Path,
    table_name: str,
    hybrid_search: bool,
    query_reformulation: bool,
    reranking: bool,
) -> None:
    df = load_and_validate_csv(csv_path)
    averages = compute_averages(df)

    log.info("Connecting to database: %s", db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")

    try:
        with con:
            # ── 1. Create / ensure the metrics table ──────────────────────────
            con.execute(METRICS_TABLE_DDL.format(table=table_name))
            log.info("Metrics table '%s' ready.", table_name)

            # ── 2. Create / ensure the index table ───────────────────────────
            con.execute(INDEX_TABLE_DDL)
            log.info("Index table ready.")

            # ── 3. Insert metrics rows ────────────────────────────────────────
            records = df.to_dict(orient="records")
            con.executemany(
                f"""
                INSERT INTO "{table_name}"
                    (timestamp, query, reformulated_query,
                     vector_chunk_ids, bm25_chunk_ids,
                     pre_rerank_chunk_ids, post_rerank_chunk_ids, final_retrieved_chunks,
                     generated_answer, qid, type, top_k, actual_chunk,
                     mrr, recall, precision, ndcg,
                     faithfulness, context_precision, answer_correctness)
                VALUES
                    (:timestamp, :query, :reformulated_query,
                     :vector_chunk_ids, :bm25_chunk_ids,
                     :pre_rerank_chunk_ids, :post_rerank_chunk_ids, :final_retrieved_chunks,
                     :generated_answer, :qid, :type, :top_k, :actual_chunk,
                     :mrr, :recall, :precision, :ndcg,
                     :faithfulness, :context_precision, :answer_correctness)
                """,
                records,
            )
            log.info("Inserted %d rows into '%s'.", len(records), table_name)

            # ── 4. Upsert index row ───────────────────────────────────────────
            
            timestamp = datetime.now(timezone.utc).strftime("%d_%m_%H_%M")
            con.execute(
                """
                INSERT INTO "index" (
                    table_name, timestamp,
                    hybrid_search, query_reformulation, reranking,
                    mrr_average, recall_average, precision_average,
                    ndcg_average, faithfulness_average,
                    context_precision_average, answer_correctness_average
                ) VALUES (
                    :table_name, :timestamp,
                    :hybrid_search, :query_reformulation, :reranking,
                    :mrr_average, :recall_average, :precision_average,
                    :ndcg_average, :faithfulness_average,
                    :context_precision_average, :answer_correctness_average
                )
                ON CONFLICT(table_name) DO UPDATE SET
                    timestamp                  = excluded.timestamp,
                    hybrid_search              = excluded.hybrid_search,
                    query_reformulation        = excluded.query_reformulation,
                    reranking                  = excluded.reranking,
                    mrr_average                = excluded.mrr_average,
                    recall_average             = excluded.recall_average,
                    precision_average          = excluded.precision_average,
                    ndcg_average               = excluded.ndcg_average,
                    faithfulness_average       = excluded.faithfulness_average,
                    context_precision_average  = excluded.context_precision_average,
                    answer_correctness_average = excluded.answer_correctness_average
                """,
                {
                    "table_name": table_name,
                    "timestamp": timestamp,
                    "hybrid_search": int(hybrid_search),
                    "query_reformulation": int(query_reformulation),
                    "reranking": int(reranking),
                    **averages,
                },
            )
            log.info("Index row upserted for table '%s'.", table_name)

    except sqlite3.Error as exc:
        log.error("Database error — transaction rolled back: %s", exc)
        sys.exit(1)
    finally:
        con.close()

    log.info("Done. Database: %s", db_path.resolve())



if __name__ == "__main__":

    ingest(
        csv_path=Path("metric.csv"),
        db_path=Path("metrics.db"),
        table_name="experiment_7",
        hybrid_search=True,
        query_reformulation=False,
        reranking=True,
    )



