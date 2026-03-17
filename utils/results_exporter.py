"""
Exports per-question evaluation results to a timestamped CSV and a SQLite database.
"""

import csv
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Retriever metric columns (order matters for CSV) ──
RETRIEVER_METRICS = ["mrr", "recall", "precision", "ndcg"]

# ── Generation metric columns ──
GENERATION_METRICS = ["faithfulness", "context_precision", "answer_correctness"]

DB_NAME = "eval_results.db"
TABLE_NAME = "evaluation_results"


# =====================================================
# CSV Export
# =====================================================

def export_per_question_csv(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Merge retrieval and generation per-query metrics by qid and write
    a new timestamped CSV to *output_dir*.

    Returns the path of the generated CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"eval_results_{timestamp}.csv"

    # Index generation results by qid for fast lookup
    gen_by_qid: Dict[str, Dict] = {
        g["qid"]: g for g in generation_per_query
    }

    # 'actual_chunk' is mapped to 'ground_truth_chunk_id', 'predicted_chunks' to 'retrieved_chunk_ids'
    header = [
        "sno",
        "qid",
        "question",
        "type",
        "actual_chunk",         # ground_truth_chunk_id
        "predicted_chunks",     # retrieved_chunk_ids
        *RETRIEVER_METRICS,
        *GENERATION_METRICS,
    ]

    rows: List[list] = []
    for idx, r in enumerate(retrieval_per_query, start=1):
        qid = r.get("qid", "")
        g = gen_by_qid.get(qid, {})

        # Map actual_chunk and predicted_chunks
        actual_chunk = r.get("ground_truth_chunk_id", "")
        predicted_chunks = r.get("retrieved_chunk_ids", [])
        # Handle empty predicted_chunks (e.g., q3_chunk7)
        predicted_chunks_str = "|".join(predicted_chunks) if predicted_chunks else ""

        row = [
            idx,
            qid,
            r.get("query", ""),
            r.get("tag", ""),
            actual_chunk,
            predicted_chunks_str,
            *[r.get(m, "") for m in RETRIEVER_METRICS],
            *[g.get(m, "") for m in GENERATION_METRICS],
        ]
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(f"CSV saved to {csv_path}")
    return str(csv_path)


# =====================================================
# SQLite Storage
# =====================================================

def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create the results table if it doesn't exist."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT    NOT NULL,
            sno             INTEGER NOT NULL,
            qid             TEXT    NOT NULL,
            question        TEXT,
            type            TEXT,
            ground_truth_chunk_id TEXT,
            retrieved_chunk_ids TEXT,
            mrr             REAL,
            recall          REAL,
            precision_score REAL,
            ndcg            REAL,
            faithfulness    REAL,
            context_precision REAL,
            answer_correctness REAL
        )
    """)
    conn.commit()


def save_to_sqlite(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Insert one row per question into a SQLite database stored at
    *output_dir*/eval_results.db.

    Returns the database file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / DB_NAME
    run_ts = datetime.now().isoformat()

    gen_by_qid: Dict[str, Dict] = {
        g["qid"]: g for g in generation_per_query
    }

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_table(conn)

        for idx, r in enumerate(retrieval_per_query, start=1):
            qid = r.get("qid", "")
            g = gen_by_qid.get(qid, {})

            # Map actual_chunk and predicted_chunks for DB
            actual_chunk = r.get("ground_truth_chunk_id", "")
            predicted_chunks = r.get("retrieved_chunk_ids", [])
            predicted_chunks_str = "|".join(predicted_chunks) if predicted_chunks else ""
            conn.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (run_timestamp, sno, qid, question, type,
                     ground_truth_chunk_id, retrieved_chunk_ids,
                     mrr, recall, precision_score, ndcg,
                     faithfulness, context_precision, answer_correctness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_ts,
                    idx,
                    qid,
                    r.get("query", ""),
                    r.get("tag", ""),
                    actual_chunk,
                    predicted_chunks_str,
                    r.get("mrr"),
                    r.get("recall"),
                    r.get("precision"),
                    r.get("ndcg"),
                    g.get("faithfulness"),
                    g.get("context_precision"),
                    g.get("answer_correctness"),
                ),
            )

        conn.commit()
        logger.info(f"SQLite results saved to {db_path} (run={run_ts})")
    finally:
        conn.close()

    return str(db_path)
