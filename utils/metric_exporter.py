import csv
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="EXPORT METRICS")

# ── Schema constants ───────────────────────────────────────────────────────────

CSV_FIELDNAMES: List[str] = [
    "sno", "question", "type",
    "mrr", "recall", "precision", "ndcg",
    "faithfulness", "context_precision", "answer_correctness",
    "actual_chunk", "predicted_chunk",
]

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp       TEXT    NOT NULL,
    sno                 INTEGER NOT NULL,
    qid                 TEXT,
    question            TEXT,
    type                TEXT,
    mrr                 REAL,
    recall              REAL,
    precision_score     REAL,
    ndcg                REAL,
    faithfulness        REAL,
    context_precision   REAL,
    answer_correctness  REAL,
    actual_chunk        TEXT,
    predicted_chunk     TEXT
)
"""

_DB_INSERT = """
INSERT INTO evaluation_results (
    run_timestamp, sno, qid, question, type,
    mrr, recall, precision_score, ndcg,
    faithfulness, context_precision, answer_correctness,
    actual_chunk, predicted_chunk
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UNSAFE_CHARS = re.compile(r"[^\w\-]")


# ── Generic writers ────────────────────────────────────────────────────────────

def save_to_csv(rows: List[Dict[str, Any]], file_path: Path, fieldnames: List[str]) -> Path:
    """Write *rows* to a CSV at *file_path* (parent dirs created automatically)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("[metric_exporter] %d rows → %s", len(rows), file_path)
    return file_path.resolve()


def save_to_sqlite(db_path: Path, schema_sql: str, insert_sql: str, params_list: List[tuple]) -> Path:
    """Ensure a table exists then batch-insert all rows into *db_path*."""
    db_path = Path(db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(schema_sql)
        conn.executemany(insert_sql, params_list)
    logger.info("[metric_exporter] %d rows → %s", len(params_list), db_path)
    return db_path


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_visionrag_lookup(visionrag_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load visionragresult.json → {qid: {ground_truth_chunk_id, retrieved_chunk_ids}}."""
    path = Path(visionrag_path)
    if not path.exists():
        logger.info("[metric_exporter] visionragresult not found: %s", path)
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {
            e["qid"]: {
                "ground_truth_chunk_id": e.get("ground_truth_chunk_id", ""),
                "retrieved_chunk_ids":   e.get("retrieved_chunk_ids", []),
            }
            for e in entries if "qid" in e
        }
    except Exception as exc:
        logger.info("[metric_exporter] Failed to load visionrag lookup: %s", exc)
        return {}


def _merge(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    visionrag_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Join retrieval + generation rows by qid into flat dicts."""
    visionrag_lookup = visionrag_lookup or {}
    gen_idx = {g["qid"]: g for g in generation_per_query}
    rows: List[Dict[str, Any]] = []

    for sno, r in enumerate(retrieval_per_query, start=1):
        qid   = r.get("qid", "")
        g     = gen_idx.get(qid, {})
        parts = qid.split("_", 1)
        vr    = visionrag_lookup.get(qid, {})

        rows.append({
            "sno":     sno,
            "qid":     parts[0],
            "chunkID": parts[1] if len(parts) > 1 else "",
            "question": r.get("query", ""),
            "type":    r.get("tag", ""),
            "mrr":       r.get("mrr"),
            "recall":    r.get("recall"),
            "precision": r.get("precision"),
            "ndcg":      r.get("ndcg"),
            "generated_answer": vr.get("generated_answer", ""),
            "faithfulness":       g.get("faithfulness"),
            "context_precision":  g.get("context_precision"),
            "answer_correctness": g.get("answer_correctness"),
            "actual_chunk":    vr.get("ground_truth_chunk_id", r.get("ground_truth_chunk_id", "")),
            "predicted_chunk": " | ".join(vr.get("retrieved_chunk_ids", r.get("retrieved_chunk_ids", []))),
        })
    return rows


def _build_rows(retrieval_per_query, generation_per_query, visionrag_path) -> List[Dict[str, Any]]:
    return _merge(retrieval_per_query, generation_per_query, _load_visionrag_lookup(visionrag_path))


def _safe_name(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", name.strip()) or "eval"


# ── Public API ─────────────────────────────────────────────────────────────────

def export_metrics_to_csv(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str | Path,
    collection_name: str,
    visionrag_path: str | Path = "outputs/visionragresult.json",
    custom_filename: Optional[str] = None,
) -> Path:
    """Export per-question metrics to a timestamped CSV (new file per run)."""
    rows = _build_rows(retrieval_per_query, generation_per_query, visionrag_path)
    if custom_filename:
        path = Path(output_dir) / custom_filename
    else:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(output_dir) / f"{_safe_name(collection_name)}_{ts}.csv"
    return save_to_csv(rows, path, CSV_FIELDNAMES)


def export_metrics_to_db(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str | Path,
    visionrag_path: str,
    db_filename: str = "eval_results.db",
) -> Path:
    """Append per-question metrics to a persistent SQLite database (rows tagged with run_timestamp)."""
    rows   = _build_rows(retrieval_per_query, generation_per_query, visionrag_path)
    run_ts = datetime.now().isoformat()
    db     = Path(output_dir) / db_filename
    params = [
        (
            run_ts,
            r["sno"], r["qid"], r["question"], r["type"],
            r["mrr"], r["recall"], r["precision"], r["ndcg"],
            r["faithfulness"], r["context_precision"], r["answer_correctness"],
            r["actual_chunk"], r["predicted_chunk"],
        )
        for r in rows
    ]
    return save_to_sqlite(db, _DB_SCHEMA, _DB_INSERT, params)





