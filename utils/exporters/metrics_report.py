
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.exporters.csv_exporter import save_to_csv
from utils.exporters.sqlite_exporter import save_to_sqlite

logger = logging.getLogger(__name__)

# ── Schema constants (single source of truth) ─────────────────────────────────

CSV_FIELDNAMES: List[str] = [
    "sno", "question", "type",
    # retriever
    "mrr", "recall", "precision", "ndcg",
    # generation
    "faithfulness", "context_precision", "answer_correctness",
    # chunk content (sourced from visionragresult.json + target_chunks.json)
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


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_chunk_lookup(target_chunks_path: str | Path) -> Dict[str, str]:
    """Load target_chunks.json → {chunk_id: chunk_text}."""
    path = Path(target_chunks_path)
    if not path.exists():
        logger.warning("[metrics_report] target_chunks file not found: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {
            cid: info.get("text", cid)
            for cid, info in raw.items()
            if isinstance(info, dict)
        }
    except Exception as e:
        logger.warning("[metrics_report] Failed to load chunk lookup: %s", e)
        return {}


def _load_visionrag_lookup(visionrag_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load visionragresult.json → {qid: {ground_truth_chunk_id, retrieved_chunk_ids}}."""
    path = Path(visionrag_path)
    if not path.exists():
        logger.warning("[metrics_report] visionragresult file not found: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        return {
            e["qid"]: {
                "ground_truth_chunk_id": e.get("ground_truth_chunk_id", ""),
                "retrieved_chunk_ids":   e.get("retrieved_chunk_ids", []),
            }
            for e in entries
            if "qid" in e
        }
    except Exception as e:
        logger.warning("[metrics_report] Failed to load visionrag lookup: %s", e)
        return {}


def _merge(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    chunk_lookup: Optional[Dict[str, str]] = None,
    visionrag_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Join retrieval and generation rows by qid into a flat list.

    Args:
        retrieval_per_query:  per-query retrieval metrics (from retriever_metrics.json).
        generation_per_query: per-query generation metrics (from generator_metrics.json).
        chunk_lookup:         chunk_id -> chunk text (from target_chunks.json).
        visionrag_lookup:     qid -> {ground_truth_chunk_id, retrieved_chunk_ids}
                              (from visionragresult.json). Used to populate
                              ``actual_chunk`` and ``predicted_chunk`` columns.
    """
    chunk_lookup     = chunk_lookup or {}
    visionrag_lookup = visionrag_lookup or {}
    gen_idx          = {g["qid"]: g for g in generation_per_query}
    rows             = []

    for sno, r in enumerate(retrieval_per_query, start=1):

        qid = r.get("qid", "")
        g   = gen_idx.get(qid, {})

        # split qid -> question id and chunk id
        question_id = qid.split("_")[0] if "_" in qid else qid
        chunk_id    = qid.split("_")[1] if "_" in qid else 0

        # ── chunk ID data from visionragresult.json ───────────────────────────
        vr = visionrag_lookup.get(qid, {})
        gt_chunk_id    = vr.get("ground_truth_chunk_id", r.get("ground_truth_chunk_id", ""))
        retrieved_ids  = vr.get("retrieved_chunk_ids",   r.get("retrieved_chunk_ids", []))

        # ── store chunk IDs directly ──────────────────────────────────────────
        actual_chunk    = gt_chunk_id
        predicted_chunk = " | ".join(retrieved_ids)

        rows.append({
            "sno": sno,
            "qid": question_id,
            "chunkID": chunk_id,
            "question": r.get("query", ""),
            "type": r.get("tag", ""),

            "mrr": r.get("mrr"),
            "recall": r.get("recall"),
            "precision": r.get("precision"),
            "ndcg": r.get("ndcg"),

            "faithfulness": g.get("faithfulness"),
            "context_precision": g.get("context_precision"),
            "answer_correctness": g.get("answer_correctness"),

            "actual_chunk": actual_chunk,
            "predicted_chunk": predicted_chunk,
        })

    return rows


# ── Public API ─────────────────────────────────────────────────────────────────

def export_metrics_to_csv(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str | Path,
    target_chunks_path: str | Path = "data/target_chunks.json",
    visionrag_path: str | Path = "outputs/visionragresult.json",
) -> Path:
    """
    Export per-question metrics to a timestamped CSV file.

    Each pipeline run creates a NEW file — nothing is ever overwritten.

    Chunk IDs are sourced from *visionrag_path* (visionragresult.json) and
    resolved to full text via *target_chunks_path* (target_chunks.json).

    Returns:
        Path of the written CSV file.
    """
    chunk_lookup     = _load_chunk_lookup(target_chunks_path)
    visionrag_lookup = _load_visionrag_lookup(visionrag_path)
    rows = _merge(retrieval_per_query, generation_per_query, chunk_lookup, visionrag_lookup)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"eval_results_{ts}.csv"
    return save_to_csv(rows, path, CSV_FIELDNAMES)


def export_metrics_to_db(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str | Path,
    db_filename: str = "eval_results.db",
    target_chunks_path: str | Path = "data/target_chunks.json",
    visionrag_path: str | Path = "outputs/visionragresult.json",
) -> Path:
    """
    Insert per-question metrics into a persistent SQLite database.

    Every run appends new rows tagged with *run_timestamp*, so the full
    history of all pipeline runs is preserved in one file.

    Chunk IDs are sourced from *visionrag_path* (visionragresult.json) and
    resolved to full text via *target_chunks_path* (target_chunks.json).

    Returns:
        Path of the SQLite database file.
    """
    chunk_lookup     = _load_chunk_lookup(target_chunks_path)
    visionrag_lookup = _load_visionrag_lookup(visionrag_path)
    rows   = _merge(retrieval_per_query, generation_per_query, chunk_lookup, visionrag_lookup)
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
