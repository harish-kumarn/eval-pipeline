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

_UNSAFE_CHARS = re.compile(r"[^\w\-]")


# ── Generic writers ────────────────────────────────────────────────────────────


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


def _load_csv_log(log_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load backend CSV log indexed by query text."""
    if not log_path.exists():
        logger.warning("[metric_exporter] CSV log not found: %s", log_path)
        return {}

    log_data = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_text = row.get("query", "").strip()
                if q_text:
                    log_data[q_text] = {
                        "reformulated_query":     row.get("reformulated_query", ""),
                        "vector_chunkid":         row.get("vector_chunk_ids", "[]"),
                        "bm25_chunkid":           row.get("bm25_chunk_ids", "[]"),
                        "pre_rank_chunkid":       row.get("pre_rerank_chunk_ids", "[]"),
                        "post_rerank_chunk_id":   row.get("post_rerank_chunk_ids", "[]"),
                        "final_retrieved_chunks": row.get("final_retrieved_chunks", "[]"),
                    }
    except Exception as e:
        logger.error("[metric_exporter] Error reading CSV log %s: %s", log_path, e)
    return log_data


def _merge(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    visionrag_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    csv_log_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Join retrieval + generation + CSV log rows by qid/query into flat dicts."""
    visionrag_lookup = visionrag_lookup or {}
    csv_log_lookup   = csv_log_lookup or {}
    gen_idx = {g["qid"]: g for g in generation_per_query}
    rows: List[Dict[str, Any]] = []

    for sno, r in enumerate(retrieval_per_query, start=1):
        qid   = r.get("qid", "")
        q_text = r.get("query", "").strip()
        g     = gen_idx.get(qid, {})
        parts = qid.split("_", 1)
        vr    = visionrag_lookup.get(qid, {})
        cl    = csv_log_lookup.get(q_text, {})

        rows.append({
            "sno":     sno,
            "qid":     parts[0],
            "chunkID": parts[1] if len(parts) > 1 else "",
            "question": q_text,
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

            "reformulated_query":      cl.get("reformulated_query", ""),
            "vector_chunkid":          cl.get("vector_chunkid", "[]"),
            "bm25_chunkid":            cl.get("bm25_chunkid", "[]"),
            "pre_rank_chunkid":        cl.get("pre_rank_chunkid", "[]"),
            "post_rerank_chunk_id":    cl.get("post_rerank_chunk_id", "[]"),
            "final_retrieved_chunks":  cl.get("final_retrieved_chunks", "[]"),
        })
    return rows


def _build_rows(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    visionrag_path: str | Path,
    log_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    visionrag_lookup = _load_visionrag_lookup(visionrag_path)
    csv_log_lookup   = _load_csv_log(log_path) if log_path else None
    return _merge(retrieval_per_query, generation_per_query, visionrag_lookup, csv_log_lookup)


def _safe_name(name: str) -> str:
    """Sanitize table name to be alphanumeric + underscores."""
    stem = Path(name).stem
    return _UNSAFE_CHARS.sub("_", stem.strip()) or "eval_results"


# ── Public API ─────────────────────────────────────────────────────────────────


def export_metrics_to_db(
    retrieval_per_query: List[Dict[str, Any]],
    generation_per_query: List[Dict[str, Any]],
    output_dir: str | Path,
    visionrag_path: str | Path,
    log_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    db_filename: str = "eval_results.db",
) -> Path:
    """Append metrics to a dedicated experiment table with refined column order."""
    rows = _build_rows(retrieval_per_query, generation_per_query, visionrag_path, log_path)
    
    experiment_name = table_name or (log_path.stem if log_path else "evaluation_results")
    safe_table_name = _safe_name(experiment_name)
    
    # 1. Generate Schema
    schema_sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_table_name}" (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        qid                     TEXT,
        timestamp               TEXT    NOT NULL,
        question                TEXT,
        reformulated_query      TEXT,
        type                    TEXT,
        vector_chunkid          TEXT,
        bm25_chunkid            TEXT,
        pre_rerank_chunkid      TEXT,
        post_rerank_chunkid     TEXT,
        final_retrieved_chunk   TEXT,
        actual_chunk            TEXT,
        predicted_chunk         TEXT,
        mrr                     REAL,
        recall                  REAL,
        precision_score         REAL,
        ndcg                    REAL,
        faithfulness            REAL,
        context_precision       REAL,
        answer_correctness      REAL
    )
    """
    
    # 2. Generate Insert SQL (Columns ordered as requested)
    columns = [
        "qid", "timestamp", "question", "reformulated_query", "type",
        "vector_chunkid", "bm25_chunkid", "pre_rerank_chunkid", "post_rerank_chunkid", "final_retrieved_chunk",
        "actual_chunk", "predicted_chunk",
        "mrr", "recall", "precision_score", "ndcg",
        "faithfulness", "context_precision", "answer_correctness"
    ]
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = f'INSERT INTO "{safe_table_name}" ({", ".join(columns)}) VALUES ({placeholders})'
    
    cur_ts = datetime.now().isoformat()
    db     = Path(output_dir) / db_filename
    
    params = [
        (
            r["qid"], cur_ts, r["question"], r["reformulated_query"], r["type"],
            r["vector_chunkid"], r["bm25_chunkid"], r["pre_rank_chunkid"], r["post_rerank_chunk_id"], r["final_retrieved_chunks"],
            r["actual_chunk"], r["predicted_chunk"],
            r["mrr"], r["recall"], r["precision"], r["ndcg"],
            r["faithfulness"], r["context_precision"], r["answer_correctness"]
        )
        for r in rows
    ]
    
    return save_to_sqlite(db, schema_sql, insert_sql, params)





