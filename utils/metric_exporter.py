"""
metric_exporter.py
------------------
Exports per-question evaluation metrics to CSV and SQLite.

Public API:
  - export_all_metrics(...)  → runs both CSV and DB exports in one call (shared data load).
  - export_to_csv(...)       → standalone CSV export.
  - export_to_db(...)        → standalone SQLite export.
"""

import pandas as pd
import ast
import csv
import json
import re
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="EXPORT METRICS")


# ── Module-level constants ────────────────────────────────────────────────────

_UNSAFE_CHARS = re.compile(r"[^\w\-]")

# Columns written to both the results table and the per-query metrics.
METRIC_COLS: Tuple[str, ...] = (
    "mrr", "recall", "precision", "ndcg",
    "faithfulness", "context_precision", "answer_correctness", "answer_relevancy",
)

# Full ordered column list for the SQLite results table.
ALL_COLS: Tuple[str, ...] = (
    "timestamp", "query", "type", "top_k",
    "reformulated_query", "vector_chunk_ids", "bm25_chunk_ids",
    "pre_rerank_chunk_ids", "post_rerank_chunk_ids",
    "final_retrieved_chunks", "actual_chunk",
    *METRIC_COLS,
)


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_table_name(name: str) -> str:
    """Sanitise *name* for use as an SQLite table name."""
    stem = Path(name).stem
    return _UNSAFE_CHARS.sub("_", stem.strip()).lower() or "eval_results"


def _get_alias(row: Dict[str, Any], *keys: str) -> str:
    """Return the first non-None value found among *keys* in *row*, or ''."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return ""


def _normalise_chunks(raw, top_k: int) -> str:
    """
    Coerce *raw* (list or string) to a string representation capped at *top_k* items.
    Handles pipe-separated strings and Python list literals.
    """
    if isinstance(raw, list):
        return str(raw[:top_k])

    if isinstance(raw, str):
        if "|" in raw:
            parts = [c.strip() for c in raw.split("|") if c.strip()]
            return str(parts[:top_k])
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return str(parsed[:top_k])
        except (ValueError, SyntaxError):
            pass

    return str(raw) if raw else ""


def _load_visionrag_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load visionragresult.json and return a qid-keyed lookup dict."""
    if not path.exists():
        logger.warning("visionragresult not found: %s", path)
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {
            e["qid"]: {
                "ground_truth_chunk_id": e.get("ground_truth_chunk_id", ""),
                "retrieved_chunk_ids":   e.get("retrieved_chunk_ids", []),
                "generated_answer":      e.get("generated_answer", ""),
            }
            for e in entries if "qid" in e
        }
    except Exception as exc:
        logger.warning("Failed to load visionrag lookup from %s: %s", path, exc)
        return {}


def _load_csv_log(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load the VisionRAG client CSV log and return a query-text-keyed lookup dict.
    Handles column-name aliases used by different backend versions.
    """
    if not path or not path.exists():
        logger.warning("CSV log not found: %s", path)
        return {}
    
    logger.info("Loading CSV log from: %s", path)

    data: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                query = (row.get("query") or row.get("question") or "").strip()
                if not query:
                    continue
                logger.info("==============================================")
                logger.info("Processing CSV log row : %s", row)
                logger.info("==============================================")

                data[query] = {
                    "question_id":           _get_alias(row, "qid"),
                    "reformulated_query":    _get_alias(row, "reformulated_query", "reformulated_question"),
                    "vector_chunk_ids":      _get_alias(row, "vector_chunk_ids", "vector_search"),
                    "bm25_chunk_ids":        _get_alias(row, "bm25_chunk_ids", "bm25"),
                    "pre_rerank_chunk_ids":  _get_alias(row, "pre_rerank_chunk_ids", "before_rr"),
                    "post_rerank_chunk_ids": _get_alias(row, "post_rerank_chunk_ids", "after_rr"),
                    "final_retrieved_chunks": _get_alias(row, "final_retrieved_chunks", "retrieved"),
                }
    except Exception as e:
        logger.error("Error reading CSV log %s: %s", path, e)

    return data


def _build_rows(
    retrieval: List[Dict[str, Any]],
    generation: List[Dict[str, Any]],
    top_k: int,
    visionrag: Dict[str, Dict[str, Any]],
    csv_log: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge retrieval, generation, visionrag, and log data into flat export rows."""

    gen_map = {g["qid"]: g for g in generation}
    rows = []

    for r in retrieval:
        qid = r.get("qid", "")
        q = (r.get("query") or "").strip()

        g = gen_map.get(qid, {})
        vr = visionrag.get(qid, {})

        # ------------------------------------------------------------------
        # FIX:
        # First try lookup by qid (recommended structure)
        # Fallback to query-text matching for backward compatibility
        # ------------------------------------------------------------------
        cl = csv_log.get(qid)

        if cl is None:
            cl = csv_log.get(q)

        # fuzzy fallback for older logs
        if cl is None:
            q_lower = q.lower()

            for k, v in csv_log.items():
                if not isinstance(k, str):
                    continue

                k_lower = k.lower()

                if q_lower in k_lower or k_lower in q_lower:
                    cl = v
                    break

        cl = cl or {}

        raw_chunks = (
            cl.get("final_retrieved_chunks")
            or vr.get("retrieved_chunk_ids")
            or r.get("retrieved_chunk_ids", [])
        )

        rows.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),

            "query": q,

            "reformulated_query":
                cl.get("reformulated_query") or q,

            "vector_chunk_ids":
                cl.get("vector_chunk_ids", ""),

            "bm25_chunk_ids":
                cl.get("bm25_chunk_ids", ""),

            "pre_rerank_chunk_ids":
                cl.get("pre_rerank_chunk_ids", ""),

            "post_rerank_chunk_ids":
                cl.get("post_rerank_chunk_ids", ""),

            "final_retrieved_chunks":
                _normalise_chunks(raw_chunks, top_k),

            "generated_answer":
                vr.get("generated_answer")
                or g.get("answer")
                or r.get("generated_answer", ""),

            "qid": qid,

            "type":
                r.get("tag", ""),

            "top_k":
                int(top_k),

            "actual_chunk":
                vr.get("ground_truth_chunk_id")
                or r.get("ground_truth_chunk_id", ""),

            # Retrieval metrics
            "mrr":
                r.get("mrr"),

            "recall":
                r.get("recall"),

            "precision":
                r.get("precision"),

            "ndcg":
                r.get("ndcg"),

            # Generation metrics
            "faithfulness":
                g.get("faithfulness"),

            "context_precision":
                g.get("context_precision"),

            "answer_correctness":
                g.get("answer_correctness"),

            "answer_relevancy":
                g.get("answer_relevancy"),
        })

    return rows

# ── Internal export helpers ───────────────────────────────────────────────────

def _write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> Path:
    """Write *rows* to *csv_path* and return the path."""
    if not rows:
        logger.warning("No rows to write — CSV will be empty: %s", csv_path)
        return csv_path

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV saved → %s (%d rows)", csv_path, len(rows))
    return csv_path


def _write_db(
    rows: List[Dict[str, Any]],
    db_path: Path,
    table: str,
    hybrid_search: bool,
    query_reformulation: bool,
    reranking: bool,
    final_csv_name: Optional[str],
) -> Path:
    """Upsert *rows* into the SQLite database at *db_path* and return the path."""

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 6) if vals else None

    with sqlite3.connect(db_path) as conn:

        # Results table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp              TEXT,
                query                  TEXT,
                type                   TEXT,
                top_k                  INTEGER,
                reformulated_query     TEXT,
                vector_chunk_ids       TEXT,
                bm25_chunk_ids         TEXT,
                pre_rerank_chunk_ids   TEXT,
                post_rerank_chunk_ids  TEXT,
                final_retrieved_chunks TEXT,
                actual_chunk           TEXT,
                mrr                    REAL,
                recall                 REAL,
                precision              REAL,
                ndcg                   REAL,
                faithfulness           REAL,
                context_precision      REAL,
                answer_correctness     REAL,
                answer_relevancy       REAL
            )
        """)
        conn.executemany(
            f'INSERT INTO "{table}" ({", ".join(ALL_COLS)}) '
            f'VALUES ({", ".join(["?"] * len(ALL_COLS))})',
            [tuple(r.get(c) for c in ALL_COLS) for r in rows],
        )

        # Run index table (one row per evaluation run)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS "index" (
                table_name                  TEXT PRIMARY KEY,
                timestamp                   TEXT,
                final_csv_name              TEXT,
                hybrid_search               INTEGER,
                query_reformulation         INTEGER,
                reranking                   INTEGER,
                mrr_average                 REAL,
                recall_average              REAL,
                precision_average           REAL,
                ndcg_average                REAL,
                faithfulness_average        REAL,
                context_precision_average   REAL,
                answer_correctness_average  REAL,
                answer_relevancy_average    REAL
            )
        """)
        conn.execute("""
            INSERT INTO "index" (
                table_name, timestamp, final_csv_name,
                hybrid_search, query_reformulation, reranking,
                mrr_average, recall_average, precision_average, ndcg_average,
                faithfulness_average, context_precision_average,
                answer_correctness_average, answer_relevancy_average
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(table_name) DO UPDATE SET
                timestamp                  = excluded.timestamp,
                final_csv_name             = excluded.final_csv_name,
                hybrid_search              = excluded.hybrid_search,
                query_reformulation        = excluded.query_reformulation,
                reranking                  = excluded.reranking,
                mrr_average                = excluded.mrr_average,
                recall_average             = excluded.recall_average,
                precision_average          = excluded.precision_average,
                ndcg_average               = excluded.ndcg_average,
                faithfulness_average       = excluded.faithfulness_average,
                context_precision_average  = excluded.context_precision_average,
                answer_correctness_average = excluded.answer_correctness_average,
                answer_relevancy_average   = excluded.answer_relevancy_average
        """, (
            table,
            datetime.now().strftime("%d_%m_%H_%M"),
            final_csv_name,
            int(hybrid_search),
            int(query_reformulation),
            int(reranking),
            _avg("mrr"), _avg("recall"), _avg("precision"), _avg("ndcg"),
            _avg("faithfulness"), _avg("context_precision"),
            _avg("answer_correctness"), _avg("answer_relevancy"),
        ))

    logger.info("DB saved → %s (table=%s)", db_path, table)
    return db_path


# ── Public API ────────────────────────────────────────────────────────────────

def export_all_metrics(
    run_output_dir: Path,
    visionrag_populated_path: Path,
    retrieval_path: Path,
    generation_path: Path,
    vrag_file_name_csv: Path,
    top_k: int,
    result_csv_filename: str,
    table_name: str,
    hybrid_search: bool,
    query_reformulation: bool,
    reranking: bool,
    db_filename: str = "../../eval_dashboard/metrics.db",
) -> Tuple[Path, Path]:
    """
    Load results once and export to both CSV and SQLite.

    Returns:
        (csv_path, db_path)
    """
    # Load per-query metric lists from JSON outputs.
    retrieval_pq: List[Dict] = []
    generation_pq: List[Dict] = []

    if retrieval_path.exists():
        retrieval_pq = (json.loads(retrieval_path.read_text(encoding="utf-8")) or {}).get("per_query_metrics", [])

    if generation_path.exists():
        generation_pq = (json.loads(generation_path.read_text(encoding="utf-8")) or {}).get("per_query_metrics", [])

    # Load shared lookup data once — reused by both exports.
    visionrag = _load_visionrag_lookup(visionrag_populated_path)
    csv_log   = _load_csv_log(vrag_file_name_csv)
    logger.info("=======================================================================================================")
    logger.info("VisionRAG CSV log: %s", csv_log)
    logger.info("VisionRAG lookup: %s", visionrag)
    logger.info("=======================================================================================================")

    rows      = _build_rows(retrieval_pq, generation_pq, top_k, visionrag, csv_log)

    logger.info("VisionRAG ROWS: %s", rows)
    logger.info("=======================================================================================================")

    # CSV export
    run_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _write_csv(rows, run_output_dir / result_csv_filename)

    remove_columns_from_csv(
        csv_path=csv_path,
        columns_to_remove=["generated_answer","faithfulness", "context_precision", "answer_correctness", "answer_relevancy"],
    )


    # DB export
    db_path = None
    # db_path = Path(run_output_dir) / db_filename
    # db_path.parent.mkdir(parents=True, exist_ok=True)
    # _write_db(
    #     rows             = rows,
    #     db_path          = db_path,
    #     table            = _safe_table_name(table_name),
    #     hybrid_search    = hybrid_search,
    #     query_reformulation = query_reformulation,
    #     reranking        = reranking,
    #     final_csv_name   = csv_path.name,
    # )

    return csv_path, db_path


def export_to_csv(
    retrieval: List[Dict[str, Any]],
    generation: List[Dict[str, Any]],
    output_dir: Path,
    visionrag_populated_path: Path,
    top_k: int,
    log_path: Optional[Path] = None,
    result_csv_filename: Optional[str] = None,
) -> Path:
    """Standalone CSV export (loads its own data — use export_all_metrics to avoid double I/O)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows     = _build_rows(retrieval, generation, top_k, _load_visionrag_lookup(visionrag_populated_path), _load_csv_log(log_path))
    csv_path = output_dir / (result_csv_filename or "evaluation_results.csv")
    return _write_csv(rows, csv_path)


def export_to_db(
    retrieval: List[Dict[str, Any]],
    generation: List[Dict[str, Any]],
    output_dir: Path,
    visionrag_populated_path: Path,
    top_k: int,
    hybrid_search: bool = True,
    query_reformulation: bool = False,
    reranking: bool = False,
    log_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    db_filename: str = "metrics.db",
    final_csv_name: Optional[str] = None,
) -> Path:
    """Standalone SQLite export (loads its own data — use export_all_metrics to avoid double I/O)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows    = _build_rows(retrieval, generation, top_k, _load_visionrag_lookup(visionrag_populated_path), _load_csv_log(log_path))
    db_path = output_dir / db_filename
    return _write_db(
        rows             = rows,
        db_path          = db_path,
        table            = _safe_table_name(table_name or "evaluation_results"),
        hybrid_search    = hybrid_search,
        query_reformulation = query_reformulation,
        reranking        = reranking,
        final_csv_name   = final_csv_name,
    )



# ── Utils ────────────────────────────────────────────────────────────────


def remove_columns_from_csv(
    csv_path: str,
    columns_to_remove: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Remove specified columns from a CSV file.

    Args:
        csv_path: Path to input CSV file.
        columns_to_remove: List of column names to remove.
                           If None or empty, file is unchanged.
        output_path: Optional output CSV path.
                     If not provided, original file is overwritten.

    Returns:
        Path to the saved CSV file.
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    # Skip if list is empty or None
    if columns_to_remove:

        # Keep only existing columns
        existing_columns = [
            col for col in columns_to_remove
            if col in df.columns
        ]

        # Drop columns
        if existing_columns:
            df = df.drop(columns=existing_columns)

    # Save output
    save_path = output_path or csv_path
    df.to_csv(save_path, index=False)

    return save_path




