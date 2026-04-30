import csv
import json
import sqlite3
import statistics
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="EXPORT METRICS")

_UNSAFE_CHARS = re.compile(r"[^\w\-]")


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    stem = Path(name).stem
    return _UNSAFE_CHARS.sub("_", stem.strip()).lower() or "eval_results"


def _load_visionrag_lookup(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        logger.warning("visionragresult not found: %s", path)
        return {}

    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {
            e["qid"]: {
                "ground_truth_chunk_id": e.get("ground_truth_chunk_id", ""),
                "retrieved_chunk_ids": e.get("retrieved_chunk_ids", []),
                "generated_answer": e.get("generated_answer", ""),
            }
            for e in entries if "qid" in e
        }
    except Exception as exc:
        logger.warning("Failed to load visionrag lookup: %s", exc)
        return {}


def _load_csv_log(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        logger.warning("CSV log not found: %s", path)
        return {}

    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Use columns found in the file
            for row in reader:
                q = (row.get("query") or row.get("question") or "").strip()
                if not q:
                    continue
                
                # Try multiple aliases for each field
                def _get_alias(keys):
                    for k in keys:
                        v = row.get(k)
                        if v is not None: return v
                    return ""

                data[q] = {
                    "reformulated_query": _get_alias(["reformulated_query", "reformulated_question"]),
                    "vector_chunk_ids": _get_alias(["vector_chunk_ids", "vector_search"]),
                    "bm25_chunk_ids": _get_alias(["bm25_chunk_ids", "bm25"]),
                    "pre_rerank_chunk_ids": _get_alias(["pre_rerank_chunk_ids", "before_rr"]),
                    "post_rerank_chunk_ids": _get_alias(["post_rerank_chunk_ids", "after_rr"]),
                    "final_retrieved_chunks": _get_alias(["final_retrieved_chunks", "retrieved"]),
                }
    except Exception as e:
        logger.error("Error reading CSV log %s: %s", path, e)

    return data


def build_rows(
    retrieval: List[Dict[str, Any]],
    generation: List[Dict[str, Any]],
    top_k: int,
    visionrag: Dict[str, Dict[str, Any]],
    csv_log: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:

    gen_map = {g["qid"]: g for g in generation}
    rows = []

    for i, r in enumerate(retrieval, 1):
        qid = r.get("qid", "")
        q = (r.get("query") or "").strip()

        g = gen_map.get(qid, {})
        vr = visionrag.get(qid, {})
        cl = csv_log.get(q, {})

        retrieved_chunks = (
            cl.get("retrieved")
            or vr.get("retrieved_chunk_ids")
            or r.get("retrieved_chunk_ids", [])
        )

        if isinstance(retrieved_chunks, list):
            retrieved_chunks = str(retrieved_chunks[:top_k])
        elif isinstance(retrieved_chunks, str):
            if "|" in retrieved_chunks:
                rc_list = [c.strip() for c in retrieved_chunks.split("|") if c.strip()]
                retrieved_chunks = str(rc_list[:top_k])
            else:
                import ast
                try:
                    rc_list = ast.literal_eval(retrieved_chunks)
                    if isinstance(rc_list, list):
                        retrieved_chunks = str(rc_list[:top_k])
                except Exception:
                    pass

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": q,
            "reformulated_query": cl.get("reformulated_query") or q,
            "vector_chunk_ids": cl.get("vector_chunk_ids", ""),
            "bm25_chunk_ids": cl.get("bm25_chunk_ids", ""),
            "pre_rerank_chunk_ids": cl.get("pre_rerank_chunk_ids", ""),
            "post_rerank_chunk_ids": cl.get("post_rerank_chunk_ids", ""),
            "final_retrieved_chunks": retrieved_chunks,
            "generated_answer": vr.get("generated_answer") or g.get("answer") or r.get("generated_answer", ""),

            "qid": qid,
            "type": r.get("tag", ""),
            "top_k": top_k,
            "actual_chunk": vr.get("ground_truth_chunk_id") or r.get("ground_truth_chunk_id", ""),

            "mrr": r.get("mrr"),
            "recall": r.get("recall"),
            "precision": r.get("precision"),
            "ndcg": r.get("ndcg"),

            "faithfulness": g.get("faithfulness"),
            "context_precision": g.get("context_precision"),
            "answer_correctness": g.get("answer_correctness"),
            "answer_relevancy": g.get("answer_relevancy"),
        })

    return rows


# ── Exporters ───────────────────────────────────────────────────────────────

def export_to_csv(
    retrieval: List[Dict[str, Any]],
    generation: List[Dict[str, Any]],
    output_dir: str | Path,
    visionrag_path: str | Path,
    top_k: int,
    log_path: Optional[Path] = None,
    csv_filename: Optional[str] = None,
) -> Path:
    """Prepare and export metrics to a CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visionrag = _load_visionrag_lookup(visionrag_path)
    csv_log = _load_csv_log(log_path)
    rows = build_rows(retrieval, generation, top_k, visionrag, csv_log)

    if not csv_filename:
        csv_filename = "evaluation_results.csv"
    
    csv_path = output_dir / csv_filename
    
    if not rows:
        return csv_path

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV saved → %s", csv_path)
    return csv_path


def export_to_db(
    retrieval: List[Dict[str, Any]],
    generation: List[Dict[str, Any]],
    output_dir: str | Path,
    visionrag_path: str | Path,
    top_k: int,
    hybrid_search: bool = True,
    query_reformulation: bool = False,
    reranking: bool = False,
    log_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    db_filename: str = "metrics.db",
    final_csv_name: Optional[str] = None,
) -> Path:
    """Prepare and export metrics to an SQLite database."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / db_filename

    visionrag = _load_visionrag_lookup(visionrag_path)
    csv_log   = _load_csv_log(log_path)
    rows      = build_rows(retrieval, generation, top_k, visionrag, csv_log)
    table     = _safe_name(table_name or "evaluation_results")

    METRIC_COLS = [
        "mrr", "recall", "precision", "ndcg",
        "faithfulness", "context_precision", "answer_correctness", "answer_relevancy",
    ]

    ALL_COLS = [
        "timestamp", "query", "type", "top_k",
        "reformulated_query", "vector_chunk_ids", "bm25_chunk_ids",
        "pre_rerank_chunk_ids", "post_rerank_chunk_ids",
        "final_retrieved_chunks", "actual_chunk",
        *METRIC_COLS,
    ]

    def avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 6) if vals else None

    with sqlite3.connect(db_path) as conn:

        # ── Main results table ───────────────────────────────────────────────
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

        # ── Index table ──────────────────────────────────────────────────────
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
            avg("mrr"), avg("recall"), avg("precision"), avg("ndcg"),
            avg("faithfulness"), avg("context_precision"),
            avg("answer_correctness"), avg("answer_relevancy"),
        ))

    logger.info("DB saved → %s", db_path)
    return db_path