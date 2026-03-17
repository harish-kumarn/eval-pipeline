"""
Retrieval and generation metrics calculators for the evaluation pipeline.

Uses RAGAS for generation metrics (faithfulness, context_precision) and
implements custom retrieval metrics (MRR, Recall, Precision, NDCG).
"""

import math
import logging
from typing import Dict, List, Any, Optional, TypeVar

from core.data_models import RetrievalMetrics, GenerationMetrics
from core.interfaces import IMetricsCalculator
from metrics.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────────

def _compute_mean(values: List[float]) -> Optional[float]:
    """Return the mean of *values*, or None if the list is empty."""
    return float(sum(values) / len(values)) if values else None


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Metrics
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalMetricsCalculator(IMetricsCalculator):
    """Calculator for retrieval-specific metrics (MRR, Recall, Precision, NDCG)."""

    def __init__(self):
        self.metric_map = {
            "mrr":       self._calculate_mrr,
            "recall":    self._calculate_recall,
            "precision": self._calculate_precision,
            "ndcg":      self._calculate_ndcg,
        }

    # ── per-query metric helpers ────────────────────────────────────────────

    def _calculate_mrr(self, rank: Optional[int], **kwargs) -> float:
        return 1 / rank if rank else 0

    def _calculate_recall(self, rank: Optional[int], **kwargs) -> float:
        return 1 if rank else 0

    def _calculate_precision(self, rels: List[bool], k: int, **kwargs) -> float:
        return sum(rels[:k]) / k

    def _calculate_ndcg(self, rels: List[bool], k: int, **kwargs) -> float:
        dcg  = self._dcg(rels, k)
        idcg = self._dcg([True] * min(sum(rels), k), k)
        return dcg / idcg if idcg else 0

    def _dcg(self, rels: List[bool], k: int) -> float:
        return sum(
            1 / math.log2(i + 2)
            for i, r in enumerate(rels[:k]) if r
        )

    # ── public API ───────────────────────────────────────────────────────────

    def compute(
        self,
        data: List[Dict[str, Any]],
        k: int = 3,
        requested_metrics: Optional[List[str]] = None,
    ) -> RetrievalMetrics:
        if not data:
            return RetrievalMetrics(k=k, total_queries=0)

        active_metrics = {
            m: []
            for m in (requested_metrics or self.metric_map.keys())
            if m in self.metric_map
        }
        ranks: List[int] = []
        per_query: List[Dict[str, Any]] = []

        for d in data:
            gt        = d.get("ground_truth_chunk_id")
            retrieved = d.get("retrieved_chunk_ids", [])
            rels      = [r == gt for r in retrieved]
            rank      = next((i + 1 for i, r in enumerate(rels[:k]) if r), None)

            query_entry: Dict[str, Any] = {
                "qid":   d.get("qid"),
                "query": d.get("question"),
                "tag":   d.get("tag"),
                "rank":  rank,
                "ground_truth_chunk_id": gt,
                "retrieved_chunk_ids": retrieved,
            }

            for m_name in active_metrics:
                val = self.metric_map[m_name](rank=rank, rels=rels, k=k)
                active_metrics[m_name].append(val)
                query_entry[m_name] = val

            if rank:
                ranks.append(rank)

            per_query.append(query_entry)

        return RetrievalMetrics(
            mrr_at_k=_compute_mean(active_metrics["mrr"])         if "mrr"       in active_metrics else None,
            recall_at_k=_compute_mean(active_metrics["recall"])   if "recall"    in active_metrics else None,
            precision_at_k=_compute_mean(active_metrics["precision"]) if "precision" in active_metrics else None,
            ndcg_at_k=_compute_mean(active_metrics["ndcg"])       if "ndcg"      in active_metrics else None,
            mean_rank=_compute_mean(ranks),
            k=k,
            total_queries=len(data),
            per_query_metrics=per_query,
        )















# ──────────────────────────────────────────────────────────────────────────────
# Generation Metrics  (RAGAS)
# ──────────────────────────────────────────────────────────────────────────────

class GenerationMetricsCalculator(IMetricsCalculator):
    """Calculator for generation metrics using the RAGAS evaluation framework."""

    def __init__(self):
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness, context_precision
        from datasets import Dataset

        self.evaluate  = evaluate
        self.RunConfig = RunConfig
        self.Dataset   = Dataset
        

        self.metric_map: Dict[str, Any] = {
            "faithfulness":      faithfulness,
            "context_precision": context_precision,
        }

    # ── public API ───────────────────────────────────────────────────────────

    def compute(
        self,
        data: List[Dict[str, Any]],
        llm_wrapper=None,
        embeddings_wrapper=None,
        requested_metrics: Optional[List[str]] = None,
        tracker: Optional[TokenTracker] = None,
    ) -> Optional[GenerationMetrics]:

        if not llm_wrapper or not embeddings_wrapper:
            raise RuntimeError("RAGAS requires llm_wrapper and embeddings_wrapper")

        # ── resolve which metrics to run ─────────────────────────────────────
        # answer_correctness is handled separately
        base_metric_names: List[str] = []   # faithfulness, context_precision
        base_metric_objs:  List[Any] = []

        if requested_metrics:
            for m in requested_metrics:
                if m in self.metric_map:
                    base_metric_objs.append(self.metric_map[m])
                    base_metric_names.append(m)
                elif m == "answer_correctness":
                    pass  # handled in separate call below
                else:
                    logger.warning("Generation metric '%s' not recognised. Skipping.", m)
        else:
            base_metric_objs  = list(self.metric_map.values())
            base_metric_names = list(self.metric_map.keys())

        run_ac = (requested_metrics is None) or ("answer_correctness" in (requested_metrics or []))
        all_metric_names = base_metric_names + (["answer_correctness"] if run_ac else [])

        if not all_metric_names:
            logger.warning("No valid generation metrics to compute.")
            return None

        # ── filter valid samples ─────────────────────────────────────────────
        valid = []
        for d in data:
            qid      = d.get("qid", "_unknown_")
            contexts = [c for c in d.get("contexts", []) if isinstance(c, str) and c.strip()]

            missing: list[str] = []
            if not contexts:
                missing.append("contexts (all empty or missing — chunk IDs not found in target_chunks.json)")
            if not d.get("question", "").strip():
                missing.append("question")
            if not d.get("answer", "").strip():
                missing.append("answer")
            if not d.get("ground_truth", "").strip():
                missing.append("ground_truth")

            if missing:
                logger.warning(
                    "Skipping qid=%s from RAGAS evaluation — missing: %s",
                    qid, ", ".join(missing),
                )
                continue

            valid.append({**d, "contexts": contexts})

        if not valid:
            logger.warning("No valid samples for RAGAS evaluation.")
            return None

        # Faithfulness + context_precision (uses ground_truth)
        results_by_q: Dict[str, Dict[str, float]] = {d.get("qid", "_unknown_"): {} for d in valid}

        if base_metric_objs:
            logger.info("↳ Evaluating base metrics (batched): %s", ", ".join(base_metric_names))
            ds_base = self.Dataset.from_dict({
                "question":    [str(d["question"])     for d in valid],
                "answer":      [str(d["answer"])       for d in valid],
                "ground_truth":[str(d["ground_truth"]) for d in valid],
                "contexts":    [d["contexts"]           for d in valid],
            })

            retries = 3
            for attempt in range(retries):
                try:
                    if tracker is not None:
                        tracker.clear_context()
                    result_base = self.evaluate(
                        ds_base,
                        metrics=base_metric_objs,
                        llm=llm_wrapper,
                        embeddings=embeddings_wrapper,
                        run_config=self.RunConfig(timeout=600),
                    )
                    df_base = result_base.to_pandas()
                    for i, d in enumerate(valid):
                        qid = d.get("qid", "_unknown_")
                        for m_name in base_metric_names:
                            if m_name in df_base.columns:
                                score = float(df_base[m_name].iloc[i])
                                results_by_q[qid][m_name] = 0.0 if math.isnan(score) else score
                            else:
                                results_by_q[qid][m_name] = 0.0
                    break
                except Exception as e:
                    logger.warning(f"Base metrics evaluation failed (attempt {attempt+1}): {e}")
                    if attempt == retries - 1:
                        for d in valid:
                            for m_name in base_metric_names:
                                results_by_q[d.get("qid", "_unknown_")][m_name] = "timeout error"

        # ── Call 2: answer_correctness  (uses answer_extract as ground_truth) ─
        if run_ac:
            from ragas.metrics import answer_correctness

            # Only samples that have a non-empty answer_extract
            valid_ac = [d for d in valid if d.get("answer_extract", "").strip()]
            skipped  = [d.get("qid") for d in valid if d not in valid_ac]
            if skipped:
                logger.warning(
                    "answer_correctness: no answer_extract for qids %s — skipping them.", skipped
                )

            if valid_ac:
                logger.info(
                    "↳ Evaluating answer_correctness with answer_extract ground truth (%d samples)",
                    len(valid_ac),
                )
                ds_ac = self.Dataset.from_dict({
                    "question":    [str(d["question"])       for d in valid_ac],
                    "answer":      [str(d["answer"])         for d in valid_ac],
                    "ground_truth":[str(d["answer_extract"]) for d in valid_ac],
                    "contexts":    [d["contexts"]             for d in valid_ac],
                })

                retries = 3
                for attempt in range(retries):
                    try:
                        result_ac = self.evaluate(
                            ds_ac,
                            metrics=[answer_correctness],
                            llm=llm_wrapper,
                            embeddings=embeddings_wrapper,
                            run_config=self.RunConfig(timeout=600),
                        )
                        df_ac = result_ac.to_pandas()
                        for i, d in enumerate(valid_ac):
                            qid = d.get("qid", "_unknown_")
                            if "answer_correctness" in df_ac.columns:
                                score = float(df_ac["answer_correctness"].iloc[i])
                                results_by_q[qid]["answer_correctness"] = 0.0 if math.isnan(score) else score
                            else:
                                results_by_q[qid]["answer_correctness"] = 0.0
                        break
                    except Exception as e:
                        logger.warning(f"answer_correctness evaluation failed (attempt {attempt+1}): {e}")
                        if attempt == retries - 1:
                            for d in valid_ac:
                                results_by_q[d.get("qid", "_unknown_")]["answer_correctness"] = "timeout error"
            else:
                logger.warning("No samples with answer_extract — answer_correctness skipped.")

        # ── aggregate ─────────────────────────────────────────────────────────
        agg: Dict[str, List[float]] = {m: [] for m in all_metric_names}
        per_query: List[Dict[str, Any]] = []

        for d in valid:
            qid      = d.get("qid", "_unknown_")
            q_scores = results_by_q.get(qid, {})

            entry: Dict[str, Any] = {
                "qid":   qid,
                "query": d.get("question", ""),
                "tag":   d.get("tag", ""),
            }
            for m_name in all_metric_names:
                score = q_scores.get(m_name, 0.0)
                entry[m_name] = score
                agg[m_name].append(score)

            per_query.append(entry)

        return GenerationMetrics(
            faithfulness=_compute_mean(agg.get("faithfulness", [])),
            context_precision=_compute_mean(agg.get("context_precision", [])),
            answer_correctness=_compute_mean(agg.get("answer_correctness", [])),
            per_query_metrics=per_query,
        )






        