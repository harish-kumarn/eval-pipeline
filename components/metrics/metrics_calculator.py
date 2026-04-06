"""
Retrieval and generation metrics calculators for the evaluation pipeline.

Uses RAGAS for generation metrics (faithfulness, context_precision) and
implements custom retrieval metrics (MRR, Recall, Precision, NDCG).
"""

import math
from typing import Dict, List, Any, Optional, TypeVar

from  utils.support_evals.data_models import RetrievalMetrics, GenerationMetrics
from utils.support_evals.interfaces import IMetricsCalculator
from components.metrics.token_tracker import TokenTracker

from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="METRICS CALCULATOR")

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
    _RETRIES = 3

    def __init__(self):
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness, context_precision
        from datasets import Dataset

        self.evaluate = evaluate
        self.RunConfig = RunConfig
        self.Dataset = Dataset

        self.metric_map = {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
        }

    # ── helpers ────────────────────────────────────────────────────────────

    def _build_dataset(self, samples, gt_field):
        return self.Dataset.from_dict({
            "question": [str(d["question"]) for d in samples],
            "answer": [str(d["answer"]) for d in samples],
            "ground_truth": [str(d[gt_field]) for d in samples],
            "contexts": [d["contexts"] for d in samples],
        })

    def _validate_samples(self, data):
        valid = []
        for i, d in enumerate(data):
            qid = d.get("qid") or f"_idx_{i}"

            contexts = [
                c for c in d.get("contexts", [])
                if isinstance(c, str) and c.strip()
            ]

            if not all([
                d.get("question", "").strip(),
                d.get("answer", "").strip(),
                d.get("ground_truth", "").strip(),
            ]):
                logger.info("Skipping qid=%s due to missing fields", qid)
                continue

            valid.append({**d, "qid": qid, "contexts": contexts})

        return valid

    def _run_ragas(
        self,
        samples,
        metrics: Dict[str, Any],
        gt_field,
        results_by_q,
        llm_wrapper,
        embeddings_wrapper,
        label,
    ):
        dataset = self._build_dataset(samples, gt_field)
        names = list(metrics.keys())
        objs = list(metrics.values())

        for attempt in range(self._RETRIES):
            try:
                result = self.evaluate(
                    dataset,
                    metrics=objs,
                    llm=llm_wrapper,
                    embeddings=embeddings_wrapper,
                    run_config=self.RunConfig(timeout=600),
                )
                df = result.to_pandas()

                for i, d in enumerate(samples):
                    qid = d["qid"]
                    for m in names:
                        score = float(df[m].iloc[i]) if m in df.columns else None
                        if score is not None and math.isnan(score):
                            score = None
                        results_by_q[qid][m] = score

                return

            except Exception as e:
                logger.info("%s failed (attempt %d): %s", label, attempt + 1, e)

        # fallback
        for d in samples:
            for m in names:
                results_by_q[d["qid"]][m] = None

    # ── main ───────────────────────────────────────────────────────────────

    def compute(
        self,
        data,
        llm_wrapper=None,
        embeddings_wrapper=None,
        requested_metrics=None,
        tracker=None,
    ):
        if not llm_wrapper or not embeddings_wrapper:
            raise RuntimeError("RAGAS requires llm_wrapper and embeddings_wrapper")

        requested = set(requested_metrics or [])

        # resolve metrics
        base_metrics = {
            m: self.metric_map[m]
            for m in (requested or self.metric_map.keys())
            if m in self.metric_map
        }

        run_ac = not requested or "answer_correctness" in requested

        valid = self._validate_samples(data)
        if not valid:
            logger.info("No valid samples.")
            return None

        results_by_q = {d["qid"]: {} for d in valid}

        metric_groups = []

        if base_metrics:
            # Only run context-dependent metrics on samples with at least one non-empty context
            valid_for_base = [d for d in valid if d.get("contexts")]
            if valid_for_base:
                metric_groups.append({
                    "metrics": base_metrics,
                    "gt_field": "ground_truth",
                    "samples": valid_for_base,
                    "label": "base_metrics",
                })

        if run_ac:
            from ragas.metrics import answer_correctness

            valid_ac = [
                d for d in valid
                if d.get("answer_extract", "").strip()
            ]

            if valid_ac:
                metric_groups.append({
                    "metrics": {"answer_correctness": answer_correctness},
                    "gt_field": "answer_extract",
                    "samples": valid_ac,
                    "label": "answer_correctness",
                })
            else:
                logger.info("No samples with answer_extract.")

        # run all metric groups
        for group in metric_groups:
            if tracker:
                tracker.clear_context()

            logger.info("Evaluating: %s", list(group["metrics"].keys()))

            self._run_ragas(
                group["samples"],
                group["metrics"],
                group["gt_field"],
                results_by_q,
                llm_wrapper,
                embeddings_wrapper,
                group["label"],
            )

        # aggregate
        agg = {m: [] for m in (*base_metrics.keys(), "answer_correctness")}
        per_query = []

        for d in valid:
            qid = d["qid"]
            scores = results_by_q[qid]

            entry = {
                "qid": qid,
                "query": d.get("question", ""),
                "tag": d.get("tag", ""),
            }

            for m in agg:
                score = scores.get(m)
                entry[m] = score
                if isinstance(score, float):
                    agg[m].append(score)

            per_query.append(entry)

        return GenerationMetrics(
            faithfulness=_compute_mean(agg.get("faithfulness", [])),
            context_precision=_compute_mean(agg.get("context_precision", [])),
            answer_correctness=_compute_mean(agg.get("answer_correctness", [])),
            per_query_metrics=per_query,
        )


        