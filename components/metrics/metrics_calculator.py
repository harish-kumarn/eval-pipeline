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
    """Calculator for generation metrics using the RAGAS evaluation framework."""

    _RETRIES = 3

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

    # ── private helpers ──────────────────────────────────────────────────────

    #TODO loop thru samples once and build dataset
    def _build_dataset(self, samples: List[Dict[str, Any]], gt_field: str):
        return self.Dataset.from_dict({
            "question":    [str(d["question"])        for d in samples],
            "answer":      [str(d["answer"])          for d in samples],
            "ground_truth":[str(d[gt_field])          for d in samples],
            "contexts":    [d["contexts"]              for d in samples],
        })

    def _run_ragas(
        self,
        samples: List[Dict[str, Any]],
        metric_names: List[str],
        metric_objs: List[Any],
        gt_field: str,
        results_by_q: Dict[str, Dict[str, Any]],
        llm_wrapper,
        embeddings_wrapper,
        label: str,
    ) -> None:
        """Evaluate *metric_objs* with retries and write scores into *results_by_q*."""
        dataset = self._build_dataset(samples, gt_field)
        for attempt in range(self._RETRIES):
            try:
                result = self.evaluate(
                    dataset,
                    metrics=metric_objs,
                    llm=llm_wrapper,
                    embeddings=embeddings_wrapper,
                    run_config=self.RunConfig(timeout=600),
                )
                df = result.to_pandas()
                for i, d in enumerate(samples):
                    qid = d.get("qid", "_unknown_")
                    for m in metric_names:
                        score = float(df[m].iloc[i]) if m in df.columns else float("nan")
                        results_by_q[qid][m] = 0.0 if math.isnan(score) else score
                return
            except Exception as e:
                logger.info("%s evaluation failed (attempt %d): %s", label, attempt + 1, e)

        # All retries exhausted
        for d in samples:
            for m in metric_names:
                results_by_q[d.get("qid", "_unknown_")][m] = "timeout error"

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
        run_ac = requested_metrics is None or "answer_correctness" in requested_metrics
        base_metrics: Dict[str, Any] = {}  # name -> ragas metric obj

        if requested_metrics:
            for m in requested_metrics:
                if m in self.metric_map:
                    base_metrics[m] = self.metric_map[m]
                elif m != "answer_correctness":
                    logger.info("Generation metric '%s' not recognised. Skipping.", m)
        else:
            base_metrics = dict(self.metric_map)

        all_metric_names = list(base_metrics) + (["answer_correctness"] if run_ac else [])
        if not all_metric_names:
            logger.info("No valid generation metrics to compute.")
            return None








        # ── filter valid samples ─────────────────────────────────────────────
        valid = []
        for d in data:
            qid      = d.get("qid", "_unknown_")
            contexts = [c for c in d.get("contexts", []) if isinstance(c, str) and c.strip()]             ## TODO : check this 
            missing  = (
                (["contexts (all empty or missing — chunk IDs not found in target_chunks.json)"] if not contexts else [])
                + (["question"]     if not d.get("question",     "").strip() else [])
                + (["answer"]       if not d.get("answer",       "").strip() else [])
                + (["ground_truth"] if not d.get("ground_truth", "").strip() else [])
            )
            if missing:
                logger.info("Skipping qid=%s from RAGAS evaluation — missing: %s", qid, ", ".join(missing))
                continue
            valid.append({**d, "contexts": contexts})

        if not valid:
            logger.info("No valid samples for RAGAS evaluation.")
            return None

        results_by_q: Dict[str, Dict[str, Any]] = {d.get("qid", "_unknown_"): {} for d in valid}















        # ── base metrics (faithfulness, context_precision) ───────────────────
        if base_metrics:
            logger.info("↳ Evaluating base metrics (batched): %s", ", ".join(base_metrics))
            if tracker is not None:
                tracker.clear_context()
            self._run_ragas(
                valid, list(base_metrics), list(base_metrics.values()),
                gt_field="ground_truth", results_by_q=results_by_q,
                llm_wrapper=llm_wrapper, embeddings_wrapper=embeddings_wrapper,
                label="Base metrics",
            )

        # ── answer_correctness (uses answer_extract as ground_truth) ─────────
        if run_ac:
            from ragas.metrics import answer_correctness

            valid_ac = [d for d in valid if d.get("answer_extract", "").strip()]
            skipped  = [d.get("qid") for d in valid if d not in valid_ac]
            if skipped:
                logger.info("answer_correctness: no answer_extract for qids %s — skipping them.", skipped)

            if valid_ac:
                logger.info(
                    "↳ Evaluating answer_correctness with answer_extract ground truth (%d samples)",
                    len(valid_ac),
                )
                self._run_ragas(
                    valid_ac, ["answer_correctness"], [answer_correctness],
                    gt_field="answer_extract", results_by_q=results_by_q,
                    llm_wrapper=llm_wrapper, embeddings_wrapper=embeddings_wrapper,
                    label="answer_correctness",
                )
            else:
                logger.info("No samples with answer_extract — answer_correctness skipped.")

        # ── aggregate ─────────────────────────────────────────────────────────
        agg: Dict[str, List[float]] = {m: [] for m in all_metric_names}
        per_query: List[Dict[str, Any]] = []


        #TODO check what this is
        for d in valid:
            qid      = d.get("qid", "_unknown_")
            q_scores = results_by_q.get(qid, {})
            entry    = {"qid": qid, "query": d.get("question", ""), "tag": d.get("tag", "")}
            for m in all_metric_names:
                score      = q_scores.get(m, 0.0)
                entry[m]   = score
                if isinstance(score, float):
                    agg[m].append(score)
            per_query.append(entry)

        return GenerationMetrics(
            faithfulness=_compute_mean(agg.get("faithfulness", [])),
            context_precision=_compute_mean(agg.get("context_precision", [])),
            answer_correctness=_compute_mean(agg.get("answer_correctness", [])),
            per_query_metrics=per_query,
        )


        