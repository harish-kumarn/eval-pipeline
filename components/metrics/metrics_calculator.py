"""
metrics_calculator.py
---------------------
Retrieval and generation metrics calculators for the evaluation pipeline.

Retrieval metrics (custom): MRR@k, Recall@k, Precision@k, NDCG@k.
Generation metrics (RAGAS): Faithfulness, Context Precision,
                             Answer Correctness, Answer Relevancy.
"""

import math
from typing import Any, Dict, List, Optional

from components.metrics.token_tracker import TokenTracker
from config.logger import setup_logger
from utils.support_evals.data_models import GenerationMetrics, RetrievalMetrics
from utils.support_evals.interfaces import IMetricsCalculator

logger = setup_logger(name="eval_pipeline", tag="METRICS CALCULATOR")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> Optional[float]:
    """Return the arithmetic mean of *values*, or None if the list is empty."""
    return float(sum(values) / len(values)) if values else None


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

class RetrievalMetricsCalculator(IMetricsCalculator):
    """Computes MRR, Recall, Precision, and NDCG for retrieval evaluation."""   ###  def __init__(self): pass


    SUPPORTED_METRICS = ("mrr", "recall", "precision", "ndcg")

    def compute(
        self,
        data: List[Dict[str, Any]],
        k: int = 3,
        requested_metrics: Optional[List[str]] = None,
    ) -> RetrievalMetrics:
       
        if not data:
            return RetrievalMetrics(k=k, total_queries=0)

   

        active = set(requested_metrics or self.SUPPORTED_METRICS) & set(self.SUPPORTED_METRICS) ## requested_metrics = ["recall", "mrr"] then in active only recall,mrr comes 
        accumulators: Dict[str, List[float]] = {m: [] for m in active}
        ranks: List[int] = []
        per_query: List[Dict[str, Any]] = []
      

        for d in data:
            gt = d.get("ground_truth_chunk_id", []) or []
            retrieved = d.get("retrieved_chunk_ids", []) or []
            retrieved_list = d.get("retrieved_list")
            if not retrieved_list:                          
                gt_set = set(gt)
                retrieved_list = [rid in gt_set for rid in retrieved]

            retrieved_position = d.get("retrieved_position") or []
          
            rank = min(retrieved_position) if retrieved_position else 0
            scores = self._score_query(rank=rank, retrieved_list=retrieved_list, k=k, active=active, gt=gt, retrieved=retrieved) # [1] , [True, False, False, False, False]  --->  should want fully not single like predictor, set of metrics

            for m, v in scores.items():
                accumulators[m].append(v)

            if rank:
                ranks.append(rank)

            per_query.append({
                "qid":                   d.get("qid"),
                "query":                 d.get("question"),
                "tag":                   d.get("tag"),
                "ground_truth":          d.get("ground_truth", ""),
                "vector_chunk_ids":      d.get("vector_chunk_ids"),
                "bm25_chunk_ids" :       d.get("bm25_chunk_ids"),
                "pre_rerank_chunk_ids":  d.get("pre_rerank_chunk_ids"),
                "post_rerank_chunk_ids": d.get("post_rerank_chunk_ids"),
                "generated_answer":      d.get("generated_answer"),
                "predicted":             d.get("predicted"),
                "retrieved_list":        retrieved_list,          # already computed above in this loop
                "retrieved_position":    d.get("retrieved_position"),
                "rank":                  rank,
                "ground_truth_chunk_id": gt,
                "retrieved_chunk_ids":   retrieved,
                **scores,
            })

       


        return RetrievalMetrics(
            mrr_at_k       = _mean(accumulators.get("mrr", [])),
            recall_at_k    = _mean(accumulators.get("recall", [])),
            precision_at_k = _mean(accumulators.get("precision", [])),
            ndcg_at_k      = _mean(accumulators.get("ndcg", [])),
            mean_rank      = _mean(ranks),
            k              = k,
            total_queries  = len(data),
            per_query_metrics = per_query,
        )
        

    def _score_query(
        self,
        rank: Optional[int],
        retrieved_list: List[bool],
        k: int,
        active: set,
        gt: Optional[List[str]] = None,
        retrieved: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        scores = {}
        n_relevant_total = len(gt) if gt else 0          # total relevant chunks that exist
        n_retrieved_at_k = min(k, len(retrieved) if retrieved is not None else k)

        if "mrr" in active:
            scores["mrr"] = 1 / rank if rank else 0.0

        if "recall" in active:
            # conventional recall@k: relevant chunks retrieved in top-k / total relevant chunks
            scores["recall"] = (sum(retrieved_list[:k]) / n_relevant_total) if n_relevant_total else 0.0

        if "precision" in active:
            # conventional precision@k: relevant chunks retrieved in top-k / chunks actually retrieved (up to k)
            scores["precision"] = (sum(retrieved_list[:k]) / n_retrieved_at_k) if n_retrieved_at_k else 0.0

        if "ndcg" in active:
            scores["ndcg"] = self._ndcg(retrieved_list, k, n_relevant_total)

        return scores

    def _ndcg(self, retrieved_list: List[bool], k: int, n_relevant_total: int) -> float:
        dcg = self._dcg(retrieved_list, k)
        ideal_hits = min(n_relevant_total, k)          # best case: all top ideal_hits are relevant
        idcg = self._dcg([True] * ideal_hits, k)
        return dcg / idcg if idcg else 0.0

    def _dcg(self, retrieved_list: List[bool], k: int) -> float:
        return sum(1 / math.log2(i + 2) for i, r in enumerate(retrieved_list[:k]) if r)


# ── Generation Metrics (RAGAS) ────────────────────────────────────────────────

# Canonical set of generation metrics evaluated through RAGAS.
GENERATION_METRICS = ("faithfulness", "context_precision", "answer_correctness", "answer_relevancy")


class GenerationMetricsCalculator(IMetricsCalculator):
    """
    Computes generation metrics via RAGAS:
      - Faithfulness   ----> Checks whether the generated answer (llm) is supported by the retrieved context(from rag).
      - Context Precision  ----> Checks whether the retrieved context is relevant to answering the question. ex : bad context precision Fuel mixing ratio is 50:1.Engine maintenance...Safety precautions...
      - Answer Correctness ----> Checks how closely the generated answer matches the ground-truth answer.
      - Answer Relevancy ----> Checks whether the generated answer is relevant to the user's question.
    """

    _RETRIES = 3

    def __init__(self) -> None:
        self._ragas_metrics = self._load_ragas_metrics()

        # Import heavy RAGAS dependencies once at construction time.
        from datasets import Dataset
        from ragas import RunConfig, evaluate
        self.evaluate  = evaluate
        self.RunConfig = RunConfig
        self.Dataset   = Dataset

    # ── Setup ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_ragas_metrics() -> Dict[str, Any]:
        """Import and return RAGAS metric objects, handling version differences."""
        from ragas.metrics import answer_correctness, context_precision, faithfulness

        try:
            from ragas.metrics import answer_relevancy  ## some ragas have this metrics some other doesn't
        except ImportError:
            try:
                from ragas.metrics import answer_relevance as answer_relevancy
            except ImportError:
                from ragas.metrics.collections import answer_relevancy  # type: ignore[no-redef]

        return {
            "faithfulness":       faithfulness,
            "context_precision":  context_precision,
            "answer_correctness": answer_correctness,
            "answer_relevancy":   answer_relevancy,
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_samples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for i, d in enumerate(data):
            qid = d.get("qid") or f"_idx_{i}"
            question     = str(d.get("question") or "").strip()
            answer       = str(d.get("answer") or "").strip()
            ground_truth = str(d.get("ground_truth") or "").strip()
            contexts = [c for c in d.get("contexts", []) if isinstance(c, str) and c.strip()]

            missing = [f for f, v in (("question", question), ("answer", answer), ("ground_truth", ground_truth)) if not v]
            if missing:
                logger.warning("qid=%s missing fields %s — keeping row, some metrics may be skipped", qid, missing)

            valid.append({
                **d,
                "qid": qid,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
                "_missing_fields": missing,   # downstream can check this
            })
        return valid

    # ── RAGAS execution ───────────────────────────────────────────────────────

    def _evaluate_sample(
        self,
        sample: Dict[str, Any],
        metrics: Dict[str, Any],
        gt_field: str,
        llm_wrapper,
        embeddings_wrapper,
    ) -> Dict[str, Optional[float]]:
        """Run RAGAS on a single sample with retry logic."""
        dataset = self.Dataset.from_dict({
            "question":     [str(sample["question"])],
            "answer":       [str(sample["answer"])],
            "ground_truth": [str(sample.get(gt_field, ""))],
            "contexts":     [sample.get("contexts", [])],
        })

        last_df = None
        for attempt in range(self._RETRIES):
            try:
                result  = self.evaluate(
                    dataset,
                    metrics    = list(metrics.values()),
                    llm        = llm_wrapper,
                    embeddings = embeddings_wrapper,
                    run_config = self.RunConfig(timeout=600),
                )
                last_df = result.to_pandas() ## convert to dataframe
              
            
                return {
                    m: (None if math.isnan(v := float(last_df[m].iloc[0])) else v)
                    if m in last_df.columns else None
                    for m in metrics
                } # how much faithfullness , answer_correctness
            except Exception as e:
                logger.warning(
                    "qid=%s | attempt %d/%d failed for %s: %s",
                    sample.get("qid"), attempt + 1, self._RETRIES, list(metrics.keys()), e,
                )

        # All retries exhausted — log the last successful DataFrame if available.
        if last_df is not None:
            logger.warning("Raw RAGAS df for qid=%s:\n%s", sample.get("qid"), last_df.to_string())

        return {m: None for m in metrics}

    def _run_metric_group(
        self,
        samples: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        gt_field: str,
        results_by_qid: Dict[str, Dict],
        llm_wrapper,
        embeddings_wrapper,
        label: str,
        tracker: Optional[TokenTracker] = None,
    ) -> None:
        logger.info("Evaluating [%s] across %d samples", label, len(samples))

        for idx, sample in enumerate(samples, 1): ##generation data ---> sample
          
            qid = sample["qid"]
            logger.info("[%d/%d] %s — qid=%s", idx, len(samples), label, qid)

            if tracker:
                tracker.set_context(metric=label, qid=qid)

            scores = self._evaluate_sample(sample, metrics, gt_field, llm_wrapper, embeddings_wrapper)
            results_by_qid[qid].update(scores)

            if tracker:
                tracker.clear_context()
                token_count = (
                    tracker.get_summary()
                    .get("per_question", {}).get(qid, {})
                    .get("per_metric", {}).get(label, {})
                    .get("total_tokens", 0)
                )
                score_display = {m: f"{v:.3f}" if v is not None else "None" for m, v in scores.items()}
                logger.info("qid=%s scores=%s tokens=%d", qid, score_display, token_count)

    # ── Metric group definitions ──────────────────────────────────────────────

    def _build_metric_groups(
        self,
        valid: List[Dict[str, Any]],
        requested: Optional[set],
    ) -> List[Dict[str, Any]]:
        all_metrics = set(GENERATION_METRICS) if requested is None else set(requested)
        groups: List[Dict[str, Any]] = []

        def has(d, field):
            return bool(str(d.get(field) or "").strip())

        # Group 1: context-dependent metrics
        context_metrics = {
            m: self._ragas_metrics[m]
            for m in ("faithfulness", "context_precision")
            if m in all_metrics and m in self._ragas_metrics
        }
        if context_metrics:
            context_samples = [
                d for d in valid
                if d.get("contexts") and has(d, "answer") and has(d, "ground_truth")
            ]
            if context_samples:
                groups.append({
                    "metrics":  context_metrics,
                    "gt_field": "ground_truth",
                    "samples":  context_samples,
                    "label":    "faithfulness+context_precision",
                })

        # Group 2: answer correctness
        if "answer_correctness" in all_metrics and "answer_correctness" in self._ragas_metrics:
            ac_samples = [d for d in valid if has(d, "ground_truth") and has(d, "answer")]
            if ac_samples:
                groups.append({
                    "metrics":  {"answer_correctness": self._ragas_metrics["answer_correctness"]},
                    "gt_field": "ground_truth",
                    "samples":  ac_samples,
                    "label":    "answer_correctness",
                })

        # Group 3: answer relevancy
        if "answer_relevancy" in all_metrics and "answer_relevancy" in self._ragas_metrics:
            ar_samples = [d for d in valid if has(d, "answer") and has(d, "question")]
            if ar_samples:
                groups.append({
                    "metrics":  {"answer_relevancy": self._ragas_metrics["answer_relevancy"]},
                    "gt_field": "ground_truth",
                    "samples":  ar_samples,
                    "label":    "answer_relevancy",
                })

        return groups   


    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        data: List[Dict[str, Any]],   ## generation_data
        llm_wrapper=None,
        embeddings_wrapper=None,
        requested_metrics: Optional[List[str]] = None,
        tracker: Optional[TokenTracker] = None,
    ) -> Optional[GenerationMetrics]:
        if not llm_wrapper or not embeddings_wrapper:
            raise RuntimeError("RAGAS requires llm_wrapper and embeddings_wrapper")

        valid = self._validate_samples(data)
       
        if not valid:
            logger.warning("No valid samples to evaluate — all samples were filtered out.")
            return None

        requested = set(requested_metrics) if requested_metrics else None

        results_by_qid: Dict[str, Dict] = {d["qid"]: {} for d in valid}
        print("results_by_qid : ",results_by_qid) ## only taking the qid and store in result_by_id like  q1_chunk1': {}, 'q2_chunk1': {}, 'q3_chunk1': {}, 'q1_chunk2': {},
        

        for group in self._build_metric_groups(valid, requested): 
            if tracker:
                tracker.clear_context() ## everything change to none
            self._run_metric_group(
                samples        = group["samples"],
                metrics        = group["metrics"],
                gt_field       = group["gt_field"],
                results_by_qid = results_by_qid,
                llm_wrapper    = llm_wrapper,
                embeddings_wrapper = embeddings_wrapper,
                label          = group["label"],
                tracker        = tracker,
            )

           
        

        return self._aggregate(valid, results_by_qid)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(
        self,
        valid: List[Dict[str, Any]],
        results_by_qid: Dict[str, Dict],
    ) -> GenerationMetrics:
        accumulators: Dict[str, List[float]] = {m: [] for m in GENERATION_METRICS}
        per_query: List[Dict[str, Any]] = []
        

        for d in valid:
            qid    = d["qid"]
            scores = results_by_qid[qid]
            entry  = {"qid": qid, "query": d.get("question", ""), "tag": d.get("tag", "")}

            for m in GENERATION_METRICS:
                score    = scores.get(m)
                entry[m] = score
                if isinstance(score, float):
                    accumulators[m].append(score)

            per_query.append(entry)

        return GenerationMetrics(
            faithfulness       = _mean(accumulators["faithfulness"]),
            context_precision  = _mean(accumulators["context_precision"]),
            answer_correctness = _mean(accumulators["answer_correctness"]),
            answer_relevancy   = _mean(accumulators["answer_relevancy"]),
            per_query_metrics  = per_query,
        )