import math
import logging
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dotenv import load_dotenv
import google.generativeai as genai

from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from core.data_models import RetrievalMetrics, GenerationMetrics
from metrics.token_tracker import TokenTracker

load_dotenv()
logger = logging.getLogger(__name__)

# =====================================================
# Gemini LLM (LangChain + RAGAS SAFE)
# =====================================================

class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash"
    tracker: Optional[Any] = None  # TokenTracker instance (Any to keep Pydantic happy)

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)

        # --- Token tracking ---
        if self.tracker is not None:
            try:
                usage = response.usage_metadata
                self.tracker.record_llm(
                    prompt_tokens=usage.prompt_token_count or 0,
                    completion_tokens=usage.candidates_token_count or 0,
                )
            except Exception as te:
                logger.debug(f"Token tracking failed: {te}")

        text = response.text if response and response.text else ""

        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]

        return text.strip()


# =====================================================
# Gemini Embeddings (RAGAS SAFE)
# =====================================================

class GeminiEmbeddings(Embeddings):

    def __init__(self, model_name: str = "gemini-embedding-001", tracker=None):
        # Embeddings is a plain ABC (not Pydantic), so we need explicit __init__
        self.model_name = model_name
        self.tracker = tracker
        self._dim: Optional[int] = None

    def _embed(self, text: str) -> List[float]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        response = genai.embed_content(
            model=f"models/{self.model_name}",
            content=text,
            task_type="retrieval_document",
        )
        vec = response["embedding"]

        if self._dim is None:
            self._dim = len(vec)

        # --- Token tracking (estimate: word count) ---
        if self.tracker is not None:
            try:
                estimated = len(text.split())
                self.tracker.record_embedding(estimated)
            except Exception as te:
                logger.debug(f"Embedding token tracking failed: {te}")

        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            if not t or not t.strip():
                vectors.append([0.0] * (self._dim or 768))
            else:
                vectors.append(self._embed(t))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * (self._dim or 768)
        return self._embed(text)


# =====================================================
# Base Calculator
# =====================================================

class MetricsCalculator(ABC):
    @abstractmethod
    def compute(self, data: List[Dict[str, Any]], **kwargs):
        pass


# =====================================================
# Retrieval Metrics
# =====================================================

class RetrievalMetricsCalculator(MetricsCalculator):
    """Calculator for retrieval-specific metrics."""

    def __init__(self):
        self.metric_map = {
            "mrr": self._calculate_mrr,
            "recall": self._calculate_recall,
            "precision": self._calculate_precision,
            "ndcg": self._calculate_ndcg,
        }

    def _calculate_mrr(self, rank: Optional[int], **kwargs) -> float:
        return 1 / rank if rank else 0

    def _calculate_recall(self, rank: Optional[int], **kwargs) -> float:
        return 1 if rank else 0

    def _calculate_precision(self, rels: List[bool], k: int, **kwargs) -> float:
        return sum(rels[:k]) / k

    def _calculate_ndcg(self, rels: List[bool], k: int, **kwargs) -> float:
        dcg = self._dcg(rels, k)
        idcg = self._dcg([True] * min(sum(rels), k), k)
        return dcg / idcg if idcg else 0

    def _dcg(self, rels: List[bool], k: int) -> float:

        return sum(
            1 / math.log2(i + 2)
            for i, r in enumerate(rels[:k]) if r
        )

    def compute(self, data: List[Dict[str, Any]], k: int = 3, requested_metrics: Optional[List[str]] = None) -> RetrievalMetrics:
        if not data:
            return RetrievalMetrics(k=k, total_queries=0)

        compute_all = requested_metrics is None
        requested = set(requested_metrics) if requested_metrics else set(self.metric_map.keys())

        # Filter valid metrics and initialize storage
        active_metrics = {m: [] for m in requested if m in self.metric_map}
        ranks = []
        per_query = []

        for d in data:
            gt = d.get("ground_truth_chunk_id")
            retrieved = d.get("retrieved_chunk_ids", [])
            rels = [r == gt for r in retrieved]
            rank = next((i + 1 for i, r in enumerate(rels[:k]) if r), None)
            
            query_metrics = {
                "qid": d.get("qid"),
                "query": d.get("question"),
                "tag": d.get("tag"),
                "rank": rank
            }

            for m_name in active_metrics:
                metric_func = self.metric_map[m_name]
                val = metric_func(rank=rank, rels=rels, k=k)
                active_metrics[m_name].append(val)
                query_metrics[m_name] = val

            if rank:
                ranks.append(rank)

            per_query.append(query_metrics)

        return RetrievalMetrics(
            mrr_at_k=sum(active_metrics["mrr"]) / len(active_metrics["mrr"]) if "mrr" in active_metrics and active_metrics["mrr"] else None,
            recall_at_k=sum(active_metrics["recall"]) / len(active_metrics["recall"]) if "recall" in active_metrics and active_metrics["recall"] else None,
            precision_at_k=sum(active_metrics["precision"]) / len(active_metrics["precision"]) if "precision" in active_metrics and active_metrics["precision"] else None,
            ndcg_at_k=sum(active_metrics["ndcg"]) / len(active_metrics["ndcg"]) if "ndcg" in active_metrics and active_metrics["ndcg"] else None,
            mean_rank=sum(ranks) / len(ranks) if ranks else None,
            k=k,
            total_queries=len(data),
            per_query_metrics=per_query,
        )


# =====================================================
# Generation Metrics (RAGAS – FINAL)
# =====================================================

class GenerationMetricsCalculator(MetricsCalculator):

    def __init__(self):
        from ragas import evaluate, RunConfig
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
        from datasets import Dataset

        self.evaluate = evaluate
        self.RunConfig = RunConfig
        self.Dataset = Dataset

        self.metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

    def compute(
        self,
        data,
        llm_wrapper,
        embeddings_wrapper,
        requested_metrics: Optional[List[str]] = None,
        tracker: Optional[TokenTracker] = None,
    ) -> GenerationMetrics:
        if not llm_wrapper or not embeddings_wrapper:
            raise RuntimeError("RAGAS requires llm_wrapper and embeddings_wrapper")

        # Resolve which metrics to run
        metrics_to_run: Dict[str, Any] = {}
        if requested_metrics:
            for m in requested_metrics:
                if m in self.metric_map:
                    metrics_to_run[m] = self.metric_map[m]
                else:
                    logger.warning(f"Generation metric '{m}' not recognized. Skipping.")
        else:
            metrics_to_run = dict(self.metric_map)

        if not metrics_to_run:
            logger.warning("No valid generation metrics to compute. Returning None.")
            return None

        # Filter valid samples
        valid = []
        for d in data:
            contexts = [c for c in d.get("contexts", []) if isinstance(c, str) and c.strip()]
            if (
                d.get("question", "").strip()
                and d.get("answer", "").strip()
                and d.get("ground_truth", "").strip()
                and contexts
            ):
                valid.append({**d, "contexts": contexts})

        if not valid:
            logger.warning("No valid samples for RAGAS evaluation. Skipping.")
            return None

        # -------------------------------------------------------
        # Evaluate ONE question × ONE metric at a time so the
        # tracker can tag every LLM/embedding call with (qid, metric).
        # -------------------------------------------------------
        # results[qid][metric] = score
        results_by_q: Dict[str, Dict[str, float]] = {
            d["qid"]: {} for d in valid if d.get("qid")
        }

        for m_name, m_obj in metrics_to_run.items():
            logger.info(f"  ↳ Evaluating metric: {m_name}")

            for d in valid:
                qid = d.get("qid") or "_unknown_"

                if tracker is not None:
                    tracker.set_context(metric=m_name, qid=qid)

                single_ds = self.Dataset.from_dict({
                    "question":    [str(d["question"])],
                    "answer":      [str(d["answer"])],
                    "ground_truth":[str(d["ground_truth"])],
                    "contexts":    [d["contexts"]],
                })

                try:
                    result = self.evaluate(
                        single_ds,
                        metrics=[m_obj],
                        llm=llm_wrapper,
                        embeddings=embeddings_wrapper,
                        run_config=self.RunConfig(timeout=600),
                    )
                    df = result.to_pandas()
                    score = float(df[m_name].iloc[0]) if m_name in df.columns else 0.0
                    if math.isnan(score):
                        logger.warning(f"Metric '{m_name}' is NaN for qid={qid}. Setting to 0.0.")
                        score = 0.0
                except Exception as e:
                    logger.error(f"Failed to evaluate '{m_name}' for qid={qid}: {e}")
                    score = 0.0

                if tracker is not None:
                    tracker.clear_context()

                if qid not in results_by_q:
                    results_by_q[qid] = {}
                results_by_q[qid][m_name] = score

        # -------------------------------------------------------
        # Build per-query list and compute aggregate means
        # -------------------------------------------------------
        per_query = []
        agg: Dict[str, List[float]] = {m: [] for m in metrics_to_run}

        for d in valid:
            qid = d.get("qid") or "_unknown_"
            q_scores = results_by_q.get(qid, {})
            entry = {
                "qid":   qid,
                "query": d.get("question", ""),
                "tag":   d.get("tag", ""),
            }
            for m_name in metrics_to_run:
                score = q_scores.get(m_name, 0.0)
                entry[m_name] = score
                agg[m_name].append(score)
            per_query.append(entry)

        def _mean(m):
            vals = agg.get(m, [])
            return float(sum(vals) / len(vals)) if vals else None

        return GenerationMetrics(
            faithfulness=_mean("faithfulness"),
            answer_relevancy=_mean("answer_relevancy"),
            context_recall=_mean("context_recall"),
            context_precision=_mean("context_precision"),
            answer_correctness=_mean("answer_correctness"),
            per_query_metrics=per_query,
        )

