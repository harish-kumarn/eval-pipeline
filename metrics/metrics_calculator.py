import math
import logging
import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dotenv import load_dotenv
import google.generativeai as genai

from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from core.data_models import RetrievalMetrics, GenerationMetrics

load_dotenv()
logger = logging.getLogger(__name__)

# =====================================================
# Gemini LLM (LangChain + RAGAS SAFE)
# =====================================================

class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash"

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
    model_name: str = "text-embedding-004"
    _dim: Optional[int] = None

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

    def compute(self, data, llm_wrapper, embeddings_wrapper, requested_metrics: Optional[List[str]] = None) -> GenerationMetrics:
        if not llm_wrapper or not embeddings_wrapper:
            raise RuntimeError("RAGAS requires llm_wrapper and embeddings_wrapper")

        metrics_to_run = []
        if requested_metrics:
            for m in requested_metrics:
                if m in self.metric_map:
                    metrics_to_run.append(self.metric_map[m])
                else:
                    logger.warning(f"Generation metric '{m}' not recognized. Skipping.")
        else:
            metrics_to_run = list(self.metric_map.values())

        if not metrics_to_run:
            logger.warning("No valid generation metrics to compute. Returning None.")
            return None

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

        dataset = self.Dataset.from_dict({
            "question": [str(d["question"]) for d in valid],
            "answer": [str(d["answer"]) for d in valid],
            "ground_truth": [str(d["ground_truth"]) for d in valid],
            "contexts": [d["contexts"] for d in valid],
        })


        results = self.evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=llm_wrapper,
            embeddings=embeddings_wrapper,
            run_config=self.RunConfig(timeout=600),
        )

        df = results.to_pandas()
        
        # Replace NaN with 0.0 for calculations and report
        # We use a copy for per_query to avoid mutating the DF if not needed, 
        # but here we want the metrics to be clean.
        per_query = []
        for i, row in df.iterrows():
            qid = valid[i].get("qid")
            metric_res = {
                "qid": qid,
                "query": valid[i].get("question"),
                "tag": valid[i].get("tag")
            }
            has_nan = False
            for m_name in self.metric_map:
                if m_name in row:
                    val = row[m_name]
                    if isinstance(val, (float, int)) and math.isnan(val):
                        logger.warning(f"Metric '{m_name}' is NaN for query ID {qid}. Setting to 0.0.")
                        val = 0.0
                        has_nan = True
                    metric_res[m_name] = float(val)
            per_query.append(metric_res)

        def get_mean(col_name):
            if col_name in df:
                # mean() ignores NaN by default. If all are NaN, it returns NaN.
                val = df[col_name].mean()
                return float(val) if not math.isnan(val) else 0.0
            return None

        return GenerationMetrics(
            faithfulness=float(df["faithfulness"].mean()) if "faithfulness" in df else None,
            answer_relevancy=float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
            context_recall=float(df["context_recall"].mean()) if "context_recall" in df else None,
            context_precision=float(df["context_precision"].mean()) if "context_precision" in df else None,
            answer_correctness=float(df["answer_correctness"].mean()) if "answer_correctness" in df else None,
            per_query_metrics=per_query,
        )
