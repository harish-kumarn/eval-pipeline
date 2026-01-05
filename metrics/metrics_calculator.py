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
# Metric Models
# =====================================================

@dataclass
class RetrievalMetrics:
    mrr_at_k: float
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mean_rank: float
    k: int
    total_queries: int
    per_query_metrics: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mrr@k": self.mrr_at_k,
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "ndcg@k": self.ndcg_at_k,
            "mean_rank": self.mean_rank,
            "k": self.k,
            "total_queries": self.total_queries,
            "per_query_metrics": self.per_query_metrics,
        }


@dataclass
class GenerationMetrics:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    per_query_metrics: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "per_query_metrics": self.per_query_metrics,
        }


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

    @staticmethod
    def _dcg(rels: List[bool], k: int) -> float:
        return sum(
            1 / math.log2(i + 2)
            for i, r in enumerate(rels[:k]) if r
        )

    def compute(self, data: List[Dict[str, Any]], k: int = 5) -> RetrievalMetrics:
        if not data:
            return RetrievalMetrics(0, 0, 0, 0, float("inf"), k, 0, [])

        mrr, recall, precision, ndcg, ranks = [], [], [], [], []
        per_query = []

        for d in data:
            gt = d.get("ground_truth_chunk_id")
            rels = [r == gt for r in d.get("retrieved_chunk_ids", [])]

            rank = next((i + 1 for i, r in enumerate(rels[:k]) if r), None)

            mrr.append(1 / rank if rank else 0)
            recall.append(1 if rank else 0)
            precision.append(sum(rels[:k]) / k)

            dcg = self._dcg(rels, k)
            idcg = self._dcg([True] * min(sum(rels), k), k)
            ndcg.append(dcg / idcg if idcg else 0)

            if rank:
                ranks.append(rank)

            per_query.append({
                "qid": d.get("qid"),
                "rank": rank,
                "mrr": mrr[-1],
                "recall": recall[-1],
                "precision": precision[-1],
                "ndcg": ndcg[-1],
            })

        return RetrievalMetrics(
            sum(mrr) / len(mrr),
            sum(recall) / len(recall),
            sum(precision) / len(precision),
            sum(ndcg) / len(ndcg),
            sum(ranks) / len(ranks) if ranks else float("inf"),
            k,
            len(data),
            per_query,
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
        )
        from datasets import Dataset

        self.evaluate = evaluate
        self.RunConfig = RunConfig
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]
        self.Dataset = Dataset

    def compute(self, data, llm_wrapper, embeddings_wrapper) -> GenerationMetrics:
        if not llm_wrapper or not embeddings_wrapper:
            raise RuntimeError("RAGAS requires llm_wrapper and embeddings_wrapper")

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
            "question": [d["question"] for d in valid],
            "answer": [d["answer"] for d in valid],
            "ground_truth": [d["ground_truth"] for d in valid],
            "contexts": [d["contexts"] for d in valid],
        })

        results = self.evaluate(
            dataset,
            metrics=self.metrics,
            llm=llm_wrapper,
            embeddings=embeddings_wrapper,
            run_config=self.RunConfig(timeout=600),
        )

        df = results.to_pandas()

        per_query = []
        for i, row in df.iterrows():
            per_query.append({
                "qid": valid[i].get("qid"),
                "faithfulness": float(row["faithfulness"]),
                "answer_relevancy": float(row["answer_relevancy"]),
                "context_recall": float(row["context_recall"]),
                "context_precision": float(row["context_precision"]),
            })

        return GenerationMetrics(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevancy=float(df["answer_relevancy"].mean()),
            context_recall=float(df["context_recall"].mean()),
            context_precision=float(df["context_precision"].mean()),
            per_query_metrics=per_query,
        )
