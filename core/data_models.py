"""Data models for the evaluation pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Question:
    """Represents a single question extracted from target chunks."""
    qid: str
    text: str
    chunk_id: str
    question_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from retriever module."""
    qid: str
    question: str
    ground_truth_chunk: str
    retrieved_chunks: List[str] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    generated_answer: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "ground_truth_chunk": self.ground_truth_chunk,
            "retrieved_chunks": self.retrieved_chunks,
            "retrieved_docs": self.retrieved_docs,
            "generated_answer": self.generated_answer,
            "error": self.error,
        }


@dataclass
class GenerationResult:
    """Result from generator module."""
    qid: str
    question: str
    ground_truth: str
    retrieved_context: str
    generated_answer: str
    retrieved_top_chunk: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "retrieved_context": self.retrieved_context,
            "generated_answer": self.generated_answer,
            "retrieved_top_chunk": self.retrieved_top_chunk,
            "error": self.error,
        }


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics."""
    mrr_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    mean_rank: Optional[float] = None
    k: int = 0
    total_queries: int = 0
    per_query_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "mrr@k": self.mrr_at_k,
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "ndcg@k": self.ndcg_at_k,
            "mean_rank": self.mean_rank,
            "k": self.k,
            "total_queries": self.total_queries,
            "per_query_metrics": self.per_query_metrics,
        }.items() if v is not None}


@dataclass
class GenerationMetrics:
    """Aggregated generation metrics using RAGAS framework."""
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    answer_correctness: Optional[float] = None
    per_query_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "faithfulness": self.faithfulness,
            "context_precision": self.context_precision,
            "answer_correctness": self.answer_correctness,
            "per_query_metrics": self.per_query_metrics,
        }.items() if v is not None}
