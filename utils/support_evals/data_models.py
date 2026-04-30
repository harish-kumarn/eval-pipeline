"""Data models for the evaluation pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Input models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Question:
    """A single evaluation question, tied to a specific source chunk."""
    qid: str
    text: str
    chunk_id: str
    question_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline result models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Output from the retriever stage."""
    qid: str
    question: str
    ground_truth_chunk: str
    retrieved_chunks: List[str] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    generated_answer: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Output from the generator stage."""
    qid: str
    question: str
    ground_truth: str
    retrieved_context: str
    generated_answer: str
    retrieved_top_chunk: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics across all queries."""
    k: int
    total_queries: int
    mrr_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    mean_rank: Optional[float] = None
    per_query_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.k < 0:
            raise ValueError(f"k must be non-negative, got {self.k}")
        if self.total_queries < 0:
            raise ValueError(f"total_queries must be non-negative, got {self.total_queries}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns all fields. Metric scores are None when not computed,
        which is preserved so callers can distinguish 'not run' from '0.0'.
        """
        return {
            "k":                  self.k,
            "total_queries":      self.total_queries,
            "mrr@k":              self.mrr_at_k,
            "recall@k":           self.recall_at_k,
            "precision@k":        self.precision_at_k,
            "ndcg@k":             self.ndcg_at_k,
            "mean_rank":          self.mean_rank,
            "per_query_metrics":  self.per_query_metrics,
        }


@dataclass
class GenerationMetrics:
    """Aggregated RAGAS generation metrics across all queries."""
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    per_query_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns all fields. Scores are None when not computed,
        which is preserved so callers can distinguish 'not run' from '0.0'.
        """
        return {
            "faithfulness":       self.faithfulness,
            "context_precision":  self.context_precision,
            "answer_correctness": self.answer_correctness,
            "answer_relevancy":   self.answer_relevancy,
            "per_query_metrics":  self.per_query_metrics,
        }