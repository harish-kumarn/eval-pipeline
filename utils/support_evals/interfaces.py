"""Abstract interfaces for pluggable components."""

from abc import ABC, abstractmethod
from typing import List
from utils.support_evals.data_models import Question, RetrievalResult, GenerationResult


class IRetriever(ABC):
    """Interface for retriever implementations."""

    @abstractmethod
    async def retrieve(self, question: Question) -> RetrievalResult:
        """Retrieve documents for a question."""
        pass


class IGenerator(ABC):
    """Interface for generator implementations."""

    @abstractmethod
    async def generate(
        self,
        question: str,
        context: str,
        ground_truth: str,
    ) -> GenerationResult:
        """Generate answer for a question."""
        pass


class IMetricsCalculator(ABC):
    """Interface for metrics calculators."""

    @abstractmethod
    def compute(self, results: List) -> dict:
        """Compute metrics from results."""
        pass
