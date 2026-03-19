from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class PerformanceCategory(Enum):
    BOTH_SUCCESS = "both_success"
    RETRIEVAL_FAIL_GEN_SUCCESS = "retrieval_fail_gen_success"
    RETRIEVAL_SUCCESS_GEN_FAIL = "retrieval_success_gen_fail"
    BOTH_FAIL = "both_fail"


@dataclass
class QueryAnalysis:
    query_id: str
    query: str
    retrieval_score: float
    generation_score: float
    category: PerformanceCategory
    retrieval_metrics: Dict
    generation_metrics: Dict
    retrieval_status: str  # MISS, HIT
    tag: str = "" # direct, indirect, image_based
    hit_type: Optional[str] = None  # early_hit, late_hit
    outcome: str = "" # hallucination_from_prior, complete_failure, etc.
