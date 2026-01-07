from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class PerformanceCategory(Enum):
    BOTH_SUCCESS = "both_success"
    RETRIEVAL_FAIL_GEN_SUCCESS = "retrieval_fail_gen_success"
    RETRIEVAL_SUCCESS_GEN_FAIL = "retrieval_success_gen_fail"
    BOTH_FAIL = "both_fail"


@dataclass
class RetrievalPattern:
    pattern_type: str
    diagnosis: str
    root_cause: str
    action: str
    severity: str


@dataclass
class GenerationPattern:
    pattern_type: str
    diagnosis: str
    inference: str
    severity: str
    production_risk: str


@dataclass
class QueryAnalysis:
    query_id: str
    query: str
    retrieval_score: float
    generation_score: float
    category: PerformanceCategory
    retrieval_metrics: Dict
    generation_metrics: Dict
    retrieval_pattern: Optional[RetrievalPattern] = None
    generation_pattern: Optional[GenerationPattern] = None
