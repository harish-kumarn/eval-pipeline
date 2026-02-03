from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class RAGFlowResult:
    retrieval_status: str  # MISS, HIT
    outcome: str  # hallucination_from_prior, complete_failure, generator_ignoring_context, partial_answer, successful_rag
    hit_type: Optional[str] = None  # early_hit, late_hit

def analyze_rag_flow(
    retrieval_metrics: Dict,
    generation_metrics: Dict,
    thresholds: Dict
) -> RAGFlowResult:

    recall = retrieval_metrics.get("recall", 0.0)
    is_hit = recall > 0.0

    relevancy = generation_metrics.get("answer_relevancy", 0.0)
    faithfulness = generation_metrics.get("faithfulness", 0.0)
    correctness = generation_metrics.get("answer_correctness", 0.0)

    f_min = thresholds.get("faithfulness", 0.0)
    c_min = thresholds.get("answer_correctness", 0.0)
    rel_min = thresholds.get("answer_relevancy", 0.0)

    if not is_hit:
        if relevancy >= rel_min and faithfulness < f_min:
            outcome = "hallucination_from_prior"
        else:
            outcome = "complete_failure"

        return RAGFlowResult(
            retrieval_status="MISS",
            outcome=outcome
        )

    # HIT
    mrr = retrieval_metrics.get("mrr", 0.0)
    hit_type = "early_hit" if mrr >= 0.999 else "late_hit"

    if faithfulness < f_min:
        outcome = "generator_ignoring_context"
    elif correctness < c_min:
        outcome = "partial_answer"
    else:
        outcome = "successful_rag"

    return RAGFlowResult(
        retrieval_status="HIT",
        hit_type=hit_type,
        outcome=outcome
    )
