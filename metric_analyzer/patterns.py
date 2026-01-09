from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class RAGFlowResult:
    retrieval_status: str  # MISS, HIT
    outcome: str  # hallucination_from_prior, complete_failure, generator_ignoring_context, partial_answer, successful_rag
    hit_type: Optional[str] = None  # early_hit, late_hit

def analyze_rag_flow(retrieval_metrics: Dict, generation_metrics: Dict, thresholds: Dict) -> RAGFlowResult:
    # 1. Was ground truth retrieved? (We can use recall or a specific 'hit' flag if we had it)

    recall = retrieval_metrics.get('recall', 0)
    is_hit = recall > 0 
    
    # Ragas metrics
    relevancy = generation_metrics.get('answer_relevancy', 0)
    faithfulness = generation_metrics.get('faithfulness', 0)
    correctness = generation_metrics.get('answer_correctness', 0)
    
    # Thresholds
    f_low = thresholds.get('faithfulness', 0.7)
    c_low = thresholds.get('answer_correctness', 0.7)
    rel_high = thresholds.get('answer_relevancy', 0.7)

    if not is_hit:
        # retrieval_status = "MISS"
        if relevancy >= rel_high:
            outcome = "hallucination_from_prior"
        else:
            outcome = "complete_failure"
        return RAGFlowResult(retrieval_status="MISS", outcome=outcome)
    else:
        # retrieval_status = "HIT"
        # Is Rank == 1? (We can use MRR to infer this: MRR=1 means Rank=1)
        mrr = retrieval_metrics.get('mrr', 0)
        hit_type = "early_hit" if mrr == 1.0 else "late_hit"
        
        if faithfulness < f_low:
            outcome = "generator_ignoring_context"
        else:
            if correctness < c_low:
                outcome = "partial_answer"
            else:
                outcome = "successful_rag"
                
        return RAGFlowResult(retrieval_status="HIT", hit_type=hit_type, outcome=outcome)
