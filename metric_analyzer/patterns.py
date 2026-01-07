from typing import Dict
from .models import RetrievalPattern, GenerationPattern


def analyze_retrieval_pattern(metrics: Dict, thresholds: Dict) -> RetrievalPattern:
    recall = metrics.get('recall', metrics.get('recall_at_k', 0))
    precision = metrics.get('precision', 0)
    mrr = metrics.get('mrr', 0)

    r_ok = recall >= thresholds['recall']
    p_ok = precision >= thresholds['precision']
    mrr_ok = mrr >= thresholds['mrr']

    if not r_ok:
        return RetrievalPattern(
            "Low Recall@k",
            "Relevant documents not being retrieved",
            "Bad embeddings / chunking / indexing strategy",
            "Change embedding model, adjust chunk size, rebuild index",
            "critical"
        )

    if r_ok and not p_ok:
        return RetrievalPattern(
            "High Recall + Low Precision",
            "Too many noisy documents retrieved",
            "Weak ranking / over-fetching",
            "Improve reranker, reduce k, add filtering",
            "warning"
        )

    if not mrr_ok:
        return RetrievalPattern(
            "Low MRR",
            "Relevant documents ranked too low",
            "Ranking mismatch between embeddings and relevance",
            "Add cross-encoder reranker, tune ranking weights",
            "warning"
        )

    return RetrievalPattern(
        "High Recall + High MRR",
        "Retrieval working well",
        "N/A",
        "Focus optimization efforts on generation stage",
        "info"
    )


def analyze_generation_pattern(metrics: Dict, thresholds: Dict) -> GenerationPattern:
    f = metrics.get('faithfulness', 0)
    cp = metrics.get('context_precision', 0)
    cr = metrics.get('context_recall', 0)
    ac = metrics.get('answer_correctness', 0)
    ar = metrics.get('answer_relevancy', 0)

    f_ok = f >= thresholds['faithfulness']
    cp_ok = cp >= thresholds['context_precision']
    cr_ok = cr >= thresholds['context_recall']
    ac_ok = ac >= thresholds['answer_correctness']
    ar_ok = ar >= thresholds['answer_relevancy']

    if not f_ok and not cp_ok and not cr_ok:
        return GenerationPattern(
            "🔴 Complete Hallucination",
            "Model hallucinating due to bad context",
            "Low faithfulness + Low context quality = Unsafe answer",
            "critical",
            "high"
        )

    if not f_ok and cp_ok and cr_ok:
        return GenerationPattern(
            "🔴 Context Ignoring",
            "Model ignoring good context (prompt failure)",
            "Good context available but model not using it",
            "critical",
            "high"
        )

    if not f_ok and cp_ok and not cr_ok:
        return GenerationPattern(
            "🔴 Gap Filling Hallucination",
            "Hallucinating to fill missing information",
            "Model making up info for incomplete context",
            "critical",
            "high"
        )

    if ac_ok and not f_ok:
        return GenerationPattern(
            "🟥 Lucky Guess (DANGEROUS)",
            "Correct answer but not grounded in context",
            "Answer appears correct by chance",
            "critical",
            "high"
        )

    if not cr_ok and f_ok:
        return GenerationPattern(
            "🟡 Retrieval Fault",
            "Context missing but model stayed grounded",
            "Retriever failed but generator didn't hallucinate",
            "warning",
            "medium"
        )

    if cr_ok and not f_ok:
        return GenerationPattern(
            "🟡 Generation Fault",
            "Generator failed (ignored good context)",
            "Retrieval succeeded but generation hallucinated",
            "critical",
            "high"
        )

    if not cp_ok and not cr_ok:
        return GenerationPattern(
            "🟢 Retriever Broken",
            "Context quality extremely poor",
            "Both precision and recall failed",
            "critical",
            "high"
        )

    if not cp_ok and cr_ok:
        return GenerationPattern(
            "🟢 Over-retrieval / Noise",
            "Too much irrelevant context retrieved",
            "Good recall but poor precision",
            "warning",
            "medium"
        )

    if cp_ok and not cr_ok:
        return GenerationPattern(
            "🟢 Under-retrieval",
            "Not enough context retrieved",
            "Precise but incomplete context",
            "warning",
            "medium"
        )

    if not ar_ok and cr_ok:
        return GenerationPattern(
            "🟣 Prompt Misalignment",
            "Answer not addressing the question",
            "Good context but answer off-topic",
            "warning",
            "medium"
        )

    if not ar_ok and not cr_ok:
        return GenerationPattern(
            "🟣 Complete Failure",
            "Bad retrieval AND bad answer",
            "Both stages failed",
            "critical",
            "high"
        )

    if f_ok and ac_ok and cr_ok:
        return GenerationPattern(
            "✅ Ideal RAG Answer",
            "Faithful, correct, well-grounded",
            "All metrics healthy",
            "info",
            "low"
        )

    return GenerationPattern(
        "Mixed Performance",
        "Some metrics passing, others failing",
        "Requires deeper investigation",
        "warning",
        "medium"
    )
