"""
prompt_comparison.py
====================
Two evaluation functions to compare RAGAS default prompts vs Maruyama custom prompts.

Functions
---------
evaluate_with_ragas_prompts(data, llm_wrapper, embeddings_wrapper, tracker)
    Runs faithfulness, context_precision and answer_correctness using the
    *default* RAGAS prompts (includes few-shot examples, generic instructions).

evaluate_with_maruyama_prompts(data, llm_wrapper, embeddings_wrapper, tracker)
    Runs the same three metrics but replaces every prompt with a short,
    Maruyama-domain-specific instruction that has NO few-shot examples.
    This is the main lever for reducing token cost.

Both functions return the same schema:
    {
        "scores": {
            "faithfulness": float,
            "context_precision": float,
            "answer_correctness": float,
        },
        "token_usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "embedding_tokens": int,
            "total_tokens": int,
        },
        "per_query": [ {qid, question, faithfulness, context_precision, answer_correctness} ]
    }

Usage
-----
    from metrics.prompt_comparison import (
        evaluate_with_ragas_prompts,
        evaluate_with_maruyama_prompts,
    )

    result_ragas    = evaluate_with_ragas_prompts(data, llm, embeddings, tracker)
    result_maruyama = evaluate_with_maruyama_prompts(data, llm, embeddings, tracker)
"""

import math
import logging
from typing import List, Dict, Any, Optional

from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, ContextPrecision, AnswerCorrectness
from datasets import Dataset

from metrics.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


# ===========================================================================
# ── HELPER ──────────────────────────────────────────────────────────────────
# ===========================================================================

def _run_evaluation(
    data: List[Dict[str, Any]],
    metric_objects: Dict[str, Any],
    llm_wrapper,
    embeddings_wrapper,
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, Any]:
    """
    Core loop: evaluate each metric × question, return aggregated results.

    Parameters
    ----------
    data            : list of dicts with keys: qid, question, answer, ground_truth, contexts
    metric_objects  : dict of {metric_name: ragas_metric_instance}
    llm_wrapper     : LangchainLLMWrapper (already wraps GeminiLLM with tracker)
    embeddings_wrapper : LangchainEmbeddingsWrapper (already wraps GeminiEmbeddings)
    tracker         : TokenTracker to record token usage per (metric, qid)
    """
    # ── validate samples ────────────────────────────────────────────────────
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
        logger.warning("No valid samples found for evaluation.")
        return {"scores": {m: None for m in metric_objects}, "token_usage": {}, "per_query": []}

    results_by_q: Dict[str, Dict[str, float]] = {d["qid"]: {} for d in valid}

    # ── run RAGAS BATCHED ───────────────────────────────────────────────────
    m_names = list(metric_objects.keys())
    m_objs  = list(metric_objects.values())
    
    logger.info(f"    ↳ Evaluating metrics (batched): {', '.join(m_names)}")
    
    # Prepare Dataset
    ds_dict = {
        "question":    [str(d["question"]) for d in valid],
        "answer":      [str(d["answer"]) for d in valid],
        "ground_truth":[str(d["ground_truth"]) for d in valid],
        "contexts":    [d["contexts"] for d in valid],
    }
    ds = Dataset.from_dict(ds_dict)

    try:
        # Clear tracker context as per-metric/per-qid tracking is impossible with parallelism
        if tracker is not None:
            tracker.clear_context()

        result = evaluate(
            ds,
            metrics=m_objs,
            llm=llm_wrapper,
            embeddings=embeddings_wrapper,
            run_config=RunConfig(timeout=600),
        )

        df = result.to_pandas()
        for i, d in enumerate(valid):
            qid = d["qid"]
            for m_name in m_names:
                score = float(df[m_name].iloc[i]) if m_name in df.columns else 0.0
                if math.isnan(score):
                    score = 0.0
                results_by_q[qid][m_name] = score

    except Exception as e:
        logger.error(f"  Batched evaluation failed: {e}")
        # Fallback to zero scores
        for d in valid:
            qid = d["qid"]
            for m_name in m_names:
                results_by_q[qid][m_name] = 0.0

    # ── token usage ──────────────────────────────────────────────────────────
    token_usage = {}
    per_question_tokens = {}
    if tracker is not None:
        summary = tracker.get_summary()
        total = summary.get("total", {})
        token_usage = {
            "prompt_tokens":     total.get("prompt_tokens", 0),
            "completion_tokens": total.get("completion_tokens", 0),
            "embedding_tokens":  total.get("embedding_tokens", 0),
            "total_tokens":      total.get("total_tokens", 0),
        }
        per_question_tokens = summary.get("per_question", {})

    # ── aggregate ────────────────────────────────────────────────────────────
    per_query = []
    agg: Dict[str, List[float]] = {m: [] for m in metric_objects}

    for d in valid:
        qid = d["qid"]
        q_scores = results_by_q.get(qid, {})
        q_tokens = per_question_tokens.get(qid, {})
        
        entry = {
            "qid": qid,
            "question": d.get("question", ""),
            "tokens": {
                "prompt": q_tokens.get("prompt_tokens", 0),
                "completion": q_tokens.get("completion_tokens", 0),
                "embedding": q_tokens.get("embedding_tokens", 0),
                "total": q_tokens.get("total_tokens", 0),
            },
            "metric_tokens": {}
        }
        for m_name in metric_objects:
            score = q_scores.get(m_name, 0.0)
            entry[m_name] = score
            agg[m_name].append(score)
            
            # Detailed per-metric tokens
            m_toks = q_tokens.get("per_metric", {}).get(m_name, {}) 
            entry["metric_tokens"][m_name] = {
                "prompt":     m_toks.get("prompt_tokens", 0),
                "completion": m_toks.get("completion_tokens", 0),
                "embedding":  m_toks.get("embedding_tokens", 0),
                "total":      m_toks.get("total_tokens", 0),
            }
        per_query.append(entry)

    def _mean(m):
        vals = agg.get(m, [])
        return float(sum(vals) / len(vals)) if vals else None

    aggregated_scores = {m: _mean(m) for m in metric_objects}

    return {
        "scores":      aggregated_scores,
        "token_usage": token_usage,
        "per_query":   per_query,
    }


# ===========================================================================
# ── FUNCTION 1: DEFAULT RAGAS PROMPTS ───────────────────────────────────────
# ===========================================================================

def evaluate_with_ragas_prompts(
    data: List[Dict[str, Any]],
    llm_wrapper,
    embeddings_wrapper,
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, Any]:
    """
    Evaluate faithfulness, context_precision and answer_correctness using
    the **default RAGAS prompts** (generic instructions + few-shot examples).

    Parameters
    ----------
    data               : list[dict] — each item must have:
                           qid, question, answer, ground_truth, contexts (list[str])
    llm_wrapper        : LangchainLLMWrapper wrapping GeminiLLM
    embeddings_wrapper : LangchainEmbeddingsWrapper wrapping GeminiEmbeddings
    tracker            : optional TokenTracker; if provided, token usage is recorded

    Returns
    -------
    {
        "scores":      {"faithfulness": float, "context_precision": float, "answer_correctness": float},
        "token_usage": {"prompt_tokens": int, "completion_tokens": int, "embedding_tokens": int, "total_tokens": int},
        "per_query":   [ {"qid", "question", "faithfulness", "context_precision", "answer_correctness"} ]
    }
    """
    logger.info("=== Evaluating with DEFAULT RAGAS prompts ===")

    # Use fresh singleton instances so we don't carry over any custom prompts
    # from a previous call to evaluate_with_maruyama_prompts.
    ragas_faithfulness       = Faithfulness()
    ragas_context_precision  = ContextPrecision()
    ragas_answer_correctness = AnswerCorrectness()

    metric_objects = {
        "faithfulness":       ragas_faithfulness,
        "context_precision":  ragas_context_precision,
        "answer_correctness": ragas_answer_correctness,
    }

    if tracker is not None:
        tracker.reset()

    return _run_evaluation(data, metric_objects, llm_wrapper, embeddings_wrapper, tracker)


# ===========================================================================
# ── FUNCTION 2: CUSTOM MARUYAMA PROMPTS ────────────────────────────────────
# ===========================================================================

def evaluate_with_maruyama_prompts(
    data: List[Dict[str, Any]],
    llm_wrapper,
    embeddings_wrapper,
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, Any]:
    """
    Evaluate the same three metrics using **custom Maruyama prompts**.

    Key differences from default RAGAS prompts:
    - NO few-shot examples  → fewer tokens per call
    - Domain-specific instruction text (Maruyama agricultural equipment manuals,
      safety procedures, boom sprayer / agrichemical context)
    - Same JSON output schema as RAGAS so scoring logic is unchanged

    Parameters / Returns: same as evaluate_with_ragas_prompts().
    """
    logger.info("=== Evaluating with CUSTOM MARUYAMA prompts ===")

    # ── import prompt classes ────────────────────────────────────────────────
    from ragas.metrics.base import MetricWithLLM
    from ragas.prompt import PydanticPrompt
    from ragas.metrics._faithfulness import (
        StatementGeneratorPrompt,
        NLIStatementPrompt,
        StatementGeneratorInput,
        StatementGeneratorOutput,
        NLIStatementInput,
        NLIStatementOutput,
    )
    from ragas.metrics._context_precision import (
        ContextPrecisionPrompt,
        QAC,
        Verification,
    )
    from ragas.metrics._answer_correctness import (
        CorrectnessClassifier,
        QuestionAnswerGroundTruth,
        ClassificationWithReason,
    )

    # ── 1. Faithfulness: statement_generator_prompt ──────────────────────────
    class MaruyamaStatementGeneratorPrompt(StatementGeneratorPrompt):
        instruction: str = (
            "You are evaluating answers from a Maruyama agricultural equipment manual "
            "(topics: boom sprayers, agrichemicals, pest control, tractor safety). "
            "Given a question and answer, split the answer into short, self-contained "
            "factual statements. Remove all pronouns and make each statement standalone. "
            "Return a JSON object with key 'statements' containing a list of strings."
        )
        examples: list = []   

    # ── 2. Faithfulness: nli_statement_prompt ───────────────────────────────
    class MaruyamaNLIStatementPrompt(NLIStatementPrompt):
        instruction: str = (
            "You are evaluating answers for a Maruyama agricultural equipment Q&A system "
            "(topics: boom sprayers, tractor safety, agrichemical use, protective equipment). "
            "Given a context (from a technical manual), a question and a list of statements, "
            "decide for each statement whether it is SUPPORTED by the context. "
            "Return a JSON with key 'statements', a list of objects each containing "
            "'statement' and 'verdict' (1 = supported, 0 = not supported)."
        )
        examples: list = []  # no few-shot examples

    # ── 3. Context Precision prompt ──────────────────────────────────────────
    class MaruyamaContextPrecisionPrompt(ContextPrecisionPrompt):
        instruction: str = (
            "You are assessing whether a retrieved passage from a Maruyama agricultural "
            "manual (covering boom sprayers, safety procedures, agrichemical handling, "
            "tractor operation) was useful for generating the given answer. "
            "Output JSON with keys 'reason' (brief explanation) and 'verdict' "
            "(1 if useful, 0 if not)."
        )
        examples: list = []   # no few-shot examples

    # ── 4. Answer Correctness: correctness_classifier ───────────────────────
    class MaruyamaCorrectnessClassifier(CorrectnessClassifier):
        instruction: str = (
            "You are fact-checking answers for Maruyama agricultural equipment Q&A "
            "(topics: sprayer tilts, protective gear, safety procedures, engine handling). "
            "Compare each statement in the answer to the ground truth and classify as: "
            "TP (present in both), FP (in answer only), FN (in ground truth only). "
            "Return JSON with keys 'TP', 'FP', 'FN', each a list of "
            "{statement, reason} objects."
        )
        examples: list = []   # no few-shot examples

    # ── 5. Answer Correctness: statement_generator_prompt ───────────────────
    class MaruyamaACStatementGeneratorPrompt(StatementGeneratorPrompt):
        instruction: str = (
            "You are evaluating answers from a Maruyama agricultural equipment manual. "
            "Decompose the given text into short, self-contained factual statements with "
            "no pronouns. Return JSON with key 'statements' as a list of strings."
        )
        examples: list = []   # no few-shot examples

    # ── Build fresh metric instances and inject custom prompts ───────────────
    m_faithfulness = Faithfulness()
    m_faithfulness.set_prompts(**{
        "statement_generator_prompt": MaruyamaStatementGeneratorPrompt(),
        "n_l_i_statement_prompt":     MaruyamaNLIStatementPrompt(),
    })

    m_context_precision = ContextPrecision()
    m_context_precision.set_prompts(**{
        "context_precision_prompt": MaruyamaContextPrecisionPrompt(),
    })

    m_answer_correctness = AnswerCorrectness()
    m_answer_correctness.set_prompts(**{
        "correctness_classifier":     MaruyamaCorrectnessClassifier(),
        "statement_generator_prompt": MaruyamaACStatementGeneratorPrompt(),
    })

    metric_objects = {
        "faithfulness":       m_faithfulness,
        "context_precision":  m_context_precision,
        "answer_correctness": m_answer_correctness,
    }

    if tracker is not None:
        tracker.reset()

    return _run_evaluation(data, metric_objects, llm_wrapper, embeddings_wrapper, tracker)
