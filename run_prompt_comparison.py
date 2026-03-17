"""
run_prompt_comparison.py
========================
Standalone runner to compare RAGAS default prompts vs custom Maruyama prompts.

Loads the existing pipeline output (outputs/visionragresult.json) and the
source chunks (data/test_small.json) to build the evaluation dataset, then
runs one of the two functions and prints a summary.

Usage
-----
# Run with DEFAULT RAGAS prompts:
    .venv/bin/python run_prompt_comparison.py --mode ragas

# Run with CUSTOM MARUYAMA prompts:
    .venv/bin/python run_prompt_comparison.py --mode maruyama
"""

import json
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from metrics.metrics_calculator import GeminiLLM, GeminiEmbeddings
from metrics.token_tracker import TokenTracker
from metrics.prompt_comparison import (
    evaluate_with_ragas_prompts,
    evaluate_with_maruyama_prompts,
)
from core.question_extractor import QuestionExtractor

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
VISIONRAG_RESULT = Path("outputs/visionragresult.json")
CHUNKS_FILE      = Path("data/target_chunks.json")      # using full chunks file
K                = 3                                   # top-k contexts to include


def build_eval_data(visionrag_file: Path, chunks_file: Path, k: int):
    """
    Build the evaluation dataset from the pipeline's retrieval output + source chunks.

    Returns a list of dicts:
        {qid, question, answer, ground_truth, contexts, tag}
    """
    extractor = QuestionExtractor(str(chunks_file))
    extractor.extract_questions()   # populates question_map

    with open(visionrag_file, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    data = []
    for d in retrieval_results:
        qid       = d.get("qid", "")
        question  = d.get("question", "")
        answer    = d.get("generated_answer", "")
        tag       = d.get("tag", "")

        # Ground truth from source chunk metadata
        ground_truth = extractor.get_ground_truth(qid) or ""

        # Contexts: text of each retrieved chunk (up to k)
        contexts = []
        for cid in d.get("retrieved_chunk_ids", [])[:k]:
            text = extractor.get_chunk_text(cid)
            if text and text.strip():
                contexts.append(text)

        # If no chunks were retrieved, fall back to the source chunk text
        if not contexts:
            chunk_id = d.get("ground_truth_chunk_id", "")
            fallback = extractor.get_chunk_text(chunk_id)
            if fallback:
                contexts = [fallback]

        if question and answer and ground_truth and contexts:
            data.append({
                "qid":          qid,
                "question":     question,
                "answer":       answer,
                "ground_truth": ground_truth,
                "contexts":     contexts,
                "tag":          tag,
            })
        else:
            logger.warning(f"Skipping {qid} — missing question/answer/ground_truth/contexts")

    return data


def print_results(mode: str, result: dict):
    """Pretty-print scores and token usage."""
    sep = "─" * 52
    print(f"\n{'═'*52}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'═'*52}")

    print("\n  SCORES")
    print(f"  {sep}")
    scores = result.get("scores", {})
    for metric, score in scores.items():
        val = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {metric:<25}  {val}")

    print(f"\n  TOKEN USAGE")
    print(f"  {sep}")
    tok = result.get("token_usage", {})
    print(f"  {'prompt_tokens':<25}  {tok.get('prompt_tokens', 0):,}")
    print(f"  {'completion_tokens':<25}  {tok.get('completion_tokens', 0):,}")
    print(f"  {'embedding_tokens':<25}  {tok.get('embedding_tokens', 0):,}")
    print(f"  {'total_tokens':<25}  {tok.get('total_tokens', 0):,}")

    per_query = result.get("per_query", [])
    if per_query:
        print(f"\n  PER-QUERY BREAKDOWN")
        print(f"  {sep}")
        for q in per_query:
            print(f"  [{q['qid']}] {q.get('question','')[:60]}...")
            # Aggregate Tokens
            t = q.get("tokens", {})
            print(f"    Total Tokens: {t.get('total',0):,} (P:{t.get('prompt',0):,}, E:{t.get('embedding',0):,}, C:{t.get('completion',0):,})")
            
            # Per-Metric Breakdown
            mt = q.get("metric_tokens", {})
            for m_name in ["faithfulness", "context_precision", "answer_correctness"]:
                score = q.get(m_name)
                s_str = f"{score:.4f}" if score is not None else "N/A"
                m_tok = mt.get(m_name, {})
                t_str = f"Tokens: {m_tok.get('total',0)} (P:{m_tok.get('prompt',0)}, E:{m_tok.get('embedding',0)}, C:{m_tok.get('completion',0)})"
                print(f"    - {m_name:<18}: {s_str} | {t_str}")
            print(f"    {'-'*48}")

    print(f"\n{'═'*52}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare RAGAS vs Maruyama prompts")
    parser.add_argument(
        "--mode",
        choices=["ragas", "maruyama", "both"],
        default="both",
        help="Which prompts to use: 'ragas', 'maruyama', or 'both' (default)",
    )
    args = parser.parse_args()

    # ── Build evaluation data ────────────────────────────────────────────────
    logger.info(f"Loading data from {VISIONRAG_RESULT} + {CHUNKS_FILE}")
    data = build_eval_data(VISIONRAG_RESULT, CHUNKS_FILE, K)
    logger.info(f"Built {len(data)} valid evaluation samples")

    if not data:
        print("No valid samples to evaluate. Check your visionragresult.json and chunks file.")
        return

    # ── Initialise LLM + Embeddings ──────────────────────────────────────────
    tracker    = TokenTracker()
    llm        = LangchainLLMWrapper(GeminiLLM(tracker=tracker))
    embeddings = LangchainEmbeddingsWrapper(GeminiEmbeddings(tracker=tracker))

    # ── Run evaluations ──────────────────────────────────────────────────────
    modes_to_run = []
    if args.mode == "both":
        modes_to_run = ["ragas", "maruyama"]
    else:
        modes_to_run = [args.mode]

    for m in modes_to_run:
        tracker.reset()
        if m == "ragas":
            result = evaluate_with_ragas_prompts(data, llm, embeddings, tracker)
            out_path = Path("outputs/ragas_results.json")
        else:
            result = evaluate_with_maruyama_prompts(data, llm, embeddings, tracker)
            out_path = Path("outputs/maruyama_results.json")

        # Save to JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved results to {out_path}")

        print_results(m, result)


if __name__ == "__main__":
    main()
