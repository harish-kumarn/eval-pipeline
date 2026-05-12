"""
orchestrator.py
---------------
Orchestrates the full evaluation pipeline:
  1. Retrieval phase   — queries VisionRAG for every question.
  2. Metrics phase     — computes retrieval + generation metrics via RAGAS.
  3. Persistence phase — saves JSON output files for downstream analysis.
"""

import ast
import csv
import json
from pathlib import Path
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from components.metrics.llm_providers import GeminiEmbeddings, GeminiLLM
from components.metrics.metrics_calculator import GenerationMetricsCalculator, RetrievalMetricsCalculator
from components.metrics.token_tracker import TokenTracker
from components.retriever.retriever import VisionRAGRetriever
from config.config_loader import load_config, load_metrics_config
from config.logger import setup_logger
from utils.data_extractor import DataExtractor
from utils.support_evals.file_handler import FileHandler

logger = setup_logger(name="eval_pipeline", tag="ORCHESTRATOR")
CONFIG = load_config("config/config.yml")


class EvaluationOrchestrator:
    """Runs the end-to-end RAG evaluation pipeline."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        logger.info("Initializing EvaluationOrchestrator")

        self.retriever = VisionRAGRetriever(
            user_id             = CONFIG["vrag_retriever"]["user_id"],
            collection_name     = CONFIG["vrag_retriever"]["collection_name"],
            query_reformulation = CONFIG["vrag_retriever"]["query_reformulation"],
            use_reranking       = CONFIG["vrag_retriever"]["use_reranking"],
            use_hybrid          = CONFIG["vrag_retriever"]["use_hybrid"],
            chunk_format        = CONFIG["vrag_retriever"]["chunk_format"],
        )

        self.data_extractor           = DataExtractor(CONFIG["target_chunks"])
        self.retrieval_calculator     = RetrievalMetricsCalculator()
        self.generation_calculator    = GenerationMetricsCalculator()

        self.output_dir               = output_dir or Path(CONFIG["results_dirs"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_answers_file   = self.output_dir / CONFIG["results_dirs"]["generated_answers_file"]
        self.visionrag_file           = self.output_dir / CONFIG["results_dirs"]["vrag_result_json"]
        self.retrieval_metric_path = self.output_dir / "retriever_metrics.json"

        logger.info("✓ EvaluationOrchestrator ready → %s", self.output_dir)

    # ── Pipeline entry point ──────────────────────────────────────────────────

    async def run(self, vrag_file_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the retrieval phase and return aggregated results."""
        logger.info("=" * 60)
        logger.info("Pipeline run started")

        questions = self.data_extractor.extract_questions()
        logger.info("Loaded %d questions for evaluation", len(questions))

        await self._run_retriever(questions, vrag_file_name=vrag_file_name)

    # ── Retrieval phase ───────────────────────────────────────────────────────

    async def _run_retriever(self, questions, vrag_file_name: Optional[str] = None) -> None:
        """Query VisionRAG for every question and persist results to disk."""
        logger.info("-" * 40)
        logger.info("RETRIEVAL PHASE — %d queries", len(questions))

        ### Initializing chat
        self.retriever.reinitialize_chat()
        results = []

        for i, q in enumerate(questions, 1):

            is_last = i == len(questions)

            logger.info(f"[{i}/{len(questions)}] Retrieving qid={q.qid} (is_last={is_last})")

            try:
                result = await self.retriever.retrieve(
                    q,
                    vrag_file_name=vrag_file_name,
                    is_last=is_last,
                )
                logger.info("Completed qid=%s successfully", q.qid)

            except Exception as e:

                logger.exception("Failed qid=%s error=%s", q.qid, str(e), )
                results.append({ "qid": q.qid, "error": str(e), })

        logger.info("Finished processing all questions")


    # ── Metrics phase ─────────────────────────────────────────────────────────

    def compute_retriever_metrics(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load retrieval output, run retrieval metrics, and save JSON reports."""
        logger.info("-" * 40)
        logger.info("RETRIEVER METRICS PHASE")

        target_file      = file_path or self.visionrag_file
        retrieval_results = FileHandler.load_json(target_file)
        logger.info("Loaded %d retrieval results from %s", len(retrieval_results), target_file)

        # Inject ground truth where missing
        for d in retrieval_results:
            if not d.get("ground_truth"):
                d["ground_truth"] = self.data_extractor.get_ground_truth(d["qid"]) or ""
                logger.info("Injected ground_truth for qid=%s", d["qid"])

        metrics_config = load_metrics_config(CONFIG["metrics_config_path"])
        retrieval_req  = metrics_config.get("retriever_metrics")
        logger.info("Metrics config loaded — retrieval_metrics=%s", retrieval_req)

        retrieval_metrics = self.retrieval_calculator.compute(
            retrieval_results,
            k                 = CONFIG["vrag_client"]["final_top_k"],
            requested_metrics = retrieval_req,
        )
        logger.info(
            "✅ Retrieval metrics — MRR@k=%.3f, Recall@k=%.3f, Precision@k=%.3f, NDCG@k=%.3f",
            retrieval_metrics.mrr_at_k or 0,
            retrieval_metrics.recall_at_k or 0,
            retrieval_metrics.precision_at_k or 0,
            retrieval_metrics.ndcg_at_k or 0,
        )

        
        retrieval_metrics_dict = retrieval_metrics.to_dict()
        FileHandler.save_json(retrieval_metrics_dict, self.retrieval_metric_path)
        logger.info("Retrieval metrics saved → %s", self.retrieval_metric_path)

        logger.info("=" * 60)
        logger.info("RETRIEVER METRICS PHASE complete")
        return {"retrieval_metrics": retrieval_metrics_dict}


    def compute_generator_metrics(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load retrieval output, run generation metrics, and save JSON reports."""
        logger.info("-" * 40)
        logger.info("GENERATOR METRICS PHASE")

        target_file       = file_path or self.visionrag_file
        retrieval_results  = FileHandler.load_json(target_file)
        logger.info("Loaded %d retrieval results from %s", len(retrieval_results), target_file)

        # Inject ground truth where missing
        for d in retrieval_results:
            if not d.get("ground_truth"):
                d["ground_truth"] = self.data_extractor.get_ground_truth(d["qid"]) or ""
                logger.info("Injected ground_truth for qid=%s", d["qid"])

        # Build generation dataset
        generation_data, empty_context_count = self._build_generation_dataset(retrieval_results)
        logger.info(
            "Generation dataset built — %d samples (%d with empty contexts)",
            len(generation_data), empty_context_count,
        )

        metrics_config = load_metrics_config(CONFIG["metrics_config_path"])
        generation_req = metrics_config.get("generation_metrics")
        logger.info("Metrics config loaded — generation_metrics=%s", generation_req)

        tracker    = TokenTracker()
        llm        = LangchainLLMWrapper(GeminiLLM(tracker=tracker))
        embeddings = LangchainEmbeddingsWrapper(GeminiEmbeddings(tracker=tracker))

        logger.info("Computing generation metrics (RAGAS)…")
        generation_metrics = self.generation_calculator.compute(
            generation_data,
            llm_wrapper        = llm,
            embeddings_wrapper = embeddings,
            requested_metrics  = generation_req,
            tracker            = tracker,
        )

        results: Dict[str, Any] = {}

        if generation_metrics:
            logger.info(
                "✅ Generation metrics — Faithfulness=%.3f, ContextPrecision=%.3f, "
                "AnswerCorrectness=%.3f, AnswerRelevancy=%.3f",
                generation_metrics.faithfulness or 0,
                generation_metrics.context_precision or 0,
                generation_metrics.answer_correctness or 0,
                generation_metrics.answer_relevancy or 0,
            )
            out_path = self.output_dir / "generator_metrics.json"
            generation_metrics_dict = generation_metrics.to_dict()
            FileHandler.save_json(generation_metrics_dict, out_path)
            logger.info("Generation metrics saved → %s", out_path)
            results["generation_metrics"] = generation_metrics_dict
        else:
            logger.warning("No valid generation metrics produced — skipping generation metrics output.")

        results["token_usage"] = self._log_and_save_token_usage(tracker)

        logger.info("=" * 60)
        logger.info("GENERATOR METRICS PHASE complete")
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_generation_dataset(
        self, retrieval_results: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Build the dataset consumed by RAGAS from retrieval results.

        Returns:
            (generation_data, empty_context_count)
        """
        generation_data: List[Dict[str, Any]] = []
        empty_context_count = 0

        for d in retrieval_results:
            retrieved_cids = d.get("retrieved_chunk_ids", [])[:CONFIG["vrag_client"]["final_top_k"]]
            contexts: List[str] = []

            for cid in retrieved_cids:
                text = self.data_extractor.get_chunk_text(cid) or ""
                if not text.strip() and cid == d.get("ground_truth_chunk_id"):
                    text = f"Ground-truth context retrieved (text missing for {cid})"
                contexts.append(text)

            non_empty = sum(1 for c in contexts if c.strip())
            if non_empty == 0:
                logger.warning(
                    "qid=%s: none of the retrieved chunks %s are in target_chunks.json — "
                    "context-dependent metrics will be skipped.",
                    d.get("qid"), retrieved_cids,
                )
                empty_context_count += 1
            else:
                logger.info(
                    "qid=%s: resolved %d/%d retrieved chunks to text.",
                    d.get("qid"), non_empty, len(retrieved_cids),
                )

            generation_data.append({
                "qid":            d.get("qid"),
                "question":       d.get("question"),
                "answer":         d.get("generated_answer"),
                "ground_truth":   d.get("ground_truth"),
                "answer_extract": self.data_extractor.get_answer_extract(d.get("qid", "")),
                "contexts":       contexts,
                "tag":            d.get("tag"),
            })

        return generation_data, empty_context_count


    def _log_and_save_token_usage(self, tracker: TokenTracker) -> Dict[str, Any]:
        """Log per-query token breakdown and save the summary JSON."""
        token_summary = tracker.get_summary()
        per_q = token_summary.get("per_question", {})

        logger.info("Starting RAGAS token calculation")
        if per_q:
            logger.info("── Token usage per query ──────────────────────────────")
            logger.info(
                "  %-40s  %8s  %10s  %9s  %7s",
                "Query ID", "Prompt", "Completion", "Embedding", "Total",
            )
            logger.info("  " + "-" * 80)
            for qid, q_data in sorted(per_q.items()):
                logger.info(
                    "  %-40s  %8d  %10d  %9d  %7d",
                    qid,
                    q_data["prompt_tokens"],
                    q_data["completion_tokens"],
                    q_data["embedding_tokens"],
                    q_data["total_tokens"],
                )
                for metric_name, m_data in sorted(q_data.get("per_metric", {}).items()):
                    logger.info(
                        "      ├─ %-36s  %8d  %10d  %9d  %7d",
                        metric_name,
                        m_data["prompt_tokens"],
                        m_data["completion_tokens"],
                        m_data["embedding_tokens"],
                        m_data["total_tokens"],
                    )
            logger.info("  " + "-" * 80)
        else:
            logger.info("No per-query token data recorded (RAGAS may have made no LLM calls).")

        out_path = self.output_dir / "token_usage.json"
        FileHandler.save_json(token_summary, out_path)
        total = token_summary["total"]
        logger.info(
            "✅ Token usage TOTAL — prompt=%d, completion=%d, embedding≈%d, total=%d",
            total["prompt_tokens"], total["completion_tokens"],
            total["embedding_tokens"], total["total_tokens"],
        )
        logger.info("Token usage saved → %s", out_path)

        return token_summary

    # ── populate_visionrag_results ────────────────────────────────────────────

    def populate_visionrag_results(self, log_path: Path) -> None:
        """Construct visionragresult.json from golden dataset, generated answers and retrieved chunks."""
        
        logger.info(f"Populating results from golden dataset, answers and log: {log_path}")
        log_data  = self._parse_retrieval_log(log_path)
        questions = self.data_extractor.extract_questions()
        all_results = self._merge_visionrag_results(questions, log_data)

        if all_results:
            FileHandler.save_json(all_results, self.visionrag_file)
            logger.info("Successfully populated %d entries into %s", len(all_results), self.visionrag_file)
        else:
            logger.warning("No entries were constructed for visionrag result.")


    def _parse_retrieval_log(self, log_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parse CSV log into:
        
        {
            query: {
                "retrieved_chunk_ids": [...],
                "generated_answer": "..."
            }
        }
        """

        log_data: Dict[str, Dict[str, Any]] = {}

        if not log_path.exists():
            logger.info(
                "Log file not found: %s — retrieved chunks will be empty",
                log_path
            )
            return log_data

        fmt = CONFIG.get("chunk_format", "chunk{id}")

        try:
            with open(log_path, "r", encoding="utf-8") as f:

                for row in csv.DictReader(f):

                    query = row.get("query", "").strip()
                    chunks_str = row.get("final_retrieved_chunks", "")
                    answer = row.get("answer", "").strip()

                    if not query:
                        continue

                    retrieved_chunks = []

                    if chunks_str:
                        try:
                            chunks = ast.literal_eval(chunks_str)

                            if isinstance(chunks, list):
                                retrieved_chunks = [
                                    fmt.format(id=cid)
                                    for cid in chunks
                                ]

                        except (ValueError, SyntaxError):
                            logger.info(
                                "Failed to parse chunks string: %.120s",
                                chunks_str
                            )

                    log_data[query] = {
                        "retrieved_chunk_ids": retrieved_chunks,
                        "generated_answer": answer,
                    }

        except Exception as e:
            logger.error(
                "Error reading log CSV %s: %s",
                log_path,
                e
            )

        return log_data


    def find_best_match(self, question: str, log_data: Dict[str, Dict[str, Any]]):
        """
        Find closest matching question from log_data keys.
        """

        keys = list(log_data.keys())

        # Try exact match first
        if question.strip() in log_data:
            return log_data[question.strip()]

        # Fuzzy match
        matches = get_close_matches(
            question,
            keys,
            n=1,
            cutoff=0.3
        )

        if matches:
            return log_data[matches[0]]

        return {}


    def _merge_visionrag_results(
        self,
        questions,
        log_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge golden dataset with retrieval log."""

        results = []

        for q in questions:

            matched_data = self.find_best_match(q.text, log_data)

            results.append({
                "qid": q.qid,
                "question": q.text,
                "ground_truth_chunk_id": q.chunk_id,
                "retrieved_chunk_ids":
                    matched_data.get("retrieved_chunk_ids", []),

                "generated_answer":
                    matched_data.get("generated_answer", ""),

                "tag": q.question_type,
            })

        return results







