"""
Orchestrates the full evaluation pipeline:
  1. Retrieval phase — queries VisionRAG for every question.
  2. Metrics phase   — computes retrieval + generation metrics via RAGAS.
  3. Saves JSON output files for downstream analysis.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

from utils.data_extractor import DataExtractor
from components.retriever.retriever import VisionRAGRetriever
from components.metrics.metrics_calculator import RetrievalMetricsCalculator, GenerationMetricsCalculator
from components.metrics.llm_providers import GeminiLLM, GeminiEmbeddings
from components.metrics.token_tracker import TokenTracker
from utils.support_evals.file_handler import FileHandler
import ast
import csv

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config.config_loader import load_metrics_config
from config.logger import setup_logger

# --------------------------
# Logging
# --------------------------
logger = setup_logger(name="eval_pipeline", tag="ORCHESTRATOR")
logger.info("Logger initialized for orchestrator")


class EvaluationOrchestrator:
    """Runs the end-to-end RAG evaluation pipeline."""

    def __init__(self, config):
        self.config = config
        logger.info(
            "Initializing EvaluationOrchestrator (user_id=%s, collection=%s, k=%s, timeout=%ds)",
            config["user_id"], config["collection_name"], config.get("k", "?"), config["timeout"],
        )

        self.retriever = VisionRAGRetriever(
            user_id=config["user_id"],
            timeout=config["timeout"],
            chunk_format=config["chunk_format"],
            collection_name=config["collection_name"],
            query_reformulation=config.get("query_reformulation", False),
            use_reranking=config.get("use_reranking", False),
            use_hybrid=config.get("use_hybrid", False),
        )

        ### Initializing required components
        self.data_extractor     = DataExtractor(config["target_chunks"])
        self.retrieval_calculator   = RetrievalMetricsCalculator()
        self.generation_calculator  = GenerationMetricsCalculator()

        ### Initializing output directory
        self.output_dir = Path(config["output_dir"]) 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visionrag_file = self.output_dir / "visionragresult.json"
        self.generated_answers_file = self.output_dir / "generated_answers.json"

        
        logger.info("✓ EvaluationOrchestrator ready")

    async def run(self, eval_file_name: Optional[str] = None) -> Dict[str, Any]:
        """Run retrieval then metrics computation and return aggregated results."""
        logger.info("=" * 60)
        logger.info("Pipeline run started")
        questions = self.data_extractor.extract_questions()
        logger.info("Loaded %d questions for evaluation", len(questions))
        await self._run_retriever(questions, eval_file_name=eval_file_name)
        return {}

    def populate_visionrag_results(self, log_path: Path) -> None:
        """Construct visionragresult.json from golden dataset, generated answers and retrieved chunks."""
        logger.info("Populating results from golden dataset, answers and log: %s", log_path)
        
        # 1. Load generated answers
        try:
            with open(self.generated_answers_file, "r", encoding="utf-8") as f:
                generated_answers_list = json.load(f)
            # Create a lookup map
            generated_answers_map = {item["qid"]: item.get("generated_answer", "") for item in generated_answers_list}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Failed to load %s: %s", self.generated_answers_file, e)
            return

        # 2. Parse the CSV log for retrieved chunks
        log_data = {}
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        query = row.get("query", "").strip()
                        chunks_str = row.get("final_retrieved_chunks", "[]")
                        try:
                            chunks = ast.literal_eval(chunks_str)
                            if isinstance(chunks, list):
                                fmt = self.config.get("chunk_format", "chunk{id}")
                                log_data[query] = [fmt.format(id=cid) for cid in chunks]
                        except (ValueError, SyntaxError):
                            logger.warning("Failed to parse chunks string: %s", chunks_str)
            except Exception as e:
                logger.error("Error reading log CSV %s: %s", log_path, e)
        else:
            logger.warning("Log file not found %s, retrieved chunks will be empty", log_path)

        # 3. Golden dataset
        questions = self.data_extractor.extract_questions()
        
        # 4. Merge all together
        all_results = []
        for q in questions:
            # Match query by text since log_data might use query text
            retrieved = log_data.get(q.text.strip(), [])
            
            entry = {
                "qid": q.qid,
                "question": q.text,
                "ground_truth_chunk_id": q.chunk_id,
                "retrieved_chunk_ids": retrieved,
                "generated_answer": generated_answers_map.get(q.qid, ""),
                "tag": q.question_type,
            }
            all_results.append(entry)

        # 5. Save back
        if len(all_results) > 0:
            with open(self.visionrag_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info("Successfully populated %d entries into %s", len(all_results), self.visionrag_file)
        else:
            logger.warning("No entries were constructed for visionrag result.")

    def recompute_metrics(self) -> Dict[str, Any]:
        """Re-run the metrics computation phase using the current visionragresult.json."""
        return self._compute_metrics(self.visionrag_file)

    # ── Retrieval Phase ───────────────────────────────────────────────────────

    async def _run_retriever(self, questions, eval_file_name: Optional[str] = None) -> None:
        """Query VisionRAG for every question and persist results to disk."""
        logger.info("-" * 40)
        logger.info("RETRIEVAL PHASE — %d queries", len(questions))
        
        all_results    = []
        errors = 0

        for i, q in enumerate(questions, 1):
            is_last = (i == len(questions))
            logger.info("[%d/%d] Retrieving qid=%s (is_last=%s)", i, len(questions), q.qid, is_last)
            result = await self.retriever.retrieve(q, eval_file_name=eval_file_name, is_last=is_last)
            result.generated_answer = self._strip_base64_images(result.generated_answer)  # TODO : check this

            if result.error:
                logger.info("qid=%s: retrieval error — %s", q.qid, result.error)
                errors += 1
            else:
                logger.info(
                    "qid=%s: retrieved %d chunks, answer length=%d chars",
                    q.qid,
                    len(result.retrieved_chunks or []),
                    len(result.generated_answer or ""),
                )

            all_results.append({
                "qid":                 result.qid,
                "question":            result.question,
                "generated_answer":    result.generated_answer,
            })
            
            with open(self.generated_answers_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(
            "Retrieval phase complete — %d succeeded, %d errors. Results → %s",
            len(all_results) - errors, errors, self.generated_answers_file,
        )

    # ── Metrics Phase ─────────────────────────────────────────────────────────

    def _compute_metrics(self, visionrag_file: Path) -> Dict[str, Any]:
        """Load retrieval output, run all metrics, and save JSON reports."""
        logger.info("-" * 40)
        logger.info("METRICS PHASE")
        
        with open(self.visionrag_file, "r", encoding="utf-8") as f:
            retrieval_results = json.load(f)
        logger.info("Loaded %d retrieval results from %s", len(retrieval_results), self.visionrag_file)

        # Inject ground truth where missing
        injected = 0
        for d in retrieval_results:                          # TODO : check this
            if not d.get("ground_truth"):
                d["ground_truth"] = self.data_extractor.get_ground_truth(d["qid"]) or ""
                injected += 1
        if injected:
            logger.info("Injected ground_truth for %d queries that had none", injected)

        # Build generation dataset
        generation_data = []
        empty_context_count = 0
        for d in retrieval_results:
            retrieved_cids = d.get("retrieved_chunk_ids", [])[:self.config["k"]]
            contexts = []
            for cid in retrieved_cids:
                text = self.data_extractor.get_chunk_text(cid) or ""
                contexts.append(text)

            non_empty = sum(1 for c in contexts if c.strip())
            if non_empty == 0:
                logger.info(
                    "qid=%s: none of the retrieved chunks %s are in target_chunks.json "
                    "— this query will be excluded from RAGAS generation metrics.",
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

        logger.info(
            "Generation dataset built — %d samples (%d will be skipped due to empty contexts)",
            len(generation_data), empty_context_count,
        )

        # Load metrics config
        metrics_config  = load_metrics_config(self.config["metrics_config_path"])
        retrieval_req   = metrics_config.get("retriever_metrics")
        generation_req  = metrics_config.get("generation_metrics")
        logger.info(
            "Metrics config loaded — retrieval_metrics=%s, generation_metrics=%s",
            retrieval_req, generation_req,
        )

        results: Dict[str, Any] = {}

        # 1. Retrieval Metrics
        logger.info("Computing retrieval metrics…")

        retrieval_metrics = self.retrieval_calculator.compute(
            retrieval_results,
            k=self.config["k"],
            requested_metrics=retrieval_req,
        )
        logger.info(
            "✅ Retrieval metrics — MRR@k=%.3f, Recall@k=%.3f, Precision@k=%.3f, NDCG@k=%.3f",
            retrieval_metrics.mrr_at_k or 0,
            retrieval_metrics.recall_at_k or 0,
            retrieval_metrics.precision_at_k or 0,
            retrieval_metrics.ndcg_at_k or 0,
        )

        retrieval_metrics_dict = retrieval_metrics.to_dict()
        FileHandler.save_json(retrieval_metrics_dict, self.output_dir / "retriever_metrics.json")
        logger.info("Retrieval metrics saved → %s", self.output_dir / "retriever_metrics.json")
        results["retrieval_metrics"] = retrieval_metrics_dict

        # 2. Generation Metrics (RAGAS)
        logger.info("Computing generation metrics (RAGAS)…")

        tracker    = TokenTracker()
        llm        = LangchainLLMWrapper(GeminiLLM(tracker=tracker))
        embeddings = LangchainEmbeddingsWrapper(GeminiEmbeddings(tracker=tracker))

        generation_metrics = self.generation_calculator.compute(
            generation_data,
            llm_wrapper=llm,
            embeddings_wrapper=embeddings,
            requested_metrics=generation_req,
            tracker=tracker,
        )

        if generation_metrics:
            logger.info(
                "✅ Generation metrics — Faithfulness=%.3f, ContextPrecision=%.3f, AnswerCorrectness=%.3f",
                generation_metrics.faithfulness or 0,
                generation_metrics.context_precision or 0,
                generation_metrics.answer_correctness or 0,
            )
            generation_metrics_dict = generation_metrics.to_dict()
            FileHandler.save_json(generation_metrics_dict, self.output_dir / "generator_metrics.json")
            logger.info("Generation metrics saved → %s", self.output_dir / "generator_metrics.json")
            results["generation_metrics"] = generation_metrics_dict
        else:
            logger.info("No valid generation metrics produced — skipping generation metrics output.")

        # 3. Token usage report
        token_summary = tracker.get_summary()
        FileHandler.save_json(token_summary, self.output_dir / "token_usage.json")
        total = token_summary["total"]
        logger.info(
            "✅ Token usage — prompt=%d, completion=%d, embedding≈%d, total=%d",
            total["prompt_tokens"], total["completion_tokens"],
            total["embedding_tokens"], total["total_tokens"],
        )
        logger.info("Token usage saved → %s", self.output_dir / "token_usage.json")
        results["token_usage"] = token_summary

        logger.info("=" * 60)
        logger.info("METRICS PHASE complete")
        return results


    @staticmethod
    def _strip_base64_images(text: str) -> str:
        """Remove inline base64 image data URIs from the given text."""
        return re.sub(
            r"data:image\/png;base64,[A-Za-z0-9+/=\n\r]+",
            "",
            text,
            flags=re.DOTALL,
        )



