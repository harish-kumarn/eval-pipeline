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

    def __init__(self, 
                 input_questions: Optional[str] = None, 
                 generated_answers_path: Optional[Path] = None, 
                 visionrag_populated_path: Optional[Path] = None, 
                 output_dir: Optional[Path] = None) -> None:

        ### Initialize Paths
        self.output_dir               = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_answers_file   = generated_answers_path
        self.visionrag_file           = visionrag_populated_path
        self.retrieval_metric_path = self.output_dir / CONFIG["results_dirs"]["retriever_metrics"]
        self.retrieval_metric_path_csv = self.output_dir / CONFIG["results_dirs"]["retriever_metrics_csv"]


        ### Initialize Metrics Calculators and Vrag Client
        self.data_extractor           = DataExtractor(input_questions)
        self.retrieval_calculator     = RetrievalMetricsCalculator()
        self.generation_calculator    = GenerationMetricsCalculator()
        self.retriever = VisionRAGRetriever(
            user_id             = CONFIG["vrag_retriever"]["user_id"],
            collection_name     = CONFIG["vrag_retriever"]["collection_name"],
            query_reformulation = CONFIG["vrag_retriever"]["query_reformulation"],
            use_reranking       = CONFIG["vrag_retriever"]["use_reranking"],
            use_hybrid          = CONFIG["vrag_retriever"]["use_hybrid"],
        )
        self.questions = self.data_extractor.extract_questions()

        logger.info("Loaded %d questions for evaluation", len(self.questions))
        logger.info("✓ Evaluation Orchestrator ready → %s", self.output_dir)
     

 
    # ── Pipeline entry point ──────────────────────────────────────────────────

    async def run(self, vrag_file_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the retrieval phase and return aggregated results."""

        logger.info("=" * 60)
        logger.info("Pipeline run started")
        await self._run_retriever(self.questions, vrag_file_name=vrag_file_name)


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

        target_file = file_path or self.visionrag_file  # visionragresult.json

        retrieval_results = FileHandler.load_json(target_file)
        logger.info("Loaded %d retrieval results from %s", len(retrieval_results), target_file)

        if not retrieval_results:
            logger.warning(
                "No retrieval results found in %s — skipping metrics computation", target_file
            )
            return {"retrieval_metrics": {}}

        metrics_config = load_metrics_config(CONFIG["metrics_config_path"])
        k = CONFIG["vrag_client"]["final_top_k"]

        retrieval_req = metrics_config.get("retriever_metrics")
        logger.info("Metrics config loaded — retrieval_metrics=%s", retrieval_req)

        all_metrics = list(self.retrieval_calculator.SUPPORTED_METRICS)  # ("mrr","recall","precision","ndcg")

        full_metrics = self.retrieval_calculator.compute(
            retrieval_results,
            k=k,
            requested_metrics=all_metrics,
        )
        logger.info(
            "✅ Retrieval metrics — MRR@k=%.3f, Recall@k=%.3f, Precision@k=%.3f, NDCG@k=%.3f",
            full_metrics.mrr_at_k or 0,
            full_metrics.recall_at_k or 0,
            full_metrics.precision_at_k or 0,
            full_metrics.ndcg_at_k or 0,
        )

        retrieval_metrics_dict = full_metrics.to_dict()  # save everything — metrics + per_query_metrics

        FileHandler.save_json(retrieval_metrics_dict, self.retrieval_metric_path)
        logger.info("Retrieval metrics saved → %s", self.retrieval_metric_path)

        self._save_retrieval_details_csv(
            retrieval_results=retrieval_results,
            per_query_metrics=full_metrics.per_query_metrics,
            metric_fields=all_metrics,
            csv_path=self.retrieval_metric_path_csv,
        )

        logger.info("=" * 60)
        logger.info("RETRIEVER METRICS PHASE complete")
        return {"retrieval_metrics": retrieval_metrics_dict}

    def _save_retrieval_details_csv(
        self,
        retrieval_results: List[Dict[str, Any]],
        per_query_metrics: List[Dict[str, Any]],
        metric_fields: List[str],
        csv_path: Path,
    ) -> Path:
        fieldnames = [
            "qid",
            "query",
            "tag",
            "ground_truth",
            "vector_chunk_ids",
            "bm25_chunk_ids",
            "pre_rerank_chunk_ids",
            "post_rerank_chunk_ids",
            "retrieved_chunk_ids",
            "ground_truth_chunk_id",
            "retrieved_list",
            "predicted",
            "retrieved_position",
            "rank",
            *metric_fields,
        ]

        rows = []
        for pq in per_query_metrics:
            row = {
                "qid":                    pq.get("qid"),
                "query":                  pq.get("query"),
                "tag":                    pq.get("tag"),
                "ground_truth":           pq.get("ground_truth", ""), 
                "vector_chunk_ids":       "; ".join(map(str, pq.get("vector_chunk_ids") or [])),
                "bm25_chunk_ids":         "; ".join(map(str, pq.get("bm25_chunk_ids") or [])),
                "pre_rerank_chunk_ids":   "; ".join(map(str, pq.get("pre_rerank_chunk_ids") or [])),
                "post_rerank_chunk_ids":  "; ".join(map(str, pq.get("post_rerank_chunk_ids") or [])),
                #"generated_answer":       pq.get("generated_answer"),
                "retrieved_chunk_ids":    "; ".join(map(str, pq.get("retrieved_chunk_ids") or [])),
                "ground_truth_chunk_id":  "; ".join(map(str, pq.get("ground_truth_chunk_id") or [])),
                "retrieved_list":         "; ".join(map(str, pq.get("retrieved_list") or [])),
                "predicted":              pq.get("predicted"),
                "retrieved_position":     "; ".join(map(str, pq.get("retrieved_position") or [])),
                "rank":                   pq.get("rank"),
            }
            for m in metric_fields:
                row[m] = pq.get(m, "")
            rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Per-query retrieval details saved → %s", csv_path)
        return csv_path





    def compute_generator_metrics(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load retrieval output, run generation metrics, and save JSON reports."""
        logger.info("-" * 40)
        logger.info("GENERATOR METRICS PHASE")
        target_file       = file_path or self.visionrag_file
     
    
        retrieval_results  = FileHandler.load_json(target_file)
        logger.info("Loaded %d retrieval results from %s", len(retrieval_results), target_file)
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
         
            # retrieved_cids =  "page_1_chunk_1", "page_1_chunk_2" , "page_2_chunk_2", "page_1_chunk_4", "page_1_chunk_3"
            
     

            for cid in retrieved_cids:
                text = self.data_extractor.get_chunk_text(cid) or ""
                if not text.strip():
                    if cid in d.get("ground_truth_chunk_id", []):
                        text = f"Ground-truth context retrieved (text missing for {cid})"
                    else:
                        logger.debug("Skipping unresolved chunk %s (not in ground truth)", cid)
                        continue   # <-- don't append bare empty strings
                contexts.append(text)
                

            non_empty = sum(1 for c in contexts if c) #Ground truth:page_1_chunk_1 , Chunk text:""
            print("non_empty :", non_empty)
             
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
                "contexts":       contexts,
                "tag":            d.get("tag"),
            })
        print("=============", contexts)
      
            
        return generation_data, empty_context_count 


    def _log_and_save_token_usage(self, tracker: TokenTracker) -> Dict[str, Any]:
        """Log per-query token breakdown and save the summary JSON."""
        token_summary = tracker.get_summary()
        per_q = token_summary.get("per_question", {})

        logger.info("Starting RAGAS token calculation")
        if per_q:
            logger.info("── Token usage per query ──────────")
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
        all_results = self._merge_visionrag_results(self.questions, log_data)

        if all_results:
            FileHandler.save_json(all_results, self.visionrag_file)
            logger.info("Successfully populated %d entries into %s", len(all_results), self.visionrag_file)
        else:
            logger.warning("No entries were constructed for visionrag result.")


    def _parse_list(self, value: str, field_name: str, elem_type: type | None = None) -> list:
        """
        Parse a string-encoded list (e.g. "['a', 'b']" or "[1, 2, 3]") into an
        actual Python list. If elem_type is given, only elements matching that
        type are kept; if elem_type is None, the parsed list is returned as-is.
        """
        result: list = []
        if value:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    result = [x for x in parsed if isinstance(x, elem_type)] if elem_type else parsed
            except (ValueError, SyntaxError):
                logger.info(
                    "Failed to parse %s string: %.120s",
                    field_name,
                    value
                )
        return result

    def _parse_retrieval_log(self, log_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parse CSV log into:

        {
            query: {
                "retrieved_chunk_ids": [...],
                "generated_answer": "...",
                "retrieved_list": [...],
                "retrieved_position": [...],
                "vector_chunk_ids": [...],
                "bm25_chunk_ids": [...],
                "pre_rerank_chunk_ids": [...],
                "post_rerank_chunk_ids": [...],
            }
        }
        """

        log_data: Dict[str, Dict[str, Any]] = {}

        if not log_path.exists():
            logger.info("Log file not found: %s — retrieved chunks will be empty", log_path)
            return log_data

     

        try:
            with open(log_path, "r", encoding="utf-8") as f:

                reader = csv.DictReader(f)
                for row in reader:

                    query = row.get("query", "").strip()
                    answer = row.get("generated_answer", "").strip()

                    chunks_str = row.get("final_retrieved_chunks", "")
                    retrieved_list_str = row.get("retrieved_list", "")
                    retrieved_position_str = row.get("retrieved_position", "")
                    vector_chunk_ids_str = row.get("vector_chunk_ids", "")
                    bm25_chunk_ids_str = row.get("bm25_chunk_ids", "")
                    pre_rerank_chunk_ids_str = row.get("pre_rerank_chunk_ids", "")
                    post_rerank_chunk_ids_str = row.get("post_rerank_chunk_ids", "")
                    predicted = row.get("predicted", "")

                    if not query:
                        continue

                    retrieved_chunks = self._parse_list(chunks_str, "final_retrieved_chunks", str)
                    retrieved_list = self._parse_list(retrieved_list_str, "retrieved_list")
                    retrieved_position = self._parse_list(retrieved_position_str, "retrieved_position", int)  ##check if it needed
                    vector_chunk_ids = self._parse_list(vector_chunk_ids_str, "vector_chunk_ids", str)
                    bm25_chunk_ids = self._parse_list(bm25_chunk_ids_str, "bm25_chunk_ids", str)
                    pre_rerank_chunk_ids = self._parse_list(pre_rerank_chunk_ids_str, "pre_rerank_chunk_ids", str)
                    post_rerank_chunk_ids = self._parse_list(post_rerank_chunk_ids_str, "post_rerank_chunk_ids", str)

                    log_data[query] = {
                        "retrieved_chunk_ids": retrieved_chunks,
                        "generated_answer": answer,
                        "retrieved_list": retrieved_list,
                        "predicted": predicted,
                        "retrieved_position": retrieved_position,
                        "vector_chunk_ids": vector_chunk_ids,
                        "bm25_chunk_ids": bm25_chunk_ids,
                        "pre_rerank_chunk_ids": pre_rerank_chunk_ids,
                        "post_rerank_chunk_ids": post_rerank_chunk_ids,

                    }

        except Exception as e:
            logger.error("Failed to parse retrieval log %s: %s", log_path, e)
        return log_data



    def find_best_match(self, question, log_data):
        """Find exact matching question from log_data keys."""
        question = (question or "").strip()

        if not question:
            print("SKIPPED: empty/blank question text")
            return {}

        if question in log_data:
            print("exact_match:", question)
            return log_data[question]

        print("NO MATCH FOUND for:", repr(question))
        return {}


# exact_match: According to the assembly instructions, what is the correct order for attaching the components to the spray hose?
# exact_match: In what sequence should the nozzle, pipe, and lever cock be connected to the spraying line?
# fuzzy_match: Before starting the equipment, what hardware check must be performed, and what is the correct sequence for assembling the spray attachments? -> According to the assembly instructions, what is the correct order for attaching the components to the spray hose?
# fuzzy_match: What pre-start safety check should be performed on the equipment's screws? -> In what sequence should the nozzle, pipe, and lever cock be connected to the spraying line?
# fuzzy_match: What fastener-related inspection must be completed before you begin operating the machinery? -> According to the assembly instructions, what is the correct order for attaching the components to the spray hose?
# fuzzy_match: Before starting the equipment, what hardware check must be performed, and what is the correct sequence for assembling the spray attachments? -> According to the assembly instructions, what is the correct order for attaching the components to the spray hose?
# fuzzy_match: What is the specified ratio of gasoline to oil for the blended fuel? -> According to the assembly instructions, what is the correct order for attaching the components to the spray hose?
# NO MATCH FOUND for: 'Which specific brand of 50:1 2-Cycle Oil is recommended for the fuel premix?'



    # both the places have the text and metadata have question 
    def _merge_visionrag_results(
        self,
        questions,
        log_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge golden dataset with retrieval log."""
 
        results = []

        for q in questions:

            matched_data = self.find_best_match(q.text, log_data)

            results.append({  ## why not groundtruth not here 
                "qid": q.qid,
                "question": q.text,
                "tag": q.question_type,
                "ground_truth_chunk_id": q.metadata.get("answer_chunk_id", q.chunk_id),
                "ground_truth": q.metadata.get("answer_extract", ""),
                "retrieved_chunk_ids":
                    matched_data.get("retrieved_chunk_ids", []),
                "retrieved_list" : matched_data.get("retrieved_list",[]),
                "predicted" : matched_data.get("predicted",[]),
                "retrieved_position" :matched_data.get("retrieved_position",[]),
                "vector_chunk_ids": matched_data.get("vector_chunk_ids", []),
                "bm25_chunk_ids": matched_data.get("bm25_chunk_ids", []),
                "pre_rerank_chunk_ids": matched_data.get("pre_rerank_chunk_ids", []),
                "post_rerank_chunk_ids": matched_data.get("post_rerank_chunk_ids", []),
                "generated_answer":matched_data.get("generated_answer", ""),
                })
        print("RESULT====",results)
    
      
        return results
    


        




