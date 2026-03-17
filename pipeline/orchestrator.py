"""
Orchestrates the full evaluation pipeline:
  1. Retrieval phase — queries VisionRAG for every question.
  2. Metrics phase   — computes retrieval + generation metrics via RAGAS.
  3. Saves JSON output files for downstream analysis.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, Any

from core.question_extractor import QuestionExtractor
from retriever.retriever import VisionRAGRetriever
from metrics.metrics_calculator import RetrievalMetricsCalculator, GenerationMetricsCalculator
from metrics.llm_providers import GeminiLLM, GeminiEmbeddings
from metrics.token_tracker import TokenTracker
from utils.file_handler import FileHandler

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config.config_loader import load_metrics_config
from rich.console import Console

logger = logging.getLogger("rich")
console = Console()


class EvaluationOrchestrator:
    """Runs the end-to-end RAG evaluation pipeline."""

    def __init__(self, config):
        self.config = config

        self.retriever = VisionRAGRetriever(
            user_id=config["user_id"], 
            timeout=config["timeout"],
            chunk_format=config["chunk_format"],
            collection_name=config["collection_name"],
        )

        ### Initializing required components
        self.question_extractor     = QuestionExtractor(config["target_chunks"])
        self.retrieval_calculator   = RetrievalMetricsCalculator()
        self.generation_calculator  = GenerationMetricsCalculator()

        ### Initializing output directory
        self.output_dir = Path(config["output_dir"]) 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visionrag_file = self.output_dir / "visionragresult.json"

    async def run(self) -> Dict[str, Any]:
        """Run retrieval then metrics computation and return aggregated results."""
        
        questions      = self.question_extractor.extract_questions()
        await self._run_retriever(questions)
        return self._compute_metrics(self.visionrag_file)

    # ── Retrieval Phase ───────────────────────────────────────────────────────

    async def _run_retriever(self, questions) -> None:
        """Query VisionRAG for every question and persist results to disk."""
        
        all_results    = []

        console.rule("[bold blue]Retrieval Phase[/bold blue]")
        console.log(f"[bold green]Retrieving results for {len(questions)} queries…[/bold green]")

        for i, q in enumerate(questions, 1):
            console.log(f"[cyan]Processing question {i}/{len(questions)}: {q.qid}[/cyan]")
            result = await self.retriever.retrieve(q)
            result.generated_answer = self._strip_base64_images(result.generated_answer)

            all_results.append({
                "qid":                 result.qid,
                "question":            result.question,
                "ground_truth_chunk_id": result.ground_truth_chunk,
                "retrieved_chunk_ids": result.retrieved_chunks,
                "generated_answer":    result.generated_answer,
                "tag":                 q.question_type,
            })
            console.log(f"[green]✓[/green] Completed {q.qid}")

            with open(self.visionrag_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Metrics Phase ─────────────────────────────────────────────────────────

    def _compute_metrics(self, visionrag_file: Path) -> Dict[str, Any]:
        """Load retrieval output, run all metrics, and save JSON reports."""
        
        with open(self.visionrag_file, "r", encoding="utf-8") as f:
            retrieval_results = json.load(f)

        # Inject ground truth where missing
        for d in retrieval_results:                          # TODO : check this
            if not d.get("ground_truth"):
                d["ground_truth"] = self.question_extractor.get_ground_truth(d["qid"]) or ""

        # Build generation dataset
        generation_data = []
        for d in retrieval_results:
            retrieved_cids = d.get("retrieved_chunk_ids", [])[:self.config.k]
            contexts = []
            for cid in retrieved_cids:
                text = self.question_extractor.get_chunk_text(cid) or ""
                contexts.append(text)

            non_empty = sum(1 for c in contexts if c.strip())
            if non_empty == 0:
                logger.warning(
                    "qid=%s: none of the retrieved chunks %s are in target_chunks.json "
                    "— this query will be excluded from RAGAS generation metrics.",
                    d.get("qid"), retrieved_cids,
                )
            else:
                logger.debug(
                    "qid=%s: resolved %d/%d retrieved chunks to text.",
                    d.get("qid"), non_empty, len(retrieved_cids),
                )

            generation_data.append({
                "qid":            d.get("qid"),
                "question":       d.get("question"),
                "answer":         d.get("generated_answer"),
                "ground_truth":   d.get("ground_truth"),
                "answer_extract": self.question_extractor.get_answer_extract(d.get("qid", "")),
                "contexts":       contexts,
                "tag":            d.get("tag"),
            })

        # Load metrics config
        metrics_config  = load_metrics_config(self.config.metrics_config_path)
        retrieval_req   = metrics_config.get("retriever_metrics")
        generation_req  = metrics_config.get("generation_metrics")

        results: Dict[str, Any] = {}

        # 1. Retrieval Metrics
        console.rule("[bold blue]Metrics Phase[/bold blue]")
        console.log("[bold green]Computing retrieval metrics…[/bold green]")

        retrieval_metrics = self.retrieval_calculator.compute(
            retrieval_results,
            k=self.config.k,
            requested_metrics=retrieval_req,
        )
        console.log(
            f"[green]✅ Retrieval Metrics Computed[/green] "
            f"[dim](MRR: {retrieval_metrics.mrr_at_k:.2f})[/dim]"
        )

        retrieval_metrics_dict = retrieval_metrics.to_dict()
        FileHandler.save_json(retrieval_metrics_dict, self.output_dir / "retriever_metrics.json")
        results["retrieval_metrics"] = retrieval_metrics_dict

        # 2. Generation Metrics (RAGAS)
        console.log("[bold green]Computing generation metrics (RAGAS)…[/bold green]")

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
            console.log("[green]✅ Generation Metrics Computed[/green]")
            generation_metrics_dict = generation_metrics.to_dict()
            FileHandler.save_json(generation_metrics_dict, self.output_dir / "generator_metrics.json")
            results["generation_metrics"] = generation_metrics_dict
        else:
            console.print("[bold yellow]No valid generation metrics — skipping.[/bold yellow]")

        # 3. Token usage report
        console.log("[bold green]Saving token usage report…[/bold green]")
        token_summary = tracker.get_summary()
        FileHandler.save_json(token_summary, self.output_dir / "token_usage.json")

        total = token_summary["total"]
        console.log(
            f"[green]✅ Token Usage Saved[/green] "
            f"[dim](prompt={total['prompt_tokens']:,}, "
            f"completion={total['completion_tokens']:,}, "
            f"embed≈{total['embedding_tokens']:,}, "
            f"total={total['total_tokens']:,})[/dim]"
        )
        results["token_usage"] = token_summary

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



