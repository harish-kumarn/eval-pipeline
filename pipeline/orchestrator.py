import re
import json
import logging
from pathlib import Path
from typing import Dict, Any

from core.data_models import PipelineConfig
from core.question_extractor import QuestionExtractor
from retriever.retriever import VisionRAGRetriever
from metrics.metrics_calculator import (
    RetrievalMetricsCalculator,
    GenerationMetricsCalculator,
    GeminiLLM,
    GeminiEmbeddings,
)
from metrics.token_tracker import TokenTracker
from utils.file_handler import FileHandler

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rich.console import Console
from rich.logging import RichHandler

# Configure logger with RichHandler for better formatting if standard logging is still used elsewhere
logger = logging.getLogger("rich")
console = Console()


class EvaluationOrchestrator:

    def __init__(self, config: PipelineConfig):
        self.config = config

        self.retriever = VisionRAGRetriever(
            user_id=config.user_id,
            timeout=config.retriever_timeout,
            chunk_format=config.chunk_format,
        )

        self.question_extractor = QuestionExtractor(config.target_chunks_file)
        self.retrieval_calculator = RetrievalMetricsCalculator()
        self.generation_calculator = GenerationMetricsCalculator()

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> Dict[str, Any]:
        questions = self.question_extractor.extract_questions()
        visionrag_file = self.output_dir / "visionragresult.json"

        # 1. Retrieval Phase
        results = await self._run_retriever(questions)
        
        # 2. Metrics Phase
        return self._compute_metrics(visionrag_file)

    # retrieve from VISIONRAG
    async def _run_retriever(self, questions):
        visionrag_file = self.output_dir / "visionragresult.json"
        all_results = []

        console.rule("[bold blue]Retrieval Phase[/bold blue]")
        console.log(f"[bold green]Retrieving results for {len(questions)} queries...[/bold green]")
        
        for i, q in enumerate(questions, 1):
            console.log(f"[cyan]Processing question {i}/{len(questions)}: {q.qid}[/cyan]")
            result = await self.retriever.retrieve(q)
            result.generated_answer = self.remove_images(result.generated_answer)
            entry = {
                "qid": result.qid,
                "question": result.question,
                "ground_truth_chunk_id": result.ground_truth_chunk,
                "retrieved_chunk_ids": result.retrieved_chunks,
                "generated_answer": result.generated_answer,
                "tag": q.question_type,
            }

            all_results.append(entry)
            console.log(f"[green]✓[/green] Completed {q.qid}")

            with open(visionrag_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        return all_results

    def remove_images(self, text: str) -> str:
        """
        This Function is used to remove images from the text
        which has the format of `....text.... data:image/png;base64,/9j/4AA...`
        """
        console.log("Removing images from text")
        console.log(text)
        return re.sub(
            r"data:image\/png;base64,[A-Za-z0-9+/=\n\r]+",
            "",
            text,
            flags=re.DOTALL
        )

    
    def _compute_metrics(self, visionrag_file: Path) -> Dict[str, Any]:
        with open(visionrag_file, "r", encoding="utf-8") as f:
            retrieval_results = json.load(f)

        for d in retrieval_results:
            if not d.get("ground_truth"):
                d["ground_truth"] = self.question_extractor.get_ground_truth(d["qid"]) or ""

        def extract_context(cid: str) -> str:
            return self.question_extractor.get_chunk_text(cid) or ""

        # Build generation dataset
        generation_data = []
        for d in retrieval_results:
            contexts = [extract_context(cid) for cid in d.get("retrieved_chunk_ids", [])[:self.config.k]]
            generation_data.append({
                "qid": d.get("qid"),
                "question": d.get("question"),
                "answer": d.get("generated_answer"),
                "ground_truth": d.get("ground_truth"),
                "contexts": contexts,
                "tag": d.get("tag"),
            })

        # Load metrics config
        metrics_config = {}
        if self.config.metrics_config_path:
            config_path = Path(self.config.metrics_config_path)
            if config_path.exists():
                console.log(f"[dim]Loading metrics config from {config_path}[/dim]")
                with open(config_path, "r") as f:
                    metrics_config = json.load(f)
            else:
                console.print(f"[bold yellow]Metrics config file {config_path} not found. Using all metrics.[/bold yellow]")

        retrieval_req = metrics_config.get("retriever_metrics")
        generation_req = metrics_config.get("generation_metrics")

        results = {}

        results = {}

        # 1. Retrieval Metrics
        console.rule("[bold blue]Metrics Phase[/bold blue]")
        console.log("[bold green]Computing retrieval metrics...[/bold green]")
        
        retrieval_metrics = self.retrieval_calculator.compute(
            retrieval_results,
            k=self.config.k,
            requested_metrics=retrieval_req
        )
        
        console.log(f"[green]✅ Retrieval Metrics Computed[/green] [dim](MRR: {retrieval_metrics.mrr_at_k:.2f})[/dim]")

        retrieval_metrics_dict = retrieval_metrics.to_dict()
        FileHandler.save_json(
            retrieval_metrics_dict,
            self.output_dir / "retriever_metrics.json",
        )
        results["retrieval_metrics"] = retrieval_metrics_dict

        # 2. Generation Metrics (RAGAS)
        console.log("[bold green]Computing generation metrics (RAGAS)...[/bold green]")

        # Create a shared token tracker and inject into Gemini LLM + Embeddings
        tracker = TokenTracker()
        llm = LangchainLLMWrapper(GeminiLLM(tracker=tracker))
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
            FileHandler.save_json(
                generation_metrics_dict,
                self.output_dir / "generator_metrics.json",
            )
            results["generation_metrics"] = generation_metrics_dict
        else:
            console.print("[bold yellow]No valid generation metrics to compute. Skipping generator metrics.[/bold yellow]")

        # 3. Save token usage report
        console.log("[bold green]Saving token usage report...[/bold green]")
        token_summary = tracker.get_summary()
        FileHandler.save_json(
            token_summary,
            self.output_dir / "token_usage.json",
        )
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





