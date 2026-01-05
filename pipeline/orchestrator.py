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
from utils.file_handler import FileHandler

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:

    def __init__(self, config: PipelineConfig):
        self.config = config

        self.retriever = VisionRAGRetriever(
            user_id=config.user_id,
            timeout=config.retriever_timeout,
            chunk_format=config.chunk_format,
        )

        self.question_extractor = QuestionExtractor(config.target_chunks_file)
        self.retrieval_calc = RetrievalMetricsCalculator()
        self.generation_calc = GenerationMetricsCalculator()

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> Dict[str, Any]:
        questions = self.question_extractor.extract_questions()
        await self._run_retriever(questions)
        return self._compute_metrics(self.output_dir / "visionragresult.json")

    async def _run_retriever(self, questions):
        visionrag_file = self.output_dir / "visionragresult.json"
        all_results = []

        for q in questions:
            result = await self.retriever.retrieve(q)

            entry = {
                "qid": result.qid,
                "question": result.question,
                "ground_truth_chunk_id": result.ground_truth_chunk,
                "retrieved_chunk_ids": result.retrieved_chunks,
                "generated_answer": result.generated_answer,
            }

            all_results.append(entry)

            with open(visionrag_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        return all_results

    def _compute_metrics(self, visionrag_file: Path) -> Dict[str, Any]:
        with open(visionrag_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for d in data:
            if not d.get("ground_truth"):
                d["ground_truth"] = self.question_extractor.get_ground_truth(d["qid"]) or ""

        def extract_context(cid: str) -> str:
            return self.question_extractor.get_chunk_text(cid) or ""

        # Build generation dataset
        gen_data = []
        for d in data:
            contexts = [extract_context(cid) for cid in d.get("retrieved_chunk_ids", [])[:self.config.k]]
            gen_data.append({
                "qid": d.get("qid"),
                "question": d.get("question"),
                "answer": d.get("generated_answer"),
                "ground_truth": d.get("ground_truth"),
                "contexts": contexts,
            })

        # RAGAS wrappers
        llm = LangchainLLMWrapper(GeminiLLM())
        embeddings = LangchainEmbeddingsWrapper(GeminiEmbeddings())

        retrieval_metrics = self.retrieval_calc.compute(data, k=self.config.k)
        generation_metrics = self.generation_calc.compute(
            gen_data,
            llm_wrapper=llm,
            embeddings_wrapper=embeddings,
        )

        retrieval_metrics_dict = retrieval_metrics.to_dict()
        generation_metrics_dict = generation_metrics.to_dict() if generation_metrics else {}

        FileHandler.save_json(
            retrieval_metrics_dict,
            self.output_dir / "retriever_metrics.json",
        )

        if generation_metrics:
            FileHandler.save_json(
                generation_metrics_dict,
                self.output_dir / "generator_metrics.json",
            )
        else:
            logger.warning("No valid generation metrics to compute. Skipping generator metrics.")

        return {
            "retrieval_metrics": retrieval_metrics_dict,
            "generation_metrics": generation_metrics_dict,
        }
