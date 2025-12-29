import os
import json
import logging
from typing import Optional, List

from dotenv import load_dotenv
from google import genai
from langchain_core.language_models.llms import LLM
from langchain_cohere import CohereEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)

logger = logging.getLogger(__name__)


class GeminiLLM(LLM):
    """Gemini LLM wrapper for RAGAS evaluation."""
    
    model_name: str = "gemini-2.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        response = genai.Client(api_key=os.getenv("GOOGLE_API_KEY")).models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip() if response and response.text else ""


def load_environment() -> None:
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = ["GOOGLE_API_KEY", "COHERE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")
    
    logger.info("Environment variables loaded successfully")


def load_data(input_file: str) -> List[dict]:
    """Load evaluation data from JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {input_file}")
    return data


def build_dataset(raw_data: List[dict]) -> Dataset:
    """Build RAGAS-compatible dataset from raw data."""
    dataset = Dataset.from_dict({
        "question": [x["question"] for x in raw_data],
        "answer": [x["answer"] for x in raw_data],
        "contexts": [x["context"] for x in raw_data],
        "ground_truths": [x["ground_truth"] for x in raw_data],
        "reference": [x["ground_truth"][0] if x["ground_truth"] else "" for x in raw_data],
    })
    logger.info("Dataset constructed successfully")
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    """Validate dataset integrity."""
    for i, row in enumerate(dataset):
        assert isinstance(row["question"], str) and row["question"].strip(), \
            f"Invalid question at index {i}"
        assert isinstance(row["answer"], str) and row["answer"].strip(), \
            f"Invalid answer at index {i}"
        assert isinstance(row["contexts"], list) and len(row["contexts"]) > 0, \
            f"Invalid contexts at index {i}"
        assert all(isinstance(c, str) and c.strip() for c in row["contexts"]), \
            f"Invalid context content at index {i}"
        assert isinstance(row["ground_truths"], list) and len(row["ground_truths"]) > 0, \
            f"Invalid ground_truths at index {i}"
        assert all(isinstance(gt, str) and gt.strip() for gt in row["ground_truths"]), \
            f"Invalid ground_truth content at index {i}"
    
    logger.info("Dataset validation passed")


def initialize_llm() -> LangchainLLMWrapper:
    """Initialize Gemini LLM wrapper."""
    llm = LangchainLLMWrapper(GeminiLLM())
    logger.info("LLM initialized")
    return llm


def initialize_embeddings() -> LangchainEmbeddingsWrapper:
    """Initialize Cohere embeddings wrapper."""
    embeddings = LangchainEmbeddingsWrapper(
        CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
    )
    logger.info("Embeddings initialized")
    return embeddings


def run_evaluation(
    dataset: Dataset,
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
) -> dict:
    """Run RAGAS evaluation on the dataset."""
    metrics = [
        answer_relevancy,
        answer_similarity,
        faithfulness,
        context_precision,
        context_recall,
        answer_correctness,
    ]
    
    logger.info("Starting RAGAS evaluation")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    logger.info("Evaluation completed")
    
    return results


def save_results(results: dict, output_dir: str = ".") -> None:
    """Save evaluation results to files."""
    df = results.to_pandas()
    
    # Per-question metrics
    df.to_json(os.path.join(output_dir, "metrics_per_question.json"), orient="records", indent=2)
    df.to_csv(os.path.join(output_dir, "metrics_per_question.csv"), index=False)
    
    # Aggregate metrics
    summary = df.mean(numeric_only=True).to_dict()
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Results saved successfully")
    logger.info(f"Aggregate metrics: {json.dumps(summary, indent=2)}")


def main(input_file: str = "ragas_ready.json", output_dir: str = ".") -> None:
    """Main evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    load_environment()
    raw_data = load_data(input_file)
    dataset = build_dataset(raw_data)
    validate_dataset(dataset)
    
    llm = initialize_llm()
    embeddings = initialize_embeddings()
    
    results = run_evaluation(dataset, llm, embeddings)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()