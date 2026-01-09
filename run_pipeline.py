import asyncio
import argparse
import sys
import logging
from pathlib import Path

from core.data_models import PipelineConfig
from pipeline.orchestrator import EvaluationOrchestrator
from metric_analyzer.analyzer import RAGMetricAnalyzer

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline (VisionRAG + RAGAS + Analysis)"
    )

    parser.add_argument("--target-chunks", default="data/target_chunks.json")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--user-id", default="43")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-format", default="chunk{id}")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--metrics-config", help="Path to metrics configuration JSON file")

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("🚀 Starting Evaluation Pipeline")

    config = PipelineConfig(
        target_chunks_file=args.target_chunks,
        output_dir=args.output_dir,
        user_id=args.user_id,
        k=args.k,
        chunk_format=args.chunk_format,
        retriever_timeout=args.timeout,
        metrics_config_path=args.metrics_config,
    )

    orchestrator = EvaluationOrchestrator(config)

    try:
        # 1. Run Pipeline (Retrieval + Metrics)
        asyncio.run(orchestrator.run())
        
        # 2. Run Analysis
        logger.info("📊 Running final performance analysis...")
        analyzer = RAGMetricAnalyzer()
        
        retrieval_path = Path(args.output_dir) / "retriever_metrics.json"
        generation_path = Path(args.output_dir) / "generator_metrics.json"
        
        analyzer.analyze(
            str(retrieval_path) if retrieval_path.exists() else None,
            str(generation_path) if generation_path.exists() else None
        )
        
        logger.info(f"✅ Pipeline and Analysis completed successfully")

    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
