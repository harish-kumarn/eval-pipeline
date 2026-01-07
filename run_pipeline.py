import asyncio
import argparse
import sys
import logging

from core.data_models import PipelineConfig
from pipeline.orchestrator import EvaluationOrchestrator

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
        description="Run evaluation pipeline (VisionRAG + RAGAS)"
    )

    parser.add_argument("--target-chunks", default="data/target_chunks.json")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--user-id", default="43")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-format", default="chunk{id}")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--metrics-config", help="Path to metrics configuration JSON file")

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("🚀 Starting evaluation pipeline")
    logger.info(
        "Config | target=%s | out=%s | user=%s | k=%d",
        args.target_chunks,
        args.output_dir,
        args.user_id,
        args.k,
    )

    config = PipelineConfig(
        target_chunks_file=args.target_chunks,
        output_dir=args.output_dir,
        user_id=args.user_id,
        k=args.k,
        chunk_format=args.chunk_format,
        retriever_timeout=args.timeout,
        save_results=not args.no_save,
        metrics_config_path=args.metrics_config,
    )

    orchestrator = EvaluationOrchestrator(config)

    try:
        asyncio.run(orchestrator.run())
        logger.info("✅ Pipeline completed successfully")
    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
