import asyncio
import argparse
import sys
import logging
from pathlib import Path
from rich.logging import RichHandler

from core.data_models import PipelineConfig
from pipeline.orchestrator import EvaluationOrchestrator
from metric_analyzer.analyzer import RAGMetricAnalyzer
from metric_analyzer.reporting import export_json, export_to_csv

# --------------------------------------------------
# Logging
# --------------------------------------------------

# Configure root logger with RichHandler
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("rich")


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
    parser.add_argument("--collection-name", default="sprayers", help="Name of the collection to use")

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
        
        if not args.metrics_config:
            logger.error("❌ --metrics-config is required for analysis")
            sys.exit(1)

        analyzer = RAGMetricAnalyzer(
            metrics_config_path=args.metrics_config
        )

        #2. Calculate the metrics
        retrieval_path = Path(args.output_dir) / "retriever_metrics.json"
        generation_path = Path(args.output_dir) / "generator_metrics.json"
        
        results = analyzer.analyze(
            str(retrieval_path) if retrieval_path.exists() else None,
            str(generation_path) if generation_path.exists() else None
        )
        
        # 3. Export Results and do analysis
        if results and "categories" in results:
            analysis_json = Path(args.output_dir) / "analysis_report.json"
            analysis_csv = Path(args.output_dir) / "analysis_report.csv"
            
            export_json(results["categories"], str(analysis_json))
            export_to_csv(results["categories"], str(analysis_csv))

        logger.info(f"✅ Pipeline and Analysis completed successfully")

    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
