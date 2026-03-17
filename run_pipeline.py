import asyncio
import sys
import logging
from pathlib import Path
from rich.logging import RichHandler

import json

# from core.data_models import PipelineConfig
from pipeline.orchestrator import EvaluationOrchestrator
from metric_analyzer.analyzer import RAGMetricAnalyzer
from metric_analyzer.reporting import export_json, export_to_csv
from utils.exporters import export_metrics_to_csv, export_metrics_to_db
from config.config_loader import load_config
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
# Main
# --------------------------------------------------

def main() -> None:

    logger.info("🚀 Starting Evaluation Pipeline")

    config = load_config("config/config.yml")
    print("config read successfuly")
    print(config["user_id"])
    try:        
        orchestrator = EvaluationOrchestrator(config)

        # 1. Run Pipeline (Retrieval + Metrics)
        asyncio.run(orchestrator.run())
        

        analyzer = RAGMetricAnalyzer(
            metrics_config_path="config/metrics_config.json"
        )

        # 2. Calculate the metrics
        retrieval_path = Path(".") / "retriever_metrics.json"
        generation_path = Path(".") / "generator_metrics.json"
        
        results = analyzer.analyze(
            str(retrieval_path) if retrieval_path.exists() else None,
            str(generation_path) if generation_path.exists() else None
        )
        
        
        # 3. Export analysis results
        if results and "categories" in results:
            analysis_json = Path(".") / "analysis_report.json"
            analysis_csv  = Path(".") / "analysis_report.csv"
            export_json(results["categories"], str(analysis_json))
            export_to_csv(results["categories"], str(analysis_csv))

        # 4. Export per-question metrics to CSV (new timestamped file each run)
        retrieval_pq  = []
        generation_pq = []

        if retrieval_path.exists():
            with open(retrieval_path, encoding="utf-8") as f:
                retrieval_pq = json.load(f).get("per_query_metrics", [])

        if generation_path.exists():
            with open(generation_path, encoding="utf-8") as f:
                generation_pq = json.load(f).get("per_query_metrics", [])

        csv_out = export_metrics_to_csv(retrieval_pq, generation_pq, ".")
        logger.info(f"📄 Per-question CSV → {csv_out}")

        # 5. Save per-question metrics to SQLite DB (all runs appended)
        db_out = export_metrics_to_db(retrieval_pq, generation_pq, ".")
        logger.info(f"🗄️  Per-question DB  → {db_out}")

        logger.info("✅ Pipeline and Analysis completed successfully")

    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()



























































# --------------------------------------------------
# CLI
# --------------------------------------------------

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Run evaluation pipeline (VisionRAG + RAGAS + Analysis)"
#     )

#     parser.add_argument("--target-chunks", default="data/target_chunks.json")
#     parser.add_argument("--output-dir", default=".")
#     parser.add_argument("--user-id", default="43")
#     parser.add_argument("--k", type=int, default=3)
#     parser.add_argument("--chunk-format", default="chunk{id}")
#     parser.add_argument("--timeout", type=int, default=300)
#     parser.add_argument("--metrics-config", help="Path to metrics configuration JSON file")
#     parser.add_argument("--collection-name", default="sprayers", help="Name of the collection to use")

#     return parser.parse_args()