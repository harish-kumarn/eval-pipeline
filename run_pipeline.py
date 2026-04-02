import asyncio
from datetime import datetime
import json
import sys
from pathlib import Path

from config.config_loader import load_config
from config.logger import setup_logger
from core.orchestrator import EvaluationOrchestrator
from utils.metric_exporter import export_metrics_to_db
from utils.support_evals.file_handler import FileHandler


# --------------------------
# Logging
# --------------------------
logger = setup_logger(name="eval_pipeline", tag="MAIN_APP")
logger.info("Logger initialized for main app")



# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    logger.info("🚀 Starting Evaluation Pipeline")

    try:
        config = load_config("config/config.yml")

        # 1) Setup Run Metadata
        safe_collection = (config.get("collection_name") or "unknown").replace("/", "_")
        csv_name_base = config.get("csv_name", "eval_run")
        ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        eval_file_name = f"{csv_name_base}_{ts}"

        csv_filename = f"eval_{safe_collection}_{csv_name_base}_{ts}.csv"
        logger.info(f"this is my csv file name {csv_filename}")

        orchestrator = EvaluationOrchestrator(config)
        asyncio.run(orchestrator.run(eval_file_name=eval_file_name))
        
        aws_config = config.get("aws", {})
        s3_bucket = aws_config.get("s3_bucket")
        
        if s3_bucket:
            FileHandler.download_eval_file_from_s3(
                s3_bucket=s3_bucket,
                eval_file_name=eval_file_name,
                local_dir=config.get("output_dir", "eval_results")
            )
        else:
            FileHandler.copy_local_evalFile(
                backend_logs_dir=config.get("backend_logs_dir", ""),
                local_dest=eval_file_name
            )

        # 1.5) Enrich results from backend logs and recompute metrics
        output_dir = Path(config["output_dir"])
        log_path_csv = output_dir / f"{eval_file_name}.csv"
        log_path_no_ext = output_dir / eval_file_name
        log_path = log_path_csv if log_path_csv.exists() else log_path_no_ext
        
        orchestrator.populate_visionrag_results(log_path)
        orchestrator.recompute_metrics()

        # 2) Analyze outputs
        retrieval_path = output_dir / "retriever_metrics.json"
        generation_path = output_dir / "generator_metrics.json"
        visionrag_path = output_dir / "visionragresult.json"

        # 4) Export per-question metrics (CSV + DB)
        retrieval_pq: list[dict] = []
        generation_pq: list[dict] = []

        if retrieval_path.exists():
            with open(retrieval_path, encoding="utf-8") as f:
                retrieval_pq = (json.load(f) or {}).get("per_query_metrics", [])

        if generation_path.exists():
            with open(generation_path, encoding="utf-8") as f:
                generation_pq = (json.load(f) or {}).get("per_query_metrics", [])

        csv_filename = f"eval_{safe_collection}_{csv_name_base}_{ts}.csv"

        db_out = export_metrics_to_db(
            retrieval_pq,
            generation_pq,
            output_dir,
            visionrag_path=visionrag_path,
        )
        logger.info("🗄️  Per-question DB  → %s", db_out)

        logger.info("✅ Pipeline and Analysis completed successfully")
    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


























































