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


# --------------------------
# Load Config
# --------------------------
CONFIG = load_config("config/config.yml")


# --------------------------
# Path configurations
# --------------------------
output_dir = Path(CONFIG["results_dirs"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
retrieval_path = output_dir / "retriever_metrics.json"
generation_path = output_dir / "generator_metrics.json"
visionrag_path = output_dir / "visionragresult.json"


# --------------------------
# Initialize Orchestrator
# --------------------------
orchestrator = EvaluationOrchestrator()



# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    try:
        

        # 1) Setup Run Metadata
        safe_collection = (CONFIG.get("collection_name") or "unknown").replace("/", "_")
        csv_name_base = CONFIG.get("csv_name", "eval_run")
        ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        eval_file_name = f"{csv_name_base}_{ts}"

        csv_filename = f"eval_{safe_collection}_{csv_name_base}_{ts}.csv"
        logger.info(f"this is my csv file name {csv_filename}")

        ### Run the pipeline
        asyncio.run(orchestrator.run(eval_file_name=eval_file_name))
        
        
        ### Download the pipeline results from S3
        if CONFIG['aws_s3_bucket']:
            FileHandler.download_eval_file_from_s3(
                s3_bucket=CONFIG['aws_s3_bucket'],
                eval_file_name=eval_file_name,
                local_dir=CONFIG["results_dirs"]["output_dir"]
            )


        ### Enrich results from backend logs and recompute metrics
        vrag_client_csv = output_dir / f"{eval_file_name}.csv"
        orchestrator.populate_visionrag_results(vrag_client_csv)
        orchestrator._compute_metrics(visionrag_path)


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
            log_path=vrag_client_csv,
            table_name=eval_file_name,
        )
        logger.info("🗄️  Per-question DB  → %s", db_out)

        logger.info("✅ Pipeline and Analysis completed successfully")
    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


























































