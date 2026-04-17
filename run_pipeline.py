import asyncio
from datetime import datetime
import json
import sys
from pathlib import Path

from config.config_loader import load_config
from config.logger import setup_logger
from core.orchestrator import EvaluationOrchestrator
from utils.metric_exporter import export_to_db, export_to_csv
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
# Main
# --------------------------------------------------

def main() -> None:

    try:
        # 1) Setup Run Metadata
        base_output_dir = Path(CONFIG["results_dirs"]["output_dir"])
        csv_name_base = CONFIG.get("csv_name", "eval_run")
        ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
        # eval_file_name will be the directory name
        eval_file_name = f"{csv_name_base}_{ts}"
        run_output_dir = base_output_dir / eval_file_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Simplified CSV filename for final export (distinct from backend log)
        csv_filename = f"results_{eval_file_name}.csv"
        logger.info(f"Main Run Folder: {eval_file_name}")
        logger.info(f"Final CSV Export: {csv_filename}")

        # 2) Initialize Orchestrator with the run-specific directory
        orchestrator = EvaluationOrchestrator(output_dir=run_output_dir)
        
        # Internal paths for metrics calculation
        retrieval_path = run_output_dir / "retriever_metrics.json"
        generation_path = run_output_dir / "generator_metrics.json"
        visionrag_path = run_output_dir / "visionragresult.json"

        ### Run the pipeline
        asyncio.run(orchestrator.run(eval_file_name=eval_file_name))
        if CONFIG['aws_s3_bucket']:
            FileHandler.download_eval_file_from_s3(
                s3_bucket=CONFIG['aws_s3_bucket'],
                eval_file_name=eval_file_name,
                local_dir=run_output_dir
            )


        ### Enrich results from backend logs and recompute metrics
        vrag_client_csv = run_output_dir / f"{eval_file_name}.csv"
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

        # csv_filename already generated above
        csv_out = export_to_csv(
            retrieval_pq,
            generation_pq,
            run_output_dir,
            visionrag_path=visionrag_path,
            top_k=CONFIG["k"],
            log_path=vrag_client_csv,
            csv_filename=csv_filename
        )
        final_csv_name = csv_out.name
        logger.info("📊 Per-question CSV → %s (final_csv_name=%s)", csv_out, final_csv_name)

        db_out = export_to_db(
            retrieval_pq,
            generation_pq,
            run_output_dir,
            visionrag_path=visionrag_path,
            top_k=CONFIG["k"],
            hybrid_search=CONFIG["use_hybrid"],
            query_reformulation=CONFIG["query_reformulation"],
            reranking=CONFIG["use_reranking"],
            log_path=vrag_client_csv,
            table_name=eval_file_name,
            db_filename="../../eval_dashboard/metrics.db", 
            final_csv_name=final_csv_name
        )
        logger.info("🗄️  Per-question DB  → %s", db_out)

        logger.info("✅ Pipeline and Analysis completed successfully")
    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


























































