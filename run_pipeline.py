import asyncio
import sys
from datetime import datetime
from pathlib import Path

from config.config_loader import load_config
from config.logger import setup_logger
from core.orchestrator import EvaluationOrchestrator
from utils.metric_exporter import export_all_metrics
from utils.support_evals.file_handler import FileHandler


def setup_run_environment(config: dict, logger) -> dict:
    """Sets up run metadata, creates directories, and returns relevant paths."""
    
    base_output_dir = Path(config["results_dirs"]["output_dir"])
    csv_name_base = config.get("csv_name", "eval_run").strip()
    now = datetime.now()
    eval_file_name = now.strftime(f"%d_%m_{csv_name_base}_%H_%M")
    run_output_dir = base_output_dir / eval_file_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory: %s", run_output_dir)

    paths = {
        "run_output_dir": run_output_dir,
        "eval_file_name": eval_file_name,
        "csv_filename": f"results_{eval_file_name}.csv",
        "retrieval_path": run_output_dir / "retriever_metrics.json",
        "generation_path": run_output_dir / "generator_metrics.json",
        "visionrag_path": run_output_dir / "visionragresult.json",
        "vrag_client_csv": run_output_dir / f"{eval_file_name}.csv"
    }
    return paths


def main() -> None:
    # Initialize Logging
    logger = setup_logger(name="eval_pipeline", tag="RUN_PIPELINE")
    logger.info("Logger initialized for run pipeline")

    try:
        ### Load Config
        config_path = "config/config.yml"
        config = load_config(config_path)
 
        ### Setup Run Environment
        paths = setup_run_environment(config, logger)
        run_output_dir = paths["run_output_dir"]
        eval_file_name = paths["eval_file_name"]
        visionrag_path = paths["visionrag_path"]
        vrag_client_csv = paths["vrag_client_csv"]

        ### Initialize Orchestrator
        orchestrator = EvaluationOrchestrator(output_dir=run_output_dir)
        
        ### Run the pipeline
        asyncio.run(orchestrator.run(eval_file_name=eval_file_name))
        
        if config.get("aws_s3_bucket"):
            FileHandler.download_eval_file_from_s3(
                s3_bucket=config["aws_s3_bucket"],
                eval_file_name=eval_file_name,
                local_dir=run_output_dir
            )

        ### Enrich results from backend logs and recompute metrics
        orchestrator.populate_visionrag_results(vrag_client_csv)
        orchestrator.compute_retriever_metrics(visionrag_path)
        # # orchestrator.compute_generator_metrics(visionrag_path)


        # Export per-question metrics (CSV + DB)
        csv_out, db_out = export_all_metrics(
            run_output_dir=run_output_dir,
            visionrag_path=visionrag_path,
            retrieval_path=paths["retrieval_path"],
            generation_path=paths["generation_path"],
            vrag_client_csv=vrag_client_csv,
            top_k=config.get("k"),
            csv_filename=paths["csv_filename"],
            table_name=eval_file_name,
            hybrid_search=config.get("use_hybrid", False),
            query_reformulation=config.get("query_reformulation", False),
            reranking=config.get("use_reranking", False),
            db_filename="../../eval_dashboard/metrics.db",
        )
        logger.info("📊 Per-question CSV → %s (final_csv_name=%s)", csv_out, csv_out.name)
        logger.info("🗄️  Per-question DB  → %s", db_out)

        logger.info("✅ Pipeline and Analysis completed successfully")
    
    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
