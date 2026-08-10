# import asyncio
# import sys
# from datetime import datetime
# from pathlib import Path

# from config.config_loader import load_config
# from config.logger import setup_logger
# from core.orchestrator import EvaluationOrchestrator
# from utils.metric_exporter import export_all_metrics
# from utils.support_evals.file_handler import FileHandler, delete_csv_from_s3


# def setup_run_environment(CONFIG: dict, logger) -> dict:
#     """Sets up run metadata, creates directories, and returns relevant paths."""

#     base_output_dir = Path(CONFIG["results_dirs"]["output_dir"])
#     csv_name_base = CONFIG.get("csv_name", "eval_run").strip()
#     vrag_file_name = datetime.now().strftime(f"%d_%m_{csv_name_base}_%H_%M")
#     output_dir = base_output_dir / vrag_file_name

#     paths = {
#         "output_dir": output_dir,
#         "vrag_file_name": vrag_file_name,
#         "result_csv_filename": f"results_{vrag_file_name}.csv",
#         "retrieval_path": output_dir / "retriever_metrics.json",
#         "generation_path": output_dir / "generator_metrics.json",
#         "visionrag_populated_path": output_dir / "visionragresult.json",
#         "vrag_file_name_csv": output_dir / f"{vrag_file_name}.csv"
#     }
#     return paths


# def main() -> None:
#     # Initialize Logging
#     logger = setup_logger(name="eval_pipeline", tag="RUN_PIPELINE")
#     logger.info("Logger initialized for run pipeline")

#     try:
#         ### Load Config
#         config_path = "config/config.yml"
#         CONFIG = load_config(config_path)
 
#         ### Setup Run Environment
#         paths = setup_run_environment(CONFIG, logger)
#         output_dir = paths["output_dir"]
#         visionrag_populated_path = paths["visionrag_populated_path"]
#         vrag_file_name = paths["vrag_file_name"]
#         vrag_file_name_csv = paths["vrag_file_name_csv"]

#         ### Initialize Orchestrator
#         orchestrator = EvaluationOrchestrator(output_dir=output_dir)
        
#         ### Run the pipeline
#         asyncio.run(orchestrator.run(vrag_file_name=vrag_file_name))

#         if CONFIG.get("aws_s3_bucket"):
#             FileHandler.download_eval_file_from_s3(
#                 s3_bucket=CONFIG["aws_s3_bucket"],
#                 vrag_file_name=vrag_file_name,
#                 local_dir=output_dir
#             )
        
#         ### Enrich results from backend logs and recompute metrics
#         orchestrator.populate_visionrag_results(vrag_file_name_csv)        
#         orchestrator.compute_retriever_metrics(visionrag_populated_path)
#         # # orchestrator.compute_generator_metrics(visionrag_populated_path)


#         # Export per-question metrics (CSV + DB)
#         csv_out, db_out = export_all_metrics(
#             run_output_dir=output_dir,
#             visionrag_populated_path=visionrag_populated_path,
#             retrieval_path=paths["retrieval_path"],
#             generation_path=paths["generation_path"],
#             vrag_file_name_csv=vrag_file_name_csv,
#             top_k=CONFIG["vrag_client"]["final_top_k"],
#             result_csv_filename=paths["result_csv_filename"],
#             table_name=vrag_file_name,
#             hybrid_search=CONFIG.get("use_hybrid", False), 
#             query_reformulation=CONFIG.get("query_reformulation", False),
#             reranking=CONFIG.get("use_reranking", False),
#             db_filename="../../eval_dashboard/metrics.db",
#         )

#         logger.info(f"📊 Per-question CSV → {csv_out} (final_csv_name={csv_out.name})")
#         logger.info(f"🗄️  Per-question DB  → {db_out}")

#         if CONFIG.get("delete_s3_csv"):
#             delete_csv_from_s3(vrag_file_name)
#             logger.info(f"🗑️  Deleted CSV from S3: {vrag_file_name}")

#         logger.info("✅ Pipeline and Analysis completed successfully")
    
#     except Exception:
#         logger.exception("❌ Pipeline failed")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()











import asyncio
import sys
from datetime import datetime
from pathlib import Path

from config.config_loader import load_config
from config.logger import setup_logger
from core.orchestrator import EvaluationOrchestrator
from utils.metric_exporter import export_all_metrics
from utils.support_evals.file_handler import FileHandler, delete_csv_from_s3
from utils.helper.retrieved_helper import run_evaluation


# Initialize Logging
logger = setup_logger(name="eval_pipeline", tag="RUN_PIPELINE")
logger.info("Logger initialized for run pipeline")


### Load Config
config_path = "config/config.yml"
CONFIG = load_config(config_path)


### Path Config
INPUT_QUESTIONS = CONFIG.get("target_chunks", "data/input_questions.json")
csv_name_base = CONFIG.get("csv_name", "eval_run").strip()
VRAG_FILE_NAME = datetime.now().strftime(f"%d_%m_{csv_name_base}_%H_%M")
base_output_dir = Path(CONFIG["results_dirs"]["output_dir"])
OUTPUT_DIR = base_output_dir / VRAG_FILE_NAME
GENERATED_ANSWERS_PATH = OUTPUT_DIR / CONFIG["results_dirs"]["generated_answers_file"]
VISIONRAG_POPULATED_PATH = OUTPUT_DIR / CONFIG["results_dirs"]["vrag_result_json"]


###
RETRIEVAL_PATH = OUTPUT_DIR / "retriever_metrics.json"
GENERATION_PATH = OUTPUT_DIR / "generator_metrics.json"
VRAG_FILE_NAME_CSV = OUTPUT_DIR / f"{VRAG_FILE_NAME}.csv"
RESULT_CSV_FILENAME = f"results_{VRAG_FILE_NAME}.csv"
RESULT_CSV_PATH = OUTPUT_DIR / RESULT_CSV_FILENAME


def main() -> None:

    try:

        ### Initialize Orchestrator
        orchestrator = EvaluationOrchestrator(
                                input_questions=INPUT_QUESTIONS,
                                output_dir=OUTPUT_DIR,
                                generated_answers_path=GENERATED_ANSWERS_PATH,
                                visionrag_populated_path=VISIONRAG_POPULATED_PATH

                                )

        # ### Run the pipeline
        # asyncio.run(orchestrator.run(vrag_file_name=VRAG_FILE_NAME))

        # if CONFIG.get("aws_s3_bucket"):
        #     FileHandler.download_eval_file_from_s3(
        #         s3_bucket=CONFIG["aws_s3_bucket"],
        #         vrag_file_name=VRAG_FILE_NAME,
        #         local_dir=OUTPUT_DIR
        #     )

        # ### Evaluate retriever results against ground truth and save the
        # ### augmented CSV (adds question_type, tag, answer_chunk_id,
        # ### predicted, retrieved_position columns).
        # logger.info("Running retriever evaluation against ground truth")
        # run_evaluation(
        #     dataset_path=INPUT_QUESTIONS,
        #     results_path=VRAG_FILE_NAME_CSV,
        #     output_path=RESULT_CSV_PATH,
        # )\
        #TODO: replace this with config
        VRAG_FILE_NAME_CSV = "retriver_res/04_08_dummy1_18_54_with_answers.csv"

        # ### Enrich results from backend logs and recompute metrics
        orchestrator.populate_visionrag_results(Path(VRAG_FILE_NAME_CSV)) #visionragresult.json


        orchestrator.compute_retriever_metrics(VISIONRAG_POPULATED_PATH)
        orchestrator.compute_generator_metrics(VISIONRAG_POPULATED_PATH)


















        # # Export per-question metrics (CSV + DB)
        # csv_out, db_out = export_all_metrics(
        #     run_output_dir=OUTPUT_DIR,
        #     visionrag_populated_path=VISIONRAG_POPULATED_PATH,
        #     retrieval_path=RETRIEVAL_PATH,
        #     generation_path=GENERATION_PATH,
        #     vrag_file_name_csv=VRAG_FILE_NAME_CSV,
        #     top_k=CONFIG["vrag_client"]["final_top_k"],
        #     result_csv_filename=RESULT_CSV_FILENAME,
        #     table_name=VRAG_FILE_NAME,
        #     hybrid_search=CONFIG.get("use_hybrid", False), 
        #     query_reformulation=CONFIG.get("query_reformulation", False),
        #     reranking=CONFIG.get("use_reranking", False),
        #     db_filename="../../eval_dashboard/metrics.db",
        # )

        # logger.info(f"📊 Per-question CSV → {csv_out} (final_csv_name={csv_out.name})")
        # logger.info(f"🗄️  Per-question DB  → {db_out}")

        # if CONFIG.get("delete_s3_csv"):
        #     delete_csv_from_s3(VRAG_FILE_NAME)
        #     logger.info(f"🗑️  Deleted CSV from S3: {VRAG_FILE_NAME}")

        # logger.info("✅ Pipeline and Analysis completed successfully")

    except Exception:
        logger.exception("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


