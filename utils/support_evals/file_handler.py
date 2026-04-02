"""File I/O utilities."""

import json
from pathlib import Path
from typing import Any
from config.logger import setup_logger
import shutil

logger = setup_logger(name="eval_pipeline", tag="FILE_HANDLER")
logger.info("Logger initialized for file handler")

class FileHandler:
    """Static helpers for reading and writing JSON files."""

    @staticmethod
    def load_json(file_path: str) -> Any:
        """Load and return the contents of a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> None:
        """Serialize *data* to a JSON file, creating parent directories as needed."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def copy_local_evalFile(backend_logs_dir: str, local_dest: str) -> None:
        backend_logs_dir = Path(backend_logs_dir)
        local_dest = Path(local_dest)

        source_path = backend_logs_dir / local_dest.name
        pipeline_eval_dir = Path("/home/krithika/Documents/Vrag-eval-pipeline/eval_results")

        target_path = pipeline_eval_dir / local_dest.name

        if not source_path.exists():
            print(f"Warning: Source not found at {source_path}")
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)

        logger.info(f"✓ Copied {source_path} → {target_path}")

    @staticmethod
    def download_eval_file_from_s3(s3_bucket: str, eval_file_name: str, local_dir: str = "eval_results") -> str:
        import boto3
        import os
        from dotenv import load_dotenv

        load_dotenv()
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2")
        )
        
        file_name_with_ext = f"{eval_file_name}.csv"
        s3_key = f"eval_results/{file_name_with_ext}"
        
        local_path = Path(local_dir) / file_name_with_ext
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading s3://{s3_bucket}/{s3_key} to {local_path}")
            s3_client.download_file(s3_bucket, s3_key, str(local_path))
            logger.info(f"✓ Downloaded {local_path} from S3")
            return str(local_path)
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            return None
