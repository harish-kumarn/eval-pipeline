"""File I/O utilities."""

import json
from pathlib import Path
from typing import Any, List, Dict


class FileHandler:
    """Handles file operations."""

    @staticmethod
    def load_json(file_path: str) -> Any:
        """Load JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> None:
        """Save data to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
        """Save list of dicts to JSONL file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
