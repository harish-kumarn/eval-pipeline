"""File I/O utilities."""

import json
from pathlib import Path
from typing import Any


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
