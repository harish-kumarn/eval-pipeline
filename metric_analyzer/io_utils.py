"""Metric analyzer I/O utilities."""

import json
from pathlib import Path


def load_metrics(path: str):
    """Load per-query metrics from a JSON file.

    Returns an empty list if *path* is None or the file does not exist.
    Unwraps the ``per_query_metrics`` key when the file contains a summary dict.
    """
    if path is None:
        return []

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parent.parent / resolved

    if not resolved.exists():
        return []

    with open(resolved) as f:
        data = json.load(f)

    if isinstance(data, dict) and "per_query_metrics" in data:
        return data["per_query_metrics"]

    return data if isinstance(data, list) else data.get("metrics", [])
