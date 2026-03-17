"""
csv_exporter.py
---------------
Generic, reusable CSV writer.
Import and call save_to_csv() from anywhere in the project.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def save_to_csv(
    rows: List[Dict[str, Any]],
    file_path: Path,
    fieldnames: List[str],
) -> Path:
    """
    Write *rows* to a CSV file at *file_path*.

    Args:
        rows:       list of dicts — each dict is one row (extra keys ignored).
        file_path:  destination file path (parent dirs created automatically).
        fieldnames: ordered list of column names used as the CSV header.

    Returns:
        Resolved Path of the written file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("[csv_exporter] %d rows → %s", len(rows), file_path)
    return file_path
