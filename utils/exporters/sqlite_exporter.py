"""
sqlite_exporter.py
------------------
Generic, reusable SQLite batch writer.
Import and call save_to_sqlite() from anywhere in the project.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def save_to_sqlite(
    db_path: Path,
    schema_sql: str,
    insert_sql: str,
    params_list: List[tuple],
) -> Path:
    """
    Ensure a table exists then batch-insert all rows.

    Args:
        db_path:     Path to the SQLite file (created if missing).
        schema_sql:  CREATE TABLE IF NOT EXISTS DDL statement.
        insert_sql:  Parameterised INSERT statement (uses ? placeholders).
        params_list: List of tuples, one per row, matching *insert_sql*.

    Returns:
        Resolved Path of the SQLite file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(schema_sql)
        conn.executemany(insert_sql, params_list)
        conn.commit()
        logger.info("[sqlite_exporter] %d rows → %s", len(params_list), db_path)
    except Exception:
        conn.rollback()
        logger.exception("[sqlite_exporter] Insert failed, rolling back.")
        raise
    finally:
        conn.close()

    return db_path
