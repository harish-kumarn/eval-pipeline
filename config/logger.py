# bc_rag/core/logger.py

import logging
import sys
import os
import datetime


def setup_logger(name="eval_pipeline", level=logging.INFO, tag=None):
    """
    Configure and return a tagged LoggerAdapter for *name*.

    The first call for a given *name* attaches two handlers:
      • StreamHandler  – writes to stdout
      • FileHandler    – writes to <project_root>/logs/<name>_YYYY-MM-DD_HH-MM-SS.log
                         (one file per pipeline run, no rotation needed)

    Subsequent calls for the same *name* reuse the existing handlers and just
    return a new adapter with the requested *tag*.
    """
    # Resolve the project root: config/logger.py → config/ → project_root/
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    base_logger = logging.getLogger(name)

    if not base_logger.handlers:
        base_logger.setLevel(level)
        base_logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(tag)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        # Per-run file handler (timestamp down to the second)
        run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, f"{name}_{run_ts}.log")
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        base_logger.addHandler(console_handler)
        base_logger.addHandler(file_handler)

    return logging.LoggerAdapter(base_logger, {"tag": tag or "GENERAL"})






# Helper functions for styled logging
def log_info(logger, message):
    """Log information with green bold formatting"""
    logger.info(f"[bold green]{message}[/bold green]")


def log_warning(logger, message):
    """Log warning with yellow bold formatting"""
    logger.info(f"[bold yellow]{message}[/bold yellow]")


def log_error(logger, message):
    """Log error with red bold formatting"""
    logger.error(f"[bold red]{message}[/bold red]")


def log_debug(logger, message):
    """Log debug with violet bold formatting"""
    logger.info(f"[bold magenta]{message}[/bold magenta]")


def log_success(logger, message):
    """Log success messages with cyan bold formatting"""
    logger.info(f"[bold cyan]{message}[/bold cyan]")


def log_processing(logger, message):
    """Log processing status with blue bold formatting"""
    logger.info(f"[bold blue]{message}[/bold blue]")