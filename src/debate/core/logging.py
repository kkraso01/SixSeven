"""Centralised logging configuration for the debate simulator.

Call ``setup_logging()`` once at application startup (CLI entry-point).
Library code should simply use ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DIR = Path("logs")


def setup_logging(
    *,
    level: int = logging.INFO,
    log_dir: Path = DEFAULT_LOG_DIR,
    log_file: str = "output.log",
) -> None:
    """Configure root logger with file + console handlers (idempotent).

    Args:
        level: Logging level (default ``INFO``).
        log_dir: Directory for the log file; created if absent.
        log_file: Name of the log file inside *log_dir*.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler(),
        ],
    )
    _CONFIGURED = True
