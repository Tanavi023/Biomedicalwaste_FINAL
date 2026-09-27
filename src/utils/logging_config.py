"""Central logging configuration for command-line project workflows."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging once, then return the root logger."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
