"""Logging utilities."""

from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: str, log_dir: Optional[str] = None, filename: str = "run.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
