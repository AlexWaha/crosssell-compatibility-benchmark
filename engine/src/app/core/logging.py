"""Logging configuration for the engine service."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application. Call once at startup.

    Args:
        level: Log level string (e.g. "INFO", "DEBUG", "WARNING").
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
