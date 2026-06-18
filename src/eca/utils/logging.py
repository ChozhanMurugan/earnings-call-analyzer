"""Lightweight logging wrapper around loguru."""
from __future__ import annotations

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {name}:{function}:{line} - {message}")

__all__ = ["logger"]
