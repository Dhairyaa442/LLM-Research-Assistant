"""
Shared utility helpers used across the project.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import wraps
from typing import Any, Callable


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def timer(fn: Callable) -> Callable:
    """Decorator that logs the execution time of a function."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.getLogger(fn.__module__).debug(
            "%s completed in %.2fs", fn.__qualname__, elapsed
        )
        return result
    return wrapper


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate a string and append ellipsis if needed."""
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def format_docs(docs: list) -> str:
    """Format a list of LangChain Documents into a readable string block."""
    if not docs:
        return "No documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[{i}] ({source})\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
