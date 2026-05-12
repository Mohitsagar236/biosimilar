"""Shared utilities."""

import hashlib
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def file_hash(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate(text: str, max_chars: int = 200) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def format_sources(docs) -> str:
    seen = set()
    lines = []
    for doc in docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        if src not in seen:
            seen.add(src)
            lines.append(f"  • {src}")
    return "\n".join(lines) if lines else "  • (no sources)"
