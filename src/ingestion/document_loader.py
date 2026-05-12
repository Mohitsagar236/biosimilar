"""Loads PDF, TXT, CSV, and Markdown files into LangChain Document objects."""

import csv
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _load_txt(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(path), "type": "txt"})]


def _load_pdf(path: Path) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": str(path), "type": "pdf", "page": i + 1},
            ))
    return docs


def _load_csv(path: Path) -> List[Document]:
    docs = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        column_summary = f"Columns: {', '.join(columns)}" if columns else ""
        for i, row in enumerate(reader):
            # Include column context so the LLM can interpret field values
            pairs = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            text = f"{column_summary}\n{pairs}" if column_summary else pairs
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(path), "type": "csv", "row": i + 1},
                ))
    return docs


def _load_md(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(path), "type": "md"})]


_LOADERS = {
    ".txt": _load_txt,
    ".pdf": _load_pdf,
    ".csv": _load_csv,
    ".md": _load_md,
}


def load_document(path: str | Path) -> List[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        logger.warning("Unsupported file type: %s — skipping.", path)
        return []
    logger.info("Loading %s", path.name)
    return loader(path)


def load_directory(directory: str | Path, recursive: bool = True) -> List[Document]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    all_docs: List[Document] = []
    for file_path in sorted(directory.glob(pattern)):
        if file_path.is_file() and file_path.suffix.lower() in _LOADERS:
            docs = load_document(file_path)
            all_docs.extend(docs)

    logger.info("Loaded %d document segments from %s", len(all_docs), directory)
    return all_docs
