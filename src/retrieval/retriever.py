"""Wraps the vector store retrieval and formats retrieved context."""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from src.vectorstore.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    db: VectorDatabase,
    k: int = None,
    method: str = "mmr",
) -> List[Document]:
    if method == "mmr":
        docs = db.mmr_search(query, k=k)
    else:
        docs = db.similarity_search(query, k=k)
    logger.info("Retrieved %d chunks for query: %r", len(docs), query[:60])
    return docs


def format_context(docs: List[Document]) -> str:
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"[{i}] Source: {source}" + (f", page {page}" if page else "")
        parts.append(f"{label}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def retrieve_with_context(
    query: str,
    db: VectorDatabase,
    k: int = None,
    method: str = "mmr",
) -> Tuple[List[Document], str]:
    docs = retrieve(query, db, k=k, method=method)
    context = format_context(docs)
    return docs, context
