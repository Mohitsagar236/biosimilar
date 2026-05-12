"""LangChain tools available to the RAG agent."""

from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.vectorstore.vector_db import VectorDatabase


def make_document_search_tool(db: "VectorDatabase"):
    @tool
    def search_documents(query: str) -> str:
        """Search the ingested document knowledge base for information relevant to a query.
        Use this whenever the user asks a question that might be answerable from the documents."""
        docs = db.mmr_search(query, k=5)
        if not docs:
            return "No relevant documents found for this query."
        parts = []
        for i, doc in enumerate(docs, 1):
            from pathlib import Path
            source = Path(doc.metadata.get("source", "unknown")).name
            page = doc.metadata.get("page", "")
            label = f"[Chunk {i} | {source}" + (f" p.{page}" if page else "") + "]"
            parts.append(f"{label}\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    return search_documents


def make_list_sources_tool(db: "VectorDatabase"):
    @tool
    def list_sources() -> str:
        """List all documents currently ingested in the knowledge base."""
        from pathlib import Path
        sources = db.list_sources()
        if not sources:
            return "No documents have been ingested yet."
        names = [Path(s).name for s in sources]
        return "Ingested documents:\n" + "\n".join(f"  • {n}" for n in names)

    return list_sources


@tool
def get_current_date() -> str:
    """Return today's date. Use when the user asks about current date or time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")
