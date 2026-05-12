"""Vector store abstraction supporting ChromaDB (default) and FAISS.

Set VECTOR_DB=faiss in .env to switch to FAISS.
ChromaDB persists automatically; FAISS saves/loads from FAISS_INDEX_PATH.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config

logger = logging.getLogger(__name__)

VECTOR_DB = os.getenv("VECTOR_DB", "chroma").lower()
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(Path(config.BASE_DIR) / "faiss_index"))


# ── ChromaDB backend ──────────────────────────────────────────────────────────

class _ChromaBackend:
    def __init__(self, embeddings: Embeddings):
        from langchain_chroma import Chroma
        self._store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    def add_documents(self, documents: List[Document]) -> None:
        self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int) -> List[Document]:
        count = self._store._collection.count()
        safe_k = min(k, count)
        if safe_k == 0:
            return []
        return self._store.similarity_search(query, k=safe_k)

    def mmr_search(self, query: str, k: int, fetch_k: int) -> List[Document]:
        count = self._store._collection.count()
        safe_k = min(k, count)
        safe_fetch_k = min(fetch_k, count)
        if safe_k == 0:
            return []
        return self._store.max_marginal_relevance_search(query, k=safe_k, fetch_k=safe_fetch_k)

    def as_retriever(self, search_type: str, k: int):
        kwargs = {"k": k}
        if search_type == "mmr":
            kwargs["fetch_k"] = k * 3
        return self._store.as_retriever(search_type=search_type, search_kwargs=kwargs)

    def count(self) -> int:
        return self._store._collection.count()

    def list_sources(self) -> List[str]:
        result = self._store._collection.get(include=["metadatas"])
        return sorted({m.get("source", "unknown") for m in result["metadatas"]})

    def get_document_stats(self) -> List[dict]:
        result = self._store._collection.get(include=["metadatas"])
        metadatas = result["metadatas"] or []
        counts: dict = {}
        for meta in metadatas:
            src = (meta or {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunks": cnt} for src, cnt in sorted(counts.items())]

    def get_chunks_for_source(self, source: str) -> List[Document]:
        result = self._store._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        docs = []
        documents  = result["documents"]  or []
        metadatas  = result["metadatas"]  or []
        for content, meta in zip(documents, metadatas):
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def delete_source(self, source: str) -> int:
        before = self.count()
        self._store._collection.delete(where={"source": source})
        return before - self.count()

    def reset(self) -> None:
        self._store._collection.delete(where={"source": {"$ne": ""}})


# ── FAISS backend ─────────────────────────────────────────────────────────────

class _FAISSBackend:
    def __init__(self, embeddings: Embeddings):
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")
        self._embeddings = embeddings
        self._FAISS = FAISS
        self._store = None
        self._index_path = Path(FAISS_INDEX_PATH)
        if self._index_path.exists():
            logger.info("Loading existing FAISS index from %s", self._index_path)
            self._store = FAISS.load_local(
                str(self._index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

    def add_documents(self, documents: List[Document]) -> None:
        if self._store is None:
            self._store = self._FAISS.from_documents(documents, self._embeddings)
        else:
            self._store.add_documents(documents)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_path))

    def similarity_search(self, query: str, k: int) -> List[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def mmr_search(self, query: str, k: int, fetch_k: int) -> List[Document]:
        if self._store is None:
            return []
        return self._store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    def as_retriever(self, search_type: str, k: int):
        if self._store is None:
            raise RuntimeError("FAISS store is empty. Run ingest.py first.")
        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    def count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def list_sources(self) -> List[str]:
        if self._store is None:
            return []
        sources = {
            doc.metadata.get("source", "unknown")
            for doc in self._store.docstore._dict.values()
        }
        return sorted(sources)

    def get_document_stats(self) -> List[dict]:
        if self._store is None:
            return []
        counts: dict = {}
        for doc in self._store.docstore._dict.values():
            src = doc.metadata.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunks": cnt} for src, cnt in sorted(counts.items())]

    def get_chunks_for_source(self, source: str) -> List[Document]:
        if self._store is None:
            return []
        return [
            doc for doc in self._store.docstore._dict.values()
            if doc.metadata.get("source") == source
        ]

    def delete_source(self, source: str) -> int:
        raise NotImplementedError(
            "Per-document deletion is not supported with FAISS. Use reset() to clear all documents."
        )

    def reset(self) -> None:
        import shutil
        if self._index_path.exists():
            shutil.rmtree(self._index_path)
        self._store = None


# ── Public facade ─────────────────────────────────────────────────────────────

class VectorDatabase:
    """Provider-agnostic vector database wrapper.

    Defaults to ChromaDB. Set VECTOR_DB=faiss in .env to use FAISS.
    """

    def __init__(self, embeddings: Embeddings):
        if VECTOR_DB == "faiss":
            logger.info("Using FAISS vector store.")
            self._backend = _FAISSBackend(embeddings)
        else:
            logger.info("Using ChromaDB vector store.")
            self._backend = _ChromaBackend(embeddings)

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            logger.warning("No documents to add.")
            return 0
        self._backend.add_documents(documents)
        count = self._backend.count()
        logger.info("Vector store now has %d chunks.", count)
        return count

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        return self._backend.similarity_search(query, k=k or config.TOP_K)

    def mmr_search(self, query: str, k: int = None, fetch_k: int = None) -> List[Document]:
        k = k or config.TOP_K
        return self._backend.mmr_search(query, k=k, fetch_k=fetch_k or k * 3)

    def as_retriever(self, search_type: str = "mmr", k: int = None):
        return self._backend.as_retriever(search_type=search_type, k=k or config.TOP_K)

    def count(self) -> int:
        try:
            return self._backend.count()
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        return self._backend.list_sources()

    def get_document_stats(self) -> List[dict]:
        return self._backend.get_document_stats()

    def get_chunks_for_source(self, source: str) -> List[Document]:
        return self._backend.get_chunks_for_source(source)

    def delete_source(self, source: str) -> int:
        removed = self._backend.delete_source(source)
        logger.info("Deleted source %s (%d chunks removed).", source, removed)
        return removed

    def reset(self) -> None:
        self._backend.reset()
        logger.info("Vector store cleared.")
