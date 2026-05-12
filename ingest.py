#!/usr/bin/env python
"""Ingestion pipeline: load → preprocess → chunk → embed → store.

Supports three source types:
  - Local file:      python ingest.py --source path/to/file.pdf
  - Local directory: python ingest.py --source data/documents/
  - Google Drive:    python ingest.py --gdrive <folder_id>   (requires setup; see gdrive_loader.py)
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from src.embeddings.embedding_generator import get_embeddings
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_directory, load_document
from src.ingestion.preprocessor import preprocess_documents
from src.utils.helpers import setup_logging
from src.vectorstore.vector_db import VectorDatabase


def run_ingestion(
    source: str = None,
    gdrive_folder_id: str = None,
    reset: bool = False,
) -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    embeddings = get_embeddings()
    db = VectorDatabase(embeddings)

    if reset:
        logger.info("Resetting vector store...")
        db.reset()

    # ── Google Drive source ──────────────────────────────────────────────────
    if gdrive_folder_id:
        logger.info("Ingesting from Google Drive folder: %s", gdrive_folder_id)
        from src.ingestion.gdrive_loader import load_from_google_drive
        raw_docs = load_from_google_drive(gdrive_folder_id, recursive=True)

    # ── Local file or directory ──────────────────────────────────────────────
    else:
        source = source or config.DOCUMENTS_DIR
        path = Path(source)
        if path.is_file():
            raw_docs = load_document(path)
        elif path.is_dir():
            raw_docs = load_directory(path)
        else:
            logger.error("Source not found: %s", source)
            sys.exit(1)

    if not raw_docs:
        logger.warning("No documents loaded. Check the source path and file formats.")
        return 0

    # Deduplicate: skip documents whose source path is already in the vector store
    existing_sources = set(db.list_sources())
    unique_docs = [d for d in raw_docs if d.metadata.get("source") not in existing_sources]
    skipped = len(raw_docs) - len(unique_docs)
    if skipped:
        logger.info("Skipping %d already-ingested document segment(s).", skipped)
    if not unique_docs:
        logger.info("All documents already ingested. Nothing new to add.")
        return db.count()

    cleaned = preprocess_documents(unique_docs)
    chunks = chunk_documents(cleaned, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    count = db.add_documents(chunks)

    logger.info("Ingestion complete. %d chunks stored in vector DB.", count)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                                  # ingest data/documents/
  python ingest.py --source my_docs/               # ingest a local folder
  python ingest.py --source report.pdf             # ingest a single file
  python ingest.py --gdrive 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs79    # Google Drive folder
  python ingest.py --reset                         # clear DB then ingest
        """,
    )
    parser.add_argument("--source", default=None,
                        help="Local file or directory to ingest (default: data/documents/)")
    parser.add_argument("--gdrive", default=None, metavar="FOLDER_ID",
                        help="Google Drive folder ID to ingest from (requires service account setup)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing vector store data before ingesting")
    args = parser.parse_args()
    run_ingestion(source=args.source, gdrive_folder_id=args.gdrive, reset=args.reset)
