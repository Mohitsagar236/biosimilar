"""Flask backend for the Agentic RAG Chatbot."""

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

BASE = Path(__file__).parent

import config
from src.embeddings.embedding_generator import get_embeddings
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_document
from src.ingestion.preprocessor import preprocess_documents
from src.utils.helpers import setup_logging
from src.vectorstore.vector_db import VectorDatabase

setup_logging("INFO")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(BASE / "templates"),
    static_folder=str(BASE / "static"),
    static_url_path="/static",
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "60 per hour"],
    storage_uri="memory://",
)

# ── Initialise shared resources ───────────────────────────────────────────────
_embeddings = get_embeddings()
_db = VectorDatabase(_embeddings)
_agent = None

MAX_BYTES = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def get_agent():
    global _agent
    if _agent is None:
        from src.agent.rag_agent import RAGAgent
        _agent = RAGAgent(_db)
    return _agent


def _process_file(f) -> tuple[str | None, str | None]:
    """Load, preprocess, chunk, and store a single uploaded file.

    Returns (filename, error_message). On success error_message is None.
    """
    if not f.filename:
        return None, "empty filename"

    suffix = Path(f.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".csv", ".md"}:
        return f.filename, f"unsupported file type '{suffix}'"

    # Validate file size without reading the entire stream twice
    f.stream.seek(0, 2)
    size = f.stream.tell()
    f.stream.seek(0)
    if size > MAX_BYTES:
        return f.filename, f"file too large ({size // (1024*1024)} MB; max {config.MAX_UPLOAD_SIZE_MB} MB)"

    try:
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / f.filename
        f.save(str(tmp_path))

        raw = load_document(tmp_path)
        if not raw:
            return f.filename, "could not extract any text"

        cleaned = preprocess_documents(raw)
        if not cleaned:
            return f.filename, "all content filtered out (too short?)"

        chunks = chunk_documents(cleaned, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        if not chunks:
            return f.filename, "chunking produced 0 chunks"

        # Normalize source metadata to just the filename so deduplication and
        # the Documents panel work correctly regardless of temp directory path.
        for chunk in chunks:
            chunk.metadata["source"] = f.filename

        # Skip if already ingested (filename-based deduplication)
        existing = set(_db.list_sources())
        new_chunks = [c for c in chunks if c.metadata.get("source") not in existing]
        if not new_chunks:
            logger.info("Skipping already-ingested file: %s", f.filename)
            return f.filename, "already ingested"

        before = _db.count()
        _db.add_documents(new_chunks)
        after = _db.count()
        if after <= before:
            return f.filename, "chunks produced but none added to DB"

        logger.info("Ingested %s → %d new chunks", f.filename, after - before)
        return f.filename, None

    except Exception as e:
        logger.exception("Error processing file %s", f.filename)
        return f.filename, str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    count = _db.count()
    sources = [Path(s).name for s in _db.list_sources()]
    model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.OPENAI_MODEL or "huggingface-local"
    return jsonify({
        "chunk_count": count,
        "sources": sources,
        "model": model,
        "provider": config.LLM_PROVIDER,
    })


@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question too long (max 2000 characters)."}), 400

    if _db.count() == 0:
        return jsonify({
            "answer": "No documents ingested yet. Please upload documents first.",
            "sources": [],
        })

    try:
        result = get_agent().chat(question)
        logger.info("Chat query answered, sources: %s", result["sources"])
        return jsonify({
            "answer": result["answer"],
            "sources": [Path(s).name for s in result["sources"]],
        })
    except Exception as e:
        logger.exception("Chat error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
@limiter.limit("10 per minute")
def ingest():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    ingested = []
    errors = []
    skipped = []
    chunks_before = _db.count()

    for f in files:
        filename, err = _process_file(f)
        if filename is None:
            continue
        if err == "already ingested":
            skipped.append(filename)
        elif err:
            errors.append(f"{filename}: {err}")
        else:
            ingested.append(filename)

    if not ingested and not skipped:
        detail = "; ".join(errors) if errors else "No valid files found."
        return jsonify({"error": f"No files ingested. {detail}"}), 400

    # Reset agent so it picks up the updated DB
    global _agent
    _agent = None

    total_chunks = _db.count()
    new_chunks = total_chunks - chunks_before
    parts = [f"Ingested {len(ingested)} file(s), {new_chunks} new chunk(s) added."]
    if skipped:
        parts.append(f"Already ingested (skipped): {', '.join(skipped)}.")
    if errors:
        parts.append(f"Skipped with errors: {'; '.join(errors)}.")

    return jsonify({
        "message": " ".join(parts),
        "files": ingested,
        "total_chunks": total_chunks,
    })


@app.route("/api/documents")
def list_documents():
    try:
        stats = _db.get_document_stats()
    except Exception as e:
        logger.exception("get_document_stats failed")
        return jsonify({"error": str(e), "documents": []}), 500
    docs = []
    for item in stats:
        src = item["source"]
        name = Path(src).name
        ext = Path(src).suffix.lstrip(".").lower() or "unknown"
        docs.append({
            "name": name,
            "source": src,
            "type": ext,
            "chunks": item["chunks"],
        })
    return jsonify({"documents": docs})


@app.route("/api/documents/content")
def document_content():
    source = request.args.get("source", "").strip()
    if not source:
        return jsonify({"error": "source parameter required"}), 400

    chunks = _db.get_chunks_for_source(source)
    if not chunks:
        return jsonify({"error": "No content found for this document."}), 404

    def sort_key(d):
        return (d.metadata.get("page", 0), d.metadata.get("row", 0))
    chunks.sort(key=sort_key)

    full_content = "\n\n".join(c.page_content for c in chunks)
    limit = 15_000
    truncated = len(full_content) > limit
    return jsonify({
        "source": source,
        "name": Path(source).name,
        "type": Path(source).suffix.lstrip(".").lower() or "unknown",
        "content": full_content[:limit],
        "truncated": truncated,
        "total_chars": len(full_content),
        "chunks": len(chunks),
    })


@app.route("/api/documents", methods=["DELETE"])
@limiter.limit("20 per minute")
def delete_document():
    source = request.args.get("source", "").strip()
    if not source:
        return jsonify({"error": "source parameter required"}), 400
    try:
        removed = _db.delete_source(source)
        global _agent
        _agent = None
        logger.info("Deleted document: %s (%d chunks removed)", source, removed)
        return jsonify({
            "message": f"Deleted {Path(source).name} ({removed} chunk(s) removed).",
            "chunks_removed": removed,
        })
    except NotImplementedError as e:
        return jsonify({"error": str(e)}), 501
    except Exception as e:
        logger.exception("Error deleting document %s", source)
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug")
def debug():
    count = _db.count()
    sources = _db.list_sources()
    test_results = []
    if count > 0:
        try:
            docs = _db.similarity_search("test", k=1)
            test_results = [{"source": d.metadata.get("source"), "snippet": d.page_content[:100]} for d in docs]
        except Exception as e:
            test_results = [{"error": str(e)}]
    return jsonify({
        "chunk_count": count,
        "sources": sources,
        "test_search": test_results,
    })


@app.route("/api/deepgram-key")
def deepgram_key():
    key = os.getenv("DEEPGRAM_API_KEY", "")
    if not key or key == "your_deepgram_api_key_here":
        return jsonify({"error": "DEEPGRAM_API_KEY not configured in .env"}), 503
    return jsonify({"key": key})


@app.route("/api/clear-memory", methods=["POST"])
def clear_memory():
    agent = get_agent()
    agent.clear_memory()
    return jsonify({"message": "Conversation memory cleared."})


@app.route("/api/reset", methods=["POST"])
@limiter.limit("5 per minute")
def reset_db():
    global _agent
    _db.reset()
    _agent = None
    logger.info("Vector store reset via API.")
    return jsonify({"message": "Vector store cleared. Please re-ingest documents."})


if __name__ == "__main__":
    print(f"\n  RAG Chatbot running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
