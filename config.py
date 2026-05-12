import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

BASE_DIR = Path(__file__).parent

# LLM provider: "groq", "openai", or "huggingface"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# OpenAI settings (used if LLM_PROVIDER=openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# HuggingFace (free local embeddings — always used when provider is groq)
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM generation settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# Vector store
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = "rag_documents"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# UI / display
SNIPPET_MAX_CHARS = int(os.getenv("SNIPPET_MAX_CHARS", "200"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))

# Paths
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents"))

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".md"}

# ── Startup validation ────────────────────────────────────────────────────────
if not (0 < CHUNK_SIZE <= 10_000):
    raise ValueError(f"CHUNK_SIZE must be between 1 and 10000, got {CHUNK_SIZE}")
if not (0 <= CHUNK_OVERLAP < CHUNK_SIZE):
    raise ValueError(f"CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE ({CHUNK_SIZE}), got {CHUNK_OVERLAP}")
if not (1 <= TOP_K <= 50):
    raise ValueError(f"TOP_K must be between 1 and 50, got {TOP_K}")
if not (0.0 <= LLM_TEMPERATURE <= 2.0):
    raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {LLM_TEMPERATURE}")
if not (1 <= MAX_TOKENS <= 32_768):
    raise ValueError(f"MAX_TOKENS must be between 1 and 32768, got {MAX_TOKENS}")
