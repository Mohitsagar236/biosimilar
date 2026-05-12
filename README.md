# Agentic RAG Chatbot

A lightweight AI agent that ingests documents, creates embeddings, stores them in a vector database, and answers questions using retrieval-augmented generation (RAG). Answers are grounded strictly in the ingested documents — the system refuses to speculate when information is not available.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                         │
│                                                                  │
│  Documents      Preprocessing    Embedding        Vector DB      │
│  PDF/TXT/CSV  → Clean & Chunk  → sentence-       → ChromaDB      │
│  MD / GDrive                     transformers       (persisted)   │
└──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                            │
│                                                                  │
│  User Query → Embed Query → MMR Retrieval → Format Context       │
│                                                   │              │
│  Response ← LLM (Groq / OpenAI) ←── System Prompt + Context     │
│                                                                  │
│  Conversation Memory: Sliding window (last 6 turns)              │
└──────────────────────────────────────────────────────────────────┘
```

### Component Map

| Module | File | Responsibility |
|--------|------|----------------|
| Document Loader | `src/ingestion/document_loader.py` | Load PDF, TXT, CSV, MD |
| Preprocessor | `src/ingestion/preprocessor.py` | Clean and normalize text |
| Chunker | `src/ingestion/chunker.py` | Recursive character text splitting |
| Embedding Generator | `src/embeddings/embedding_generator.py` | HuggingFace or OpenAI embeddings |
| Vector Database | `src/vectorstore/vector_db.py` | ChromaDB with persistence |
| Retriever | `src/retrieval/retriever.py` | MMR or similarity search + context formatting |
| RAG Agent | `src/agent/rag_agent.py` | Orchestrates retrieval + LLM + memory |
| Agent Tools | `src/agent/tools.py` | LangChain tools for agentic ReAct loop |
| Conversation Memory | `src/memory/conversation_memory.py` | Sliding-window conversation history |
| Google Drive Loader | `src/ingestion/gdrive_loader.py` | Bonus: ingest from Google Drive |
| Flask Web UI | `web/app.py` | Full web interface with chat, upload, voice |
| CLI | `cli/main.py` | Terminal interface using Rich |

---

## Features

### Core (Required)
- **Document ingestion pipeline** — PDF, TXT, CSV, Markdown
- **Text chunking** — `RecursiveCharacterTextSplitter` with configurable size and overlap
- **Embedding generation** — `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- **Vector database** — ChromaDB with persistent storage
- **Query interface** — Flask web UI + CLI
- **LLM response generation** — Groq (Llama 3.1, free) or OpenAI

### Bonus Features
- **Multi-source ingestion** — local files, directories, and Google Drive (via service account)
- **Conversation memory** — sliding window of last 6 turns injected into every prompt
- **Agentic behavior** — LangGraph ReAct agent with `search_documents`, `list_sources`, and `get_current_date` tools; the agent autonomously decides when and what to retrieve
- **Web UI** — Flask-based chat interface with document manager, voice input (Web Speech API), and text-to-speech output

---

## Setup

### 1. Clone / open the project
```bash
cd biosimilar
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
Create a `.env` file in the project root:
```env
# Required: LLM provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here   # free at console.groq.com
GROQ_MODEL=llama-3.1-8b-instant

# Optional: use OpenAI instead
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...

# Vector store (defaults work out of the box)
CHROMA_PERSIST_DIR=./chroma_db
DOCUMENTS_DIR=./data/documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

> **No API key at all?** Set `LLM_PROVIDER=huggingface` in `.env` — the system falls back to `TinyLlama` for generation (requires `pip install transformers torch`). Embeddings always use `sentence-transformers` locally at no cost.

---

## Usage

### Step 1 — Ingest Documents

The `data/documents/` directory includes 11 sample documents covering AI fundamentals, machine learning, NLP, RAG, vector databases, LLM prompting, deep learning, AI ethics, MLOps, Python best practices, and transformer architecture.

```bash
# Ingest all documents in data/documents/ (default)
python ingest.py

# Ingest a specific file
python ingest.py --source path/to/your/file.pdf

# Ingest an entire folder
python ingest.py --source my_docs/

# Ingest from Google Drive (requires service account — see src/ingestion/gdrive_loader.py)
python ingest.py --gdrive YOUR_FOLDER_ID

# Clear the database and re-ingest
python ingest.py --reset
```

### Step 2 — Ask Questions

**Option A: Web UI (recommended)**
```bash
python web/app.py
```
Open [http://localhost:5000](http://localhost:5000). Features: multi-turn chat, document upload, document manager (view/delete), voice input, and text-to-speech.

**Option B: CLI**
```bash
python cli/main.py
```
Commands at the prompt:
- `sources` — list ingested documents
- `clear` — reset conversation memory
- `quit` — exit

### Step 3 — Run Tests
```bash
pytest tests/ -v
```

---

## Example Queries

After ingesting the sample documents:

```
What is retrieval-augmented generation and why does it reduce hallucinations?
What are the three main types of machine learning?
Compare FAISS and ChromaDB as vector databases.
Explain the transformer self-attention mechanism.
What is MMR retrieval and how does it differ from similarity search?
How does chain-of-thought prompting improve LLM reasoning?
What are the key steps in a machine learning project according to the MLOps SOP?
What ethical concerns exist around large language models?
```

---

## Architecture Decisions

### 1. ChromaDB over FAISS
ChromaDB provides persistence, metadata filtering, and a simple Python API without manual index save/load. FAISS is faster at scale but requires more plumbing. For a prototype with document-level metadata needs, ChromaDB is the right tradeoff.

### 2. MMR as default retrieval strategy
Maximum Marginal Relevance penalizes chunks too similar to already-selected ones, resulting in broader context coverage. Pure cosine similarity tends to return near-duplicate chunks from the same paragraph. MMR is the default; similarity search is the fallback if MMR raises an exception.

### 3. Recursive Character Splitter
`RecursiveCharacterTextSplitter` splits at natural boundaries (paragraphs → sentences → words) before character-level splitting. This preserves semantic coherence better than naive fixed-size windows, leading to more coherent retrieved chunks.

### 4. Groq + free local embeddings
Groq provides fast, free LLM inference via Llama 3.1. `sentence-transformers/all-MiniLM-L6-v2` runs locally at zero cost for embeddings. This makes the entire system free to operate with no credit card. OpenAI is supported as an optional upgrade.

### 5. Sliding-window conversation memory (6 turns)
Unlimited history would overflow the LLM's context window. A window of 6 turns (12 messages) captures enough recent context for multi-turn conversations without increasing cost or latency.

### 6. Strict system prompt for hallucination prevention
The LLM is instructed to answer only from retrieved context and to explicitly say "I don't have enough information" rather than speculate. Temperature is set to `0.1` for deterministic, factual outputs.

### 7. Two agent modes
The primary `RAGAgent` always retrieves before responding — reliable and fast. The optional `create_agentic_executor` uses LangGraph's ReAct loop with three tools, giving the model control over when and what to retrieve for more complex multi-step queries.

---

## Limitations

1. **Retrieval quality ceiling** — if the relevant information is not retrieved (poor chunking, semantic mismatch, or insufficient Top-K), the LLM correctly says "I don't know" but cannot recover the answer.

2. **Chunk boundary artifacts** — splitting can sever cross-sentence context. Increasing chunk size reduces this but raises embedding cost and may dilute retrieval relevance.

3. **No multi-hop reasoning** — if the answer requires connecting information from two distant parts of different documents, a single retrieval step may miss the link. Multi-hop RAG (retrieve → reason → retrieve again) is not implemented.

4. **Embedding model locked at ingestion** — changing the embedding model after ingestion requires a full re-ingest, since query and document embeddings must be in the same vector space.

5. **Single language** — optimized for English; other languages will have reduced embedding and retrieval quality without a multilingual model.

6. **Scalability** — ChromaDB is a local store. It is not suitable for large (millions of chunks) or multi-user production deployments without replacing it with a managed vector database.

---

## Scaling Suggestions

1. **Managed vector database** — Replace ChromaDB with Pinecone, Qdrant Cloud, or Weaviate for horizontal scaling, multi-tenancy, and enterprise SLAs.

2. **Reranking stage** — Add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second retrieval stage to improve top-K precision.

3. **Hybrid search** — Combine dense (embedding) and sparse (BM25/TF-IDF) retrieval for better coverage on exact-match queries like product codes or proper names.

4. **Semantic caching** — Cache query embeddings + responses for frequently asked questions to reduce API cost and latency.

5. **Async ingestion** — Process large document sets with async embedding calls or batch requests to the embedding API to reduce ingestion time.

6. **Fine-tuned embeddings** — Train a domain-specific embedding model on the document corpus for significantly better retrieval in specialized fields.

7. **Observability** — Integrate RAGAS or TruLens to continuously evaluate retrieval faithfulness and answer relevancy in production.

8. **Document versioning** — Track document versions and update only changed chunks rather than re-ingesting the entire corpus.

---

## Project Structure

```
biosimilar/
├── config.py                        # Centralized configuration (reads .env)
├── ingest.py                        # Ingestion pipeline CLI
├── requirements.txt
├── .env                             # API keys and settings (not committed)
├── README.md
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py       # PDF, TXT, CSV, MD loading
│   │   ├── chunker.py               # RecursiveCharacterTextSplitter wrapper
│   │   ├── preprocessor.py          # Text cleaning and normalization
│   │   └── gdrive_loader.py         # Google Drive ingestion (bonus)
│   ├── embeddings/
│   │   └── embedding_generator.py   # HuggingFace or OpenAI embeddings
│   ├── vectorstore/
│   │   └── vector_db.py             # ChromaDB wrapper (add, search, delete)
│   ├── retrieval/
│   │   └── retriever.py             # MMR / similarity retrieval + formatting
│   ├── agent/
│   │   ├── rag_agent.py             # RAGAgent + LangGraph ReAct agent
│   │   └── tools.py                 # search_documents, list_sources, get_current_date
│   ├── memory/
│   │   └── conversation_memory.py   # Sliding-window conversation history
│   └── utils/
│       └── helpers.py               # Logging setup
│
├── web/
│   ├── app.py                       # Flask backend (chat, ingest, document APIs)
│   ├── templates/
│   │   └── index.html               # Single-page web UI
│   └── static/
│       ├── app.js                   # Chat, upload, voice input, TTS logic
│       └── style.css                # UI styling
│
├── cli/
│   └── main.py                      # Rich terminal interface
│
├── data/
│   └── documents/                   # 11 sample documents
│       ├── 01_ai_fundamentals.txt
│       ├── 02_machine_learning_guide.txt
│       ├── 03_nlp_overview.txt
│       ├── 04_rag_explained.txt
│       ├── 05_vector_databases.txt
│       ├── 06_llm_prompting_guide.txt
│       ├── 08_deep_learning_guide.txt
│       ├── 09_ai_ethics_research.txt
│       ├── 10_mlops_sop.txt
│       ├── 11_python_best_practices.txt
│       └── 12_transformer_architecture_paper.txt
│
└── tests/
    ├── test_pipeline.py             # Unit tests
    └── test_e2e.py                  # End-to-end tests
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| LLM | Groq Llama-3.1-8b-instant (free) / OpenAI GPT (optional) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, free) |
| Vector DB | ChromaDB (persisted) |
| Orchestration | LangChain + LangGraph |
| Web UI | Flask + vanilla JS/HTML |
| CLI | Rich + Click |
| Testing | pytest |
| Multi-source | Google Drive API (service account) |

---

*Built for AI Internship Assignment — Agentic RAG System*
