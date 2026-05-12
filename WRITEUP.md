# Agentic RAG System — Technical Write-Up

## 1. Architecture Decisions

### Document Ingestion Pipeline
The ingestion pipeline follows a clean, three-stage design: **load → preprocess → chunk**. Each stage is a separate Python module, making it easy to swap implementations (e.g., replace the PDF loader or add a new file type) without touching other stages.

**Why recursive character splitting?**
The `RecursiveCharacterTextSplitter` attempts to split at the most natural boundary available — paragraph breaks first, then sentence boundaries, then words, and finally characters as a last resort. This preserves semantic coherence within chunks far better than naively slicing at a fixed character count. Chunk size (1000 chars) and overlap (200 chars) were chosen as a balance: small enough to be specific, large enough to carry complete thoughts. The 200-char overlap ensures context is not lost at chunk boundaries.

### Embedding Strategy
Two providers are supported via a provider abstraction:
- **OpenAI `text-embedding-3-small`**: 1536-dimensional embeddings with state-of-the-art quality. Requires an API key but provides the best retrieval accuracy.
- **HuggingFace `all-MiniLM-L6-v2`**: 384-dimensional embeddings. Runs entirely locally with no API cost, enabling zero-cost operation for users without an OpenAI key.

The embedding model is abstracted behind `get_embeddings()` so the rest of the system is provider-agnostic.

### Vector Database: ChromaDB
ChromaDB was selected over FAISS for the following reasons:
1. **Persistence by default**: ChromaDB persists to disk automatically. FAISS requires manual save/load logic.
2. **Metadata filtering**: ChromaDB supports filtering by metadata (source, page, type) alongside vector similarity.
3. **Simple API**: A single `Chroma(persist_directory=...)` call handles both creation and loading of an existing store.
4. **No server required**: ChromaDB runs embedded in the Python process, removing deployment complexity.

### Retrieval: Maximum Marginal Relevance (MMR)
The default retrieval strategy is MMR rather than pure cosine similarity. Reason: similarity search tends to return near-duplicate chunks — all from the same paragraph or page — which wastes the context window. MMR iteratively selects chunks that are (a) relevant to the query AND (b) diverse with respect to already-selected chunks. This yields broader, more informative context for multi-faceted questions.

### Hallucination Prevention
Four complementary mechanisms minimize hallucinations:
1. **Strict system prompt**: The LLM is explicitly instructed to answer only from the provided context and to say "I don't have enough information" if the context is insufficient.
2. **Low temperature (0.1)**: Makes the model more deterministic and less likely to "creatively" fill in missing information.
3. **Empty context detection**: If retrieval returns no chunks (e.g., the query is entirely out of scope), the system returns a canned "not enough information" message without calling the LLM at all.
4. **RAGAS evaluation support**: The system includes RAGAS as a dependency to measure **faithfulness** (are all answer claims supported by retrieved context?) and **answer relevancy** (does the answer address the question?). This provides quantitative, reproducible proof that hallucination is minimized — not just a qualitative claim.

### Streaming Responses
The RAG agent exposes a `chat_stream()` generator that yields LLM tokens as they are produced. The Flask web UI simulates streaming by displaying a typing animation followed by the full response, while the terminal CLI displays the complete answer immediately. A `NotImplementedError` fallback ensures compatibility with local HuggingFace models that do not support streaming.

### Conversation Memory
A sliding-window memory stores the last 6 conversation turns (12 messages). This is injected into each prompt as "CONVERSATION HISTORY" to provide continuity in multi-turn conversations. The window prevents unbounded context growth that would eventually exceed the LLM's context window.

---

## 2. Limitations

### Retrieval is the Bottleneck
The system is only as good as its retrieval. If the relevant information exists in the documents but is not retrieved (because the query embedding is semantically distant from the chunk embeddings), the LLM correctly says "I don't know" — but the information was there. This can happen when:
- The user uses different terminology than the document.
- The answer is spread across multiple non-adjacent chunks.
- The document has low-quality text (scanned PDFs with OCR errors).

### No Multi-Hop Reasoning
If a question requires connecting information from two different documents or two distant sections of one document, a single retrieval step may not capture both pieces. For example: "What does the SOP say about model training, and how does that relate to the ML pipeline described in the guide?" — this requires reading from both documents simultaneously.

### Static Embeddings After Ingestion
Embeddings are computed once and stored. If the embedding model changes (e.g., upgrade from `text-embedding-3-small` to `text-embedding-3-large`), the entire vector store must be rebuilt, since query and document embeddings must come from the same model.

### Single-Stage Retrieval
The system retrieves once and passes the result to the LLM. There is no reranking step to filter out retrieved chunks that are tangentially related. A cross-encoder reranker would improve precision.

### Chunk Boundary Artifacts
Even with overlap, the splitter occasionally cuts across a sentence. This can make individual chunks harder for the LLM to interpret and slightly reduces answer quality.

### RAGAS Evaluation Depends on LLM Quality
The RAGAS faithfulness and answer relevancy metrics are themselves LLM-based — they use the same configured LLM to judge whether claims are supported. A weaker model (e.g., TinyLlama) may produce unreliable evaluation scores. For accurate evaluation, use GPT-3.5-turbo or Llama-3.1-8b-instant or better.

### No Document Update / Deduplication
If the same document is ingested twice, duplicate chunks are added to the vector store. The `--reset` flag clears all documents, but there is no per-document deduplication.

---

## 3. Suggestions for Scaling

### Infrastructure
- **Replace ChromaDB with Pinecone or Qdrant** for distributed vector storage supporting millions of documents.
- **Batch embedding API calls** during ingestion to maximize throughput and reduce API latency.
- **Async retrieval and streaming LLM responses** to reduce perceived latency in the UI.
- **Docker + Kubernetes deployment** for horizontal scaling of the API layer.

### Retrieval Quality
- **Add a reranking stage**: Use a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank the top-20 retrieved chunks down to the top-5 before passing to the LLM.
- **Hybrid search**: Combine dense vector search with sparse BM25 keyword search. This is especially important for queries containing product codes, proper nouns, or technical terms.
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer with the LLM, then embed that answer as the retrieval query. This bridges the query-document semantic gap.
- **Multi-query retrieval**: Generate 3–5 paraphrased versions of the user's query and retrieve for each, then deduplicate results.

### Document Management
- **Versioned ingestion**: Track document checksums; only re-embed changed chunks.
- **Metadata enrichment**: Extract and store document metadata (title, author, date, section headings) to enable filtered retrieval.
- **Hierarchical chunking**: Create both small chunks (for precise retrieval) and larger parent chunks (for richer LLM context), and return the parent when a child chunk matches.

### Evaluation and Monitoring
- **RAGAS integration**: Continuously evaluate context recall, context precision, answer faithfulness, and answer relevance.
- **Logging with LangSmith or Weights & Biases**: Trace every retrieval and generation call for debugging.
- **A/B testing**: Experiment with different chunk sizes, overlap, top-K values, and retrieval strategies.

### Fine-Tuning
- **Domain-specific embedding model**: Fine-tune an embedding model on your specific document corpus to significantly improve retrieval quality for specialized domains (medical, legal, engineering).
- **Instruction-tuned LLM**: Fine-tune the generation model on (question, context, answer) triples from your domain.

---

*Submitted by: [Your Name] | AI Internship Assignment — Agentic RAG*
