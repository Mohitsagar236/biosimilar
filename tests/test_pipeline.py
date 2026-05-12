"""Unit + integration tests covering every core requirement of the assignment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.preprocessor import clean_text, preprocess_documents
from src.memory.conversation_memory import ConversationMemory


# ── Requirement: Text preprocessing ──────────────────────────────────────────

class TestPreprocessor:
    def test_clean_text_normalizes_line_endings(self):
        raw = "Hello    world\r\n\r\nTest"
        result = clean_text(raw)
        assert "    " not in result
        assert "\r" not in result

    def test_clean_text_collapses_blank_lines(self):
        raw = "A\n\n\n\nB"
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_preprocess_drops_near_empty_docs(self):
        docs = [
            Document(page_content="Hi", metadata={}),
            Document(page_content="A" * 100, metadata={}),
        ]
        result = preprocess_documents(docs)
        assert len(result) == 1

    def test_preprocess_preserves_metadata(self):
        docs = [Document(page_content="A" * 100, metadata={"source": "test.txt", "page": 1})]
        result = preprocess_documents(docs)
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["page"] == 1


# ── Requirement: Text chunking ────────────────────────────────────────────────

class TestChunker:
    def test_single_long_doc_is_split(self):
        doc = Document(page_content="A" * 3000, metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_size_is_respected(self):
        doc = Document(page_content="word " * 1000, metadata={})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.page_content) <= 250

    def test_empty_input_returns_empty(self):
        assert chunk_documents([], chunk_size=500, chunk_overlap=50) == []

    def test_metadata_propagated_to_chunks(self):
        doc = Document(page_content="word " * 500, metadata={"source": "doc.pdf", "page": 3})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["source"] == "doc.pdf"

    def test_overlap_creates_shared_content(self):
        content = " ".join(f"word{i}" for i in range(200))
        doc = Document(page_content=content, metadata={})
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=30)
        if len(chunks) >= 2:
            end_of_first = chunks[0].page_content[-40:]
            start_of_second = chunks[1].page_content[:80]
            assert any(w in start_of_second for w in end_of_first.split())


# ── Requirement: Document loading (multi-format) ─────────────────────────────

class TestDocumentLoader:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("This is a test document with enough content to be processed correctly.")
        from src.ingestion.document_loader import load_document
        docs = load_document(f)
        assert len(docs) == 1
        assert "test document" in docs[0].page_content
        assert docs[0].metadata["type"] == "txt"

    def test_load_csv_produces_one_doc_per_row(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        from src.ingestion.document_loader import load_document
        docs = load_document(f)
        assert len(docs) == 2
        assert "Alice" in docs[0].page_content
        assert docs[0].metadata["type"] == "csv"

    def test_load_directory_finds_all_supported_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Content of file A " * 10)
        (tmp_path / "b.txt").write_text("Content of file B " * 10)
        (tmp_path / "skip.xyz").write_text("Should be skipped")
        from src.ingestion.document_loader import load_directory
        docs = load_directory(tmp_path, recursive=False)
        assert len(docs) == 2

    def test_unsupported_extension_returns_empty(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        from src.ingestion.document_loader import load_document
        docs = load_document(f)
        assert docs == []

    def test_source_metadata_is_set(self, tmp_path):
        f = tmp_path / "myfile.txt"
        f.write_text("Some content here that is long enough.")
        from src.ingestion.document_loader import load_document
        docs = load_document(f)
        assert str(f) == docs[0].metadata["source"]


# ── Requirement: Conversation memory ─────────────────────────────────────────

class TestConversationMemory:
    def test_add_and_retrieve_messages(self):
        mem = ConversationMemory(window=4)
        mem.add_user("Hello")
        mem.add_assistant("Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_window_limits_returned_history(self):
        mem = ConversationMemory(window=2)
        for i in range(10):
            mem.add_user(f"Q{i}")
            mem.add_assistant(f"A{i}")
        assert len(mem.get_history()) == 4  # window=2 → 2 turns = 4 messages

    def test_clear_empties_history(self):
        mem = ConversationMemory()
        mem.add_user("test")
        mem.clear()
        assert len(mem) == 0

    def test_to_langchain_messages_format(self):
        mem = ConversationMemory()
        mem.add_user("Hi")
        mem.add_assistant("Hello")
        msgs = mem.to_langchain_messages()
        assert msgs[0] == {"role": "user", "content": "Hi"}
        assert msgs[1] == {"role": "assistant", "content": "Hello"}

    def test_full_conversation_round_trip(self):
        mem = ConversationMemory(window=10)
        for i in range(5):
            mem.add_user(f"Question {i}")
            mem.add_assistant(f"Answer {i}")
        assert len(mem) == 10
        assert mem.get_history()[-1].content == "Answer 4"


# ── Requirement: Retriever formatting ────────────────────────────────────────

class TestRetriever:
    def test_format_context_empty(self):
        from src.retrieval.retriever import format_context
        assert format_context([]) == ""

    def test_format_context_includes_source(self):
        from src.retrieval.retriever import format_context
        docs = [Document(
            page_content="Deep learning uses neural networks.",
            metadata={"source": "/data/dl.txt"},
        )]
        ctx = format_context(docs)
        assert "dl.txt" in ctx
        assert "Deep learning" in ctx

    def test_format_context_includes_page_number(self):
        from src.retrieval.retriever import format_context
        docs = [Document(
            page_content="Transformers are powerful.",
            metadata={"source": "/data/paper.pdf", "page": 5},
        )]
        ctx = format_context(docs)
        assert "page 5" in ctx

    def test_format_context_separates_multiple_docs(self):
        from src.retrieval.retriever import format_context
        docs = [
            Document(page_content="Doc one.", metadata={"source": "a.txt"}),
            Document(page_content="Doc two.", metadata={"source": "b.txt"}),
        ]
        ctx = format_context(docs)
        assert "---" in ctx


# ── Requirement: Config loads correctly ──────────────────────────────────────

class TestConfig:
    def test_config_has_required_attributes(self):
        import config
        assert hasattr(config, "LLM_PROVIDER")
        assert hasattr(config, "GROQ_API_KEY")
        assert hasattr(config, "GROQ_MODEL")
        assert hasattr(config, "CHUNK_SIZE")
        assert hasattr(config, "CHUNK_OVERLAP")
        assert hasattr(config, "TOP_K")
        assert hasattr(config, "DOCUMENTS_DIR")
        assert hasattr(config, "CHROMA_PERSIST_DIR")

    def test_chunk_size_is_positive(self):
        import config
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_top_k_is_positive(self):
        import config
        assert config.TOP_K > 0


# ── Requirement: Hallucination guard (no-context path) ───────────────────────

class TestHallucinationGuard:
    def test_agent_returns_fallback_when_db_empty(self):
        """Agent must say 'I don't know' rather than hallucinate when DB has no matches."""
        import warnings
        warnings.filterwarnings("ignore")

        from unittest.mock import MagicMock, patch
        from src.agent.rag_agent import RAGAgent

        mock_db = MagicMock()
        mock_db.mmr_search.return_value = []   # simulate no relevant docs found

        with patch("src.agent.rag_agent._get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            agent = RAGAgent(mock_db)

        result = agent.chat("What is the capital of Mars?")

        # LLM should NOT have been called — agent short-circuits
        mock_llm.invoke.assert_not_called()
        assert "don't have enough information" in result["answer"].lower() or \
               "not" in result["answer"].lower()
        assert result["sources"] == []
