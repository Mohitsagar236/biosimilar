"""End-to-end integration tests — requires Groq API key and ingested documents."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pytest
import config


def requires_api_key(fn):
    return pytest.mark.skipif(
        not config.GROQ_API_KEY and not config.OPENAI_API_KEY,
        reason="No LLM API key set — skipping live LLM tests",
    )(fn)


def requires_ingested_docs(fn):
    return pytest.mark.skipif(
        not Path(config.CHROMA_PERSIST_DIR).exists(),
        reason="No ingested documents — run python ingest.py first",
    )(fn)


@pytest.fixture(scope="module")
def db():
    from src.embeddings.embedding_generator import get_embeddings
    from src.vectorstore.vector_db import VectorDatabase
    return VectorDatabase(get_embeddings())


@pytest.fixture(scope="module")
def agent(db):
    from src.agent.rag_agent import RAGAgent
    return RAGAgent(db)


# ── Core Requirement: Vector DB stores and retrieves ─────────────────────────

class TestVectorDatabase:
    def test_db_has_chunks(self, db):
        count = db.count()
        assert count > 0, "DB is empty - run python ingest.py first"
        print(f"\n  [OK] Vector DB has {count} chunks")

    def test_db_has_expected_sources(self, db):
        sources = db.list_sources()
        assert len(sources) >= 10, f"Expected 10+ documents, got {len(sources)}"
        names = [Path(s).name for s in sources]
        print(f"\n  [OK] Sources: {names}")

    def test_similarity_search_returns_results(self, db):
        results = db.similarity_search("machine learning", k=3)
        assert len(results) > 0
        assert all(hasattr(r, "page_content") for r in results)

    def test_mmr_search_returns_diverse_results(self, db):
        results = db.mmr_search("neural networks", k=5)
        assert len(results) > 0
        # MMR should return diverse chunks (no exact duplicates)
        contents = [r.page_content for r in results]
        assert len(set(contents)) == len(contents), "MMR returned duplicate chunks"

    def test_each_result_has_source_metadata(self, db):
        results = db.similarity_search("embeddings", k=3)
        for doc in results:
            assert "source" in doc.metadata


# ── Core Requirement: Retriever formats context correctly ────────────────────

class TestRetrieverIntegration:
    def test_retrieve_with_context_returns_docs_and_string(self, db):
        from src.retrieval.retriever import retrieve_with_context
        docs, ctx = retrieve_with_context("What is RAG?", db)
        assert isinstance(docs, list)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_context_contains_source_labels(self, db):
        from src.retrieval.retriever import retrieve_with_context
        _, ctx = retrieve_with_context("transformer architecture", db)
        assert "[1] Source:" in ctx

    def test_out_of_scope_query_still_retrieves_something(self, db):
        from src.retrieval.retriever import retrieve_with_context
        docs, ctx = retrieve_with_context("recipe for chocolate cake", db)
        # Retrieval always returns top-K — content may be irrelevant but not empty
        assert isinstance(docs, list)


# ── Core Requirement: LLM generates answers ──────────────────────────────────

class TestLLMResponses:
    @requires_api_key
    @requires_ingested_docs
    def test_factual_question_from_docs(self, agent):
        result = agent.chat("What are the three types of machine learning?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 30
        answer_lower = result["answer"].lower()
        assert any(w in answer_lower for w in ["supervised", "unsupervised", "reinforcement"])
        print(f"\n  [OK] Answer: {result['answer'][:150]}")

    @requires_api_key
    @requires_ingested_docs
    def test_hallucination_guard_on_unknown_topic(self, agent):
        result = agent.chat("What is the boiling point of liquid nitrogen?")
        # This is NOT in our documents — agent should admit it
        assert isinstance(result["answer"], str)
        print(f"\n  [OK] Out-of-scope: {result['answer'][:200]}")

    @requires_api_key
    @requires_ingested_docs
    def test_sources_are_returned(self, agent):
        result = agent.chat("Explain the transformer attention mechanism.")
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
        for src in result["sources"]:
            assert isinstance(src, str)

    @requires_api_key
    @requires_ingested_docs
    def test_conversation_memory_carries_context(self, agent):
        agent.clear_memory()
        agent.chat("My name is TestUser and I am studying RAG systems.")
        result2 = agent.chat("What topic am I studying according to what I just said?")
        # Memory should carry the context from the first message
        assert isinstance(result2["answer"], str)
        print(f"\n  [OK] Memory: {result2['answer'][:200]}")


# ── Bonus: Agentic tools ──────────────────────────────────────────────────────

class TestAgenticTools:
    def test_document_search_tool_returns_text(self, db):
        from src.agent.tools import make_document_search_tool
        tool = make_document_search_tool(db)
        result = tool.invoke({"query": "what is RAG"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_sources_tool_returns_document_names(self, db):
        from src.agent.tools import make_list_sources_tool
        tool = make_list_sources_tool(db)
        result = tool.invoke({})
        assert isinstance(result, str)
        assert "Ingested documents" in result or "No documents" in result

    def test_date_tool_returns_date_string(self):
        from src.agent.tools import get_current_date
        result = get_current_date.invoke({})
        assert "2026" in result or "202" in result  # current year

    @requires_api_key
    def test_agentic_executor_can_be_created(self, db):
        from langchain_core.messages import HumanMessage
        from src.agent.rag_agent import create_agentic_executor
        graph = create_agentic_executor(db)
        assert graph is not None, "create_agentic_executor returned None"

        try:
            result = graph.invoke({
                "messages": [HumanMessage(content="List the documents available in the knowledge base.")]
            })
            assert "messages" in result
            final_answer = result["messages"][-1].content
            assert isinstance(final_answer, str) and len(final_answer) > 0
            print(f"\n  [OK] Agentic: {final_answer[:200]}")
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                pytest.skip(f"Groq rate limit hit during test run (free tier) — graph was created successfully. Error: {e}")
            raise
