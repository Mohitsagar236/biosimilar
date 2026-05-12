"""RAG agent — combines retrieval, memory, and LLM into a coherent chatbot.

Two modes:
  RAGAgent          — direct retrieval → LLM (reliable, fast, default)
  create_agentic_executor — LangChain AgentExecutor with tools (true agentic loop)
"""

import logging
from typing import Dict, Generator, List

from langchain_core.messages import HumanMessage, SystemMessage

import config
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.retriever import retrieve_with_context
from src.vectorstore.vector_db import VectorDatabase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document-based assistant. You answer questions ONLY using the context retrieved from the ingested documents.

Rules you must follow:
1. Base your answer strictly on the provided CONTEXT. Do not add information not present in the context.
2. If the context does not contain enough information to answer, say clearly: "I don't have enough information in the ingested documents to answer this question."
3. Always cite the source document name(s) at the end of your answer.
4. Be concise and factual. Never speculate or hallucinate.
5. If the user's question is ambiguous, ask for clarification.
"""

ANSWER_TEMPLATE = """CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Answer based strictly on the context above. If the context lacks the information needed, say so explicitly.
"""


def _get_llm():
    from langchain_openai import ChatOpenAI

    if config.LLM_PROVIDER == "groq" and config.GROQ_API_KEY:
        return ChatOpenAI(
            model=config.GROQ_MODEL,
            openai_api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

    if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

    # HuggingFace fallback
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline
        logger.info("Loading HuggingFace LLM — this may take a moment.")
        pipe = hf_pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        raise RuntimeError(
            "No LLM available. Set GROQ_API_KEY or OPENAI_API_KEY in .env, or install transformers."
        ) from e


# ── Direct RAG Agent (default) ────────────────────────────────────────────────

class RAGAgent:
    """Simple, reliable RAG agent: retrieve context → build prompt → call LLM."""

    def __init__(self, db: VectorDatabase, memory_window: int = 6):
        self._db = db
        self._memory = ConversationMemory(window=memory_window)
        self._llm = _get_llm()
        self._last_sources: List[str] = []

    def _build_history_str(self) -> str:
        messages = self._memory.get_history()
        if not messages:
            return "(no prior conversation)"
        lines = []
        for m in messages:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def chat(self, question: str) -> Dict:
        try:
            docs, context = retrieve_with_context(question, self._db, method="mmr")
        except Exception:
            docs, context = retrieve_with_context(question, self._db, method="similarity")

        if not context:
            answer = (
                "I don't have enough information in the ingested documents to answer this question. "
                "Please make sure relevant documents have been ingested."
            )
            self._memory.add_user(question)
            self._memory.add_assistant(answer)
            return {"answer": answer, "sources": [], "context": ""}

        prompt_text = ANSWER_TEMPLATE.format(
            context=context,
            history=self._build_history_str(),
            question=question,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]

        response = self._llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        self._memory.add_user(question)
        self._memory.add_assistant(answer)

        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        return {"answer": answer, "sources": sources, "context": context}

    def chat_stream(self, question: str) -> Generator[str, None, None]:
        """Stream response tokens. After the generator is exhausted, self.last_sources is populated."""
        try:
            docs, context = retrieve_with_context(question, self._db, method="mmr")
        except Exception:
            docs, context = retrieve_with_context(question, self._db, method="similarity")

        if not context:
            answer = (
                "I don't have enough information in the ingested documents to answer this question. "
                "Please make sure relevant documents have been ingested."
            )
            self._memory.add_user(question)
            self._memory.add_assistant(answer)
            self._last_sources = []
            yield answer
            return

        prompt_text = ANSWER_TEMPLATE.format(
            context=context,
            history=self._build_history_str(),
            question=question,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]

        full_answer = ""
        try:
            for chunk in self._llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_answer += token
                yield token
        except NotImplementedError:
            # HuggingFace local models don't support streaming — fall back to full invoke
            response = self._llm.invoke(messages)
            full_answer = response.content if hasattr(response, "content") else str(response)
            yield full_answer

        self._memory.add_user(question)
        self._memory.add_assistant(full_answer)
        self._last_sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    @property
    def last_sources(self) -> List[str]:
        return self._last_sources

    def clear_memory(self) -> None:
        self._memory.clear()

    @property
    def memory(self) -> ConversationMemory:
        return self._memory


# ── Agentic RAG (bonus: ReAct tool-calling agent via LangGraph) ──────────────

def create_agentic_executor(db: VectorDatabase):
    """Create a LangGraph ReAct agent that uses tools for agentic behaviour.

    The agent autonomously decides WHEN to search documents, list sources,
    or answer directly — rather than always retrieving first.

    Returns a compiled LangGraph. Call .invoke({"messages": [("user", question)]}).
    The response dict has key "messages"; the last message is the final answer.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent

    from src.agent.tools import (
        get_current_date,
        make_document_search_tool,
        make_list_sources_tool,
    )

    llm = _get_llm()
    tools = [
        make_document_search_tool(db),
        make_list_sources_tool(db),
        get_current_date,
    ]

    system_prompt = (
        "You are a precise document-based assistant with access to tools.\n"
        "Always use the 'search_documents' tool to retrieve relevant information before answering.\n"
        "Only answer based on the retrieved document content.\n"
        "If the documents don't contain the answer, say so clearly.\n"
        "Always cite the source document name(s) in your final answer."
    )

    return create_react_agent(llm, tools, prompt=system_prompt)
