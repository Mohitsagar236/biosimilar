"""Returns the appropriate embeddings model based on LLM_PROVIDER.

  LLM_PROVIDER=groq        → HuggingFace local embeddings (Groq has no embeddings API)
  LLM_PROVIDER=openai      → OpenAI text-embedding-3-small
  LLM_PROVIDER=huggingface → HuggingFace local embeddings (fully free, no API key)
"""

import logging

from langchain_core.embeddings import Embeddings

import config

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenAI embeddings: %s", config.OPENAI_EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
    # Groq and HuggingFace both use free local embeddings
    return _hf_embeddings()


def _hf_embeddings() -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("Using HuggingFace local embeddings: %s", config.HF_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
