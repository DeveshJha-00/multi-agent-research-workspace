"""Lazy OpenAI model clients with no import-time network work."""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.core.config import settings


@lru_cache
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=settings.llm_temperature,
        max_retries=2,
        timeout=60,
    )


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dimensions,
        max_retries=2,
        request_timeout=60,
    )


# Backward-compatible lazy proxy is intentionally omitted: callers must use factories.
