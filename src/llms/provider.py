"""Provider-neutral factories for chat, embeddings, and local reranking."""

import asyncio
import hashlib
import math
import re
from functools import lru_cache
from typing import Any, Protocol

from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from src.core.config import settings


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


@lru_cache
def get_rate_limiter() -> InMemoryRateLimiter:
    """Serialize requests enough to stay usable on Groq's free-tier burst limits."""
    return InMemoryRateLimiter(
        requests_per_second=settings.groq_requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


@lru_cache
def get_llm() -> ChatGroq:
    """Return the configured Groq chat model without doing import-time network work."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_chat_model,
        temperature=settings.llm_temperature,
        reasoning_effort="low",
        max_retries=0,
        max_tokens=settings.groq_max_output_tokens,
        timeout=60,
        rate_limiter=get_rate_limiter(),
    )


def get_structured_llm(schema: type[Any]):
    """Use Groq's native JSON-schema output instead of fragile forced tool calls."""
    return get_llm().with_structured_output(schema, method="json_schema")


class FastEmbedEmbeddings:
    """Small async adapter around FastEmbed's synchronous ONNX inference."""

    def __init__(self) -> None:
        self._model = TextEmbedding(
            model_name=settings.fastembed_model,
            cache_dir=settings.fastembed_cache_dir,
            lazy_load=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.embed(texts, batch_size=settings.embedding_batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


class HashEmbeddings:
    """Tiny deterministic embeddings for memory-constrained demo deployments.

    This is a signed hashing vectorizer, not a neural embedding model. It keeps the
    same vector-store interface without loading ONNX models, making small Render
    instances much less likely to be killed while indexing documents.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions or settings.embedding_dimensions

    def _features(self, text: str) -> list[str]:
        lowered = text.casefold()
        words = re.findall(r"[\w\u0900-\u097F]+", lowered, flags=re.UNICODE)
        features: list[str] = []
        features.extend(f"w:{word}" for word in words)
        features.extend(
            f"b:{left}_{right}" for left, right in zip(words, words[1:], strict=False)
        )
        compact = re.sub(r"\s+", "", lowered)
        if compact:
            ngram_size = 3 if len(compact) < 80 else 4
            features.extend(
                f"c:{compact[index : index + ngram_size]}"
                for index in range(max(0, len(compact) - ngram_size + 1))
            )
        return features or ["empty"]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big", signed=False)
            index = value % self.dimensions
            sign = 1.0 if (value >> 63) else -1.0
            weight = 1.5 if feature.startswith("w:") else 1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


@lru_cache
def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "hash":
        return HashEmbeddings()
    return FastEmbedEmbeddings()


@lru_cache
def get_reranker() -> Ranker:
    return Ranker(
        model_name=settings.reranker_model,
        cache_dir=settings.reranker_cache_dir,
        log_level=settings.log_level,
    )


async def rerank_passages(query: str, passages: list[str]) -> list[tuple[int, float]]:
    """Return original passage indexes and normalized cross-encoder relevance scores."""
    request = RerankRequest(
        query=query,
        passages=[{"id": index, "text": passage} for index, passage in enumerate(passages)],
    )
    ranked = await asyncio.to_thread(get_reranker().rerank, request)
    return [(int(item["id"]), float(item["score"])) for item in ranked]
