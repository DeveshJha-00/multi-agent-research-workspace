"""Immutable Chat response snapshots used by optional evaluation jobs."""

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING

from src.core.config import settings
from src.db.mongo_client import db

collection = db["rag_responses"]


async def initialize_rag_response_store() -> None:
    await collection.create_index("response_id", unique=True)
    await collection.create_index(
        [("session_id", ASCENDING), ("created_at", DESCENDING)]
    )


def _bounded_contexts(contexts: list[dict]) -> list[dict]:
    output = []
    used = 0
    for item in contexts[: settings.ragas_max_contexts]:
        remaining = settings.ragas_max_context_chars - used
        if remaining <= 0:
            break
        content = str(item.get("content", "")).strip()[:remaining]
        if not content:
            continue
        output.append(
            {
                "content": content,
                "source": str(item.get("source") or "Unknown source")[:500],
                "document_id": item.get("document_id"),
                "page": item.get("page"),
                "url": str(item["url"])[:2000] if item.get("url") else None,
            }
        )
        used += len(content)
    return output


async def save_rag_response(
    *,
    response_id: str,
    session_id: str,
    question: str,
    answer: str,
    route: str,
    sources: list[dict],
    contexts: list[dict],
) -> None:
    document = {
        "response_id": response_id,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "route": route,
        "sources": sources,
        "contexts": _bounded_contexts(contexts),
        "model": settings.groq_chat_model,
        "retrieval_config": {
            "retrieval_top_k": settings.retrieval_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "retrieval_score_threshold": settings.retrieval_score_threshold,
            "rerank_relevance_threshold": settings.rerank_relevance_threshold,
        },
        "created_at": datetime.now(timezone.utc),
    }
    await collection.update_one(
        {"response_id": response_id}, {"$setOnInsert": document}, upsert=True
    )


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


async def get_rag_response(response_id: str, session_id: str) -> dict | None:
    return _public(
        await collection.find_one({"response_id": response_id, "session_id": session_id})
    )


async def clear_rag_responses(session_id: str) -> None:
    await collection.delete_many({"session_id": session_id})
