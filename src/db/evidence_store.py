"""MongoDB evidence ledger shared by specialist agents."""

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from src.db.mongo_client import db

collection = db["evidence"]


async def initialize_evidence_store() -> None:
    await collection.create_index([("task_id", 1), ("created_at", 1)])
    await collection.create_index([("session_id", 1), ("source", 1)])
    await collection.create_index(
        [("task_id", 1), ("operation_key", 1)],
        unique=True,
        partialFilterExpression={"operation_key": {"$type": "string"}},
    )


async def add_evidence(
    *,
    task_id: str,
    session_id: str,
    agent: str,
    content: str,
    source: str,
    url: str | None = None,
    document_id: str | None = None,
    page: int | None = None,
    confidence: float = 0.7,
    metadata: dict | None = None,
    operation_key: str | None = None,
) -> str:
    stable_key = operation_key or str(uuid5(NAMESPACE_URL, f"{task_id}:{source}:{content}"))
    evidence_id = str(uuid5(NAMESPACE_URL, f"evidence:{task_id}:{stable_key}"))
    await collection.update_one(
        {"task_id": task_id, "operation_key": stable_key},
        {
            "$setOnInsert": {
                "evidence_id": evidence_id,
                "task_id": task_id,
                "operation_key": stable_key,
                "session_id": session_id,
                "agent": agent,
                "content": content[:12000],
                "source": source,
                "url": url,
                "document_id": document_id,
                "page": page,
                "confidence": max(0.0, min(confidence, 1.0)),
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    return evidence_id


async def get_evidence(task_id: str, limit: int = 100) -> list[dict]:
    cursor = collection.find({"task_id": task_id}, {"_id": 0}).sort("created_at", 1).limit(limit)
    return await cursor.to_list(length=limit)


async def clear_evidence(task_id: str) -> None:
    await collection.delete_many({"task_id": task_id})
