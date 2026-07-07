"""MongoDB-backed generated artifacts."""

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from bson import Binary

from src.db.mongo_client import db

collection = db["artifacts"]


async def initialize_artifact_store() -> None:
    await collection.create_index([("task_id", 1), ("created_at", 1)])
    await collection.create_index("artifact_id", unique=True)
    await collection.create_index(
        [("task_id", 1), ("operation_key", 1)],
        unique=True,
        partialFilterExpression={"operation_key": {"$type": "string"}},
    )


async def save_artifact(
    *,
    task_id: str,
    session_id: str,
    name: str,
    media_type: str,
    content: bytes,
    operation_key: str | None = None,
) -> str:
    stable_key = operation_key or str(uuid5(NAMESPACE_URL, f"{task_id}:{name}"))
    artifact_id = str(uuid5(NAMESPACE_URL, f"artifact:{task_id}:{stable_key}"))
    await collection.update_one(
        {"task_id": task_id, "operation_key": stable_key},
        {
            "$setOnInsert": {
                "artifact_id": artifact_id,
                "task_id": task_id,
                "operation_key": stable_key,
                "session_id": session_id,
                "name": name,
                "media_type": media_type,
                "content": Binary(content),
                "size": len(content),
                "created_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    return artifact_id


async def get_artifact(artifact_id: str, session_id: str) -> dict | None:
    return await collection.find_one(
        {"artifact_id": artifact_id, "session_id": session_id},
        {"_id": 0},
    )


async def list_artifacts(task_id: str) -> list[dict]:
    return await collection.find(
        {"task_id": task_id},
        {"_id": 0, "content": 0},
    ).to_list(length=50)
