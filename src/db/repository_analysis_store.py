"""Durable internal stage state for repository-analysis specialists."""

from datetime import datetime, timezone
from typing import Any

from pymongo import ReturnDocument

from src.db.mongo_client import db

collection = db["repository_analysis_checkpoints"]


async def initialize_repository_analysis_store() -> None:
    await collection.create_index([("task_id", 1), ("analysis_key", 1)], unique=True)
    await collection.create_index([("repository_id", 1), ("updated_at", -1)])


async def get_repository_analysis_checkpoint(
    task_id: str, analysis_key: str
) -> dict[str, Any] | None:
    return await collection.find_one({"task_id": task_id, "analysis_key": analysis_key}, {"_id": 0})


async def save_repository_analysis_stage(
    *,
    task_id: str,
    analysis_key: str,
    repository_id: str,
    stage: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    document = await collection.find_one_and_update(
        {"task_id": task_id, "analysis_key": analysis_key},
        {
            "$setOnInsert": {
                "task_id": task_id,
                "analysis_key": analysis_key,
                "repository_id": repository_id,
                "created_at": now,
            },
            "$set": {"stage": stage, "updated_at": now, **values},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
        projection={"_id": 0},
    )
    return document
