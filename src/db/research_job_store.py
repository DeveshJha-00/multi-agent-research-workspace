"""Durable research jobs, leases, and replayable progress events."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from src.core.config import settings
from src.core.idempotency import canonical_hash
from src.db.mongo_client import db

jobs = db["research_jobs"]
events = db["research_events"]
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class IdempotencyConflictError(ValueError):
    """Raised when one idempotency key is reused with a different request."""


class ResearchJobCancelled(RuntimeError):
    """Raised at a graph boundary after cancellation is requested."""


async def initialize_research_job_store() -> None:
    await jobs.create_index("task_id", unique=True)
    await jobs.create_index(
        [("session_id", ASCENDING), ("idempotency_key", ASCENDING)],
        unique=True,
        partialFilterExpression={"idempotency_key": {"$type": "string"}},
    )
    await jobs.create_index([("status", ASCENDING), ("lease_expires_at", ASCENDING)])
    await events.create_index([("task_id", ASCENDING), ("sequence", ASCENDING)], unique=True)


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


async def create_research_job(
    *,
    session_id: str,
    objective: str,
    available_data: list[str],
    idempotency_key: str | None,
) -> tuple[dict[str, Any], bool]:
    request_hash = canonical_hash(
        {"session_id": session_id, "objective": objective, "available_data": available_data}
    )
    normalized_key = (idempotency_key.strip() or None) if idempotency_key else None
    if normalized_key:
        existing = await jobs.find_one(
            {"session_id": session_id, "idempotency_key": normalized_key}
        )
        if existing:
            if existing["request_hash"] != request_hash:
                raise IdempotencyConflictError(
                    "Idempotency-Key was already used for a different research request"
                )
            return _public(existing), True

    now = datetime.now(timezone.utc)
    document = {
        "task_id": str(uuid4()),
        "session_id": session_id,
        "idempotency_key": normalized_key,
        "request_hash": request_hash,
        "objective": objective,
        "available_data": available_data,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "attempts": 0,
        "event_sequence": 0,
        "result": None,
        "error": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
    }
    try:
        await jobs.insert_one(document)
    except DuplicateKeyError:
        existing = await jobs.find_one(
            {"session_id": session_id, "idempotency_key": normalized_key}
        )
        if existing and existing["request_hash"] == request_hash:
            return _public(existing), True
        raise IdempotencyConflictError(
            "Idempotency-Key was already used for a different research request"
        ) from None
    await append_event(
        document["task_id"],
        event="job_queued",
        stage="queued",
        progress=0,
        message="Research job queued",
    )
    return _public(document), False


async def get_research_job(task_id: str, session_id: str | None = None) -> dict | None:
    query: dict[str, Any] = {"task_id": task_id}
    if session_id:
        query["session_id"] = session_id
    return _public(await jobs.find_one(query))


async def list_research_jobs(session_id: str, limit: int = 20) -> list[dict]:
    cursor = (
        jobs.find({"session_id": session_id}, {"result": 0, "request_hash": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return [_public(item) async for item in cursor]


async def claim_research_job(worker_id: str) -> dict | None:
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(seconds=settings.research_job_lease_seconds)
    await jobs.update_many(
        {
            "attempts": {"$gte": settings.research_job_max_attempts},
            "$or": [
                {"status": "queued"},
                {"status": "running", "lease_expires_at": {"$lt": now}},
            ],
        },
        {
            "$set": {
                "status": "failed",
                "stage": "failed",
                "error": "Research job exceeded its automatic recovery attempts",
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    document = await jobs.find_one_and_update(
        {
            "attempts": {"$lt": settings.research_job_max_attempts},
            "$or": [
                {"status": "queued"},
                {"status": "running", "lease_expires_at": {"$lt": now}},
            ],
        },
        {
            "$set": {
                "status": "running",
                "stage": "starting",
                "lease_owner": worker_id,
                "lease_expires_at": lease_until,
                "updated_at": now,
                "started_at": now,
                "error": None,
            },
            "$inc": {"attempts": 1},
        },
        sort=[("created_at", ASCENDING)],
        return_document=ReturnDocument.AFTER,
    )
    return _public(document)


async def heartbeat_job(task_id: str, worker_id: str) -> bool:
    now = datetime.now(timezone.utc)
    result = await jobs.update_one(
        {"task_id": task_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "lease_expires_at": now + timedelta(seconds=settings.research_job_lease_seconds),
                "updated_at": now,
            }
        },
    )
    return result.modified_count == 1


async def release_job(task_id: str, worker_id: str) -> bool:
    """Release a job promptly during a graceful worker shutdown."""
    now = datetime.now(timezone.utc)
    result = await jobs.update_one(
        {"task_id": task_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "status": "queued",
                "stage": "queued",
                "lease_owner": None,
                "lease_expires_at": None,
                "updated_at": now,
            },
            "$inc": {"attempts": -1},
        },
    )
    return result.modified_count == 1


async def append_event(
    task_id: str,
    *,
    event: str,
    stage: str,
    progress: int,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict | None:
    now = datetime.now(timezone.utc)
    job = await jobs.find_one_and_update(
        {"task_id": task_id},
        {
            "$inc": {"event_sequence": 1},
            "$set": {
                "stage": stage,
                "progress": max(0, min(progress, 100)),
                "updated_at": now,
            },
        },
        return_document=ReturnDocument.AFTER,
    )
    if job is None:
        return None
    document = {
        "task_id": task_id,
        "sequence": job["event_sequence"],
        "event": event,
        "stage": stage,
        "progress": max(0, min(progress, 100)),
        "message": message,
        "details": details or {},
        "created_at": now,
    }
    await events.insert_one(document)
    return _public(document)


async def get_research_events(task_id: str, after: int = 0, limit: int = 100) -> list[dict]:
    cursor = (
        events.find({"task_id": task_id, "sequence": {"$gt": after}}, {"_id": 0})
        .sort("sequence", 1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def is_cancel_requested(task_id: str) -> bool:
    document = await jobs.find_one({"task_id": task_id}, {"status": 1})
    return bool(document and document["status"] in {"cancel_requested", "cancelled"})


async def ensure_job_not_cancelled(task_id: str) -> None:
    if await is_cancel_requested(task_id):
        raise ResearchJobCancelled("Research job was cancelled")


async def request_job_cancellation(task_id: str, session_id: str) -> dict | None:
    now = datetime.now(timezone.utc)
    document = await jobs.find_one_and_update(
        {"task_id": task_id, "session_id": session_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
        [
            {
                "$set": {
                    "status": {
                        "$cond": [{"$eq": ["$status", "queued"]}, "cancelled", "cancel_requested"]
                    },
                    "stage": {"$cond": [{"$eq": ["$status", "queued"]}, "cancelled", "cancelling"]},
                    "updated_at": now,
                    "completed_at": {
                        "$cond": [{"$eq": ["$status", "queued"]}, now, "$completed_at"]
                    },
                }
            }
        ],
        return_document=ReturnDocument.AFTER,
    )
    return _public(document)


async def complete_job(task_id: str, worker_id: str, result: dict[str, Any]) -> bool:
    now = datetime.now(timezone.utc)
    updated = await jobs.update_one(
        {"task_id": task_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "status": "completed",
                "stage": "completed",
                "progress": 100,
                "result": result,
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    return updated.modified_count == 1


async def fail_job(task_id: str, worker_id: str, error: str) -> bool:
    now = datetime.now(timezone.utc)
    updated = await jobs.update_one(
        {"task_id": task_id, "lease_owner": worker_id, "status": "running"},
        {
            "$set": {
                "status": "failed",
                "stage": "failed",
                "error": error[:2000],
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    return updated.modified_count == 1


async def cancel_job(task_id: str, worker_id: str) -> bool:
    now = datetime.now(timezone.utc)
    updated = await jobs.update_one(
        {"task_id": task_id, "lease_owner": worker_id, "status": "cancel_requested"},
        {
            "$set": {
                "status": "cancelled",
                "stage": "cancelled",
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    return updated.modified_count == 1


async def retry_job(task_id: str, session_id: str) -> dict | None:
    now = datetime.now(timezone.utc)
    document = await jobs.find_one_and_update(
        {"task_id": task_id, "session_id": session_id, "status": {"$in": ["failed", "cancelled"]}},
        {
            "$set": {
                "status": "queued",
                "stage": "queued",
                "error": None,
                "completed_at": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "updated_at": now,
                "attempts": 0,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    return _public(document)
