"""Durable, leased RAGAS evaluation jobs with per-metric checkpoints."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from src.core.config import settings
from src.core.idempotency import canonical_hash
from src.db.mongo_client import db

jobs = db["evaluation_jobs"]
TERMINAL_STATUSES = {"completed", "failed"}


class EvaluationIdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different evaluation."""


async def initialize_evaluation_job_store() -> None:
    await jobs.create_index("evaluation_id", unique=True)
    await jobs.create_index(
        [("session_id", ASCENDING), ("idempotency_key", ASCENDING)],
        unique=True,
        partialFilterExpression={"idempotency_key": {"$type": "string"}},
    )
    await jobs.create_index([("status", ASCENDING), ("lease_expires_at", ASCENDING)])
    await jobs.create_index(
        [("session_id", ASCENDING), ("response_id", ASCENDING), ("created_at", DESCENDING)]
    )


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


async def create_evaluation_job(
    *,
    session_id: str,
    response_id: str,
    reference: str | None,
    metric_names: list[str],
    idempotency_key: str | None,
) -> tuple[dict, bool]:
    normalized_reference = reference.strip() if reference and reference.strip() else None
    request_hash = canonical_hash(
        {
            "session_id": session_id,
            "response_id": response_id,
            "reference": normalized_reference,
        }
    )
    normalized_key = (idempotency_key.strip() or None) if idempotency_key else None
    if normalized_key:
        existing = await jobs.find_one(
            {"session_id": session_id, "idempotency_key": normalized_key}
        )
        if existing:
            if existing["request_hash"] != request_hash:
                raise EvaluationIdempotencyConflictError(
                    "Idempotency-Key was already used for a different evaluation request"
                )
            return _public(existing), True

    now = datetime.now(timezone.utc)
    document = {
        "evaluation_id": str(uuid4()),
        "session_id": session_id,
        "response_id": response_id,
        "reference": normalized_reference,
        "reference_supplied": normalized_reference is not None,
        "metric_names": metric_names,
        "metrics": {},
        "idempotency_key": normalized_key,
        "request_hash": request_hash,
        "status": "queued",
        "progress": 0,
        "attempts": 0,
        "error": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
    }
    try:
        await jobs.insert_one(document)
    except DuplicateKeyError:
        existing = await jobs.find_one(
            {"session_id": session_id, "idempotency_key": normalized_key}
        )
        if existing and existing["request_hash"] == request_hash:
            return _public(existing), True
        raise EvaluationIdempotencyConflictError(
            "Idempotency-Key was already used for a different evaluation request"
        ) from None
    return _public(document), False


async def get_evaluation_job(
    evaluation_id: str, session_id: str | None = None
) -> dict | None:
    query: dict[str, Any] = {"evaluation_id": evaluation_id}
    if session_id:
        query["session_id"] = session_id
    return _public(await jobs.find_one(query))


async def list_evaluation_jobs(
    session_id: str, response_id: str | None = None, limit: int = 20
) -> list[dict]:
    query = {"session_id": session_id}
    if response_id:
        query["response_id"] = response_id
    cursor = jobs.find(query, {"request_hash": 0, "reference": 0}).sort(
        "created_at", DESCENDING
    ).limit(limit)
    return [_public(item) async for item in cursor]


async def claim_evaluation_job(worker_id: str) -> dict | None:
    now = datetime.now(timezone.utc)
    await jobs.update_many(
        {
            "attempts": {"$gte": settings.evaluation_job_max_attempts},
            "$or": [
                {"status": "queued"},
                {"status": "running", "lease_expires_at": {"$lt": now}},
            ],
        },
        {
            "$set": {
                "status": "failed",
                "error": "Evaluation exceeded its automatic recovery attempts",
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    return _public(
        await jobs.find_one_and_update(
            {
                "attempts": {"$lt": settings.evaluation_job_max_attempts},
                "$or": [
                    {"status": "queued"},
                    {"status": "running", "lease_expires_at": {"$lt": now}},
                ],
            },
            {
                "$set": {
                    "status": "running",
                    "lease_owner": worker_id,
                    "lease_expires_at": now
                    + timedelta(seconds=settings.evaluation_job_lease_seconds),
                    "updated_at": now,
                    "started_at": now,
                    "error": None,
                },
                "$inc": {"attempts": 1},
            },
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
    )


async def heartbeat_evaluation_job(evaluation_id: str, worker_id: str) -> bool:
    result = await jobs.update_one(
        {"evaluation_id": evaluation_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "lease_expires_at": datetime.now(timezone.utc)
                + timedelta(seconds=settings.evaluation_job_lease_seconds),
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    return result.modified_count == 1


async def release_evaluation_job(evaluation_id: str, worker_id: str) -> bool:
    result = await jobs.update_one(
        {"evaluation_id": evaluation_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "status": "queued",
                "lease_owner": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(timezone.utc),
            },
            "$inc": {"attempts": -1},
        },
    )
    return result.modified_count == 1


async def record_metric_result(
    evaluation_id: str, worker_id: str, metric_name: str, result: dict
) -> bool:
    document = await jobs.find_one(
        {"evaluation_id": evaluation_id, "status": "running", "lease_owner": worker_id},
        {"metric_names": 1, "metrics": 1},
    )
    if not document:
        return False
    completed = sum(
        1
        for name in document["metric_names"]
        if name == metric_name or name in document.get("metrics", {})
    )
    total = max(1, len(document["metric_names"]))
    progress = min(99, int(completed * 100 / total))
    update = await jobs.update_one(
        {"evaluation_id": evaluation_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                f"metrics.{metric_name}": result,
                "progress": progress,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    return update.modified_count == 1


async def complete_evaluation_job(
    evaluation_id: str, worker_id: str, duration_seconds: float
) -> bool:
    now = datetime.now(timezone.utc)
    result = await jobs.update_one(
        {"evaluation_id": evaluation_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "status": "completed",
                "progress": 100,
                "duration_seconds": duration_seconds,
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    return result.modified_count == 1


async def fail_evaluation_job(
    evaluation_id: str, worker_id: str, error: str, duration_seconds: float
) -> bool:
    now = datetime.now(timezone.utc)
    result = await jobs.update_one(
        {"evaluation_id": evaluation_id, "status": "running", "lease_owner": worker_id},
        {
            "$set": {
                "status": "failed",
                "error": error[:1000],
                "duration_seconds": duration_seconds,
                "completed_at": now,
                "updated_at": now,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    return result.modified_count == 1


async def clear_evaluation_jobs(session_id: str) -> None:
    await jobs.delete_many({"session_id": session_id})
