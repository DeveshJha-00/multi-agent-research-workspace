"""Async durable multi-agent research API."""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from src.core.config import settings
from src.data.ingestion import ingest_dataset
from src.db.artifact_store import get_artifact
from src.db.dataset_store import list_datasets
from src.db.research_job_store import (
    TERMINAL_STATUSES,
    IdempotencyConflictError,
    append_event,
    create_research_job,
    get_research_events,
    get_research_job,
    list_research_jobs,
    request_job_cancellation,
    retry_job,
)
from src.models.api import (
    DatasetUploadResponse,
    ResearchEventResponse,
    ResearchJobCreated,
    ResearchJobStatusResponse,
    ResearchRequest,
    ResearchResponse,
)

router = APIRouter(prefix="/agents", tags=["multi-agent"])


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
    description: Annotated[str, Header(alias="X-Description", max_length=500)] = "",
) -> DatasetUploadResponse:
    result = await ingest_dataset(file=file, session_id=session_id, description=description)
    return DatasetUploadResponse(**result)


@router.get("/datasets")
async def get_datasets(
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> list[dict]:
    return await list_datasets(session_id)


@router.post("/research", response_model=ResearchJobCreated, status_code=status.HTTP_202_ACCEPTED)
async def run_research(
    request: ResearchRequest,
    idempotency_key: Annotated[
        str | None, Header(alias="Idempotency-Key", min_length=1, max_length=200)
    ] = None,
) -> ResearchJobCreated:
    try:
        job, reused = await create_research_job(
            session_id=request.session_id,
            objective=request.objective,
            available_data=request.available_data,
            idempotency_key=idempotency_key,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ResearchJobCreated(task_id=job["task_id"], status=job["status"], reused=reused)


async def _require_job(task_id: str, session_id: str) -> dict:
    job = await get_research_job(task_id, session_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")
    return job


def _status_response(job: dict) -> ResearchJobStatusResponse:
    return ResearchJobStatusResponse(**job)


@router.get("/tasks", response_model=list[ResearchJobStatusResponse])
async def get_tasks(
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> list[ResearchJobStatusResponse]:
    return [_status_response(job) for job in await list_research_jobs(session_id)]


@router.get("/tasks/{task_id}", response_model=ResearchJobStatusResponse)
async def get_task(
    task_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> ResearchJobStatusResponse:
    return _status_response(await _require_job(task_id, session_id))


@router.get("/tasks/{task_id}/events", response_model=list[ResearchEventResponse])
async def get_task_events(
    task_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
    after: int = 0,
) -> list[ResearchEventResponse]:
    await _require_job(task_id, session_id)
    return [ResearchEventResponse(**item) for item in await get_research_events(task_id, after)]


@router.get("/tasks/{task_id}/events/stream")
async def stream_task_events(
    task_id: str,
    request: Request,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
    after: int = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    await _require_job(task_id, session_id)

    async def generate():
        try:
            header_sequence = int(last_event_id or 0)
        except ValueError:
            header_sequence = 0
        sequence = max(0, after, header_sequence)
        idle_ticks = 0
        while not await request.is_disconnected():
            batch = await get_research_events(task_id, sequence)
            for item in batch:
                sequence = item["sequence"]
                payload = json.dumps(jsonable_encoder(item), separators=(",", ":"))
                yield f"id: {sequence}\nevent: {item['event']}\ndata: {payload}\n\n"
            job = await get_research_job(task_id, session_id)
            if job is None or (job["status"] in TERMINAL_STATUSES and not batch):
                break
            idle_ticks += 1
            if idle_ticks % 30 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(settings.research_event_poll_seconds)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tasks/{task_id}/result", response_model=ResearchResponse)
async def get_task_result(
    task_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> ResearchResponse:
    job = await _require_job(task_id, session_id)
    if job["status"] != "completed" or not job.get("result"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Research result is not ready; current status is {job['status']}",
        )
    return ResearchResponse(**job["result"])


@router.delete("/tasks/{task_id}", response_model=ResearchJobStatusResponse)
async def cancel_task(
    task_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> ResearchJobStatusResponse:
    existing = await _require_job(task_id, session_id)
    job = await request_job_cancellation(task_id, session_id)
    if job is None:
        job = existing
    else:
        await append_event(
            task_id,
            event="cancellation_requested",
            stage=job["stage"],
            progress=job["progress"],
            message="Cancellation requested",
        )
    return _status_response(job)


@router.post("/tasks/{task_id}/retry", response_model=ResearchJobStatusResponse)
async def retry_task(
    task_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> ResearchJobStatusResponse:
    await _require_job(task_id, session_id)
    job = await retry_job(task_id, session_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed or cancelled research jobs can be retried",
        )
    await append_event(
        task_id,
        event="job_requeued",
        stage="queued",
        progress=job["progress"],
        message="Research job queued for retry",
    )
    return _status_response(job)


@router.get("/artifacts/{artifact_id}")
async def download_artifact(
    artifact_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> Response:
    artifact = await get_artifact(artifact_id, session_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(
        content=bytes(artifact["content"]),
        media_type=artifact["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{artifact["name"]}"'},
    )
