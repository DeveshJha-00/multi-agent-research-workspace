"""Synchronous Phase-1 multi-agent research API."""

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Header, HTTPException, Response, UploadFile, status
from groq import APIError, RateLimitError

from src.core.config import settings
from src.core.integration_errors import groq_error_detail, groq_rate_limit_detail
from src.data.ingestion import ingest_dataset
from src.db.artifact_store import get_artifact
from src.db.dataset_store import list_datasets
from src.models.api import DatasetUploadResponse, ResearchRequest, ResearchResponse
from src.orchestration.research_graph import research_orchestrator

logger = logging.getLogger(__name__)
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


@router.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    task_id = str(uuid4())
    try:
        result = await research_orchestrator.ainvoke(
            {
                "task_id": task_id,
                "session_id": request.session_id,
                "objective": request.objective,
                "available_data": request.available_data,
                "worker_results": [],
            },
            config={"recursion_limit": settings.graph_recursion_limit},
        )
    except Exception as exc:
        logger.exception("multi_agent_research_failed task_id=%s", task_id)
        if isinstance(exc, RateLimitError):
            detail = groq_rate_limit_detail(exc)
        elif isinstance(exc, APIError):
            detail = groq_error_detail(exc)
        else:
            detail = "The research orchestration service is temporarily unavailable"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc
    return ResearchResponse(
        task_id=task_id,
        content=result["final_answer"],
        worker_results=result.get("worker_results", []),
        critique=result["critique"],
        artifacts=result.get("artifacts", []),
    )


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
