"""Public API response models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.agent import AgentResult, ArtifactRecord, Critique


class Source(BaseModel):
    source: str
    document_id: str | None = None
    page: int | None = None
    url: str | None = None


class QueryResponse(BaseModel):
    content: str
    route: Literal["index", "general", "search"]
    sources: list[Source] = Field(default_factory=list)


class UploadResponse(BaseModel):
    status: bool
    document_id: str
    filename: str
    chunks_indexed: int


class DeleteResponse(BaseModel):
    status: bool
    document_id: str


class ResearchRequest(BaseModel):
    objective: str = Field(min_length=10, max_length=2000)
    session_id: str = Field(min_length=8, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
    available_data: list[str] = Field(default_factory=list, max_length=20)


class ResearchResponse(BaseModel):
    task_id: str
    content: str
    worker_results: list[AgentResult]
    critique: Critique
    artifacts: list[ArtifactRecord]


ResearchJobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "failed",
]


class ResearchJobCreated(BaseModel):
    task_id: str
    status: ResearchJobStatus
    reused: bool = False


class ResearchJobStatusResponse(BaseModel):
    task_id: str
    objective: str
    status: ResearchJobStatus
    stage: str
    progress: int = Field(ge=0, le=100)
    attempts: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ResearchEventResponse(BaseModel):
    task_id: str
    sequence: int
    event: str
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    description: str
    columns: list[str]
    row_count: int
