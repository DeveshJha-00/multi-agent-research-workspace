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
    response_id: str
    content: str
    route: Literal["index", "general", "search"]
    sources: list[Source] = Field(default_factory=list)


class UploadResponse(BaseModel):
    status: bool
    document_id: str
    filename: str
    chunks_indexed: int
    parser_provider: str = "local"
    detected_language: str = "en-IN"
    script: str = "Latn"
    warnings: list[str] = Field(default_factory=list)


class IndexedDocumentResponse(BaseModel):
    document_id: str
    filename: str
    description: str = ""
    chunks_indexed: int
    parser_provider: str = "unknown"
    detected_language: str = "unknown"
    script: str = "unknown"


class DeleteResponse(BaseModel):
    status: bool
    document_id: str


class EvaluationRequest(BaseModel):
    response_id: str = Field(min_length=8, max_length=200)
    reference: str | None = Field(default=None, max_length=12_000)


EvaluationJobStatus = Literal["queued", "running", "completed", "failed"]


class EvaluationMetricResult(BaseModel):
    name: str
    status: Literal["completed", "failed", "not_applicable"]
    score: float | None = None
    reason: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


class EvaluationCreated(BaseModel):
    evaluation_id: str
    response_id: str
    status: EvaluationJobStatus
    reused: bool = False


class EvaluationStatusResponse(BaseModel):
    evaluation_id: str
    response_id: str
    status: EvaluationJobStatus
    progress: int = Field(ge=0, le=100)
    attempts: int
    metric_names: list[str]
    metrics: dict[str, EvaluationMetricResult] = Field(default_factory=dict)
    context_count: int = 0
    reference_supplied: bool = False
    duration_seconds: float | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


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


class RepositoryUploadResponse(BaseModel):
    repository_id: str
    filename: str
    description: str
    file_count: int
    total_bytes: int
    languages: dict[str, int] = Field(default_factory=dict)
    reused: bool = False
