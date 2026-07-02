"""Public API response models."""

from typing import Literal

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
    objective: str = Field(min_length=10, max_length=8000)
    session_id: str = Field(min_length=8, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
    available_data: list[str] = Field(default_factory=list, max_length=20)


class ResearchResponse(BaseModel):
    task_id: str
    content: str
    worker_results: list[AgentResult]
    critique: Critique
    artifacts: list[ArtifactRecord]
