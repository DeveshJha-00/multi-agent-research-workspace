"""Public API response models."""

from typing import Literal

from pydantic import BaseModel, Field


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
