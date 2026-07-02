"""Structured reranking models."""

from pydantic import BaseModel, Field


class RerankItem(BaseModel):
    document_index: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class RerankResult(BaseModel):
    rankings: list[RerankItem]


class Grade(BaseModel):
    """Retained compatibility model for external imports."""

    binary_score: str
