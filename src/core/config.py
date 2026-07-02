"""Validated application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secrets are supplied only through environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Adaptive RAG API"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    tavily_api_key: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "adaptive_rag_documents"
    qdrant_timeout_seconds: int = Field(default=20, gt=0)

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "adaptive_rag"
    mongodb_timeout_ms: int = Field(default=5000, gt=0)

    retrieval_top_k: int = Field(default=12, ge=2, le=50)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.2, ge=-1.0, le=1.0)
    rerank_relevance_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_retrieval_retries: int = Field(default=1, ge=0, le=3)
    graph_recursion_limit: int = Field(default=15, ge=5, le=50)

    chunk_size: int = Field(default=1000, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    embedding_batch_size: int = Field(default=64, ge=1, le=256)
    max_history_messages: int = Field(default=30, ge=2, le=200)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def validate_runtime(self) -> None:
        """Fail fast when a required integration is not configured."""
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
