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

    llm_provider: Literal["groq"] = "groq"
    groq_api_key: str = ""
    groq_chat_model: str = "openai/gpt-oss-20b"
    embedding_provider: Literal["fastembed"] = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: str = ".cache/fastembed"
    embedding_dimensions: int = Field(default=384, gt=0)
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    reranker_cache_dir: str = ".cache/flashrank"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    groq_max_output_tokens: int = Field(default=1200, ge=128, le=4096)
    groq_requests_per_second: float = Field(default=0.2, gt=0.0, le=1.0)
    tavily_api_key: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "agentic_workspace_bge_v1"
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
    research_worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    research_job_lease_seconds: int = Field(default=60, ge=15, le=600)
    research_job_max_attempts: int = Field(default=3, ge=1, le=10)
    research_event_poll_seconds: float = Field(default=0.5, ge=0.1, le=10.0)

    chunk_size: int = Field(default=1000, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    embedding_batch_size: int = Field(default=64, ge=1, le=256)
    max_history_messages: int = Field(default=30, ge=2, le=200)
    agent_max_iterations: int = Field(default=3, ge=1, le=12)
    supervisor_max_workers: int = Field(default=2, ge=1, le=8)
    agent_max_revisions: int = Field(default=1, ge=0, le=3)
    max_dataset_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_dataset_rows: int = Field(default=10_000, ge=100, le=100_000)
    max_dataset_columns: int = Field(default=100, ge=2, le=500)
    max_repository_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_repository_files: int = Field(default=1000, ge=10, le=10_000)
    max_repository_file_bytes: int = Field(default=512 * 1024, ge=1024)
    max_repository_total_bytes: int = Field(default=20 * 1024 * 1024, ge=10_000)
    repository_search_max_matches: int = Field(default=50, ge=5, le=200)
    repository_explanation_context_chars: int = Field(default=14_000, ge=4000, le=60_000)
    repository_explanation_output_tokens: int = Field(default=1800, ge=512, le=4096)
    agent_tool_result_chars: int = Field(default=8000, ge=1000, le=16000)
    critic_results_chars: int = Field(default=3000, ge=1000, le=12000)
    critic_evidence_chars: int = Field(default=6000, ge=2000, le=16000)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def validate_runtime(self) -> None:
        """Fail fast when a required integration is not configured."""
        missing = []
        if not self.groq_configured:
            missing.append("GROQ_API_KEY")
        if not self.tavily_configured:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    @staticmethod
    def _credential_is_configured(value: str | None) -> bool:
        if not value:
            return False
        normalized = value.strip().lower()
        placeholders = ("replace-me", "replace-with", "your-")
        return not any(marker in normalized for marker in placeholders)

    @property
    def groq_configured(self) -> bool:
        return self._credential_is_configured(self.groq_api_key)

    @property
    def tavily_configured(self) -> bool:
        return self._credential_is_configured(self.tavily_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
