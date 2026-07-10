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
    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: str = "/models/fastembed"
    embedding_dimensions: int = Field(default=384, gt=0)
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    reranker_cache_dir: str = "/models/flashrank"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    groq_max_output_tokens: int = Field(default=1200, ge=128, le=4096)
    groq_requests_per_second: float = Field(default=0.2, gt=0.0, le=1.0)
    tavily_api_key: str = ""
    sarvam_api_key: str = ""
    document_parser_provider: Literal["auto", "local", "sarvam"] = "auto"
    enable_multilingual_docs: bool = True
    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_document_language: str = "auto"
    sarvam_document_output_format: Literal["md", "html", "json"] = "md"
    sarvam_max_pages_per_job: int = Field(default=10, ge=1, le=10)
    sarvam_document_download_max_bytes: int = Field(
        default=8 * 1024 * 1024,
        ge=256 * 1024,
        le=50 * 1024 * 1024,
    )
    sarvam_document_max_output_chars: int = Field(default=60_000, ge=5_000, le=500_000)
    sarvam_job_poll_seconds: float = Field(default=2.0, ge=0.5, le=30.0)
    sarvam_job_timeout_seconds: int = Field(default=60, ge=30, le=1200)
    enable_voice_features: bool = True
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: Literal["transcribe", "translate", "verbatim", "translit", "codemix"] = "transcribe"
    sarvam_stt_language: str = "unknown"
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_default_speaker: str = "auto"
    sarvam_tts_default_pace: float = Field(default=1.0, ge=0.5, le=2.0)
    sarvam_tts_audio_format: Literal["wav", "mp3", "linear16", "mulaw", "alaw", "opus", "flac", "aac"] = "wav"
    sarvam_tts_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    sarvam_tts_max_chars: int = Field(default=2500, ge=100, le=2500)
    sarvam_tts_long_answer_char_limit: int = Field(default=700, ge=100, le=2500)
    default_ui_language: str = "en-IN"
    default_answer_language: str = "auto"
    ragas_enabled: bool = True
    ragas_do_not_track: bool = True
    ragas_judge_model: str | None = None
    ragas_judge_base_url: str = "https://api.groq.com/openai/v1"
    ragas_max_contexts: int = Field(default=3, ge=1, le=10)
    ragas_max_context_chars: int = Field(default=12_000, ge=1000, le=60_000)
    evaluation_worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    evaluation_job_lease_seconds: int = Field(default=600, ge=30, le=3600)
    evaluation_job_max_attempts: int = Field(default=3, ge=1, le=10)
    evaluation_metric_delay_seconds: float = Field(default=1.0, ge=0.0, le=30.0)

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
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    max_history_messages: int = Field(default=30, ge=2, le=200)
    agent_max_iterations: int = Field(default=3, ge=1, le=12)
    supervisor_max_workers: int = Field(default=2, ge=1, le=8)
    agent_max_revisions: int = Field(default=1, ge=0, le=3)
    max_dataset_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_dataset_rows: int = Field(default=10_000, ge=100, le=100_000)
    max_dataset_columns: int = Field(default=100, ge=2, le=500)
    max_repository_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)
    max_repository_files: int = Field(default=1000, ge=10, le=10_000)
    max_repository_file_bytes: int = Field(default=512 * 1024, ge=1024)
    max_repository_total_bytes: int = Field(default=20 * 1024 * 1024, ge=10_000)
    repository_search_max_matches: int = Field(default=50, ge=5, le=200)
    repository_explanation_context_chars: int = Field(default=14_000, ge=4000, le=60_000)
    repository_explanation_output_tokens: int = Field(default=1800, ge=512, le=4096)
    agent_tool_result_chars: int = Field(default=8000, ge=1000, le=16000)
    deliverable_results_chars: int = Field(default=3000, ge=1000, le=12000)
    deliverable_evidence_chars: int = Field(default=6000, ge=2000, le=16000)

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

    @property
    def sarvam_configured(self) -> bool:
        return self._credential_is_configured(self.sarvam_api_key)

    @property
    def effective_ragas_judge_model(self) -> str:
        return self.ragas_judge_model or self.groq_chat_model


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
