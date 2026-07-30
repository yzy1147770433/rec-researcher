"""Environment-backed application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["mock", "siliconflow", "none"]
RerankerProvider = Literal["mock", "siliconflow", "none"]
VectorStore = Literal["memory", "milvus", "none"]
RetrievalMode = Literal["snippet", "hybrid"]


def _is_configured(value: object) -> bool:
    """Check presence while keeping secret values inside ``SecretStr``."""

    if value is None:
        return False
    if isinstance(value, SecretStr):
        return bool(value.get_secret_value())
    return value != ""


class Settings(BaseSettings):
    """RecResearcher settings loaded from ``REC_`` variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="REC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: Literal["mock", "real"] = "mock"
    output_dir: Path = Path("outputs")
    max_concurrency: int = Field(default=3, ge=1)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_response_bytes: int = Field(default=5_000_000, ge=1)
    max_sources_per_query: int = Field(default=5, ge=1)
    max_total_sources: int = Field(default=30, ge=1)
    max_tasks: int = Field(default=5, ge=1)
    embedding_provider: EmbeddingProvider = "siliconflow"
    reranker_provider: RerankerProvider = "siliconflow"
    vector_store: VectorStore = "milvus"
    retrieval_mode: RetrievalMode = "snippet"

    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_api_key: SecretStr | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_api_key: SecretStr | None = None
    embedding_model: str | None = None
    reranker_model: str | None = None
    milvus_uri: str = "./data/rec_researcher.db"
    milvus_collection: str = "rec_passages"

    chunk_size: int = Field(default=1200, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)
    rrf_k: int = Field(default=60, ge=1)
    retrieval_top_k: int = Field(default=20, ge=1)
    rerank_top_k: int = Field(default=10, ge=1)
    mmr_top_k: int = Field(default=8, ge=1)
    mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    evidence_excerpt_length: int = Field(default=600, ge=1)

    def safe_summary(self) -> dict[str, bool]:
        """Return presence flags only, never configuration or secret values."""

        return {name: _is_configured(value) for name, value in self.__dict__.items()}

    def missing_real_configuration(
        self, *, search_provider: Literal["tavily"] = "tavily"
    ) -> list[str]:
        """List settings required by the selected real-provider composition."""

        required = {
            "llm_base_url": self.llm_base_url,
            "llm_api_key": self.llm_api_key,
            "llm_model": self.llm_model,
        }
        if search_provider == "tavily":
            required["tavily_api_key"] = self.tavily_api_key
        return [name for name, value in required.items() if not _is_configured(value)]
