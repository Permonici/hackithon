from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "XDENT AI Support Assistant"
    environment: str = "local"
    api_prefix: str = "/api"

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=1536, validation_alias="OPENAI_EMBEDDING_DIMENSIONS")
    openai_chat_input_price_per_1m: float = Field(default=0.0, validation_alias="OPENAI_CHAT_INPUT_PRICE_PER_1M")
    openai_chat_output_price_per_1m: float = Field(default=0.0, validation_alias="OPENAI_CHAT_OUTPUT_PRICE_PER_1M")
    openai_embedding_price_per_1m: float = Field(default=0.0, validation_alias="OPENAI_EMBEDDING_PRICE_PER_1M")
    pricing_currency: str = Field(default="USD", validation_alias="PRICING_CURRENCY")
    pricing_reference_url: str = Field(
        default="https://platform.openai.com/docs/pricing",
        validation_alias="PRICING_REFERENCE_URL",
    )

    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_collection: str = Field(default="xdent_transcripts", validation_alias="QDRANT_COLLECTION")

    data_dir: Path = Field(
        default=Path("data"),
        validation_alias="DATA_DIR",
    )
    logs_dir: Path = Field(default=Path("logs"), validation_alias="LOGS_DIR")

    chunk_size: int = Field(default=950, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, validation_alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")
    strict_min_score: float = Field(default=0.35, validation_alias="STRICT_MIN_SCORE")
    lenient_min_score: float = Field(default=0.18, validation_alias="LENIENT_MIN_SCORE")
    broad_min_score: float = Field(default=0.10, validation_alias="BROAD_MIN_SCORE")
    qa_match_min_score: float = Field(default=0.55, validation_alias="QA_MATCH_MIN_SCORE")

    cors_origins: str = Field(default="http://localhost:5173,http://localhost:8080", validation_alias="CORS_ORIGINS")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def interactions_log_path(self) -> Path:
        return self.logs_dir / "interactions.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
