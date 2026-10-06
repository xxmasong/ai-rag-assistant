"""Application settings, loaded from the environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql://rag:rag@localhost:5432/rag",
        description="Postgres connection string; the database needs the pgvector extension.",
    )

    anthropic_api_key: str = Field(default="", description="Key for the answering model.")
    answer_model: str = "claude-sonnet-5-5"
    embedding_model: str = "voyage-3"
    voyage_api_key: str = Field(default="")

    # Chunking. Tokens are approximated as characters / 4.
    chunk_size: int = 1600
    chunk_overlap: int = 200

    # Retrieval.
    top_k: int = 8
    rerank_keep: int = 4
    min_score: float = 0.25

    max_upload_bytes: int = 20 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
