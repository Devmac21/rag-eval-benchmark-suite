from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_EVAL_", env_file=".env", extra="ignore")

    runs_dir: Path = Path("./runs")
    reports_dir: Path = Path("./reports")
    api_key: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_api_version: str | None = None
    judge_model: str = "gpt-4o-mini"
    judge_timeout_s: float = 120.0
    judge_max_tokens: int = 450


@lru_cache
def get_settings() -> Settings:
    return Settings()
