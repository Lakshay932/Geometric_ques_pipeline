"""Environment-driven settings. Single place every component reads config from."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vlm_provider: str = "stub"
    llm_provider: str = "stub"
    openrouter_api_key: str | None = None
    openrouter_vlm_model: str | None = None
    openrouter_llm_model: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VRE_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
