"""Application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings validated at startup."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(..., alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    database_path: Path = Field(default=Path("assistant.db"), alias="DATABASE_PATH")
    signal_cli_path: str = Field(default="signal-cli", alias="SIGNAL_CLI_PATH")
    signal_account: str = Field(..., alias="SIGNAL_ACCOUNT")
    signal_owner_number: str = Field(..., alias="SIGNAL_OWNER_NUMBER")
    signal_poll_interval_seconds: float = Field(default=2.0, alias="SIGNAL_POLL_INTERVAL_SECONDS")
    memory_window_messages: int = Field(default=20, alias="MEMORY_WINDOW_MESSAGES")
    memory_summary_trigger_messages: int = Field(default=40, alias="MEMORY_SUMMARY_TRIGGER_MESSAGES")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")


def load_settings() -> Settings:
    """Load and validate settings."""

    return Settings()
