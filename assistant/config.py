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
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_owner_id: str = Field(..., alias="TELEGRAM_OWNER_ID")
    # Comma-separated Telegram user IDs allowed to send commands (defaults to owner only).
    telegram_allowed_sender_ids: str = Field(default="", alias="TELEGRAM_ALLOWED_SENDER_IDS")
    telegram_poll_timeout: int = Field(default=30, alias="TELEGRAM_POLL_TIMEOUT")
    memory_window_messages: int = Field(default=20, alias="MEMORY_WINDOW_MESSAGES")
    memory_summary_trigger_messages: int = Field(default=40, alias="MEMORY_SUMMARY_TRIGGER_MESSAGES")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")
    kagi_api_key: str = Field(..., alias="KAGI_API_KEY")
    jina_api_key: str = Field(default="", alias="JINA_API_KEY")
    bigquery_project_id: str = Field(default="", alias="BIGQUERY_PROJECT_ID")
    bigquery_dataset_id: str = Field(default="economics", alias="BIGQUERY_DATASET_ID")
    bigquery_table_id: str = Field(default="german_shopping_receipts", alias="BIGQUERY_TABLE_ID")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")


def load_settings() -> Settings:
    """Load and validate settings."""

    return Settings()


def allowed_telegram_senders(settings: Settings) -> frozenset[str]:
    """Return the set of Telegram user IDs permitted to send commands.

    Always includes the owner. Additional IDs can be added via the
    TELEGRAM_ALLOWED_SENDER_IDS env var as a comma-separated list.
    """
    extra = {n.strip() for n in settings.telegram_allowed_sender_ids.split(",") if n.strip()}
    return frozenset({settings.telegram_owner_id} | extra)
