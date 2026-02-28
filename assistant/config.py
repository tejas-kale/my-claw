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
    # Comma-separated E.164 numbers allowed to send commands (defaults to owner only).
    signal_allowed_senders: str = Field(default="", alias="SIGNAL_ALLOWED_SENDERS")
    signal_poll_interval_seconds: float = Field(default=2.0, alias="SIGNAL_POLL_INTERVAL_SECONDS")
    memory_window_messages: int = Field(default=20, alias="MEMORY_WINDOW_MESSAGES")
    memory_summary_trigger_messages: int = Field(default=40, alias="MEMORY_SUMMARY_TRIGGER_MESSAGES")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")
    memory_root: Path = Field(
        default=Path.home() / ".my-claw" / "memory",
        alias="MY_CLAW_MEMORY",
    )


def load_settings() -> Settings:
    """Load and validate settings."""

    return Settings()


def allowed_senders(settings: Settings) -> frozenset[str]:
    """Return the set of E.164 numbers permitted to send commands.

    Always includes the owner. Additional numbers can be added via the
    SIGNAL_ALLOWED_SENDERS env var as a comma-separated list.
    """
    extra = {n.strip() for n in settings.signal_allowed_senders.split(",") if n.strip()}
    return frozenset({settings.signal_owner_number} | extra)
