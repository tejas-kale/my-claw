"""Application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from tejas_config import load_config as _load_tejas_config
from tejas_config import get_secret


class Settings(BaseModel):
    """Application settings."""

    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    database_path: Path = Path("assistant.db")
    telegram_bot_token: str
    telegram_owner_id: str
    # Comma-separated Telegram user IDs allowed to send commands (defaults to owner only).
    telegram_allowed_sender_ids: str = ""
    telegram_poll_timeout: int = 30
    memory_window_messages: int = 20
    memory_summary_trigger_messages: int = 40
    request_timeout_seconds: float = 30.0
    kagi_api_key: str
    jina_api_key: str = ""
    bigquery_project_id: str = ""
    bigquery_dataset_id: str = "economics"
    bigquery_table_id: str = "german_shopping_receipts"
    gemini_api_key: str = ""
    meal_nutrition_memory_path: str = "~/.claw/meal_nutrition_memory.md"
    health_bigquery_dataset_id: str = "health"
    health_bigquery_table_id: str = "meals"
    meal_summary_timezone: str = "UTC"


def load_settings() -> Settings:
    """Load and validate settings."""

    cfg = _load_tejas_config("claw")
    return Settings(
        openrouter_api_key=cfg.openrouter.api_key,
        openrouter_model=cfg.openrouter.model,
        openrouter_base_url=cfg.openrouter.base_url,
        telegram_bot_token=get_secret("telegram_bot_token") or "",
        telegram_owner_id=cfg.claw.telegram_owner_id,
        telegram_allowed_sender_ids=cfg.claw.telegram_allowed_sender_ids,
        telegram_poll_timeout=cfg.claw.telegram_poll_timeout,
        database_path=cfg.claw.database_path,
        memory_window_messages=cfg.claw.memory_window_messages,
        memory_summary_trigger_messages=cfg.claw.memory_summary_trigger_messages,
        request_timeout_seconds=cfg.claw.request_timeout_seconds,
        kagi_api_key=get_secret("kagi_api_key") or "",
        jina_api_key=get_secret("jina_api_key") or "",
        gemini_api_key=get_secret("gemini_api_key") or "",
        bigquery_project_id=cfg.claw.bigquery_project_id,
        bigquery_dataset_id=cfg.claw.bigquery_dataset_id,
        bigquery_table_id=cfg.claw.bigquery_table_id,
        meal_nutrition_memory_path=getattr(cfg.claw, "meal_nutrition_memory_path", None) or "~/.claw/meal_nutrition_memory.md",
        health_bigquery_dataset_id=getattr(cfg.claw, "health_bigquery_dataset_id", None) or "health",
        health_bigquery_table_id=getattr(cfg.claw, "health_bigquery_table_id", None) or "meals",
        meal_summary_timezone=getattr(cfg.claw, "meal_summary_timezone", None) or "UTC",
    )


def allowed_telegram_senders(settings: Settings) -> frozenset[str]:
    """Return the set of Telegram user IDs permitted to send commands.

    Always includes the owner. Additional IDs can be added via the
    TELEGRAM_ALLOWED_SENDER_IDS env var as a comma-separated list.
    """
    extra = {n.strip() for n in settings.telegram_allowed_sender_ids.split(",") if n.strip()}
    return frozenset({settings.telegram_owner_id} | extra)
