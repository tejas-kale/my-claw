from __future__ import annotations

from assistant.config import Settings


def test_settings_meal_fields_have_defaults() -> None:
    s = Settings(
        openrouter_api_key="k",
        openrouter_model="m",
        telegram_bot_token="t",
        telegram_owner_id="1",
        kagi_api_key="k2",
    )
    assert s.meal_nutrition_memory_path == "~/.claw/meal_nutrition_memory.md"
    assert s.health_bigquery_dataset_id == "health"
    assert s.health_bigquery_table_id == "meals"
    assert s.meal_summary_timezone == "UTC"
