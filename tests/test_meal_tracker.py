from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.tools.meal_tracker import MealTracker


def _make_tracker(tmp_path: Path) -> MealTracker:
    config = MagicMock()
    config.meal_nutrition_memory_path = str(tmp_path / "memory.md")
    config.health_bigquery_dataset_id = "health"
    config.health_bigquery_table_id = "meals"
    config.bigquery_project_id = ""
    config.meal_summary_timezone = "UTC"
    llm = AsyncMock()
    kagi = AsyncMock()
    return MealTracker(config=config, llm=llm, kagi=kagi, bq_client=None)


@pytest.mark.asyncio
async def test_check_memory_returns_none_when_file_absent(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    result = await tracker._check_memory("dal makhani")
    assert result is None


@pytest.mark.asyncio
async def test_check_memory_finds_matching_entry(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\n"
        "Per 100gms: 151 kcal, 12g carbs.\n"
        "Source: Kagi search.\n"
    )
    result = await tracker._check_memory("dal makhani")
    assert result is not None
    assert "151 kcal" in result


@pytest.mark.asyncio
async def test_check_memory_is_case_insensitive(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\n"
        "Per 100gms: 151 kcal.\n"
        "Source: x.\n"
    )
    result = await tracker._check_memory("DAL  MAKHANI")
    assert result is not None


@pytest.mark.asyncio
async def test_check_memory_returns_none_for_missing_meal(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\n"
        "Per 100gms: 151 kcal.\n"
        "Source: x.\n"
    )
    result = await tracker._check_memory("chicken biryani")
    assert result is None


@pytest.mark.asyncio
async def test_save_memory_creates_file_if_absent(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients = {
        "kcal": 302.0, "carbs_g": 24.0, "fats_g": 14.0, "sugars_g": 4.0,
        "proteins_g": 16.0, "fiber_g": 6.0, "sodium_mg": 410.0,
        "cholesterol_mg": 22.0, "potassium_mg": 380.0, "iron_mg": 3.2,
        "calcium_mg": 85.0, "vitamin_c_mg": 12.0,
    }
    await tracker._save_memory("Dal Makhani", nutrients, "100gms", "Kagi search via nutritionix.")
    memory_file = tmp_path / "memory.md"
    assert memory_file.exists()
    content = memory_file.read_text()
    assert "## Dal Makhani" in content
    assert "Per 100gms" in content
    assert "Kagi search via nutritionix." in content


@pytest.mark.asyncio
async def test_save_memory_appends_to_existing_file(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text("## Samosa\nPer 1 unit: 130 kcal.\nSource: x.\n")
    nutrients = {"kcal": 151.0, "carbs_g": 12.0, "fats_g": 7.0, "sugars_g": 2.0,
                 "proteins_g": 8.0, "fiber_g": 3.0, "sodium_mg": 410.0,
                 "cholesterol_mg": 22.0, "potassium_mg": 380.0, "iron_mg": 3.2,
                 "calcium_mg": 85.0, "vitamin_c_mg": 12.0}
    await tracker._save_memory("Dal Makhani", nutrients, "100gms", "Source summary.")
    content = memory_file.read_text()
    assert "## Samosa" in content
    assert "## Dal Makhani" in content
