from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from assistant.models import LLMResponse, LLMToolCall
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


def _make_llm_response(content: str = "", tool_calls: list | None = None) -> MagicMock:
    resp = MagicMock(spec=LLMResponse)
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


def _make_tool_call(name: str, args: dict) -> MagicMock:
    tc = MagicMock(spec=LLMToolCall)
    tc.name = name
    tc.arguments = args
    tc.call_id = "call-1"
    return tc


@pytest.mark.asyncio
async def test_call_gemini_cache_hit_no_kagi_call(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients_json = json.dumps({f: 1.0 for f in [
        "kcal","carbs_g","fats_g","sugars_g","proteins_g",
        "fiber_g","sodium_mg","cholesterol_mg","potassium_mg","iron_mg","calcium_mg","vitamin_c_mg"
    ]})
    tracker._llm.generate = AsyncMock(return_value=_make_llm_response(content=nutrients_json))
    memory_entry = "## Dal Makhani\nPer 100gms: 151 kcal.\nSource: x.\n"

    result = await tracker._call_gemini("dal makhani", 200.0, "gms", memory_entry)

    # LLM called once (scaling only), Kagi never called
    tracker._llm.generate.assert_called_once()
    tracker._kagi.run.assert_not_called()
    assert result["kcal"] == 1.0


@pytest.mark.asyncio
async def test_call_gemini_cache_miss_calls_kagi_and_returns_nutrients(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients_json = json.dumps({
        "kcal": 302.0, "carbs_g": 24.0, "fats_g": 14.0, "sugars_g": 4.0,
        "proteins_g": 16.0, "fiber_g": 6.0, "sodium_mg": 410.0,
        "cholesterol_mg": 22.0, "potassium_mg": 380.0, "iron_mg": 3.2,
        "calcium_mg": 85.0, "vitamin_c_mg": 12.0,
    })
    kagi_tool_call = _make_tool_call("kagi_search", {"query": "dal makhani nutrition"})
    tracker._llm.generate = AsyncMock(side_effect=[
        _make_llm_response(tool_calls=[kagi_tool_call]),   # turn 1: search
        _make_llm_response(content=nutrients_json),         # turn 2: answer
    ])
    tracker._kagi.run = AsyncMock(return_value="search result text")

    result = await tracker._call_gemini("dal makhani", 200.0, "gms", None)

    assert tracker._llm.generate.call_count == 2
    tracker._kagi.run.assert_called_once_with(query="dal makhani nutrition")
    assert result["kcal"] == 302.0
    assert result["proteins_g"] == 16.0


@pytest.mark.asyncio
async def test_call_gemini_stops_kagi_after_3_searches(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients_json = json.dumps({"kcal": 100.0, "carbs_g": None, "fats_g": None,
                                  "sugars_g": None, "proteins_g": None, "fiber_g": None,
                                  "sodium_mg": None, "cholesterol_mg": None,
                                  "potassium_mg": None, "iron_mg": None,
                                  "calcium_mg": None, "vitamin_c_mg": None})
    kagi_call = _make_tool_call("kagi_search", {"query": "q"})
    tracker._llm.generate = AsyncMock(side_effect=[
        _make_llm_response(tool_calls=[kagi_call]),
        _make_llm_response(tool_calls=[kagi_call]),
        _make_llm_response(tool_calls=[kagi_call]),
        _make_llm_response(content=nutrients_json),  # 4th turn: no tools offered
    ])
    tracker._kagi.run = AsyncMock(return_value="result")

    result = await tracker._call_gemini("samosa", 2.0, "units", None)

    assert tracker._kagi.run.call_count == 3
    assert tracker._llm.generate.call_count == 4
    assert result["kcal"] == 100.0


@pytest.mark.asyncio
async def test_call_gemini_returns_all_nulls_on_bad_json(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    tracker._llm.generate = AsyncMock(
        return_value=_make_llm_response(content="not valid json at all!!!")
    )
    result = await tracker._call_gemini("unknown dish", 1.0, "units", None)
    assert all(v is None for v in result.values())
