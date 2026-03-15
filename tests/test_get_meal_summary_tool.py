"""Tests for GetMealSummaryTool."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.tools.get_meal_summary_tool import GetMealSummaryTool


def _make_tool(meals: list | None = None) -> GetMealSummaryTool:
    tracker = MagicMock()
    tracker.get_summary = AsyncMock(return_value=meals or [])
    return GetMealSummaryTool(tracker=tracker)


def test_tool_has_correct_name_and_schema() -> None:
    tool = _make_tool()
    assert tool.name == "get_meal_summary"
    assert "date" in tool.parameters_schema["properties"]
    assert "date" in tool.parameters_schema["required"]


@pytest.mark.asyncio
async def test_run_passes_parsed_date_to_tracker() -> None:
    tool = _make_tool(
        meals=[
            {
                "meal_name": "Dal Makhani",
                "portion_amount": 200.0,
                "portion_unit": "gms",
                "kcal": 302.0,
                "proteins_g": 16.0,
                "carbs_g": 24.0,
                "fats_g": 14.0,
                "source": "search",
            },
        ]
    )
    result = await tool.run(date="2026-03-15")
    tool._tracker.get_summary.assert_called_once_with(datetime.date(2026, 3, 15))
    assert "Dal Makhani" in result
    assert "302" in result


@pytest.mark.asyncio
async def test_run_returns_no_meals_message_when_empty() -> None:
    tool = _make_tool(meals=[])
    result = await tool.run(date="2026-03-15")
    assert "no meals" in result.lower()
