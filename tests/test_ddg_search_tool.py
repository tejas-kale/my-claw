"""Tests for DdgSearchTool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from assistant.tools.ddg_search_tool import DdgSearchTool

# Patch path must match the import in the module under test
_DDGS_PATH = "assistant.tools.ddg_search_tool.DDGS"


def _ddg_results(*items: tuple[str, str, str]) -> list[dict]:
    return [{"title": t, "href": h, "body": b} for t, h, b in items]


@pytest.mark.asyncio
async def test_run_returns_formatted_results():
    results = _ddg_results(
        ("Result One", "https://one.com", "First body text"),
        ("Result Two", "https://two.com", "Second body text"),
    )
    mock_ddgs = MagicMock()
    mock_ddgs.text = MagicMock(return_value=results)

    with patch(_DDGS_PATH, return_value=mock_ddgs):
        tool = DdgSearchTool()
        result = await tool.run(query="test query")

    assert "Result One" in result
    assert "https://one.com" in result
    assert "First body text" in result
    assert "Result Two" in result
    assert "https://two.com" in result
    assert "Second body text" in result
    assert "test query" in result


@pytest.mark.asyncio
async def test_run_returns_no_results_message():
    mock_ddgs = MagicMock()
    mock_ddgs.text = MagicMock(return_value=[])

    with patch(_DDGS_PATH, return_value=mock_ddgs):
        tool = DdgSearchTool()
        result = await tool.run(query="nothing")

    assert result == "No results found."


@pytest.mark.asyncio
async def test_run_caps_limit_at_20():
    mock_ddgs = MagicMock()
    mock_ddgs.text = MagicMock(return_value=[])

    with patch(_DDGS_PATH, return_value=mock_ddgs):
        tool = DdgSearchTool()
        await tool.run(query="test", limit=99)

    mock_ddgs.text.assert_called_once_with("test", max_results=20, backend="duckduckgo")
