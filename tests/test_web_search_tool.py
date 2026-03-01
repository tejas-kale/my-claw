"""Tests for KagiSearchTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.tools.web_search_tool import KagiSearchTool


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _search_item(title: str, url: str, snippet: str = "", published: str = "") -> dict:
    return {"t": 0, "title": title, "url": url, "snippet": snippet, "published": published}


@pytest.mark.asyncio
async def test_run_returns_formatted_results():
    payload = {
        "data": [
            _search_item("Result One", "https://one.com", "First snippet", "2024-01-01"),
            _search_item("Result Two", "https://two.com", "Second snippet"),
        ],
        "meta": {"api_balance": "9.50"},
    }
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.web_search_tool.httpx.AsyncClient", return_value=mock_client):
        tool = KagiSearchTool(api_key="test-key")
        result = await tool.run(query="python async")

    assert "Result One" in result
    assert "https://one.com" in result
    assert "First snippet" in result
    assert "Published: 2024-01-01" in result
    assert "Result Two" in result
    assert "https://two.com" in result
    assert "Second snippet" in result
    assert "$9.50" in result


@pytest.mark.asyncio
async def test_run_skips_non_search_items():
    payload = {
        "data": [
            _search_item("Real Result", "https://real.com", "Good snippet"),
            {"t": 1, "title": "Related Search", "url": "https://related.com"},
        ],
        "meta": {"api_balance": "9.00"},
    }
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.web_search_tool.httpx.AsyncClient", return_value=mock_client):
        tool = KagiSearchTool(api_key="test-key")
        result = await tool.run(query="something")

    assert "Real Result" in result
    assert "Related Search" not in result


@pytest.mark.asyncio
async def test_run_returns_no_results_message():
    payload = {"data": [], "meta": {"api_balance": "9.00"}}
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.web_search_tool.httpx.AsyncClient", return_value=mock_client):
        tool = KagiSearchTool(api_key="test-key")
        result = await tool.run(query="nothing")

    assert result == "No results found."


@pytest.mark.asyncio
async def test_run_caps_limit_at_20():
    payload = {"data": [], "meta": {}}
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.web_search_tool.httpx.AsyncClient", return_value=mock_client):
        tool = KagiSearchTool(api_key="test-key")
        await tool.run(query="test", limit=99)

    call_kwargs = mock_client.get.call_args
    assert call_kwargs.kwargs["params"]["limit"] == 20


@pytest.mark.asyncio
async def test_run_handles_non_200():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response({}, status_code=401))

    with patch("assistant.tools.web_search_tool.httpx.AsyncClient", return_value=mock_client):
        tool = KagiSearchTool(api_key="bad-key")
        result = await tool.run(query="test")

    assert "401" in result
    assert "KAGI_API_KEY" in result


@pytest.mark.asyncio
async def test_run_strips_html_from_snippets():
    payload = {
        "data": [
            _search_item("Tagged Result", "https://tagged.com", "<b>Bold</b> and <em>italic</em> text"),
        ],
        "meta": {"api_balance": "9.00"},
    }
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.web_search_tool.httpx.AsyncClient", return_value=mock_client):
        tool = KagiSearchTool(api_key="test-key")
        result = await tool.run(query="tagged")

    assert "<b>" not in result
    assert "<em>" not in result
    assert "Bold" in result
    assert "italic" in result
