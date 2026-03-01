"""Tests for ReadUrlTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.tools.read_url_tool import ReadUrlTool


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    return resp


@pytest.mark.asyncio
async def test_run_returns_formatted_markdown():
    payload = {
        "data": {
            "title": "Example Page",
            "content": "This is the page body.",
            "usage": {"tokens": 42},
        }
    }
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.read_url_tool.httpx.AsyncClient", return_value=mock_client):
        tool = ReadUrlTool(api_key="")
        result = await tool.run(url="https://example.com")

    assert "# Example Page" in result
    assert "Source: https://example.com" in result
    assert "Tokens: 42" in result
    assert "This is the page body." in result


@pytest.mark.asyncio
async def test_run_handles_non_200():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response({}, status_code=404))

    with patch("assistant.tools.read_url_tool.httpx.AsyncClient", return_value=mock_client):
        tool = ReadUrlTool(api_key="")
        result = await tool.run(url="https://missing.com/page")

    assert "404" in result
    assert "https://missing.com/page" in result


@pytest.mark.asyncio
async def test_run_omits_auth_header_without_api_key():
    payload = {"data": {"title": "T", "content": "C", "usage": {"tokens": 1}}}
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.read_url_tool.httpx.AsyncClient", return_value=mock_client):
        tool = ReadUrlTool(api_key="")
        await tool.run(url="https://example.com")

    headers = mock_client.get.call_args.kwargs["headers"]
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_run_sends_auth_header_with_api_key():
    payload = {"data": {"title": "T", "content": "C", "usage": {"tokens": 1}}}
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_response(payload))

    with patch("assistant.tools.read_url_tool.httpx.AsyncClient", return_value=mock_client):
        tool = ReadUrlTool(api_key="abc123")
        await tool.run(url="https://example.com")

    headers = mock_client.get.call_args.kwargs["headers"]
    assert headers.get("Authorization") == "Bearer abc123"
