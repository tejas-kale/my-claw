# tests/test_openrouter.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from assistant.llm.openrouter import OpenRouterProvider
from assistant.config import Settings


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="key",
        openrouter_model="default/model",
        openrouter_base_url="https://openrouter.ai/api/v1",
        telegram_bot_token="tok",
        telegram_owner_id="123",
        kagi_api_key="kagi",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_generate_uses_settings_model_by_default() -> None:
    settings = _make_settings(openrouter_model="default/model")
    provider = OpenRouterProvider(settings)
    captured = {}

    async def fake_post(url, headers, json, **kwargs):
        captured["model"] = json["model"]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop"}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client_cls.return_value = mock_client
        await provider.generate([{"role": "user", "content": "hi"}])

    assert captured["model"] == "default/model"


@pytest.mark.asyncio
async def test_generate_model_override_replaces_settings_model() -> None:
    settings = _make_settings(openrouter_model="default/model")
    provider = OpenRouterProvider(settings)
    captured = {}

    async def fake_post(url, headers, json, **kwargs):
        captured["model"] = json["model"]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop"}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client_cls.return_value = mock_client
        await provider.generate(
            [{"role": "user", "content": "hi"}],
            model="google/gemini-3.1-pro-preview-customtools",
        )

    assert captured["model"] == "google/gemini-3.1-pro-preview-customtools"
