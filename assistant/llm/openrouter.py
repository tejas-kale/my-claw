"""OpenRouter implementation of LLMProvider."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from assistant.config import Settings
from assistant.llm.base import LLMProvider
from assistant.models import LLMResponse, LLMToolCall

_LOGGER = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = [5, 15, 45]


class OpenRouterProvider(LLMProvider):
    """LLM provider using OpenRouter's OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._settings.openrouter_model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        timeout = httpx.Timeout(self._settings.request_timeout_seconds)
        async with httpx.AsyncClient(base_url=self._settings.openrouter_base_url, timeout=timeout) as client:
            for attempt in range(_MAX_RETRIES + 1):
                response = await client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code == 429 and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF_SECONDS[attempt]
                    _LOGGER.warning(
                        "OpenRouter rate limited (429), retrying in %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                break
            data = response.json()

        choice = data["choices"][0]["message"]
        finish_reason = data["choices"][0].get("finish_reason")
        content = choice.get("content") or ""
        _LOGGER.info(
            "LLM response: finish_reason=%r content=%r tool_calls=%r",
            finish_reason,
            content[:200] if content else "",
            choice.get("tool_calls"),
        )

        parsed_tool_calls: list[LLMToolCall] = []
        for tool_call in choice.get("tool_calls", []):
            function_data = tool_call.get("function", {})
            parsed_tool_calls.append(
                LLMToolCall(
                    name=function_data.get("name", ""),
                    arguments=_safe_json_loads(function_data.get("arguments", "{}")),
                    call_id=tool_call.get("id"),
                )
            )

        return LLMResponse(content=content, tool_calls=parsed_tool_calls, raw=data)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
