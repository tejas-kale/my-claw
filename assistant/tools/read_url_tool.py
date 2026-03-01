"""Jina Reader URL fetching tool."""

from __future__ import annotations

from typing import Any

import httpx

from assistant.tools.base import Tool

JINA_BASE_URL = "https://r.jina.ai"


class ReadUrlTool(Tool):
    """Fetch a web page and return it as clean markdown."""

    name = "read_url"
    description = (
        "Fetch a web page and convert it to clean markdown for reading. "
        "Use this after web_search when you need the full content of a "
        "specific page — not for every result, only the most relevant ones. "
        "Typical latency is 5–10 seconds."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The full URL to read."},
            "max_tokens": {
                "type": "integer",
                "description": "Max tokens of content to return (default 5000).",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    async def run(self, **kwargs: Any) -> str:
        url = str(kwargs["url"]).strip()
        max_tokens = int(kwargs.get("max_tokens") or 5000)

        headers = {
            "Accept": "application/json",
            "X-Retain-Images": "none",
            "X-Remove-Selector": "nav, footer, .sidebar, .ads",
            "X-Token-Budget": str(max_tokens),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{JINA_BASE_URL}/{url}",
                headers=headers,
                timeout=20.0,
            )
            if resp.status_code != 200:
                return f"Failed to read URL (HTTP {resp.status_code}): {url}"
            data = resp.json()

        content = data.get("data", {})
        title = content.get("title", "Untitled")
        body = content.get("content", "")
        tokens_used = content.get("usage", {}).get("tokens", "?")

        return f"# {title}\nSource: {url}\nTokens: {tokens_used}\n\n{body}"
