"""Kagi web search tool."""

from __future__ import annotations

import re
from typing import Any

import httpx

from assistant.tools.base import Tool

KAGI_URL = "https://kagi.com/api/v0/search"
RESULT_TYPE_SEARCH = 0


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


class KagiSearchTool(Tool):
    """Search the web using the Kagi API."""

    name = "web_search"
    description = (
        "Search the web using Kagi. Returns titles, URLs, and text snippets "
        "for each result. Use this when you need current information, facts, "
        "documentation, or anything not in your training data."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5, max 20).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def run(self, **kwargs: Any) -> str:
        query = str(kwargs["query"]).strip()
        limit = min(int(kwargs.get("limit") or 5), 20)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                KAGI_URL,
                params={"q": query, "limit": limit},
                headers={"Authorization": f"Bot {self._api_key}"},
                timeout=15.0,
            )
            if resp.status_code != 200:
                return f"Search failed (HTTP {resp.status_code}): check KAGI_API_KEY and account status."
            data = resp.json()

        results = []
        for item in data.get("data", []):
            if item.get("t") != RESULT_TYPE_SEARCH:
                continue
            entry = f"**{item['title']}**\n{item['url']}"
            if published := item.get("published", ""):
                entry += f"\nPublished: {published}"
            if snippet := _strip_html(item.get("snippet", "")):
                entry += f"\n{snippet}"
            results.append(entry)

        if not results:
            return "No results found."

        balance = data.get("meta", {}).get("api_balance", "unknown")
        header = f"Search results for: {query} (API balance: ${balance})\n\n"
        return header + "\n\n---\n\n".join(results)
