"""DuckDuckGo search tool."""

from __future__ import annotations

import asyncio
from typing import Any

from ddgs import DDGS

from assistant.tools.base import Tool


class DdgSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key required)."""

    name = "ddg_search"
    description = "Search the web using DuckDuckGo."
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

    async def run(self, **kwargs: Any) -> str:
        query = str(kwargs["query"]).strip()
        limit = min(int(kwargs.get("limit") or 5), 20)

        results = await asyncio.to_thread(
            lambda: DDGS().text(query, max_results=limit, backend="duckduckgo")
        )

        if not results:
            return "No results found."

        entries = [
            f"**{r['title']}**\n{r['href']}\n{r['body']}"
            for r in results
        ]
        header = f"DDG results for: {query}\n\n"
        return header + "\n\n---\n\n".join(entries)
