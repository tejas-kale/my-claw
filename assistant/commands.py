"""Command dispatcher for @-prefixed messages.

Commands bypass the LLM and invoke tools directly.
An unrecognised @command returns None, letting it fall through to the LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from assistant.models import Message
from assistant.tools.podcast_tool import PODCAST_TYPES

if TYPE_CHECKING:
    from assistant.db import Database
    from assistant.llm.base import LLMProvider
    from assistant.tools.ddg_search_tool import DdgSearchTool
    from assistant.tools.podcast_tool import PodcastTool
    from assistant.tools.price_tracker_tool import PriceTrackerTool
    from assistant.tools.read_url_tool import ReadUrlTool
    from assistant.tools.web_search_tool import KagiSearchTool

LOGGER = logging.getLogger(__name__)

_PODCAST_USAGE = f"Usage: @podcast <type> [url]\nValid types: {', '.join(PODCAST_TYPES)}"


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Split an @-prefixed message into (command, args).

    Returns:
        A (command, args) tuple where command is lowercased, or None if text
        is not a valid @command.
    """
    text = text.strip()
    if not text.startswith("@"):
        return None
    parts = text[1:].split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


class CommandDispatcher:
    """Routes @-prefixed messages to tool handlers, bypassing the LLM.

    Returns None for unrecognised commands so the caller can fall through.
    """

    def __init__(
        self,
        podcast_tool: PodcastTool | None = None,
        kagi_search_tool: KagiSearchTool | None = None,
        ddg_search_tool: DdgSearchTool | None = None,
        read_url_tool: ReadUrlTool | None = None,
        llm: LLMProvider | None = None,
        db: Database | None = None,
        price_tracker_tool: PriceTrackerTool | None = None,
    ) -> None:
        self._podcast_tool = podcast_tool
        self._kagi_search_tool = kagi_search_tool
        self._ddg_search_tool = ddg_search_tool
        self._read_url_tool = read_url_tool
        self._llm = llm
        self._db = db
        self._price_tracker_tool = price_tracker_tool

    async def dispatch(self, message: Message) -> str | None:
        """Dispatch a message to a command handler.

        Returns:
            A reply string for recognised commands, or None for unknown ones.
        """
        parsed = parse_command(message.text)
        if parsed is None:
            return None
        command, args = parsed
        LOGGER.info("Command dispatch: command=%r args=%r", command, args)
        if command == "podcast":
            return await self._handle_podcast(args, message)
        if command == "websearch":
            return await self._handle_websearch(args)
        if command == "clear":
            return self._handle_clear(message.group_id)
        if command == "trackprice":
            return await self._handle_trackprice(message)
        return None

    def _handle_clear(self, group_id: str) -> str:
        if self._db is None:
            return "History clearing is not available."
        self._db.clear_history(group_id)
        return "Conversation history cleared."

    async def _handle_websearch(self, args: list[str]) -> str:
        if not args:
            return "Usage: @websearch [ddg] <query>"

        if args[0].lower() == "ddg":
            tool = self._ddg_search_tool
            query = " ".join(args[1:]).strip()
            provider = "DDG"
        else:
            tool = self._kagi_search_tool
            query = " ".join(args).strip()
            provider = "Kagi"

        if not query:
            return "Usage: @websearch [ddg] <query>"
        if tool is None:
            return f"{provider} search is not configured."

        # Step 1: Generate sub-queries via LLM.
        sub_queries = [query]
        if self._llm:
            sub_queries = await self._generate_sub_queries(query)

        # Step 2: Run all sub-queries in parallel.
        search_results = await asyncio.gather(*[tool.run(query=q) for q in sub_queries])
        combined_results = "\n\n=====\n\n".join(search_results)

        if self._llm is None:
            return combined_results

        # Step 3: Rank results — LLM returns top URLs in order.
        ranked_urls = await self._rank_results(query, combined_results)

        # Step 4: Fetch up to 2 pages via Jina from ranked URLs.
        jina_pages: list[str] = []
        fetched_urls: list[str] = []
        if self._read_url_tool:
            for url in ranked_urls[:5]:
                if len(jina_pages) >= 2:
                    break
                try:
                    page = await self._read_url_tool.run(url=url)
                except Exception as exc:
                    LOGGER.warning("Jina error for %s: %s", url, exc)
                    continue
                if not page.startswith("Failed to read URL"):
                    jina_pages.append(page)
                    fetched_urls.append(url)
                else:
                    LOGGER.warning("Jina failed for %s", url)

        # Step 5: Synthesize final answer.
        context = combined_results
        if jina_pages:
            pages_section = "\n\n".join(
                f"--- Page {i + 1} ---\n{p}" for i, p in enumerate(jina_pages)
            )
            context += f"\n\n=== Full page content ===\n{pages_section}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search result summariser. "
                    "Answer using ONLY the information in the search results provided. "
                    "Do not add facts from your training data. "
                    "If the results do not contain enough information to answer fully, say so explicitly. "
                    "Reply in plain text only — no markdown, no bullet points, no headers."
                ),
            },
            {
                "role": "user",
                "content": f"Search results for '{query}':\n\n{context}\n\nSummarise the key information.",
            },
        ]
        response = await self._llm.generate(messages)
        answer = response.content

        reference_urls = fetched_urls or ranked_urls[:3]
        if reference_urls:
            refs = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(reference_urls))
            return f"{answer}\n\nReferences:\n{refs}"
        return answer

    async def _generate_sub_queries(self, query: str) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 1-5 precise sub-queries optimised for search engines. "
                    "Always prefer fewer queries — only add more if the original query is complex "
                    "and genuinely benefits from multiple angles. "
                    'Return JSON only: {"queries": ["query1", ...]}'
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            response = await self._llm.generate(
                messages, response_format={"type": "json_object"}
            )
            data = json.loads(response.content)
            queries = [str(q) for q in data.get("queries", []) if q][:5]
            return queries or [query]
        except Exception:
            LOGGER.warning("Sub-query generation failed, falling back to original query")
            return [query]

    async def _rank_results(self, query: str, combined_results: str) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ranking search results by relevance. "
                    "Return the top 5 most relevant URLs in descending order of relevance. "
                    'Return JSON only: {"urls": ["url1", ...]}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nSearch results:\n{combined_results}",
            },
        ]
        try:
            response = await self._llm.generate(
                messages, response_format={"type": "json_object"}
            )
            data = json.loads(response.content)
            return [str(u) for u in data.get("urls", []) if u][:5]
        except Exception:
            LOGGER.warning("Result ranking failed, falling back to URL order from results")
            return re.findall(r"^(https?://\S+)$", combined_results, re.MULTILINE)[:5]

    async def _handle_trackprice(self, message: Message) -> str:
        if not message.attachments:
            return "Please attach a receipt image or PDF."
        if self._price_tracker_tool is None:
            return "Price tracker is not configured."
        attachment = message.attachments[0]
        path = attachment.get("local_path", "")
        content_type = attachment.get("content_type", "image/jpeg")
        result = await self._price_tracker_tool.run(path, content_type)
        if "error" in result:
            return f"Price tracking failed: {result['error']}"
        return result.get("message", "Receipt saved.")

    async def _handle_podcast(self, args: list[str], message: Message) -> str:
        if self._podcast_tool is None:
            return "Podcast tool is not configured."
        if not args:
            return _PODCAST_USAGE

        podcast_type = args[0]
        if podcast_type not in PODCAST_TYPES:
            return f"Unknown podcast type '{podcast_type}'.\n{_PODCAST_USAGE}"

        # URL wins over attachment when both are present.
        source_url: str | None = next((a for a in args[1:] if a.startswith("http")), None)
        attachment_path: str | None = (
            None if source_url else (
                message.attachments[0]["local_path"] if message.attachments else None
            )
        )

        if not source_url and not attachment_path:
            return f"Attach a PDF or provide a URL.\n{_PODCAST_USAGE}"

        kwargs: dict[str, Any] = {
            "group_id": message.group_id,
            "is_group": message.is_group,
            "podcast_type": podcast_type,
        }
        if source_url:
            kwargs["source_url"] = source_url
        if attachment_path:
            kwargs["attachment_path"] = attachment_path

        result = await self._podcast_tool.run(**kwargs)
        if "error" in result:
            return f"Podcast failed: {result['error']}"
        return result.get("message", "Podcast generation started.")
