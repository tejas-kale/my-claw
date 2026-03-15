"""Command dispatcher for /-prefixed messages.

Commands bypass the LLM and invoke tools directly.
An unrecognised /command returns None, letting it fall through to the LLM.
"""

from __future__ import annotations

import asyncio
import datetime as _dt_mod
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assistant.models import Message
from assistant.tools.podcast_tool import PODCAST_TYPES

if TYPE_CHECKING:
    from assistant.db import Database
    from assistant.llm.base import LLMProvider
    from assistant.tools.citation_tracker_tool import CitationTrackerTool
    from assistant.tools.ddg_search_tool import DdgSearchTool
    from assistant.tools.magazine_tool import MagazineTool
    from assistant.tools.meal_tracker import MealTracker
    from assistant.tools.podcast_tool import PodcastTool
    from assistant.tools.price_tracker_tool import PriceTrackerTool
    from assistant.tools.read_url_tool import ReadUrlTool
    from assistant.tools.web_search_tool import KagiSearchTool

LOGGER = logging.getLogger(__name__)

_PODCAST_USAGE = f"Usage: /podcast <type> [url]\nValid types: {', '.join(PODCAST_TYPES)}"

TRANSIENT_COMMANDS: frozenset[str] = frozenset({"commands"})

_ALIASES: dict[str, str] = {
    "ws": "websearch",
    "tp": "trackprice",
    "pc": "podcast",
    "mg": "magazine",
    "ct": "cite",
    "cl": "clear",
    "cm": "commands",
    "tm": "trackmeal",
}

# Matches: 200gms, 200gm, 200g, 1.5cups, 1.5cup
_PORTION_RE = re.compile(r"^(\d+(?:\.\d+)?)(gms?|gm?|cups?)$", re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
_TIME_TOKEN_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _parse_portion(
    args: list[str],
) -> tuple[str, float | None, str | None, _dt_mod.datetime | None]:
    """Split args into (meal_name, portion_amount, portion_unit, logged_at).

    After the portion token, optional date (D.M / DD.MM) and/or time (H:MM / HH:MM)
    tokens are parsed. logged_at is a UTC-aware datetime or None if not provided.
    Unit normalization: g / gm / gms → "gms"; cup / cups → "cups".
    Plain number with no suffix → "units". Returns ("", None, None, None) if invalid.
    """
    if not args:
        return "", None, None, None

    # Work backwards from the end to extract date/time tokens
    remaining = list(args)
    date_part = None
    time_part = None

    while remaining:
        token = remaining[-1]
        dm = _DATE_TOKEN_RE.match(token)
        tm = _TIME_TOKEN_RE.match(token)
        if dm and date_part is None:
            day, month = int(dm.group(1)), int(dm.group(2))
            try:
                date_part = _dt_mod.date(_dt_mod.date.today().year, month, day)
                remaining.pop()
                continue
            except ValueError:
                pass
        if tm and time_part is None:
            hour, minute = int(tm.group(1)), int(tm.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                time_part = _dt_mod.time(hour, minute)
                remaining.pop()
                continue
        break  # not a date/time token, stop scanning

    if not remaining:
        return "", None, None, None

    last = remaining[-1]
    m = _PORTION_RE.match(last)
    if m:
        amount = float(m.group(1))
        raw_unit = m.group(2).lower()
        unit = "gms" if raw_unit in ("g", "gm", "gms") else "cups"
        meal_name = " ".join(remaining[:-1]).strip()
    else:
        try:
            amount = float(last)
            unit = "units"
            meal_name = " ".join(remaining[:-1]).strip()
        except ValueError:
            return "", None, None, None

    # Build logged_at from date/time parts. The entered time is stored as-is
    # with a UTC timezone marker; no local offset conversion is applied.
    logged_at = None
    if date_part is not None or time_part is not None:
        d = date_part or _dt_mod.date.today()
        t = time_part or _dt_mod.time(0, 0)
        logged_at = _dt_mod.datetime.combine(d, t, tzinfo=_dt_mod.timezone.utc)

    return meal_name, amount, unit, logged_at


def _format_summary_raw(meals: list[dict], date: _dt_mod.date) -> str:
    """Fallback summary format when no LLM is available."""
    lines = [f"Meals logged for {date.strftime('%d %b %Y')}:"]
    total = 0.0
    for m in meals:
        kcal = m.get("kcal") or 0.0
        total += kcal
        lines.append(
            f"• {m['meal_name']} {m['portion_amount']:.0f}{m['portion_unit']} — {kcal:.0f} kcal"
        )
    lines.append(f"Total: {total:.0f} kcal / 2300 goal")
    return "\n".join(lines)


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Split a /-prefixed message into (command, args), resolving aliases.

    Returns:
        A (command, args) tuple where command is the canonical lowercased name,
        or None if text is not a valid /command.
    """
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split()
    if not parts:
        return None
    cmd = parts[0].lower()
    return _ALIASES.get(cmd, cmd), parts[1:]


class CommandDispatcher:
    """Routes /-prefixed messages to tool handlers, bypassing the LLM.

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
        magazine_tool: MagazineTool | None = None,
        citation_tracker_tool: CitationTrackerTool | None = None,
        meal_tracker: MealTracker | None = None,
    ) -> None:
        self._podcast_tool = podcast_tool
        self._kagi_search_tool = kagi_search_tool
        self._ddg_search_tool = ddg_search_tool
        self._read_url_tool = read_url_tool
        self._llm = llm
        self._db = db
        self._price_tracker_tool = price_tracker_tool
        self._magazine_tool = magazine_tool
        self._citation_tracker_tool = citation_tracker_tool
        self._meal_tracker = meal_tracker
        self._pending_epub: dict[str, str] = {}  # group_id -> epub

    async def dispatch(self, message: Message) -> str | None:
        """Dispatch a message to a command handler.

        Returns:
            A reply string for recognised commands, or None for unknown ones.
        """
        # Plain chapter number after a chapter-listing response.
        stripped = message.text.strip()
        if stripped.isdigit() and message.group_id in self._pending_epub:
            epub = self._pending_epub.pop(message.group_id)
            if self._magazine_tool is None:
                return "Magazine tool is not configured."
            return await self._magazine_tool.start_generation(
                group_id=message.group_id,
                is_group=message.is_group,
                epub=epub,
                chapter=stripped,
            )

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
        if command == "magazine":
            return await self._handle_magazine(args, message)
        if command == "cite":
            return await self._handle_cite(args)
        if command == "commands":
            return self._handle_commands()
        if command == "trackmeal":
            return await self._handle_trackmeal(args)
        return None

    async def _handle_cite(self, args: list[str]) -> str:
        if self._citation_tracker_tool is None:
            return "Citation tracker is not configured."
        _USAGE = "Usage: /cite <status|list|add <url-or-doi>|run [id]|citations <id>>"
        if not args:
            return _USAGE
        sub = args[0].lower()
        if sub == "status":
            return await self._citation_tracker_tool.status()
        if sub == "list":
            return await self._citation_tracker_tool.list_papers()
        if sub == "add":
            if len(args) < 2:
                return "Usage: @cite add <url-or-doi>"
            return await self._citation_tracker_tool.add_paper(args[1])
        if sub == "run":
            paper_id = args[1] if len(args) > 1 else None
            return await self._citation_tracker_tool.run(paper_id)
        if sub == "citations":
            if len(args) < 2:
                return "Usage: @cite citations <id>"
            return await self._citation_tracker_tool.citations(args[1])
        return f"Unknown subcommand '{sub}'. {_USAGE}"

    def _handle_commands(self) -> str:
        data_path = Path(__file__).parent / "commands_help.json"
        entries = json.loads(data_path.read_text())
        lines = ["Available commands:"] + [
            f"{e['usage']}  — {e['description']}" for e in entries
        ]
        return "\n".join(lines)

    def _handle_clear(self, group_id: str) -> str:
        if self._db is None:
            return "History clearing is not available."
        return "Conversation context cleared."

    async def _handle_websearch(self, args: list[str]) -> str:
        if not args:
            return "Usage: /websearch [ddg] <query>"

        if args[0].lower() == "ddg":
            tool = self._ddg_search_tool
            query = " ".join(args[1:]).strip()
            provider = "DDG"
        else:
            tool = self._kagi_search_tool
            query = " ".join(args).strip()
            provider = "Kagi"

        if not query:
            return "Usage: /websearch [ddg] <query>"
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
        from datetime import date

        messages = [
            {
                "role": "system",
                "content": (
                    f"Today's date is {date.today().isoformat()}. "
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

    async def _handle_magazine(self, args: list[str], message: Message) -> str:
        if self._magazine_tool is None:
            return "Magazine tool is not configured."
        if not args:
            return "Usage: /magazine <epub> [chapter-number]"

        # If the last arg is a chapter number, split it off; otherwise all args
        # form the epub name/ID. This lets multi-word source names work correctly
        # (e.g. "/magazine The Blizzard" or "/magazine The Blizzard 3").
        if len(args) > 1 and args[-1].isdigit():
            epub = " ".join(args[:-1])
            chapter = args[-1]
            return await self._magazine_tool.start_generation(
                group_id=message.group_id,
                is_group=message.is_group,
                epub=epub,
                chapter=chapter,
            )

        epub = " ".join(args)
        chapters = await self._magazine_tool.list_chapters(epub)
        self._pending_epub[message.group_id] = epub
        return f"{chapters}\n\nReply with a chapter number to generate audio."

    async def _handle_trackmeal(self, args: list[str]) -> str:
        _USAGE = (
            "Usage: /tm <meal> <portion>\n"
            "Examples: /tm dal makhani 200gms  |  /tm chicken biryani 1.5cups  |  /tm samosa 2"
        )
        _SUMMARY_USAGE = (
            "Usage: /tm summary [month day]\n"
            "Examples: /tm summary  |  /tm summary mar 15"
        )
        if not args:
            return _USAGE

        # 1. Summary subcommand must be checked first.
        if args[0].lower() == "summary":
            return await self._handle_trackmeal_summary(args[1:], _SUMMARY_USAGE)

        # 2. Parse portion from last token.
        meal_name, portion_amount, portion_unit, logged_at = _parse_portion(args)
        if portion_amount is None:
            return _USAGE
        if not meal_name:
            return _USAGE

        if self._meal_tracker is None:
            return "Meal tracker is not configured."
        return await self._meal_tracker.track(
            meal_name=meal_name,
            portion_amount=portion_amount,
            portion_unit=portion_unit,
            logged_at=logged_at,
        )

    async def _handle_trackmeal_summary(self, args: list[str], usage: str) -> str:
        import calendar

        if not args:
            target_date = _dt_mod.date.today()
        else:
            month_abbrevs = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            month_str = args[0].lower()
            if month_str not in month_abbrevs:
                return usage
            if len(args) < 2:
                return usage
            try:
                day = int(args[1])
            except ValueError:
                return usage
            month = month_abbrevs[month_str]
            year = _dt_mod.date.today().year
            _, max_day = calendar.monthrange(year, month)
            if day < 1 or day > max_day:
                month_name = _dt_mod.date(year, month, 1).strftime("%B")
                return f"Invalid date: {month_name} doesn't have {day} days."
            target_date = _dt_mod.date(year, month, day)

        if self._meal_tracker is None:
            return "Meal tracker is not configured."

        meals = await self._meal_tracker.get_summary(target_date)
        if not meals:
            return f"No meals logged for {target_date.strftime('%d %b %Y')}."

        # Pass raw data to LLM for analysis and tips.
        if self._llm is None:
            return _format_summary_raw(meals, target_date)

        return await self._generate_summary_text(meals, target_date)

    async def _generate_summary_text(
        self, meals: list[dict], target_date: _dt_mod.date
    ) -> str:
        from assistant.tools.get_meal_summary_tool import _format_meals

        raw = _format_meals(meals)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a nutrition assistant. Analyze the meal log and provide a concise summary. "
                    "Flag: fiber <25g (low), sodium >2000mg (high), protein <50g (low). "
                    "End with exactly 3 concrete actionable tips for tomorrow. "
                    "Format for Telegram: bullet points, no markdown tables."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Meals logged for {target_date.strftime('%d %b %Y')} "
                    f"(goal: 2300 kcal):\n\n{raw}"
                ),
            },
        ]
        response = await self._llm.generate(messages)
        return response.content

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
