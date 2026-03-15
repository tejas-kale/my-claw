"""Meal tracking: nutrition lookup via Gemini + Kagi, memory cache, BigQuery storage."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from assistant.config import Settings
    from assistant.llm.openrouter import OpenRouterProvider
    from assistant.tools.web_search_tool import KagiSearchTool

LOGGER = logging.getLogger(__name__)

_GEMINI_MODEL = "google/gemini-3.1-pro-preview-customtools"
_MEMORY_LOCK = asyncio.Lock()

_NUTRIENT_FIELDS = [
    "kcal", "carbs_g", "fats_g", "sugars_g", "proteins_g",
    "fiber_g", "sodium_mg", "cholesterol_mg", "potassium_mg",
    "iron_mg", "calcium_mg", "vitamin_c_mg",
]


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


class MealTracker:
    """Track meals: look up nutrition via Gemini+Kagi, cache in memory file, store in BigQuery."""

    def __init__(
        self,
        config: Settings,
        llm: OpenRouterProvider,
        kagi: KagiSearchTool,
        bq_client: Any | None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._kagi = kagi
        self._bq_client = bq_client

    def _memory_path(self) -> Path:
        return Path(self._config.meal_nutrition_memory_path).expanduser()

    async def _check_memory(self, meal_name: str) -> str | None:
        """Return the cached memory block for meal_name, or None if not found."""
        async with _MEMORY_LOCK:
            path = self._memory_path()
            if not path.exists():
                return None
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")

        target = _normalize(meal_name)
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("## ") and _normalize(line[3:]) == target:
                # Collect this block (up to 3 lines: heading, nutrients, source)
                block_lines = [line]
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].startswith("## "):
                        break
                    block_lines.append(lines[j])
                return "\n".join(block_lines)
            i += 1
        return None

    async def _save_memory(
        self,
        meal_name: str,
        nutrients: dict[str, float | None],
        base_unit: str,
        source_summary: str,
    ) -> None:
        """Append a new meal entry to the memory file."""
        parts = []
        for field in _NUTRIENT_FIELDS:
            val = nutrients.get(field)
            if val is None:
                continue
            unit = _nutrient_unit(field)
            parts.append(f"{val:.1f}{unit}")
        nutrient_line = f"Per {base_unit}: {', '.join(parts)}."
        entry = f"## {meal_name.title()}\n{nutrient_line}\nSource: {source_summary}\n"

        async with _MEMORY_LOCK:
            path = self._memory_path()
            await asyncio.to_thread(_append_memory_entry, path, entry)

    # Placeholder methods — implemented in subsequent tasks
    async def track(self, meal_name: str, portion_amount: float, portion_unit: str) -> str:
        raise NotImplementedError

    async def _call_gemini(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        memory_entry: str | None,
    ) -> dict[str, float | None]:
        """Call Gemini to get nutritional data. Uses cache hit path or search path."""
        if memory_entry is not None:
            return await self._gemini_scale(meal_name, portion_amount, portion_unit, memory_entry)
        return await self._gemini_search(meal_name, portion_amount, portion_unit)

    async def _gemini_scale(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        memory_entry: str,
    ) -> dict[str, float | None]:
        """Single-turn Gemini call to scale cached values to requested portion."""
        fields = ", ".join(_NUTRIENT_FIELDS)
        messages = [
            {
                "role": "system",
                "content": (
                    f"Scale the nutritional data below to the requested portion. "
                    f"Return only a JSON object with exactly these fields "
                    f"(null if a value cannot be computed): {fields}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Nutritional data: {memory_entry}\n"
                    f"Target portion: {portion_amount} {portion_unit} of {meal_name}."
                ),
            },
        ]
        response = await self._llm.generate(messages, model=_GEMINI_MODEL)
        return _parse_nutrients(response.content)

    async def _gemini_search(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
    ) -> dict[str, float | None]:
        """Multi-turn Gemini call with Kagi search tool to find nutritional data."""
        fields = ", ".join(_NUTRIENT_FIELDS)
        memory_context = await self._load_memory_context()
        system_prompt = (
            f"Today is {datetime.date.today().isoformat()}.\n"
            f"You are a nutrition assistant. Find accurate nutritional information for the meal below.\n"
            f"You may use the kagi_search tool to look up information. You may search at most 3 times.\n"
            f"Return your final answer as a JSON object with exactly these fields (null if unknown):\n"
            f"{fields}\n"
            f"All values must be for the exact portion the user specifies.\n\n"
            f"Previously cached meals (for reference, do not re-search these):\n{memory_context}"
        )
        kagi_tool_spec = [
            {
                "type": "function",
                "function": {
                    "name": "kagi_search",
                    "description": "Search the web for nutritional information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"What is the nutritional breakdown of {meal_name}, "
                    f"{portion_amount} {portion_unit}?"
                ),
            },
        ]
        search_count = 0
        while True:
            tools = kagi_tool_spec if search_count < 3 else None
            response = await self._llm.generate(messages, tools=tools, model=_GEMINI_MODEL)
            if not response.tool_calls:
                return _parse_nutrients(response.content)
            for tool_call in response.tool_calls:
                if tool_call.name == "kagi_search" and search_count < 3:
                    query = tool_call.arguments.get("query", "")
                    result = await self._kagi.run(query=query)
                    search_count += 1
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": "kagi_search",
                                "arguments": json.dumps(tool_call.arguments),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": str(result),
                    })

    async def _load_memory_context(self) -> str:
        """Load full memory file as context string."""
        path = self._memory_path()
        if not path.exists():
            return "No meals cached yet."
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    async def _insert_bigquery(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        nutrients: dict[str, float | None],
        source: str,
    ) -> None:
        raise NotImplementedError

    async def get_summary(self, date: datetime.date) -> list[dict[str, Any]]:
        raise NotImplementedError


def _parse_nutrients(content: str) -> dict[str, float | None]:
    """Extract the 12-field nutrient JSON from Gemini's text response."""
    text = content.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.warning("Gemini returned unparseable JSON: %r", content[:200])
        return {f: None for f in _NUTRIENT_FIELDS}
    return {f: (float(data[f]) if data.get(f) is not None else None) for f in _NUTRIENT_FIELDS}


def _append_memory_entry(path: Path, entry: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    separator = "\n" if existing and not existing.endswith("\n\n") else ""
    path.write_text(existing + separator + entry + "\n", encoding="utf-8")


def _nutrient_unit(field: str) -> str:
    if field == "kcal":
        return " kcal"
    if field.endswith("_g"):
        return "g"
    if field.endswith("_mg"):
        return "mg"
    return ""
