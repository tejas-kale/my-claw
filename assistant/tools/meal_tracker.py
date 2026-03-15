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
        raise NotImplementedError

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
