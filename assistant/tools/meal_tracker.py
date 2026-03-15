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

    async def track(self, meal_name: str, portion_amount: float, portion_unit: str) -> str:
        """Full meal tracking flow: memory check → Gemini → BigQuery → formatted reply."""
        memory_entry = await self._check_memory(meal_name)
        source = "memory" if memory_entry is not None else "search"
        nutrients = await self._call_gemini(meal_name, portion_amount, portion_unit, memory_entry)

        if source == "search":
            source_summary = await self._fetch_source_summary(meal_name)
            # Only save to memory if at least some nutrients were found.
            # An all-null result would write garbage ("Per X: .") to the memory file.
            if any(v is not None for v in nutrients.values()):
                base_unit = _base_unit_for(portion_unit)
                await self._save_memory(
                    meal_name,
                    _scale_to_base(nutrients, portion_amount, portion_unit),
                    base_unit,
                    source_summary,
                )

        try:
            await self._insert_bigquery(
                meal_name=meal_name,
                portion_amount=portion_amount,
                portion_unit=portion_unit,
                nutrients=nutrients,
                source=source,
            )
        except Exception as exc:
            LOGGER.error("BigQuery insert failed for %r: %s", meal_name, exc)
            return (
                f"Logged {meal_name} to memory but could not write to the database.\n"
                + _format_nutrients(meal_name, portion_amount, portion_unit, nutrients)
            )

        all_null = all(v is None for v in nutrients.values())
        if all_null:
            return f"Logged {meal_name} ({portion_amount} {portion_unit}) but nutritional data could not be parsed."

        return _format_nutrients(meal_name, portion_amount, portion_unit, nutrients)

    async def _fetch_source_summary(self, meal_name: str) -> str:
        """Ask Gemini to summarise the sources it used in 1-2 sentences."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"In 1-2 sentences, summarize the source(s) you used to find "
                    f"nutritional information for {meal_name}. "
                    f"Do not include any nutritional values."
                ),
            }
        ]
        response = await self._llm.generate(messages, model=_GEMINI_MODEL)
        return response.content.strip()

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
        """Insert one meal row into BigQuery. No-op if bq_client is None."""
        if self._bq_client is None:
            return
        row: dict[str, Any] = {
            "meal_name": meal_name,
            "portion_amount": portion_amount,
            "portion_unit": portion_unit,
            "source": source,
            "logged_at": datetime.datetime.now(timezone.utc).isoformat(),
        }
        for field in _NUTRIENT_FIELDS:
            row[field] = nutrients.get(field)

        project = self._config.bigquery_project_id
        dataset = self._config.health_bigquery_dataset_id
        table = self._config.health_bigquery_table_id
        table_ref = f"{project}.{dataset}.{table}"

        await asyncio.to_thread(self._bq_insert, table_ref, [row])

    def _bq_insert(self, table_ref: str, rows: list[dict[str, Any]]) -> None:
        try:
            self._bq_client.get_table(table_ref)
        except Exception:
            schema = _build_bq_schema()
            from google.cloud.bigquery import Table  # type: ignore[import-untyped]
            table_obj = Table(table_ref, schema=schema)
            self._bq_client.create_table(table_obj, exists_ok=True)

        errors = self._bq_client.insert_rows_json(table_ref, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")

    async def get_summary(self, date: datetime.date) -> list[dict[str, Any]]:
        """Return all meal rows for the given date from BigQuery."""
        if self._bq_client is None:
            return []
        project = self._config.bigquery_project_id
        if not project:
            return []
        dataset = self._config.health_bigquery_dataset_id
        table = self._config.health_bigquery_table_id
        tz = self._config.meal_summary_timezone
        return await asyncio.to_thread(self._bq_query_summary, project, dataset, table, tz, date)

    def _bq_query_summary(
        self,
        project: str,
        dataset: str,
        table: str,
        tz: str,
        date: datetime.date,
    ) -> list[dict[str, Any]]:
        from google.cloud import bigquery  # type: ignore[import-untyped]

        query = f"""
            SELECT *
            FROM `{project}.{dataset}.{table}`
            WHERE DATE(logged_at, @timezone) = @date
            ORDER BY logged_at ASC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("timezone", "STRING", tz),
                bigquery.ScalarQueryParameter("date", "DATE", date.isoformat()),
            ]
        )
        result = self._bq_client.query(query, job_config=job_config).result()
        return [dict(row) for row in result]


def _build_bq_schema() -> list[Any]:
    from google.cloud import bigquery  # type: ignore[import-untyped]

    cols = [
        ("meal_name", "STRING", "REQUIRED"),
        ("portion_amount", "FLOAT64", "REQUIRED"),
        ("portion_unit", "STRING", "REQUIRED"),
        *[(f, "FLOAT64", "NULLABLE") for f in _NUTRIENT_FIELDS],
        ("source", "STRING", "REQUIRED"),
        ("logged_at", "TIMESTAMP", "REQUIRED"),
    ]
    return [bigquery.SchemaField(name, typ, mode=mode) for name, typ, mode in cols]


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


def _base_unit_for(portion_unit: str) -> str:
    if portion_unit == "gms":
        return "100gms"
    if portion_unit == "cups":
        return "1 cup"
    return "1 unit"


def _scale_to_base(
    nutrients: dict[str, float | None],
    portion_amount: float,
    portion_unit: str,
) -> dict[str, float | None]:
    """Scale portion-level nutrients back to base unit for memory storage."""
    if portion_amount == 0:
        return nutrients
    factor = 100.0 / portion_amount if portion_unit == "gms" else 1.0 / portion_amount
    return {
        f: (round(v * factor, 1) if v is not None else None)
        for f, v in nutrients.items()
    }


def _format_nutrients(
    meal_name: str,
    portion_amount: float,
    portion_unit: str,
    nutrients: dict[str, float | None],
) -> str:
    """Format nutritional data as a readable Telegram message."""
    kcal = nutrients.get("kcal")
    kcal_str = f"{kcal:.0f} kcal" if kcal is not None else "? kcal"
    lines = [f"*{meal_name.title()}* — {portion_amount:.0f} {portion_unit} — {kcal_str}"]
    field_labels = {
        "carbs_g": "Carbs", "proteins_g": "Protein", "fats_g": "Fat",
        "sugars_g": "Sugars", "fiber_g": "Fiber", "sodium_mg": "Sodium",
        "cholesterol_mg": "Cholesterol", "potassium_mg": "Potassium",
        "iron_mg": "Iron", "calcium_mg": "Calcium", "vitamin_c_mg": "Vitamin C",
    }
    for field, label in field_labels.items():
        val = nutrients.get(field)
        if val is None:
            continue
        unit = _nutrient_unit(field).strip()
        lines.append(f"• {label}: {val:.1f} {unit}")
    return "\n".join(lines)
