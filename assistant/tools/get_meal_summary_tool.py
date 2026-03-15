"""Tool that fetches meal data for a given date — used by the LLM for daily summary generation."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from assistant.tools.base import Tool

if TYPE_CHECKING:
    from assistant.tools.meal_tracker import MealTracker


class GetMealSummaryTool(Tool):
    """Fetch all meals logged for a specific date for nutrition summary generation."""

    name = "get_meal_summary"
    description = (
        "Fetch all meals logged for a specific date. "
        "Returns meal names, portions, and nutritional data. "
        "Use this to generate the daily nutrition summary."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "The date to fetch meals for, in YYYY-MM-DD format.",
            }
        },
        "required": ["date"],
        "additionalProperties": False,
    }

    def __init__(self, tracker: MealTracker) -> None:
        self._tracker = tracker

    async def run(self, **kwargs: Any) -> str:
        date_str = str(kwargs["date"])
        try:
            date = datetime.date.fromisoformat(date_str)
        except ValueError:
            return f"Invalid date format: {date_str!r}. Use YYYY-MM-DD."
        meals = await self._tracker.get_summary(date)
        if not meals:
            return f"No meals logged for {date_str}."
        return _format_meals(meals)


def _format_meals(meals: list[dict[str, Any]]) -> str:
    lines = []
    total_kcal = 0.0
    for m in meals:
        kcal = m.get("kcal") or 0.0
        total_kcal += kcal
        name = m.get("meal_name", "?")
        amt = m.get("portion_amount", 0)
        unit = m.get("portion_unit", "")
        lines.append(f"• {name} — {amt:.0f} {unit} — {kcal:.0f} kcal")
    lines.append(f"\nTotal: {total_kcal:.0f} kcal")
    # Include key nutrient totals
    for field, label in [
        ("proteins_g", "Protein"),
        ("carbs_g", "Carbs"),
        ("fats_g", "Fat"),
        ("fiber_g", "Fiber"),
        ("sodium_mg", "Sodium"),
    ]:
        total = sum((m.get(field) or 0.0) for m in meals)
        unit = "g" if field.endswith("_g") else "mg"
        lines.append(f"{label}: {total:.1f}{unit}")
    return "\n".join(lines)
