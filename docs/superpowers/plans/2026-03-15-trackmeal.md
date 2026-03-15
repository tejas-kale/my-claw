# /trackmeal Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/tm` (`/trackmeal`) command that logs meals with Gemini-sourced nutritional data (via Kagi search), caches results in a memory file, stores in BigQuery, and sends a daily 9PM summary.

**Architecture:** A `MealTracker` class handles the lookup/cache/BQ pipeline; it's wired into `CommandDispatcher` for `/tm` commands and a `GetMealSummaryTool` exposes its data to the LLM for the scheduled daily summary. The existing `OpenRouterProvider` gains an optional `model` override parameter so Gemini can be called without a second provider instance.

**Tech Stack:** Python 3.11+, `google-cloud-bigquery`, `zoneinfo` (stdlib), `asyncio.Lock`, OpenRouter API (via existing `OpenRouterProvider`), existing `KagiSearchTool`.

---

## Chunk 1: Foundations — LLM override, config, DB

### Task 1: Add model override to `OpenRouterProvider.generate()` and `LLMProvider` ABC

**Files:**
- Modify: `assistant/llm/base.py` — add `model` param to ABC
- Modify: `assistant/llm/openrouter.py:28-37`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_runtime.py` (or create `tests/test_openrouter.py`):

```python
# tests/test_openrouter.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from assistant.llm.openrouter import OpenRouterProvider
from assistant.config import Settings


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="key",
        openrouter_model="default/model",
        openrouter_base_url="https://openrouter.ai/api/v1",
        telegram_bot_token="tok",
        telegram_owner_id="123",
        kagi_api_key="kagi",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_generate_uses_settings_model_by_default() -> None:
    settings = _make_settings(openrouter_model="default/model")
    provider = OpenRouterProvider(settings)
    captured = {}

    async def fake_post(url, headers, json, **kwargs):
        captured["model"] = json["model"]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop"}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client_cls.return_value = mock_client
        await provider.generate([{"role": "user", "content": "hi"}])

    assert captured["model"] == "default/model"


@pytest.mark.asyncio
async def test_generate_model_override_replaces_settings_model() -> None:
    settings = _make_settings(openrouter_model="default/model")
    provider = OpenRouterProvider(settings)
    captured = {}

    async def fake_post(url, headers, json, **kwargs):
        captured["model"] = json["model"]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop"}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client_cls.return_value = mock_client
        await provider.generate(
            [{"role": "user", "content": "hi"}],
            model="google/gemini-3.1-pro-preview-customtools",
        )

    assert captured["model"] == "google/gemini-3.1-pro-preview-customtools"
```

- [ ] **Step 2: Run to confirm partial failure**

```bash
uv run pytest tests/test_openrouter.py -v
```
Expected: `test_generate_uses_settings_model_by_default` PASSES (it tests existing behavior); `test_generate_model_override_replaces_settings_model` FAILS with `TypeError: generate() got an unexpected keyword argument 'model'`. Only the second test is the red test.

- [ ] **Step 3: Add `model` parameter to both the ABC and the concrete implementation**

First, update `assistant/llm/base.py` to add `model` to the abstract signature:
```python
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Generate a model response."""
```

Then update `assistant/llm/openrouter.py`:
```python
async def generate(
    self,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> LLMResponse:
    payload: dict[str, Any] = {
        "model": self._settings.openrouter_model,
```
To:
```python
async def generate(
    self,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    model: str | None = None,
) -> LLMResponse:
    payload: dict[str, Any] = {
        "model": model or self._settings.openrouter_model,
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/test_openrouter.py -v
```
Expected: 2 tests PASS

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
uv run pytest -q
```
Expected: all existing tests pass

- [ ] **Step 6: Commit**

Also update `FakeLLM` in `tests/test_commands.py` if it exists (search for `async def generate` in that file) — add `model: str | None = None` to its signature to match the updated ABC, so it remains a valid subtype.

```bash
git add assistant/llm/base.py assistant/llm/openrouter.py tests/test_openrouter.py
git commit -m "feat: add optional model override to OpenRouterProvider.generate()"
```

---

### Task 2: Add meal-tracking Settings fields

**Files:**
- Modify: `assistant/config.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_config.py (create if it doesn't exist)
from __future__ import annotations

from assistant.config import Settings


def test_settings_meal_fields_have_defaults() -> None:
    s = Settings(
        openrouter_api_key="k",
        openrouter_model="m",
        telegram_bot_token="t",
        telegram_owner_id="1",
        kagi_api_key="k2",
    )
    assert s.meal_nutrition_memory_path == "~/.claw/meal_nutrition_memory.md"
    assert s.health_bigquery_dataset_id == "health"
    assert s.health_bigquery_table_id == "meals"
    assert s.meal_summary_timezone == "UTC"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_config.py::test_settings_meal_fields_have_defaults -v
```
Expected: `AttributeError` or `ValidationError`

- [ ] **Step 3: Add the four new fields to `Settings`**

In `assistant/config.py`, add after the existing `gemini_api_key` field:
```python
    meal_nutrition_memory_path: str = "~/.claw/meal_nutrition_memory.md"
    health_bigquery_dataset_id: str = "health"
    health_bigquery_table_id: str = "meals"
    meal_summary_timezone: str = "UTC"
```

Also add to `load_settings()`. These are new optional YAML keys that users may not have yet. Use a `try/except AttributeError` block to safely fall back to the Pydantic defaults when the keys are absent from the YAML:

```python
        meal_nutrition_memory_path=getattr(cfg.claw, "meal_nutrition_memory_path", None) or "~/.claw/meal_nutrition_memory.md",
        health_bigquery_dataset_id=getattr(cfg.claw, "health_bigquery_dataset_id", None) or "health",
        health_bigquery_table_id=getattr(cfg.claw, "health_bigquery_table_id", None) or "meals",
        meal_summary_timezone=getattr(cfg.claw, "meal_summary_timezone", None) or "UTC",
```

`getattr(..., None)` returns `None` if the attribute is missing (regardless of whether `tejas_config` raises `AttributeError` or returns `None`); the `or` then falls back to the default. This is safe for all `tejas_config` behaviours.

- [ ] **Step 4: Run test — expect pass**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/config.py tests/test_config.py
git commit -m "feat: add meal tracking Settings fields with defaults"
```

---

### Task 3: Add `get_scheduled_meal_summary_task()` to `Database`

**Files:**
- Modify: `assistant/db.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_db.py (create if needed, or add to existing)
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from assistant.db import Database


def _make_db() -> Database:
    tmp = tempfile.mktemp(suffix=".db")
    db = Database(Path(tmp))
    db.initialize()
    return db


def test_get_scheduled_meal_summary_task_returns_none_when_empty() -> None:
    db = _make_db()
    now = datetime.now(timezone.utc)
    result = db.get_scheduled_meal_summary_task(
        window_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
        window_end=now.replace(hour=23, minute=59, second=59, microsecond=0),
    )
    assert result is None


def test_get_scheduled_meal_summary_task_returns_pending_task_in_window() -> None:
    db = _make_db()
    now = datetime.now(timezone.utc)
    nine_pm = now.replace(hour=21, minute=0, second=0, microsecond=0)
    db.upsert_group("owner-123")
    db.create_scheduled_task(
        group_id="owner-123",
        prompt="__meal_summary__ Generate today's meal nutrition summary.",
        run_at=nine_pm,
    )
    result = db.get_scheduled_meal_summary_task(
        window_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
        window_end=now.replace(hour=23, minute=59, second=59, microsecond=0),
    )
    assert result is not None
    assert "__meal_summary__" in result["prompt"]


def test_get_scheduled_meal_summary_task_ignores_completed_tasks() -> None:
    db = _make_db()
    now = datetime.now(timezone.utc)
    nine_pm = now.replace(hour=21, minute=0, second=0, microsecond=0)
    db.upsert_group("owner-123")
    task_id = db.create_scheduled_task(
        group_id="owner-123",
        prompt="__meal_summary__ Generate today's meal nutrition summary.",
        run_at=nine_pm,
    )
    db.mark_task_status(task_id, "completed")
    result = db.get_scheduled_meal_summary_task(
        window_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
        window_end=now.replace(hour=23, minute=59, second=59, microsecond=0),
    )
    assert result is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_db.py -v -k "meal_summary"
```
Expected: `AttributeError: 'Database' object has no attribute 'get_scheduled_meal_summary_task'`

- [ ] **Step 3: Add the method to `Database`**

No schema migration needed — `scheduled_tasks` table already exists in `SCHEMA_VERSION = 2`. `SCHEMA_VERSION` stays at 2. Only a new query method is added.

In `assistant/db.py`, add after `mark_task_status`:
```python
    def get_scheduled_meal_summary_task(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, Any] | None:
        """Return a pending/running meal summary task within the given UTC window, or None.

        Uses INSTR instead of LIKE to avoid SQLite treating '_' as a wildcard.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, group_id, prompt, run_at, status
                FROM scheduled_tasks
                WHERE INSTR(prompt, '__meal_summary__') > 0
                AND status IN ('pending', 'running')
                AND run_at >= ?
                AND run_at < ?
                LIMIT 1
                """,
                (
                    window_start.astimezone(timezone.utc).isoformat(),
                    window_end.astimezone(timezone.utc).isoformat(),
                ),
            ).fetchone()
        return dict(row) if row else None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/test_db.py -v -k "meal_summary"
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/db.py tests/test_db.py
git commit -m "feat: add get_scheduled_meal_summary_task() to Database"
```

---

## Chunk 2: MealTracker

### Task 4: MealTracker — memory file operations

**Files:**
- Create: `assistant/tools/meal_tracker.py` (skeleton + `_check_memory` + `_save_memory`)
- Create: `tests/test_meal_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_meal_tracker.py
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.tools.meal_tracker import MealTracker


def _make_tracker(tmp_path: Path) -> MealTracker:
    config = MagicMock()
    config.meal_nutrition_memory_path = str(tmp_path / "memory.md")
    config.health_bigquery_dataset_id = "health"
    config.health_bigquery_table_id = "meals"
    config.bigquery_project_id = ""
    config.meal_summary_timezone = "UTC"
    llm = AsyncMock()
    kagi = AsyncMock()
    return MealTracker(config=config, llm=llm, kagi=kagi, bq_client=None)


@pytest.mark.asyncio
async def test_check_memory_returns_none_when_file_absent(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    result = await tracker._check_memory("dal makhani")
    assert result is None


@pytest.mark.asyncio
async def test_check_memory_finds_matching_entry(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\n"
        "Per 100gms: 151 kcal, 12g carbs.\n"
        "Source: Kagi search.\n"
    )
    result = await tracker._check_memory("dal makhani")
    assert result is not None
    assert "151 kcal" in result


@pytest.mark.asyncio
async def test_check_memory_is_case_insensitive(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\n"
        "Per 100gms: 151 kcal.\n"
        "Source: x.\n"
    )
    result = await tracker._check_memory("DAL  MAKHANI")
    assert result is not None


@pytest.mark.asyncio
async def test_check_memory_returns_none_for_missing_meal(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\n"
        "Per 100gms: 151 kcal.\n"
        "Source: x.\n"
    )
    result = await tracker._check_memory("chicken biryani")
    assert result is None


@pytest.mark.asyncio
async def test_save_memory_creates_file_if_absent(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients = {
        "kcal": 302.0, "carbs_g": 24.0, "fats_g": 14.0, "sugars_g": 4.0,
        "proteins_g": 16.0, "fiber_g": 6.0, "sodium_mg": 410.0,
        "cholesterol_mg": 22.0, "potassium_mg": 380.0, "iron_mg": 3.2,
        "calcium_mg": 85.0, "vitamin_c_mg": 12.0,
    }
    await tracker._save_memory("Dal Makhani", nutrients, "100gms", "Kagi search via nutritionix.")
    memory_file = tmp_path / "memory.md"
    assert memory_file.exists()
    content = memory_file.read_text()
    assert "## Dal Makhani" in content
    assert "Per 100gms" in content
    assert "Kagi search via nutritionix." in content


@pytest.mark.asyncio
async def test_save_memory_appends_to_existing_file(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text("## Samosa\nPer 1 unit: 130 kcal.\nSource: x.\n")
    nutrients = {"kcal": 151.0, "carbs_g": 12.0, "fats_g": 7.0, "sugars_g": 2.0,
                 "proteins_g": 8.0, "fiber_g": 3.0, "sodium_mg": 410.0,
                 "cholesterol_mg": 22.0, "potassium_mg": 380.0, "iron_mg": 3.2,
                 "calcium_mg": 85.0, "vitamin_c_mg": 12.0}
    await tracker._save_memory("Dal Makhani", nutrients, "100gms", "Source summary.")
    content = memory_file.read_text()
    assert "## Samosa" in content
    assert "## Dal Makhani" in content
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_meal_tracker.py -v
```
Expected: `ModuleNotFoundError: No module named 'assistant.tools.meal_tracker'`

- [ ] **Step 3: Create `meal_tracker.py` with skeleton + memory methods**

```python
# assistant/tools/meal_tracker.py
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
```

- [ ] **Step 4: Run memory tests — expect pass**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "memory"
```
Expected: all memory tests PASS

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/tools/meal_tracker.py tests/test_meal_tracker.py
git commit -m "feat: add MealTracker with memory file operations"
```

---

### Task 5: MealTracker — Gemini nutrition lookup (`_call_gemini`)

**Files:**
- Modify: `assistant/tools/meal_tracker.py`
- Modify: `tests/test_meal_tracker.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_meal_tracker.py`:

```python
from unittest.mock import AsyncMock, MagicMock, call, patch
from assistant.models import LLMResponse, LLMToolCall


def _make_llm_response(content: str = "", tool_calls: list | None = None) -> MagicMock:
    resp = MagicMock(spec=LLMResponse)
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


def _make_tool_call(name: str, args: dict) -> MagicMock:
    tc = MagicMock(spec=LLMToolCall)
    tc.name = name
    tc.arguments = args
    tc.call_id = "call-1"
    return tc


@pytest.mark.asyncio
async def test_call_gemini_cache_hit_no_kagi_call(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients_json = json.dumps({f: 1.0 for f in [
        "kcal","carbs_g","fats_g","sugars_g","proteins_g",
        "fiber_g","sodium_mg","cholesterol_mg","potassium_mg","iron_mg","calcium_mg","vitamin_c_mg"
    ]})
    tracker._llm.generate = AsyncMock(return_value=_make_llm_response(content=nutrients_json))
    memory_entry = "## Dal Makhani\nPer 100gms: 151 kcal.\nSource: x.\n"

    result = await tracker._call_gemini("dal makhani", 200.0, "gms", memory_entry)

    # LLM called once (scaling only), Kagi never called
    tracker._llm.generate.assert_called_once()
    tracker._kagi.run.assert_not_called()
    assert result["kcal"] == 1.0


@pytest.mark.asyncio
async def test_call_gemini_cache_miss_calls_kagi_and_returns_nutrients(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients_json = json.dumps({
        "kcal": 302.0, "carbs_g": 24.0, "fats_g": 14.0, "sugars_g": 4.0,
        "proteins_g": 16.0, "fiber_g": 6.0, "sodium_mg": 410.0,
        "cholesterol_mg": 22.0, "potassium_mg": 380.0, "iron_mg": 3.2,
        "calcium_mg": 85.0, "vitamin_c_mg": 12.0,
    })
    kagi_tool_call = _make_tool_call("kagi_search", {"query": "dal makhani nutrition"})
    tracker._llm.generate = AsyncMock(side_effect=[
        _make_llm_response(tool_calls=[kagi_tool_call]),   # turn 1: search
        _make_llm_response(content=nutrients_json),         # turn 2: answer
    ])
    tracker._kagi.run = AsyncMock(return_value="search result text")

    result = await tracker._call_gemini("dal makhani", 200.0, "gms", None)

    assert tracker._llm.generate.call_count == 2
    tracker._kagi.run.assert_called_once_with(query="dal makhani nutrition")
    assert result["kcal"] == 302.0
    assert result["proteins_g"] == 16.0


@pytest.mark.asyncio
async def test_call_gemini_stops_kagi_after_3_searches(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients_json = json.dumps({"kcal": 100.0, "carbs_g": None, "fats_g": None,
                                  "sugars_g": None, "proteins_g": None, "fiber_g": None,
                                  "sodium_mg": None, "cholesterol_mg": None,
                                  "potassium_mg": None, "iron_mg": None,
                                  "calcium_mg": None, "vitamin_c_mg": None})
    kagi_call = _make_tool_call("kagi_search", {"query": "q"})
    tracker._llm.generate = AsyncMock(side_effect=[
        _make_llm_response(tool_calls=[kagi_call]),
        _make_llm_response(tool_calls=[kagi_call]),
        _make_llm_response(tool_calls=[kagi_call]),
        _make_llm_response(content=nutrients_json),  # 4th turn: no tools offered
    ])
    tracker._kagi.run = AsyncMock(return_value="result")

    result = await tracker._call_gemini("samosa", 2.0, "units", None)

    assert tracker._kagi.run.call_count == 3
    assert tracker._llm.generate.call_count == 4
    assert result["kcal"] == 100.0


@pytest.mark.asyncio
async def test_call_gemini_returns_all_nulls_on_bad_json(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    tracker._llm.generate = AsyncMock(
        return_value=_make_llm_response(content="not valid json at all!!!")
    )
    result = await tracker._call_gemini("unknown dish", 1.0, "units", None)
    assert all(v is None for v in result.values())
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "call_gemini"
```
Expected: `NotImplementedError`

- [ ] **Step 3: Implement `_call_gemini`**

Replace the `_call_gemini` stub in `meal_tracker.py` with:

```python
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
```

Also add the `_parse_nutrients` module-level helper:

```python
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
```

- [ ] **Step 4: Run Gemini tests — expect pass**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "call_gemini"
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/tools/meal_tracker.py tests/test_meal_tracker.py
git commit -m "feat: implement MealTracker Gemini nutrition lookup with Kagi search"
```

---

### Task 6: MealTracker — BigQuery insert and summary query

**Files:**
- Modify: `assistant/tools/meal_tracker.py`
- Modify: `tests/test_meal_tracker.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_meal_tracker.py`:

```python
@pytest.mark.asyncio
async def test_insert_bigquery_skipped_when_no_project(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)  # bq_client=None already
    # Should complete without error
    await tracker._insert_bigquery("dal makhani", 200.0, "gms", {"kcal": 302.0}, "search")


@pytest.mark.asyncio
async def test_insert_bigquery_inserts_correct_row(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    mock_bq = MagicMock()
    mock_bq.get_table.side_effect = Exception("not found")
    mock_bq.create_table.return_value = None
    mock_bq.insert_rows_json.return_value = []
    tracker._bq_client = mock_bq
    tracker._config.health_bigquery_dataset_id = "health"
    tracker._config.health_bigquery_table_id = "meals"
    tracker._config.bigquery_project_id = "my-project"

    nutrients = {f: 1.0 for f in _NUTRIENT_FIELDS}
    await tracker._insert_bigquery("dal makhani", 200.0, "gms", nutrients, "search")

    mock_bq.insert_rows_json.assert_called_once()
    rows = mock_bq.insert_rows_json.call_args[0][1]
    assert len(rows) == 1
    row = rows[0]
    assert row["meal_name"] == "dal makhani"
    assert row["portion_amount"] == 200.0
    assert row["portion_unit"] == "gms"
    assert row["source"] == "search"
    assert row["kcal"] == 1.0
    assert "logged_at" in row


@pytest.mark.asyncio
async def test_get_summary_queries_bigquery_for_date(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {"meal_name": "Dal Makhani", "portion_amount": 200.0, "portion_unit": "gms",
         "kcal": 302.0, "proteins_g": 16.0, "source": "search", "logged_at": "2026-03-15T18:00:00Z"},
    ]
    mock_bq.query.return_value = mock_job
    tracker._bq_client = mock_bq
    tracker._config.bigquery_project_id = "my-project"

    rows = await tracker.get_summary(datetime.date(2026, 3, 15))

    mock_bq.query.assert_called_once()
    call_kwargs = mock_bq.query.call_args
    assert "2026-03-15" in str(call_kwargs)
    assert len(rows) == 1
    assert rows[0]["meal_name"] == "Dal Makhani"


@pytest.mark.asyncio
async def test_get_summary_returns_empty_when_no_bq_client(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)  # bq_client=None
    rows = await tracker.get_summary(datetime.date(2026, 3, 15))
    assert rows == []
```

Also add this import at the top of the test file:
```python
from assistant.tools.meal_tracker import _NUTRIENT_FIELDS
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "bigquery or summary"
```
Expected: `NotImplementedError`

- [ ] **Step 3: Implement `_insert_bigquery` and `get_summary`**

Replace their stubs in `meal_tracker.py`:

```python
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
        from google.cloud import bigquery  # type: ignore[import-untyped]

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
```

Also add this module-level helper:

```python
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
```

- [ ] **Step 4: Run BQ tests — expect pass**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "bigquery or summary"
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/tools/meal_tracker.py tests/test_meal_tracker.py
git commit -m "feat: implement MealTracker BigQuery insert and summary query"
```

---

### Task 7: MealTracker — `track()` orchestrator

**Files:**
- Modify: `assistant/tools/meal_tracker.py`
- Modify: `tests/test_meal_tracker.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_meal_tracker.py`:

```python
@pytest.mark.asyncio
async def test_track_cache_miss_calls_gemini_and_saves_memory(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    nutrients = {f: 1.0 for f in _NUTRIENT_FIELDS}
    tracker._call_gemini = AsyncMock(return_value=nutrients)
    tracker._save_memory = AsyncMock()
    tracker._insert_bigquery = AsyncMock()

    # Provide a second Gemini call for source summary
    tracker._llm.generate = AsyncMock(
        return_value=_make_llm_response(content="Source: Kagi via healthline.")
    )

    reply = await tracker.track("dal makhani", 200.0, "gms")

    tracker._call_gemini.assert_called_once_with("dal makhani", 200.0, "gms", None)
    tracker._save_memory.assert_called_once()
    tracker._insert_bigquery.assert_called_once()
    args = tracker._insert_bigquery.call_args[1]
    assert args["source"] == "search"
    assert "dal makhani" in reply.lower() or "200" in reply


@pytest.mark.asyncio
async def test_track_cache_hit_skips_search_and_memory_save(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    # Pre-populate memory file
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "## Dal Makhani\nPer 100gms: 151 kcal.\nSource: x.\n"
    )
    nutrients = {f: 1.0 for f in _NUTRIENT_FIELDS}
    tracker._call_gemini = AsyncMock(return_value=nutrients)
    tracker._save_memory = AsyncMock()
    tracker._insert_bigquery = AsyncMock()

    await tracker.track("dal makhani", 200.0, "gms")

    # _call_gemini called with memory_entry (not None)
    call_args = tracker._call_gemini.call_args
    assert call_args[0][3] is not None  # memory_entry is the 4th positional arg
    # _save_memory NOT called
    tracker._save_memory.assert_not_called()
    # BQ insert called with source="memory"
    args = tracker._insert_bigquery.call_args[1]
    assert args["source"] == "memory"


@pytest.mark.asyncio
async def test_track_null_nutrients_still_inserts_row_but_skips_memory(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    tracker._call_gemini = AsyncMock(return_value={f: None for f in _NUTRIENT_FIELDS})
    tracker._save_memory = AsyncMock()
    tracker._insert_bigquery = AsyncMock()
    tracker._llm.generate = AsyncMock(
        return_value=_make_llm_response(content="Source: unknown.")
    )

    reply = await tracker.track("mystery dish", 1.0, "units")

    # BQ insert still happens (with null values)
    tracker._insert_bigquery.assert_called_once()
    # Memory NOT saved when all nutrients are null (would write garbage to file)
    tracker._save_memory.assert_not_called()
    assert "incomplete" in reply.lower() or "mystery dish" in reply.lower()
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "test_track"
```
Expected: `NotImplementedError`

- [ ] **Step 3: Implement `track()`**

Replace the `track` stub in `meal_tracker.py`:

```python
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
                await self._save_memory(meal_name, _scale_to_base(nutrients, portion_amount, portion_unit), base_unit, source_summary)

        try:
            await self._insert_bigquery(meal_name, portion_amount, portion_unit, nutrients, source)
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
```

Add these module-level helpers after `_nutrient_unit`:

```python
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
```

- [ ] **Step 4: Run track tests — expect pass**

```bash
uv run pytest tests/test_meal_tracker.py -v -k "test_track"
```

- [ ] **Step 5: Run all meal tracker tests**

```bash
uv run pytest tests/test_meal_tracker.py -v
```

- [ ] **Step 6: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add assistant/tools/meal_tracker.py tests/test_meal_tracker.py
git commit -m "feat: implement MealTracker.track() orchestrator"
```

---

## Chunk 3: Tool, Command Handler, Wiring, Docs

### Task 8: `GetMealSummaryTool`

**Files:**
- Create: `assistant/tools/get_meal_summary_tool.py`
- Create: `tests/test_get_meal_summary_tool.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_get_meal_summary_tool.py
from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.tools.get_meal_summary_tool import GetMealSummaryTool


def _make_tool(meals: list | None = None) -> GetMealSummaryTool:
    tracker = MagicMock()
    tracker.get_summary = AsyncMock(return_value=meals or [])
    return GetMealSummaryTool(tracker=tracker)


def test_tool_has_correct_name_and_schema() -> None:
    tool = _make_tool()
    assert tool.name == "get_meal_summary"
    assert "date" in tool.parameters_schema["properties"]
    assert "date" in tool.parameters_schema["required"]


@pytest.mark.asyncio
async def test_run_passes_parsed_date_to_tracker() -> None:
    tool = _make_tool(meals=[
        {"meal_name": "Dal Makhani", "portion_amount": 200.0, "portion_unit": "gms",
         "kcal": 302.0, "proteins_g": 16.0, "carbs_g": 24.0, "fats_g": 14.0,
         "source": "search"},
    ])
    result = await tool.run(date="2026-03-15")
    tool._tracker.get_summary.assert_called_once_with(datetime.date(2026, 3, 15))
    assert "Dal Makhani" in result
    assert "302" in result


@pytest.mark.asyncio
async def test_run_returns_no_meals_message_when_empty() -> None:
    tool = _make_tool(meals=[])
    result = await tool.run(date="2026-03-15")
    assert "no meals" in result.lower()
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_get_meal_summary_tool.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `GetMealSummaryTool`**

```python
# assistant/tools/get_meal_summary_tool.py
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
    for field, label in [("proteins_g", "Protein"), ("carbs_g", "Carbs"),
                          ("fats_g", "Fat"), ("fiber_g", "Fiber"), ("sodium_mg", "Sodium")]:
        total = sum((m.get(field) or 0.0) for m in meals)
        unit = "g" if field.endswith("_g") else "mg"
        lines.append(f"{label}: {total:.1f}{unit}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/test_get_meal_summary_tool.py -v
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/tools/get_meal_summary_tool.py tests/test_get_meal_summary_tool.py
git commit -m "feat: add GetMealSummaryTool for LLM-driven daily nutrition summary"
```

---

### Task 9: `/tm` command handler in `commands.py`

**Files:**
- Modify: `assistant/commands.py`
- Modify: `tests/test_commands.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_commands.py`:

```python
import re
import datetime
from unittest.mock import AsyncMock, MagicMock

from assistant.commands import CommandDispatcher, parse_command
from assistant.models import Message


def _msg(text: str) -> Message:
    return Message(
        group_id="g1", sender_id="s1", text=text,
        timestamp=datetime.datetime.now(datetime.timezone.utc), attachments=[],
    )


def _make_dispatcher_with_tracker(tracker=None) -> CommandDispatcher:
    mock_tracker = tracker or AsyncMock()
    mock_tracker.track = AsyncMock(return_value="Logged dal makhani.")
    return CommandDispatcher(meal_tracker=mock_tracker)


# --- Alias ---

def test_parse_command_tm_resolves_to_trackmeal() -> None:
    result = parse_command("/tm dal makhani 200gms")
    assert result is not None
    assert result[0] == "trackmeal"


# --- Portion parsing ---

@pytest.mark.asyncio
async def test_trackmeal_parses_gms_portion() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm dal makhani 200gms"))
    tracker.track.assert_called_once_with("dal makhani", 200.0, "gms")


@pytest.mark.asyncio
async def test_trackmeal_parses_g_normalizes_to_gms() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm samosa 150g"))
    tracker.track.assert_called_once_with("samosa", 150.0, "gms")


@pytest.mark.asyncio
async def test_trackmeal_parses_cups_portion() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm chicken biryani 1.5cups"))
    tracker.track.assert_called_once_with("chicken biryani", 1.5, "cups")


@pytest.mark.asyncio
async def test_trackmeal_parses_plain_number_as_units() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm samosa 2"))
    tracker.track.assert_called_once_with("samosa", 2.0, "units")


@pytest.mark.asyncio
async def test_trackmeal_missing_portion_returns_usage_hint() -> None:
    d = _make_dispatcher_with_tracker()
    result = await d.dispatch(_msg("/tm dal makhani"))
    assert result is not None
    assert "usage" in result.lower()


@pytest.mark.asyncio
async def test_trackmeal_empty_meal_name_returns_usage_hint() -> None:
    d = _make_dispatcher_with_tracker()
    result = await d.dispatch(_msg("/tm 200gms"))
    assert result is not None
    assert "usage" in result.lower()


# --- Summary is checked before portion parsing ---

@pytest.mark.asyncio
async def test_trackmeal_summary_checked_before_portion_parsing() -> None:
    tracker = AsyncMock()
    tracker.get_summary = AsyncMock(return_value=[])
    d = CommandDispatcher(meal_tracker=tracker, llm=None)
    result = await d.dispatch(_msg("/tm summary"))
    # Should route to summary, not try to parse "summary" as a meal name
    assert result is not None
    assert "no meals" in result.lower()


@pytest.mark.asyncio
async def test_trackmeal_summary_today_calls_get_summary(tmp_path) -> None:
    tracker = AsyncMock()
    tracker.get_summary = AsyncMock(return_value=[])
    d = CommandDispatcher(meal_tracker=tracker, llm=None)
    await d.dispatch(_msg("/tm summary"))
    tracker.get_summary.assert_called_once()
    date_arg = tracker.get_summary.call_args[0][0]
    assert date_arg == datetime.date.today()


@pytest.mark.asyncio
async def test_trackmeal_summary_specific_date(tmp_path) -> None:
    tracker = AsyncMock()
    tracker.get_summary = AsyncMock(return_value=[])
    d = CommandDispatcher(meal_tracker=tracker, llm=None)
    await d.dispatch(_msg("/tm summary mar 15"))
    tracker.get_summary.assert_called_once()
    date_arg = tracker.get_summary.call_args[0][0]
    assert date_arg.month == 3
    assert date_arg.day == 15


@pytest.mark.asyncio
async def test_trackmeal_summary_invalid_day_returns_error() -> None:
    d = _make_dispatcher_with_tracker()
    result = await d.dispatch(_msg("/tm summary feb 30"))
    assert result is not None
    assert "invalid" in result.lower() or "february" in result.lower()


@pytest.mark.asyncio
async def test_trackmeal_summary_month_without_day_returns_usage_hint() -> None:
    d = _make_dispatcher_with_tracker()
    result = await d.dispatch(_msg("/tm summary mar"))
    assert result is not None
    assert "usage" in result.lower()
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_commands.py -v -k "trackmeal or tm"
```
Expected: failures due to missing alias and handler

- [ ] **Step 3: Add alias and handler to `commands.py`**

In `_ALIASES`, add:
```python
    "tm": "trackmeal",
```

In `TYPE_CHECKING` block, add:
```python
    from assistant.tools.meal_tracker import MealTracker
```

In `CommandDispatcher.__init__`, add:
```python
        meal_tracker: MealTracker | None = None,
```
and in the body:
```python
        self._meal_tracker = meal_tracker
```

In `dispatch()`, add the trackmeal branch in the `command == "..."` chain — **after** the existing `isdigit()` magazine guard at the top of `dispatch()`, not before it. The correct position is alongside the other `if command == "..."` checks (e.g., after `if command == "cite"`):
```python
        if command == "trackmeal":
            return await self._handle_trackmeal(args)
```

Add the handler and helpers:

```python
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
        last = args[-1]
        meal_name, portion_amount, portion_unit = _parse_portion(args)
        if portion_amount is None:
            return _USAGE
        if not meal_name:
            return _USAGE

        if self._meal_tracker is None:
            return "Meal tracker is not configured."
        return await self._meal_tracker.track(meal_name, portion_amount, portion_unit)

    async def _handle_trackmeal_summary(self, args: list[str], usage: str) -> str:
        import calendar

        if not args:
            target_date = datetime.date.today()
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
            year = datetime.date.today().year
            _, max_day = calendar.monthrange(year, month)
            if day < 1 or day > max_day:
                month_name = datetime.date(year, month, 1).strftime("%B")
                return f"Invalid date: {month_name} doesn't have {day} days."
            target_date = datetime.date(year, month, day)

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
        self, meals: list[dict], target_date: datetime.date
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
```

Add module-level helpers (after the existing imports, before the class):

```python
import datetime as _dt_mod  # avoid shadowing the 'datetime' class

# Matches: 200gms, 200gm, 200g, 1.5cups, 1.5cup
_PORTION_RE = re.compile(r"^(\d+(?:\.\d+)?)(gms?|gm?|cups?)$", re.IGNORECASE)


def _parse_portion(
    args: list[str],
) -> tuple[str, float | None, str | None]:
    """Split args into (meal_name, portion_amount, portion_unit).

    Unit normalization: g / gm / gms → "gms"; cup / cups → "cups".
    Plain number with no suffix → "units". Returns ("", None, None) if invalid.
    """
    last = args[-1]
    m = _PORTION_RE.match(last)
    if m:
        amount = float(m.group(1))
        raw_unit = m.group(2).lower()
        unit = "gms" if raw_unit in ("g", "gm", "gms") else "cups"
        meal_name = " ".join(args[:-1]).strip()
        return meal_name, amount, unit
    # Try plain number → units
    try:
        amount = float(last)
        meal_name = " ".join(args[:-1]).strip()
        return meal_name, amount, "units"
    except ValueError:
        return "", None, None


def _format_summary_raw(meals: list[dict], date: _dt_mod.date) -> str:
    """Fallback summary format when no LLM is available."""
    lines = [f"Meals logged for {date.strftime('%d %b %Y')}:"]
    total = 0.0
    for m in meals:
        kcal = m.get("kcal") or 0.0
        total += kcal
        lines.append(f"• {m['meal_name']} {m['portion_amount']:.0f}{m['portion_unit']} — {kcal:.0f} kcal")
    lines.append(f"Total: {total:.0f} kcal / 2300 goal")
    return "\n".join(lines)
```

Also add `import datetime` at the top of `commands.py` if not present.

- [ ] **Step 4: Run command tests — expect pass**

```bash
uv run pytest tests/test_commands.py -v -k "trackmeal or tm"
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/commands.py tests/test_commands.py
git commit -m "feat: add /tm (/trackmeal) command handler with portion parsing and summary"
```

---

### Task 9b: Optional date/time for backdated meal entries

**Files:**
- Modify: `assistant/tools/meal_tracker.py` — `track()` accepts optional `logged_at`
- Modify: `assistant/commands.py` — `_parse_portion` and `_handle_trackmeal` parse date/time tokens
- Modify: `tests/test_meal_tracker.py`
- Modify: `tests/test_commands.py`

**Feature:** Allow logging meals for previous days/times:
```
/tm dal makhani 200g 15.03 17:00   → March 15, 17:00 local time
/tm samosa 2 14.03                 → March 14, time = now
/tm chicken biryani 1.5cups 9:30   → today, 09:30 local time
```

**Parsing rules (after portion, remaining tokens):**
- Date token: matches `\d{1,2}\.\d{1,2}` (D.M, DD.M, D.MM, DD.MM) — current year
- Time token: matches `\d{1,2}:\d{2}` (H:MM, HH:MM) — 24-hour
- Date always comes before time if both present
- Both are optional; if absent, `logged_at = None` → `MealTracker.track()` uses `datetime.now(UTC)`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_commands.py`:

```python
_DATE_RE_STR = r"\d{1,2}\.\d{1,2}"
_TIME_RE_STR = r"\d{1,2}:\d{2}"

@pytest.mark.asyncio
async def test_trackmeal_parses_date_and_time() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm dal makhani 200gms 15.03 17:00"))
    tracker.track.assert_called_once()
    kwargs = tracker.track.call_args[1]
    assert kwargs["meal_name"] == "dal makhani"
    assert kwargs["portion_amount"] == 200.0
    assert kwargs["portion_unit"] == "gms"
    logged_at = kwargs["logged_at"]
    assert logged_at is not None
    assert logged_at.month == 3
    assert logged_at.day == 15
    assert logged_at.hour == 17
    assert logged_at.minute == 0


@pytest.mark.asyncio
async def test_trackmeal_parses_date_only() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm samosa 2 14.03"))
    kwargs = tracker.track.call_args[1]
    assert kwargs["logged_at"].month == 3
    assert kwargs["logged_at"].day == 14


@pytest.mark.asyncio
async def test_trackmeal_parses_time_only() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm samosa 2 9:30"))
    kwargs = tracker.track.call_args[1]
    logged_at = kwargs["logged_at"]
    assert logged_at is not None
    assert logged_at.hour == 9
    assert logged_at.minute == 30
    assert logged_at.day == datetime.date.today().day


@pytest.mark.asyncio
async def test_trackmeal_no_date_time_passes_none_logged_at() -> None:
    tracker = AsyncMock()
    tracker.track = AsyncMock(return_value="ok")
    d = CommandDispatcher(meal_tracker=tracker)
    await d.dispatch(_msg("/tm samosa 2"))
    kwargs = tracker.track.call_args[1]
    assert kwargs["logged_at"] is None
```

Add to `tests/test_meal_tracker.py`:

```python
@pytest.mark.asyncio
async def test_track_uses_provided_logged_at(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    tracker._call_gemini = AsyncMock(return_value={f: 1.0 for f in _NUTRIENT_FIELDS})
    tracker._save_memory = AsyncMock()
    tracker._insert_bigquery = AsyncMock()
    tracker._llm.generate = AsyncMock(return_value=_make_llm_response(content="Source."))

    custom_time = datetime.datetime(2026, 3, 14, 17, 0, tzinfo=datetime.timezone.utc)
    await tracker.track("dal makhani", 200.0, "gms", logged_at=custom_time)

    call_kwargs = tracker._insert_bigquery.call_args[1]
    assert call_kwargs["logged_at"] == custom_time


@pytest.mark.asyncio
async def test_track_uses_utcnow_when_logged_at_is_none(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    tracker._call_gemini = AsyncMock(return_value={f: 1.0 for f in _NUTRIENT_FIELDS})
    tracker._save_memory = AsyncMock()
    tracker._insert_bigquery = AsyncMock()
    tracker._llm.generate = AsyncMock(return_value=_make_llm_response(content="Source."))

    before = datetime.datetime.now(datetime.timezone.utc)
    await tracker.track("dal makhani", 200.0, "gms", logged_at=None)
    after = datetime.datetime.now(datetime.timezone.utc)

    call_kwargs = tracker._insert_bigquery.call_args[1]
    assert before <= call_kwargs["logged_at"] <= after
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_commands.py -v -k "date_and_time or date_only or time_only or logged_at"
uv run pytest tests/test_meal_tracker.py -v -k "logged_at"
```

- [ ] **Step 3: Update `_parse_portion` in `commands.py` to return optional `logged_at`**

Add regex constants:
```python
_DATE_TOKEN_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
_TIME_TOKEN_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
```

Update `_parse_portion` signature and return type:
```python
def _parse_portion(
    args: list[str],
) -> tuple[str, float | None, str | None, _dt_mod.datetime | None]:
    """Split args into (meal_name, portion_amount, portion_unit, logged_at).

    After the portion token, optional date (D.M / DD.MM) and/or time (H:MM / HH:MM)
    tokens are parsed. logged_at is a UTC-aware datetime or None if not provided.
    """
    if not args:
        return "", None, None, None

    # Find the portion token (last positional arg that looks like a portion).
    # Date/time tokens may follow it.
    portion_idx = None
    for i, token in enumerate(args):
        if _PORTION_RE.match(token):
            portion_idx = i
        else:
            try:
                float(token)
                portion_idx = i
            except ValueError:
                pass  # might be date/time token after portion

    if portion_idx is None:
        # Last token might be plain number — check
        try:
            float(args[-1])
            portion_idx = len(args) - 1
        except ValueError:
            return "", None, None, None

    last = args[portion_idx]
    m = _PORTION_RE.match(last)
    if m:
        amount = float(m.group(1))
        raw_unit = m.group(2).lower()
        unit = "gms" if raw_unit in ("g", "gm", "gms") else "cups"
    else:
        try:
            amount = float(last)
            unit = "units"
        except ValueError:
            return "", None, None, None

    meal_name = " ".join(args[:portion_idx]).strip()
    remaining = args[portion_idx + 1:]

    # Parse optional date and time from remaining tokens.
    logged_at = _parse_datetime_tokens(remaining)
    return meal_name, amount, unit, logged_at


def _parse_datetime_tokens(tokens: list[str]) -> _dt_mod.datetime | None:
    """Parse optional [date] [time] tokens into a UTC-aware datetime.

    Date format: D.M or DD.MM (current year).
    Time format: H:MM or HH:MM (24-hour, local time).
    """
    import zoneinfo as _zi

    date_part: _dt_mod.date | None = None
    time_part: _dt_mod.time | None = None

    for token in tokens:
        dm = _DATE_TOKEN_RE.match(token)
        if dm and date_part is None:
            day, month = int(dm.group(1)), int(dm.group(2))
            try:
                date_part = _dt_mod.date(_dt_mod.date.today().year, month, day)
            except ValueError:
                pass  # invalid date — ignore token
            continue
        tm = _TIME_TOKEN_RE.match(token)
        if tm and time_part is None:
            hour, minute = int(tm.group(1)), int(tm.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                time_part = _dt_mod.time(hour, minute)
            continue

    if date_part is None and time_part is None:
        return None

    date_part = date_part or _dt_mod.date.today()
    time_part = time_part or _dt_mod.time(0, 0)

    # Combine using the local timezone from the settings (not available here).
    # Use UTC as the default timezone for parsing; main.py can pass settings.meal_summary_timezone
    # if local time parsing is needed. For simplicity, treat user input as local machine time.
    naive = _dt_mod.datetime.combine(date_part, time_part)
    return naive.astimezone(_dt_mod.timezone.utc)
```

Update `_handle_trackmeal` to pass `logged_at` as a keyword arg:
```python
        meal_name, portion_amount, portion_unit, logged_at = _parse_portion(args)
        if portion_amount is None:
            return _USAGE
        if not meal_name:
            return _USAGE
        ...
        return await self._meal_tracker.track(
            meal_name, portion_amount, portion_unit, logged_at=logged_at
        )
```

- [ ] **Step 4: Update `MealTracker.track()` signature to accept `logged_at`**

```python
    async def track(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        logged_at: datetime.datetime | None = None,
    ) -> str:
```

Pass `logged_at` through to `_insert_bigquery`:
```python
        await self._insert_bigquery(
            meal_name, portion_amount, portion_unit, nutrients, source,
            logged_at=logged_at,
        )
```

Update `_insert_bigquery` signature:
```python
    async def _insert_bigquery(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        nutrients: dict[str, float | None],
        source: str,
        logged_at: datetime.datetime | None = None,
    ) -> None:
        ...
        row["logged_at"] = (logged_at or datetime.datetime.now(timezone.utc)).isoformat()
```

- [ ] **Step 5: Run tests — expect pass**

```bash
uv run pytest tests/test_commands.py -v -k "date_and_time or date_only or time_only or logged_at"
uv run pytest tests/test_meal_tracker.py -v -k "logged_at"
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add assistant/commands.py assistant/tools/meal_tracker.py tests/test_commands.py tests/test_meal_tracker.py
git commit -m "feat: support optional date/time for backdated meal entries"
```

---

### Task 10: Wire everything up in `main.py`

**Files:**
- Modify: `assistant/main.py`

No tests for this task — `main.py` is the integration layer. Verify manually after wiring.

- [ ] **Step 1: Add imports**

Add to the imports in `main.py`:
```python
import zoneinfo
from datetime import timedelta

from assistant.tools.meal_tracker import MealTracker
from assistant.tools.get_meal_summary_tool import GetMealSummaryTool
```

- [ ] **Step 2: Instantiate `MealTracker` and register `GetMealSummaryTool`**

After the `price_tracker_tool` block (around line 86), add:

```python
    if settings.bigquery_project_id:
        from google.cloud import bigquery as _bq  # type: ignore[import-untyped]
        _bq_client = _bq.Client(project=settings.bigquery_project_id)
    else:
        _bq_client = None

    # MealTracker gets its own KagiSearchTool instance (same API key, separate object).
    # This avoids coupling with the CommandDispatcher's search tool.
    meal_tracker = MealTracker(
        config=settings,
        llm=provider,
        kagi=KagiSearchTool(api_key=settings.kagi_api_key),
        bq_client=_bq_client,
    )
    meal_summary_tool = GetMealSummaryTool(tracker=meal_tracker)
    tools.register(meal_summary_tool)
```

- [ ] **Step 3: Pass `meal_tracker` to `CommandDispatcher`**

In the `CommandDispatcher(...)` call, add:
```python
        meal_tracker=meal_tracker,
```

Note: `_schedule_meal_summary_if_needed` and `_reschedule_meal_summary` both call `db.create_scheduled_task()` directly — they do NOT use the `TaskScheduler` object (which is just a polling wrapper). No `scheduler` parameter is needed on these functions.

- [ ] **Step 4: Schedule daily 9PM summary on startup**

Define this as a **module-level function** in `main.py` (outside `run()`):

```python
def _schedule_meal_summary_if_needed(db: Database, settings: Settings) -> None:
    """Create a 9PM meal summary task if none exists for today/tomorrow."""
    tz = zoneinfo.ZoneInfo(settings.meal_summary_timezone)
    now_local = datetime.now(tz)
    today_9pm = now_local.replace(hour=21, minute=0, second=0, microsecond=0)
    target = today_9pm if now_local < today_9pm else today_9pm + timedelta(days=1)
    target_utc = target.astimezone(timezone.utc)

    # Window: midnight to midnight on target's local date
    target_local = target.astimezone(tz)
    day_start = target_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    day_end = target_local.replace(hour=23, minute=59, second=59, microsecond=0).astimezone(timezone.utc)

    existing = db.get_scheduled_meal_summary_task(window_start=day_start, window_end=day_end)
    if existing:
        LOGGER.info("Meal summary task already scheduled: %s", existing["run_at"])
        return

    db.upsert_group(settings.telegram_owner_id)
    db.create_scheduled_task(
        group_id=settings.telegram_owner_id,
        prompt="__meal_summary__ Generate today's meal nutrition summary.",
        run_at=target_utc,
    )
    LOGGER.info("Scheduled meal summary for %s UTC", target_utc.isoformat())

    # Note: db.get_scheduled_meal_summary_task uses INSTR (not LIKE) to match
    # '__meal_summary__' literally, avoiding SQLite's '_' wildcard in LIKE patterns.
```

Call it in `run()` after `db.initialize()`:
```python
    _schedule_meal_summary_if_needed(db, settings)
```

- [ ] **Step 5: Reschedule after each summary fires**

Define `_reschedule_meal_summary` at **module level** in `main.py` (outside `run()`), so it can be called by the closure without scoping issues:

```python
def _reschedule_meal_summary(db: Database, settings: Settings) -> None:
    """Schedule the next 9PM meal summary for tomorrow."""
    tz = zoneinfo.ZoneInfo(settings.meal_summary_timezone)
    tomorrow_9pm = (
        datetime.now(tz).replace(hour=21, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    )
    target_utc = tomorrow_9pm.astimezone(timezone.utc)
    day_start = tomorrow_9pm.replace(hour=0, minute=0, second=0).astimezone(timezone.utc)
    day_end = tomorrow_9pm.replace(hour=23, minute=59, second=59).astimezone(timezone.utc)
    if not db.get_scheduled_meal_summary_task(day_start, day_end):
        db.upsert_group(settings.telegram_owner_id)
        db.create_scheduled_task(
            group_id=settings.telegram_owner_id,
            prompt="__meal_summary__ Generate today's meal nutrition summary.",
            run_at=target_utc,
        )
        LOGGER.info("Rescheduled meal summary for %s UTC", target_utc.isoformat())
```

Then modify `handle_scheduled_prompt` (inside `run()`) to call it:

```python
    async def handle_scheduled_prompt(group_id: str, prompt: str) -> None:
        response = await runtime.handle_message(
            Message(
                group_id=group_id,
                sender_id="scheduler",
                text=prompt,
                timestamp=datetime.now(timezone.utc),
                is_group=True,
            )
        )
        await telegram_adapter.send_message(group_id, response, is_group=True)
        # Reschedule daily meal summary for next day after it fires.
        if "__meal_summary__" in prompt:
            _reschedule_meal_summary(db, settings)
```

`_reschedule_meal_summary` is module-level so it's accessible from the closure.

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
uv run pytest -q
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add assistant/main.py
git commit -m "feat: wire MealTracker, GetMealSummaryTool, and daily 9PM scheduler in main.py"
```

---

### Task 11: Update docs and commands help

**Files:**
- Modify: `assistant/commands_help.json`
- Modify: `README.md`

- [ ] **Step 1: Add `/tm` entry to `commands_help.json`**

Add to the JSON array (e.g. at the end, before the closing `]`):
```json
  {
    "usage": "/trackmeal (/tm) <meal> <portion>",
    "description": "Track a meal and look up nutritional info (e.g. /tm dal makhani 200gms or /tm samosa 2). Use /tm summary [mar 15] for a daily nutrition report."
  }
```

- [ ] **Step 2: Add `/tm` section to `README.md`**

Find the existing command table or commands section in README.md and add:

```markdown
### `/trackmeal` (`/tm`) — Meal Nutrition Tracker

Log a meal and get its nutritional breakdown.

**Usage:**
```
/tm <meal name> <portion>
/tm summary [month day]
```

**Portion formats:**
- `200gms` or `200g` — grams
- `1.5cups` — cups
- `2` — units (plain number, no suffix)

**Examples:**
```
/tm dal makhani 200gms
/tm chicken biryani 1.5cups
/tm samosa 2
/tm summary
/tm summary mar 15
```

**How it works:**
1. Checks a local memory file (`~/.claw/meal_nutrition_memory.md`) for cached data
2. If not cached: calls Gemini (`google/gemini-3.1-pro-preview-customtools`) via OpenRouter, which can search Kagi up to 3 times
3. Saves the result to BigQuery (`health.meals` table) and to the memory file
4. Every day at 9PM (configured timezone), sends a summary of all meals vs. the 2300 kcal goal with 3 tips

**Config** (add to `~/.config/tejas/config.yaml` under `claw:`):
```yaml
meal_nutrition_memory_path: ~/.claw/meal_nutrition_memory.md
health_bigquery_dataset_id: health
health_bigquery_table_id: meals
meal_summary_timezone: Europe/Berlin   # IANA timezone
```
```

- [ ] **Step 3: Run full suite one final time**

```bash
uv run pytest -q
```
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add assistant/commands_help.json README.md
git commit -m "docs: add /tm command documentation to README and commands help"
```

---

## Config reminder for Tejas

After implementation is complete, add these lines to `~/.config/tejas/config.yaml` under the `claw:` section:

```yaml
meal_nutrition_memory_path: ~/.claw/meal_nutrition_memory.md
health_bigquery_dataset_id: health
health_bigquery_table_id: meals
meal_summary_timezone: Europe/Berlin   # or Asia/Kolkata, UTC, etc.
```
