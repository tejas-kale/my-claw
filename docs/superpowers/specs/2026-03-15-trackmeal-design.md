# /trackmeal (`/tm`) — Design Spec

**Date:** 2026-03-15
**Status:** Approved

---

## Overview

Add a `/trackmeal` (alias `/tm`) command to the Telegram assistant that logs meals with nutritional information. The nutritional lookup calls `google/gemini-3.1-pro-preview-customtools` via the existing `OpenRouterProvider` (with a model override — no new SDK or dependency), with Kagi search available as a tool (up to 3 calls). Results are cached in a local memory file and stored in BigQuery. A daily 9PM summary (in a configured IANA timezone) reports calorie intake vs. a 2300 kcal daily goal along with nutrient analysis and 3 tips.

---

## Command Interface

### Logging a meal

```
/tm <meal name> <portion>
/trackmeal <meal name> <portion>
```

**Parse order in `_handle_trackmeal()`:**

1. If first token is `"summary"` → route to summary handler (checked before any portion parsing)
2. Attempt portion parsing on the last token:
   - Matches `\d+(\.\d+)?(gms?|g|cups?)` → use that unit (`g`, `gms` → `"gms"`; `cup`, `cups` → `"cups"`)
   - Is a plain integer or float (no unit suffix) → unit is `"units"`
   - Otherwise → reply with usage hint, stop
3. Everything before the last token is the meal name. If empty → reply with usage hint, stop.

**Usage hint:**
```
Usage: /tm <meal> <portion>
Examples: /tm dal makhani 200gms  |  /tm chicken biryani 1.5cups  |  /tm samosa 2
```

**Parsed examples:**
```
/tm dal makhani 200gms       → meal="dal makhani", portion=200.0, unit="gms"
/tm chicken biryani 1.5cups  → meal="chicken biryani", portion=1.5, unit="cups"
/tm samosa 2                 → meal="samosa", portion=2.0, unit="units"
```

### Summary subcommand

```
/tm summary           → today's summary
/tm summary mar 15    → March 15 of the current year
```

**Date parsing:**
- Month: 3-letter abbreviation (jan–dec), case-insensitive
- Day: integer, validated with `calendar.monthrange` for the given month and current year
- Year: always current year (not specifiable by user)

**Error routing (in order):**
1. First token is `"summary"`, no further tokens → today's date
2. First token is `"summary"`, second token is a recognized month abbreviation, third token is an integer day → parse date
3. Second token is a recognized month abbreviation but no valid day follows → reply with summary usage hint
4. Any other combination → reply with summary usage hint
5. Day is out of range for the given month (e.g. `feb 30`) → reply: *"Invalid date: February doesn't have 30 days."*
6. No meals found for the date → reply: *"No meals logged for {readable date}."* — do not call LLM

**Summary usage hint:**
```
Usage: /tm summary [month day]
Examples: /tm summary  |  /tm summary mar 15
```

### Alias registration

`"tm": "trackmeal"` added to the `ALIASES` dict in `commands.py`.

---

## OpenRouterProvider Model Override

`OpenRouterProvider.generate()` currently always uses `self._settings.openrouter_model`. To call Gemini specifically, add an optional `model: str | None = None` parameter to `generate()`. When provided, it overrides `self._settings.openrouter_model` for that call only:

```python
async def generate(
    self,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    model: str | None = None,   # new optional parameter
) -> LLMResponse:
    effective_model = model or self._settings.openrouter_model
    payload = {"model": effective_model, "messages": messages, ...}
```

This is a backwards-compatible change — all existing callers continue working unchanged. `MealTracker` receives the same shared `OpenRouterProvider` instance (type `OpenRouterProvider`, not the abstract `LLMProvider`) and passes `model="google/gemini-3.1-pro-preview-customtools"` on each Gemini call.

---

## Nutrition Memory File

**Path:** `Settings.meal_nutrition_memory_path` (default `~/.claw/meal_nutrition_memory.md`)

**Format — one block per meal, separated by a blank line:**

```markdown
## Dal Makhani
Per 100gms: 151 kcal, 12g carbs, 7g fat, 2g sugars, 8g protein, 3g fiber, 410mg sodium, 22mg cholesterol, 380mg potassium, 3.2mg iron, 85mg calcium, 12mg vitamin C.
Source: Kagi search via nutritionix.com and USDA database (searched 2026-03-15).

## Samosa
Per 1 unit: 130 kcal, 18g carbs, 6g fat, 1g sugars, 3g protein, 2g fiber, 210mg sodium, 5mg cholesterol, 200mg potassium, 1mg iron, 20mg calcium, 2mg vitamin C.
Source: Kagi search via healthline.com (searched 2026-03-15).
```

**Format rules:**
- `##` heading uses the title-cased meal name for readability; lookup normalizes both sides.
- **Normalization:** `re.sub(r'\s+', ' ', name.lower().strip())` — lowercase, collapse internal whitespace, strip edges. "Dal  Makhani" and "dal makhani" both normalize to "dal makhani".
- Nutritional line always stores values **per base unit**:
  - `unit="gms"` → "Per 100gms" (divide each nutrient value by `portion_amount / 100`)
  - `unit="cups"` → "Per 1 cup" (divide each nutrient value by `portion_amount`)
  - `unit="units"` → "Per 1 unit" (divide each nutrient value by `portion_amount`)
  - All values formatted to 1 decimal place (e.g. `3.2mg`)
- Source line: the 1-2 sentence summary returned by the second Gemini call.
- File ends with a trailing newline.

**Concurrency:** `_MEMORY_LOCK = asyncio.Lock()` defined at **module level** in `meal_tracker.py`. All `_check_memory` and `_save_memory` calls acquire this lock. Only one `MealTracker` instance is created (in `main.py`). This serializes file access within a single process. Known limitation: running two bot instances concurrently would bypass this protection — not an issue for the current deployment.

---

## Nutrition Lookup Flow

### Step 1: Memory check (in `track()`)

1. `_check_memory(meal_name)` → returns the 3-line block or `None`
2. **Cache hit:** set `source = "memory"`. Call `_call_gemini(..., memory_entry=<block>)`. Skip Step 3.
3. **Cache miss:** set `source = "search"`. Call `_call_gemini(..., memory_entry=None)`. Proceed to Step 3.
4. `source` is determined here in `track()`, not inside `_call_gemini`. Both BQ insert and memory update receive the correct `source` value.

### Step 2: Inside `_call_gemini`

`_call_gemini` calls `self._llm.generate(..., model="google/gemini-3.1-pro-preview-customtools")` on every call (both cache-hit scaling calls and cache-miss search calls).

**Cache hit (memory_entry is not None) — single-turn:**
- System prompt: *"Scale the nutritional data below to the requested portion. Return only a JSON object with exactly these fields (null if a value cannot be computed): kcal, carbs_g, fats_g, sugars_g, proteins_g, fiber_g, sodium_mg, cholesterol_mg, potassium_mg, iron_mg, calcium_mg, vitamin_c_mg."*
- User message: *"Nutritional data: {memory_entry}\nTarget portion: {portion_amount} {portion_unit} of {meal_name}."*
- No tools.

**Cache miss (memory_entry is None) — multi-turn search loop:**

System prompt:
```
Today is {date}. Location: {location}.
You are a nutrition assistant. Find accurate nutritional information for the meal specified.
You may use the kagi_search tool to look up information. You may search at most 3 times.
Return your final answer as a JSON object with exactly these fields (null if unknown):
kcal, carbs_g, fats_g, sugars_g, proteins_g, fiber_g, sodium_mg, cholesterol_mg,
potassium_mg, iron_mg, calcium_mg, vitamin_c_mg
All values must be for the exact portion the user specifies (not per 100g).

Previously cached meals (for reference, do not re-search these):
{memory file contents, or "No meals cached yet."}
```

User message: `"What is the nutritional breakdown of {meal_name}, {portion_amount} {portion_unit}?"`

Tool spec exposed to Gemini:
```json
{
  "type": "function",
  "function": {
    "name": "kagi_search",
    "description": "Search the web for nutritional information.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"}
      },
      "required": ["query"]
    }
  }
}
```

Note: `KagiSearchTool` has `name = "web_search"` internally. When Gemini returns a tool call named `"kagi_search"`, `_call_gemini` maps it to `KagiSearchTool.run(query=...)` (passing `query` as a keyword argument matching `KagiSearchTool`'s `**kwargs` interface).

**Tool-call loop:**
- Track `search_count = 0`
- On each turn: if `LLMResponse.tool_calls` contains a `kagi_search` call and `search_count < 3`:
  - Execute `KagiSearchTool.run(query=tool_call.arguments["query"])`
  - Append tool result to message history
  - Increment `search_count`
  - If `search_count == 3`: send next turn **without** the tool spec (forces Gemini to answer)
  - Call `generate()` again
- If response has no tool calls (text reply): exit loop, parse JSON

**Response parsing:** Strip markdown code fences (` ```json ... ``` `) if present. Parse JSON. On failure: log warning, return all-null dict.

### Step 3: Memory update (cache miss only, in `track()`)

Second single-turn call to Gemini (same model):
> *"In 1-2 sentences, summarize the source(s) you used to find nutritional information for {meal_name}. Do not include any nutritional values."*

Then call `_save_memory(meal_name, nutrients, base_unit, source_summary)` which appends the entry to the memory file (acquiring `_MEMORY_LOCK`).

---

## BigQuery Schema

**Gate:** if `Settings.bigquery_project_id` is empty or blank, all BigQuery operations in `MealTracker` are skipped silently (same pattern as `PriceTrackerTool`).

**Dataset:** `Settings.health_bigquery_dataset_id` (default `"health"`) — a new field, separate from the existing `Settings.bigquery_dataset_id` used by `PriceTrackerTool` for shopping receipts.

**Table:** `Settings.health_bigquery_table_id` (default `"meals"`) — a new field, separate from `Settings.bigquery_table_id`.

Table created automatically on first insert via `_ensure_table()` (same pattern as `PriceTrackerTool`).

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `meal_name` | STRING | NO | Original input as typed |
| `portion_amount` | FLOAT64 | NO | |
| `portion_unit` | STRING | NO | `"gms"`, `"cups"`, or `"units"` |
| `kcal` | FLOAT64 | YES | |
| `carbs_g` | FLOAT64 | YES | |
| `fats_g` | FLOAT64 | YES | |
| `sugars_g` | FLOAT64 | YES | |
| `proteins_g` | FLOAT64 | YES | |
| `fiber_g` | FLOAT64 | YES | |
| `sodium_mg` | FLOAT64 | YES | |
| `cholesterol_mg` | FLOAT64 | YES | |
| `potassium_mg` | FLOAT64 | YES | |
| `iron_mg` | FLOAT64 | YES | |
| `calcium_mg` | FLOAT64 | YES | |
| `vitamin_c_mg` | FLOAT64 | YES | |
| `source` | STRING | NO | `"memory"` (cache hit) or `"search"` (cache miss) |
| `logged_at` | TIMESTAMP | NO | UTC, set to `datetime.now(timezone.utc)` on insert |

**`get_summary(date)` query — parameterized (no string interpolation of user values):**
```python
query = """
    SELECT *
    FROM `{project}.{dataset}.{table}`
    WHERE DATE(logged_at, @timezone) = @date
    ORDER BY logged_at ASC
""".format(project=..., dataset=..., table=...)  # server-side config values only

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("timezone", "STRING", settings.meal_summary_timezone),
        bigquery.ScalarQueryParameter("date", "DATE", date.isoformat()),
    ]
)
```

`date` is a `datetime.date` object (already validated before reaching this method). `meal_summary_timezone` is a server-side config value. Both are passed as typed BigQuery parameters, not interpolated into SQL.

---

## Daily Summary

### Timezone

Configured explicitly as an IANA identifier via `Settings.meal_summary_timezone` (e.g. `"Europe/Berlin"`, `"Asia/Kolkata"`). Default: `"UTC"`. Set once in `config.yaml` — no inference from location strings.

### Startup scheduling (in `main.py`)

```python
tz = zoneinfo.ZoneInfo(settings.meal_summary_timezone)
now_local = datetime.now(tz)
today_9pm_local = now_local.replace(hour=21, minute=0, second=0, microsecond=0)
if now_local >= today_9pm_local:
    target = today_9pm_local + timedelta(days=1)
else:
    target = today_9pm_local
target_utc = target.astimezone(timezone.utc)

# Query SQLite for existing pending/running task
existing = await db.get_scheduled_meal_summary_task()  # new DB method
if not existing:
    await db.create_scheduled_task(
        group_id=settings.telegram_owner_id,
        prompt="__meal_summary__ Generate today's meal nutrition summary.",
        run_at=target_utc,
    )
```

`db.get_scheduled_meal_summary_task()` is a new `Database` method that queries:
```sql
SELECT * FROM scheduled_tasks
WHERE prompt LIKE '%__meal_summary__%'
AND status IN ('pending', 'running')
AND run_at >= {local_midnight_utc}
AND run_at < {next_local_midnight_utc}
LIMIT 1
```

If status is stuck at `'running'` (crash during execution), the existing task is found and no duplicate is created. The stuck task will be picked up and retried by the scheduler normally.

### After a summary fires

In `main.py`'s `handle_scheduled_prompt()` function (the callback passed to `TaskScheduler`), after calling `runtime.handle_message()` and sending the Telegram reply, check if `task.prompt` contains `"__meal_summary__"`. If yes, schedule the next task for tomorrow's 9PM (same logic as startup, always +1 day from `task.run_at`).

### Summary generation

**9PM scheduled path:**
1. Scheduler routes `"__meal_summary__ Generate today's meal nutrition summary."` through `AgentRuntime`
2. `AgentRuntime` calls the regular OpenRouter model (not Gemini) with `GetMealSummaryTool` available
3. LLM calls `get_meal_summary` tool with today's date → tool returns formatted meal data
4. LLM generates analysis + 3 tips and returns the summary

**`/tm summary [date]` path:**
1. `_handle_trackmeal()` in `commands.py` calls `await meal_tracker.get_summary(date)` directly
2. If result is empty → reply without LLM
3. Otherwise pass raw meal list to regular OpenRouter model with analysis prompt
4. Return formatted summary

Both paths produce identical output format.

**Analysis prompt (passed to regular OpenRouter model):**
```
Here are the meals logged for {readable_date}:
{meal list: "• {meal_name} {portion_amount}{portion_unit} — {kcal} kcal" per line}

Total: {sum_kcal} kcal out of a 2300 kcal daily goal.

Nutrient thresholds to flag: fiber <25g (low), sodium >2000mg (high), protein <50g (low).

Please provide:
1. A brief assessment of today's nutrient intake
2. Three concrete, actionable tips for a better meal plan tomorrow based on today's gaps

Format for Telegram: use bullet points, no markdown tables.
```

---

## Architecture

### New files

```
assistant/tools/meal_tracker.py
assistant/tools/get_meal_summary_tool.py
tests/test_meal_tracker.py
tests/test_get_meal_summary_tool.py
```

### `MealTracker` class (`meal_tracker.py`)

```python
_MEMORY_LOCK = asyncio.Lock()  # module-level; single MealTracker instance assumed

class MealTracker:
    def __init__(
        self,
        config: Settings,
        llm: OpenRouterProvider,          # concrete type; model override used per call
        kagi: KagiSearchTool,
        bq_client: bigquery.Client | None,  # None when bigquery_project_id is blank
    ): ...

    async def track(self, meal_name: str, portion_amount: float, portion_unit: str) -> str: ...
    async def _check_memory(self, meal_name: str) -> str | None: ...
    async def _save_memory(
        self, meal_name: str, nutrients: dict, base_unit: str, source_summary: str
    ): ...
    async def _call_gemini(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        memory_entry: str | None,
    ) -> dict: ...  # returns nutrients dict with 12 keys; source determined by caller
    async def _insert_bigquery(
        self,
        meal_name: str,
        portion_amount: float,
        portion_unit: str,
        nutrients: dict,
        source: str,           # "memory" or "search", passed in by track()
    ): ...
    async def get_summary(self, date: datetime.date) -> list[dict]: ...
```

### `GetMealSummaryTool` (`get_meal_summary_tool.py`)

```python
name = "get_meal_summary"
description = "Fetch all meals logged for a specific date to generate a nutrition summary."
parameters_schema = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "The date to fetch meals for, in YYYY-MM-DD format."
        }
    },
    "required": ["date"]
}

def run(self, date: str, **kwargs) -> str:
    parsed = datetime.date.fromisoformat(date)
    meals = asyncio.get_event_loop().run_until_complete(self._tracker.get_summary(parsed))
    if not meals:
        return f"No meals logged for {date}."
    return _format_meals_for_llm(meals)  # formats as readable text for LLM to analyse
```

Constructor receives a `MealTracker` instance (injected in `main.py`).

### Changes to existing files

| File | Change |
|---|---|
| `assistant/llm/openrouter.py` | Add `model: str \| None = None` parameter to `generate()`; use `model or self._settings.openrouter_model` |
| `assistant/commands.py` | Add `"tm": "trackmeal"` to `ALIASES`; add `meal_tracker: MealTracker \| None = None` to `CommandDispatcher.__init__`; add `_handle_trackmeal()` |
| `assistant/main.py` | Instantiate `MealTracker`; pass to `CommandDispatcher`; register `GetMealSummaryTool`; schedule 9PM task on startup; reschedule after summary fires in `handle_scheduled_prompt()` |
| `assistant/db.py` | Add `get_scheduled_meal_summary_task(run_at_start, run_at_end)` method |
| `assistant/config.py` | Add four new `Settings` fields: `meal_nutrition_memory_path`, `health_bigquery_dataset_id`, `health_bigquery_table_id`, `meal_summary_timezone` |
| `commands_help.json` | Add `/tm` entry |
| `README.md` | Add `/tm` command section |

**`commands_help.json` entry:**
```json
"/tm": "Track a meal and get nutritional info. Usage: /tm <meal> <portion> (e.g. /tm dal makhani 200gms, /tm samosa 2). Use /tm summary [mar 15] for a daily nutrition report."
```

**Config additions under `claw:` in `~/.config/tejas/config.yaml`:**
```yaml
meal_nutrition_memory_path: ~/.claw/meal_nutrition_memory.md
health_bigquery_dataset_id: health
health_bigquery_table_id: meals
meal_summary_timezone: Europe/Berlin   # IANA timezone identifier
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Gemini returns unparseable JSON | All-null nutrients; log warning; still insert to BQ; reply: *"Logged {meal} but nutritional data could not be parsed."* |
| BQ insert fails | Log error; reply: *"Meal saved to memory but could not be written to the database."* |
| `bigquery_project_id` empty | Skip all BQ operations silently |
| Memory file missing on read | Treat as empty; no error |
| Memory file missing on write | Create file and parent directory before writing |
| No meals for summary date | Reply: *"No meals logged for {readable date}."* without calling LLM |
| Invalid portion | Reply with logging usage hint |
| Empty meal name after parsing | Reply with logging usage hint |
| Invalid summary date (day out of range) | Reply with specific error |
| Malformed summary args | Reply with summary usage hint |

---

## Testing

**Validation boundary:** All input parsing and validation occurs in `_handle_trackmeal()` in `commands.py` before calling `MealTracker.track()`. `track()` receives pre-validated, typed arguments.

**`test_meal_tracker.py`:**
- `test_track_cache_miss_calls_gemini_with_search`
- `test_track_cache_hit_calls_gemini_with_scaling_only`
- `test_track_cache_hit_skips_kagi_search`
- `test_track_cache_hit_skips_memory_save`
- `test_track_cache_hit_different_portion_scales_values`
- `test_track_inserts_to_bigquery_source_memory_on_cache_hit`
- `test_track_inserts_to_bigquery_source_search_on_cache_miss`
- `test_track_gemini_null_nutrients_still_inserts_row`
- `test_track_saves_memory_on_cache_miss`
- `test_get_summary_returns_meals_for_date`
- `test_get_summary_empty_date_returns_empty_list`

**`test_get_meal_summary_tool.py`:**
- `test_tool_name_description_schema_are_correct`
- `test_run_parses_date_and_calls_get_summary`
- `test_run_returns_no_meals_message_when_empty`

**`tests/test_commands.py` additions:**
- `test_trackmeal_parses_gms_portion`
- `test_trackmeal_parses_g_normalizes_to_gms`
- `test_trackmeal_parses_cups_portion`
- `test_trackmeal_parses_plain_number_as_units`
- `test_trackmeal_missing_portion_returns_usage_hint`
- `test_trackmeal_empty_meal_name_returns_usage_hint`
- `test_trackmeal_summary_checked_before_portion_parsing`
- `test_trackmeal_summary_today`
- `test_trackmeal_summary_specific_date`
- `test_trackmeal_summary_invalid_day_for_month_returns_error`
- `test_trackmeal_summary_month_without_day_returns_usage_hint`
- `test_trackmeal_alias_tm`
