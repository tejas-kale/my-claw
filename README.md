# my-claw

A personal AI assistant that lives in your Telegram inbox. Send it a message (or an image, PDF, or receipt) from your phone; it responds through an LLM via OpenRouter, calls tools, remembers things across conversations, searches the web, generates NotebookLM podcasts, and tracks grocery prices in BigQuery — all without leaving Telegram.

## Table of contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Telegram setup (one-time)](#telegram-setup-one-time)
- [Configuration](#configuration)
- [Running](#running)
- [Features](#features)
  - [Conversation memory](#conversation-memory)
  - [Notes](#notes)
  - [Web search](#web-search)
  - [Task scheduler](#task-scheduler)
  - [Podcast generation](#podcast-generation)
  - [Grocery price tracker](#grocery-price-tracker)
- [Commands reference](#commands-reference)
- [Adding a tool](#adding-a-tool)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
Telegram (phone)
    │  Bot API long-polling (httpx)
    ▼
TelegramAdapter        ← polls getUpdates, normalises Message objects, sends replies
    │
    ├── CommandDispatcher  ← @-prefixed commands bypass the LLM entirely
    │       ├── @podcast      → PodcastTool (NotebookLM background task)
    │       ├── @websearch    → Kagi/DDG multi-query pipeline + Jina page fetch
    │       ├── @trackprice   → PriceTrackerTool (vision LLM → BigQuery)
    │       └── @clear        → wipe conversation history for this group
    │
    └── AgentRuntime          ← memory management, two-pass LLM call, tool dispatch
            ├── LLMProvider   ← OpenRouter (OpenAI-compatible chat completions)
            ├── ToolRegistry  ← validates args, executes tools, logs audit trail
            ├── Database      ← SQLite: messages, summaries, notes, scheduled tasks
            └── TaskScheduler ← background asyncio loop for scheduled prompts

Tools available to the LLM (called autonomously, not via @commands):
    web_search        ← Kagi search (requires approval gate)
    read_url          ← Jina Reader: fetches a URL and returns clean markdown
    write_note        ← save a short note to SQLite
    list_notes        ← read back saved notes from SQLite
    get_current_time  ← current UTC time
    create_podcast    ← start NotebookLM podcast generation (background task)
```

---

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | 3.12+ | `brew install python@3.12` |
| uv | any | `brew install uv` |
| NotebookLM CLI | any | `uv tool install notebooklm-mcp-cli` (optional, for `@podcast`) |
| Google Cloud SDK | any | `brew install google-cloud-sdk` (optional, for `@trackprice`) |

---

## Installation

```bash
git clone https://github.com/you/my-claw
cd my-claw

# Create virtual environment and install dependencies
uv sync

# Install dev dependencies too (for tests)
uv sync --extra dev
```

---

## Telegram setup (one-time)

### 1. Create a bot

Message [@BotFather](https://t.me/BotFather) on Telegram:

```
/newbot
```

Follow the prompts to pick a name and username. BotFather will give you a **bot token** — save it as `TELEGRAM_BOT_TOKEN`.

### 2. Find your Telegram user ID

Message [@userinfobot](https://t.me/userinfobot) — it replies with your numeric user ID. Save it as `TELEGRAM_OWNER_ID`.

### 3. Start a conversation with your bot

Open the bot's profile and tap **Start**. This is required before the bot can message you.

### 4. Groups (optional)

Add the bot to a group. Either disable Privacy Mode for the bot via BotFather (`/mybots → Bot Settings → Group Privacy → Turn off`), or make the bot an admin, so it can see all messages — not just commands.

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then fill in your values:

```dotenv
# ── Required ──────────────────────────────────────────────────────────────────

# Your OpenRouter API key (https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-...

# Model identifier — any OpenRouter-supported model, e.g.:
#   anthropic/claude-3-5-sonnet          (recommended)
#   openai/gpt-4o
#   google/gemini-2.0-flash-thinking-exp
OPENROUTER_MODEL=anthropic/claude-3-5-sonnet

# Telegram bot token from @BotFather
TELEGRAM_BOT_TOKEN=123456789:AAF...

# Your Telegram user ID (from @userinfobot) — always allowed to send commands
TELEGRAM_OWNER_ID=123456789

# Your Kagi API key (https://kagi.com/settings?p=api) — required for web search
KAGI_API_KEY=...

# ── Optional ──────────────────────────────────────────────────────────────────

# OpenRouter base URL (change only if using a proxy)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# SQLite database file path
DATABASE_PATH=assistant.db

# Additional Telegram user IDs allowed to send commands, comma-separated
# TELEGRAM_ALLOWED_SENDER_IDS=987654321,111222333

# Long-poll timeout in seconds (how long each getUpdates call blocks)
TELEGRAM_POLL_TIMEOUT=30

# Sliding context window — how many recent messages to include in the LLM context
MEMORY_WINDOW_MESSAGES=20

# When the message count hits this number, compress history into a rolling summary
MEMORY_SUMMARY_TRIGGER_MESSAGES=40

# LLM call timeout in seconds
REQUEST_TIMEOUT_SECONDS=30

# Jina Reader API key for full-page fetches during @websearch (https://jina.ai/reader)
# Leave blank for anonymous (rate-limited) access.
# JINA_API_KEY=...

# ── BigQuery (for @trackprice) ─────────────────────────────────────────────────
# Leave BIGQUERY_PROJECT_ID unset to disable the @trackprice command.

# GCP project that owns the BigQuery dataset
# BIGQUERY_PROJECT_ID=my-gcp-project

# Dataset and table (created automatically on first use)
# BIGQUERY_DATASET_ID=economics
# BIGQUERY_TABLE_ID=german_shopping_receipts
```

---

## Running

```bash
uv run claw
```

The assistant starts long-polling Telegram. Send it a message from your phone to test.

To run in the background with logging to file:

```bash
nohup uv run claw >> ~/.my-claw/claw.log 2>&1 &
```

---

## Features

### Conversation memory

my-claw maintains per-group (or per-DM) conversation history in SQLite. The last `MEMORY_WINDOW_MESSAGES` messages are included verbatim in every LLM context. When the buffer reaches `MEMORY_SUMMARY_TRIGGER_MESSAGES`, the older portion is automatically summarised by the LLM and stored as a rolling summary — keeping context bounded while preserving long-term continuity.

---

### Notes

Quick per-group notes stored in SQLite. Good for session context:

> "Remember that the dinner is on Friday"

> "What did I ask you to remember about dinner?"

The LLM calls `write_note` / `list_notes` automatically when you ask it to remember or recall things.

---

### Web search

Web search operates in two modes: LLM-initiated (via a tool call) and command-initiated (via `@websearch`).

#### LLM-initiated search (with approval gate)

When the LLM determines a web search is needed, it proposes the query and waits for approval before executing:

```
You: What's the latest news on the EU AI Act?
Claw: I'd like to search the web to answer this. Proposed:
      - EU AI Act latest news 2025
      Reply ok to proceed.
You: ok
Claw: [summarised answer with references]
```

Reply with any of: `ok`, `yes`, `sure`, `yep`, `yeah`, `proceed`, `go`, `go ahead`, `approve`, `do it`.

#### `@websearch` command (direct, no approval gate)

```
@websearch <query>
@websearch ddg <query>
```

The command pipeline:
1. Generates 1–5 focused sub-queries via LLM
2. Runs all sub-queries in parallel against Kagi (default) or DuckDuckGo (`ddg` prefix)
3. LLM ranks the combined results by relevance and selects the top URLs
4. Fetches up to 2 of the top pages via Jina Reader (full content as clean markdown)
5. LLM synthesises a final answer with numbered references

**Kagi** (default) provides higher-quality, ad-free results and requires a paid API key. **DuckDuckGo** requires no key and is a free fallback.

Examples:

```
@websearch best Python async HTTP client 2025
@websearch ddg openrouter model pricing
```

---

### Task scheduler

Schedule a future prompt to be injected back into the conversation at a specific time:

> "Remind me in 30 minutes to take a break"

> "At 9am tomorrow, ask me what I'm working on today"

> "In 2 hours, summarise what we've discussed so far"

Scheduled tasks are stored in SQLite and survive restarts. The scheduler runs in a background asyncio task that polls every 2 seconds. When a task fires, it runs the stored prompt through the full runtime (including tools) and sends the reply to the originating group or DM.

---

### Podcast generation

Send a PDF as a Telegram attachment (or a URL in the message body), along with an `@podcast` command, and my-claw will generate a NotebookLM deep-dive audio overview and send the `.m4a` back when ready.

#### One-time setup

```bash
uv tool install notebooklm-mcp-cli
nlm login     # opens Chrome — log in with your Google account
```

#### Usage

Attach a PDF and send a caption:

```
@podcast econpod
```

Or provide a URL in the command:

```
@podcast cspod https://arxiv.org/pdf/2501.12345
@podcast ddpod https://papers.ssrn.com/sol3/papers.cfm?abstract_id=12345
```

If both a URL and an attachment are present, the URL takes precedence.

#### Podcast types

| Type | Focus |
|---|---|
| `econpod` | Planet Money style — economic story, narrative-first |
| `cspod` | CS algorithm episode — intuition, mechanics, complexity |
| `ddpod` | Deep-dive paper review — contribution, methods, results |

Focus prompts are in [`assistant/tools/podcast_tool.py`](assistant/tools/podcast_tool.py) in the `PODCAST_TYPES` dict. Edit them freely to tune the output style.

#### What happens behind the scenes

1. Verifies `nlm` is installed and on PATH
2. Creates a temporary NotebookLM notebook
3. Adds the PDF (or URL) as a source and waits for processing
4. Starts a deep-dive audio generation with the type's focus prompt
5. Polls generation status every 30 seconds (up to 60 minutes)
6. Downloads the `.m4a` when ready and sends it to the Telegram chat
7. Deletes the notebook and temp file

Generation typically takes 2–5 minutes. The assistant replies immediately with a confirmation and sends the audio file in a follow-up message when done.

---

### Grocery price tracker

Send a photo or PDF scan of a German supermarket receipt with the `@trackprice` command and my-claw will extract every line item via LLM vision, persist the data to BigQuery, and reply with a confirmation plus the 5 most recently inserted rows.

#### One-time setup

**1. Enable BigQuery in GCP and create a dataset:**

```bash
# Create dataset (table is created automatically on first insert)
bq mk --dataset my-gcp-project:economics
```

**2. Authenticate with Application Default Credentials:**

```bash
gcloud auth application-default login
```

Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` in `.env` to use a service account key.

**3. Set env vars in `.env`:**

```dotenv
BIGQUERY_PROJECT_ID=my-gcp-project
BIGQUERY_DATASET_ID=personal
BIGQUERY_TABLE_ID=german_shopping_receipts
```

If `BIGQUERY_PROJECT_ID` is not set, the `@trackprice` command is disabled (returns "not configured").

#### Usage

Attach a receipt image (JPEG, PNG) or a PDF scan and send:

```
@trackprice
```

The assistant replies with something like:

```
Saved 14 items from Rewe (2024-01-15), total: €23.45

Last 5 rows:
German                    English                   Price
------------------------------------------------------------
Vollmilch 3,5%            Whole Milk 3.5%            1.19
Bio Eier 6er              Organic Eggs 6-pack        2.49
Dinkel Brot               Spelt Bread                2.89
Hähnchenbrust             Chicken Breast             4.99
Griechischer Joghurt      Greek Yoghurt              1.79
```

#### BigQuery table schema

The table is created automatically in `<BIGQUERY_PROJECT_ID>.<BIGQUERY_DATASET_ID>.<BIGQUERY_TABLE_ID>`:

| Column | Type | Notes |
|---|---|---|
| `supermarket` | STRING | Store name extracted from receipt |
| `date` | DATE | Receipt date (YYYY-MM-DD) |
| `item_name_german` | STRING | Original German item name |
| `item_name_english` | STRING | LLM English translation |
| `price` | FLOAT64 | Per-item price in EUR |
| `total_price` | FLOAT64 | Receipt grand total, repeated per row |
| `inserted_at` | TIMESTAMP | UTC insertion timestamp |

#### How PDF receipts work

PDFs are automatically converted to an image (first page only, via PyMuPDF) before being sent to the LLM. The LLM receives a base64-encoded PNG in a vision message and returns structured JSON; no OCR library is required.

#### Querying your data

```sql
-- Monthly spend by supermarket
SELECT
    supermarket,
    FORMAT_DATE('%Y-%m', date) AS month,
    ROUND(SUM(price), 2) AS total_spend
FROM `my-gcp-project.personal.german_shopping_receipts`
GROUP BY 1, 2
ORDER BY 2 DESC, 3 DESC;

-- Most expensive items across all receipts
SELECT item_name_english, price, supermarket, date
FROM `my-gcp-project.personal.german_shopping_receipts`
ORDER BY price DESC
LIMIT 20;
```

---

## Commands reference

Commands are @-prefixed messages that bypass the LLM entirely. They are processed synchronously (except `@podcast`, which spawns a background task).

| Command | Usage | Description |
|---|---|---|
| `@podcast` | `@podcast <type> [url]` | Generate a NotebookLM podcast. Attach a PDF or provide a URL. Types: `econpod`, `cspod`, `ddpod`. |
| `@websearch` | `@websearch [ddg] <query>` | Direct web search with LLM synthesis. Omit `ddg` for Kagi; add `ddg` for DuckDuckGo. |
| `@trackprice` | `@trackprice` (with attachment) | Extract receipt items and save to BigQuery. Attach a receipt image or PDF. |
| `@clear` | `@clear` | Wipe all message history and conversation summaries for the current group. |

Unrecognised `@commands` fall through to the LLM — they are not errors.

---

## Adding a tool

1. Create `assistant/tools/my_tool.py` and subclass `Tool`:

```python
from assistant.tools.base import Tool
from typing import Any

class MyTool(Tool):
    name = "my_tool"
    description = "One-sentence description the LLM sees."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string"},
            "input":    {"type": "string", "description": "The thing to process."},
        },
        "required": ["group_id", "input"],
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": kwargs["input"].upper()}
```

2. Register it in `assistant/main.py`:

```python
from assistant.tools.my_tool import MyTool
# ...
tools.register(MyTool())
```

`group_id` and `is_group` are auto-injected by the runtime — include them in `required` to receive them in `run()`.

For tools that need to send messages back to Telegram (e.g. from a background task), accept a `signal_adapter` parameter in `__init__` — any object with a `send_message(chat_id, text, is_group, attachment_path)` coroutine works. See `PodcastTool` as the reference example.

Command-only tools (not callable by the LLM) do not need to subclass `Tool`. See `PriceTrackerTool` as an example: it is instantiated directly and called by `CommandDispatcher._handle_trackprice()`.

---

## Testing

```bash
uv run pytest -q                                        # all tests
uv run pytest tests/test_price_tracker_tool.py -v      # price tracker only
uv run pytest tests/test_podcast_tool.py -v            # podcast tool only
```

All external dependencies (BigQuery, LLM, Telegram API, fitz, nlm) are mocked in tests — no credentials or external services needed to run the suite.

---

## Troubleshooting

**Bot doesn't receive messages in a group**

Make sure Privacy Mode is disabled for the bot (via BotFather: `/mybots → Bot Settings → Group Privacy → Turn off`), or promote the bot to admin in the group.

**Bot receives stale messages after a restart**

On startup, my-claw drains old updates by calling `getUpdates?offset=-1` before entering the main loop. Messages that arrived while the process was down are intentionally discarded.

**LLM calls time out**

Increase `REQUEST_TIMEOUT_SECONDS` in `.env`. Some models (especially large reasoning models) can take 20–60 seconds on first token.

**`nlm` not found when triggering `@podcast`**

Install the NotebookLM CLI: `uv tool install notebooklm-mcp-cli`, then run `nlm login`. The `nlm` binary must be on your PATH.

**NotebookLM authentication expires**

Run `nlm login` again. Cookies last approximately 2–4 weeks.

**Podcast generation times out**

NotebookLM deep-dive generation usually takes 2–5 minutes but can take longer for large PDFs. The assistant polls for up to 60 minutes. If it consistently times out, check NotebookLM's status or try a shorter source.

**`@trackprice` returns "not configured"**

Set `BIGQUERY_PROJECT_ID` in `.env`. The command is disabled when this variable is absent.

**`@trackprice` fails with a credentials error**

Run `gcloud auth application-default login`, or set `GOOGLE_APPLICATION_CREDENTIALS` to the path of a service account JSON key that has `bigquery.dataEditor` and `bigquery.jobUser` roles on the target project.

**`@trackprice` returns "LLM returned invalid JSON"**

The vision model failed to extract structured data from the receipt image. This can happen with very blurry, heavily skewed, or partially cropped receipts. Try a clearer photo taken flat-on under good lighting.

**BigQuery table not found error**

The table is created automatically on the first successful insert. If you see this error repeatedly, check that the dataset exists and that your credentials have `bigquery.tables.create` permission on it.

**Kagi search returns an error about API balance**

Your Kagi API account has run out of credits. Top up at https://kagi.com/settings?p=billing, or switch to DuckDuckGo with `@websearch ddg <query>`.

**Assistant repeats itself or loses context**

Lower `MEMORY_WINDOW_MESSAGES` or `MEMORY_SUMMARY_TRIGGER_MESSAGES` if the context is overflowing the model's window. Increase them if the assistant forgets things too quickly.
