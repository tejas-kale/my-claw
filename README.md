# my-claw

A personal AI assistant that lives in your Signal inbox. Send it a message (or a PDF) from your phone; it responds through an LLM via OpenRouter, calls tools, remembers things across conversations, and can generate full NotebookLM podcasts and send the audio back — all without leaving Signal.

## Table of contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Signal setup (one-time)](#signal-setup-one-time)
- [Configuration](#configuration)
- [Running](#running)
- [Features](#features)
  - [Conversation memory](#conversation-memory)
  - [Notes and memory tools](#notes-and-memory-tools)
  - [Search tools](#search-tools)
  - [Task scheduler](#task-scheduler)
  - [Podcast generation](#podcast-generation)
- [Usage examples](#usage-examples)
- [Adding a tool](#adding-a-tool)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
Signal (phone)
    │  signal-cli subprocess
    ▼
SignalAdapter          ← polls receive, normalises Message objects, sends replies
    │
    ▼
AgentRuntime           ← memory management, two-pass LLM call, tool dispatch
    ├── LLMProvider    ← OpenRouter (OpenAI-compatible chat completions)
    ├── ToolRegistry   ← validates args, executes tools, logs audit trail
    ├── Database       ← SQLite: messages, summaries, notes, scheduled tasks
    └── TaskScheduler  ← background loop for scheduled prompts

PodcastTool (registered in ToolRegistry)
    └── asyncio background task → NotebookLM CLI → sends .m4a back to Signal
```

For a fully annotated code walkthrough see [docs/walkthrough.md](docs/walkthrough.md).

---

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | 3.12+ | `brew install python@3.12` |
| uv | any | `brew install uv` |
| JRE | 21+ | `brew install openjdk@21` |
| signal-cli | latest | `brew install signal-cli` |
| ripgrep | any | `brew install ripgrep` (optional, for search tool) |
| fzf | any | `brew install fzf` (optional, for fuzzy filter tool) |
| NotebookLM CLI | any | `uv tool install notebooklm-mcp-cli` (optional, for podcast tool) |

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

## Signal setup (one-time)

### 1. Register a phone number

my-claw needs its own Signal number. A spare SIM or a VoIP number (e.g. Google Voice) works fine.

```bash
# Get a captcha token by visiting:
# https://signalcaptchas.org/registration/generate.html
# Solve it, right-click "Open Signal", copy the URL — use the full URL as the captcha value.

signal-cli -a +15555550123 register --captcha "signalcaptcha://..."
signal-cli -a +15555550123 verify 123456      # SMS code sent to the number
```

### 2. Set a display name

```bash
signal-cli -a +15555550123 updateProfile --given-name "Claw" --family-name ""
```

### 3. Add yourself as a contact

This is required for UUID-to-phone-number resolution when the assistant sends direct messages:

```bash
signal-cli -a +15555550123 addContact +19999999999
```

### 4. Find your group ID (if using a group chat)

```bash
signal-cli -a +15555550123 listGroups -d
```

Copy the Base64 group ID — you will use it to test with `signal-cli send` later.

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then fill in your values:

```dotenv
# Required ─────────────────────────────────────────────────────────────────────

# Your OpenRouter API key (https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-...

# Model identifier — any OpenRouter-supported model, e.g.:
#   anthropic/claude-3-5-sonnet          (recommended)
#   openai/gpt-4o
#   google/gemini-2.0-flash-thinking-exp
OPENROUTER_MODEL=anthropic/claude-3-5-sonnet

# The Signal number registered to my-claw (E.164 format)
SIGNAL_ACCOUNT=+15555550123

# Your personal Signal number — always allowed to send commands
SIGNAL_OWNER_NUMBER=+19999999999

# Optional ──────────────────────────────────────────────────────────────────────

# OpenRouter base URL (change only if using a proxy)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Path to the signal-cli binary (default: signal-cli, assumed on PATH)
SIGNAL_CLI_PATH=signal-cli

# SQLite database file path
DATABASE_PATH=assistant.db

# Additional E.164 numbers allowed to send commands, comma-separated
# SIGNAL_ALLOWED_SENDERS=+12125551234,+14155559876

# How often (seconds) to poll Signal for new messages
SIGNAL_POLL_INTERVAL_SECONDS=2

# Sliding context window — how many messages to keep in the LLM context
MEMORY_WINDOW_MESSAGES=20

# When the message count hits this number, compress history into a summary
MEMORY_SUMMARY_TRIGGER_MESSAGES=40

# LLM call timeout in seconds
REQUEST_TIMEOUT_SECONDS=30

# Directory for markdown-file memory (daily notes + topic notes)
# MY_CLAW_MEMORY=~/.my-claw/memory
```

---

## Running

```bash
uv run claw
```

The assistant starts polling Signal every `SIGNAL_POLL_INTERVAL_SECONDS`. Send it a message from your phone to test.

To run in the background with logging to file:

```bash
nohup uv run claw >> ~/.my-claw/claw.log 2>&1 &
```

---

## Features

### Conversation memory

my-claw maintains per-group conversation history in SQLite. The last `MEMORY_WINDOW_MESSAGES` messages are included in every LLM context. When the buffer reaches `MEMORY_SUMMARY_TRIGGER_MESSAGES`, the assistant automatically summarises the conversation and stores the summary — this keeps context from growing unbounded while preserving long-term continuity.

### Notes and memory tools

The assistant can save and recall notes in two ways:

**SQLite notes** — ephemeral per-group notes stored in the database:

> "remember that the dinner is on Friday"

> "what did I ask you to remember about dinner?"

**Markdown file memory** — durable notes written to `~/.my-claw/memory/`:

- `daily/YYYY-MM-DD.md` — running timestamped daily log (append-only)
- `topics/<slug>.md` — subject-specific notes

> "save a note about the project: we decided to use Postgres"

> "read my notes on the project"

> "what's in today's log?"

The memory files are plain markdown and can be edited directly with any text editor.

### Search tools

Two search tools operate over the project directory and the memory notes folder:

**ripgrep search** — regex search across file contents:

> "search my notes for mentions of 'postgres'"

> "find all Python files that import httpx"

**fuzzy filter** — fzf-powered approximate matching on a list of strings:

> "fuzzy match 'postgr' in this list: ['postgresql', 'mysql', 'mongodb', 'redis']"

Both tools restrict paths to the project directory and `~/.my-claw/memory` — the LLM cannot access arbitrary filesystem paths.

### Task scheduler

Schedule future prompts to be injected into any conversation:

> "remind me in 30 minutes to take a break"

> "at 9am tomorrow send me a summary of what I was working on"

Scheduled tasks are stored in the database and survive restarts. The scheduler runs in a background asyncio task alongside the main poll loop.

### Podcast generation

Send a PDF (as a Signal attachment) or a URL to a PDF, along with a podcast type keyword, and my-claw will:

1. Verify `nlm` (NotebookLM CLI) is installed
2. Create a temporary NotebookLM notebook
3. Add the PDF as a source
4. Start a deep-dive audio overview with a type-specific focus prompt
5. Poll generation status every 30 seconds (up to 10 minutes)
6. Send the `.m4a` audio file back to the Signal chat
7. Delete the notebook and temp file

**Setup (one-time):**

```bash
uv tool install notebooklm-mcp-cli
nlm login     # opens Chrome — log in to your Google account
```

**Usage:**

Attach a PDF and send a caption:

```
podcast econpod
```

Or send a URL in the message body:

```
podcast cspod https://arxiv.org/pdf/2501.12345
```

**Podcast types** are configured in [assistant/tools/podcast_tool.py](assistant/tools/podcast_tool.py) in the `PODCAST_TYPES` dict. Edit the focus prompt strings there:

```python
PODCAST_TYPES: dict[str, str] = {
    "econpod": "YOUR ECONPOD PROMPT HERE",
    "cspod":   "YOUR CSPOD PROMPT HERE",
    "ddpod":   "YOUR DDPOD PROMPT HERE",
}
```

---

## Usage examples

All of these are sent as plain Signal messages to the assistant's number (or group):

```
What time is it?
```

```
Remember that I prefer dark roast coffee.
```

```
What do you know about my coffee preferences?
```

```
Remind me in 2 hours to call the dentist.
```

```
Search my notes for anything about the Q1 budget.
```

```
Save a daily note: finished the API refactor, deploying tomorrow.
```

```
Read my notes on architecture decisions.
```

```
podcast ddpod
[PDF attachment: paper.pdf]
```

```
podcast econpod https://papers.ssrn.com/sol3/papers.cfm?abstract_id=12345
```

---

## Adding a tool

1. Create a file in `assistant/tools/` and subclass `Tool`:

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

`group_id` is auto-injected by the runtime — include it in `required` and you get it for free in `run()`.

For tools that need to send messages back to Signal (e.g. from a background task), accept `signal_adapter: SignalAdapter` in `__init__` — see `PodcastTool` as the reference example.

---

## Testing

```bash
uv run pytest -q              # all tests
uv run pytest tests/test_podcast_tool.py -v   # podcast tool only
```

---

## Troubleshooting

**signal-cli fails to receive messages**

Make sure the daemon is not already running from another process, or switch to daemon mode. signal-cli only allows one active session per account.

**`signal-cli send` exits with UUID error**

Run `signal-cli -a <account> addContact <your-number>` so signal-cli can resolve your number's UUID.

**LLM calls time out**

Increase `REQUEST_TIMEOUT_SECONDS` in `.env`. Some models (especially large reasoning models) can take 20–60 seconds on first token.

**`nlm` not found when triggering podcast**

Install the NotebookLM CLI: `uv tool install notebooklm-mcp-cli`, then run `nlm login`. The `nlm` binary must be on your PATH.

**NotebookLM authentication expires**

Run `nlm login` again. Cookies last approximately 2–4 weeks.

**Podcast generation times out**

NotebookLM deep-dive generation usually takes 2–5 minutes but can take longer for large PDFs. The assistant polls for up to 10 minutes. If it consistently times out, check NotebookLM's status or try a shorter PDF.

**Assistant repeats itself or loses context**

Lower `MEMORY_WINDOW_MESSAGES` or `MEMORY_SUMMARY_TRIGGER_MESSAGES` if the context is overflowing the model's window. Increase them if the assistant forgets things too quickly.
