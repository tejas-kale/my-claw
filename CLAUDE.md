# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the assistant
python -m assistant.main

# Run all tests
pytest -q

# Run a single test
pytest tests/test_foo.py::test_function_name -q
```

Dependencies are managed with `uv` — do not run `uv` commands directly; ask Tejas to run them.

## Architecture

This is a personal Signal assistant with a clean layered architecture:

```
Signal Adapter → AgentRuntime → LLMProvider → OpenRouter
                      ↓
              ToolRegistry / SQLite (Database)
              TaskScheduler
```

**Key layers:**

- **`assistant/signal_adapter.py`** — wraps `signal-cli` subprocess; polls for inbound messages and sends replies
- **`assistant/agent_runtime.py`** — core loop: loads message history from DB, calls LLM, dispatches tool calls, saves messages. Per-group isolation. Auto-summarizes history when it exceeds `summary_trigger_messages`.
- **`assistant/llm/base.py`** — `LLMProvider` ABC. The runtime depends on this interface only; swap providers by implementing it.
- **`assistant/llm/openrouter.py`** — current implementation using OpenRouter API
- **`assistant/tools/base.py`** — `Tool` ABC with `name`, `description`, `parameters_schema`, and `async run(**kwargs)`
- **`assistant/tools/registry.py`** — `ToolRegistry` holds registered tools; `list_tool_specs()` feeds tool definitions to LLM; `execute()` dispatches calls
- **`assistant/db.py`** — SQLite persistence for messages, summaries, notes, and scheduled tasks
- **`assistant/scheduler.py`** — `TaskScheduler` runs scheduled prompts in a background loop, injecting them as synthetic messages

**Adding a tool:**
1. Subclass `assistant.tools.base.Tool`
2. Set `name`, `description`, `parameters_schema`
3. Implement `async def run(**kwargs)`
4. Register in `assistant/main.py` via `tools.register(...)`

Only explicitly registered tools are available to the model.

## Configuration

Settings are loaded from `.env` via `pydantic-settings`. Required vars:
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `SIGNAL_ACCOUNT`

See `.env.example` for the full list. Key tunables: `MEMORY_WINDOW_MESSAGES`, `MEMORY_SUMMARY_TRIGGER_MESSAGES`, `REQUEST_TIMEOUT_SECONDS`.

Runtime behavior is also governed by `openclaw.config.json` (compaction, context pruning, memory paths).

## Workspace Memory Discipline

This repo doubles as a workspace for Claude Code agents operating within it. Boot sequence from `AGENTS.md`:

1. Read `USER.md`, `learnings/LEARNINGS.md`, today's `memory/YYYY-MM-DD.md`, `MEMORY.md`, `PROTOCOL_COST_EFFICIENCY.md`
2. Print: `LOADED: USER | LEARNINGS | DAILY | MEMORY | PROTOCOL`

After tasks: log decisions to `memory/YYYY-MM-DD.md`. Append mistakes as one-liners to `learnings/LEARNINGS.md`. Do not write to `MEMORY.md` during active tasks — only during periodic reviews. Verbose reference material lives in `docs/`.
