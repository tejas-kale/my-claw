# my-claw

Personal single-user Signal assistant inspired by NanoClaw, implemented in Python with clean modular boundaries.

## Architecture

```text
Signal Adapter Layer
    ↓
Core Agent Runtime
    ↓
LLM Provider Interface
    ↓
OpenRouter Adapter

Persistence Layer (SQLite)
Task Scheduler
Tool Registry
```

The runtime is vendor-neutral at the architecture layer: `assistant.llm.base.LLMProvider` isolates model invocation from the rest of the system.

## Project layout

```text
assistant/
  main.py
  config.py
  db.py
  models.py
  signal_adapter.py
  agent_runtime.py
  scheduler.py
  llm/
    base.py
    openrouter.py
  tools/
    base.py
    registry.py
    time_tool.py
    notes_tool.py
tests/
requirements.txt
.env.example
```

## Prerequisites

- Python 3.11+
- `signal-cli` installed and available on PATH

### Install signal-cli

Follow the official install guide for your platform:
- Linux/macOS package options: https://github.com/AsamK/signal-cli

## Register a Signal number

Example flow:

```bash
signal-cli -a +15555550123 register
signal-cli -a +15555550123 verify <code-from-sms-or-voice>
```

Use that number as `SIGNAL_ACCOUNT`.

## Configure OpenRouter

1. Create an OpenRouter API key.
2. Copy `.env.example` to `.env`.
3. Set:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL` (no model hardcoded in code)
   - optionally `OPENROUTER_BASE_URL`

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m assistant.main
```

## Add tools

1. Create a class implementing `assistant.tools.base.Tool`.
2. Define `name`, `description`, and JSON `parameters_schema`.
3. Implement `async def run(...)`.
4. Register it in `assistant/main.py` via `ToolRegistry.register(...)`.

Only explicitly registered tools can run.

## Testing

```bash
pytest -q
```
