# my-claw

Personal single-user Signal assistant. Sends and receives messages via `signal-cli`, processes them through an LLM via OpenRouter, and supports tools, persistent memory, and scheduled prompts.

## Architecture

```
Signal Adapter → AgentRuntime → LLMProvider → OpenRouter
                      ↓
              ToolRegistry / SQLite
              TaskScheduler
```

## Prerequisites

- Python 3.12+
- JRE 21+ (`brew install openjdk@25` on macOS)
- `signal-cli` on PATH

## Signal setup (one-time)

```bash
# Register
signal-cli -a +15555550123 register --captcha <captcha>
signal-cli -a +15555550123 verify <sms-code>

# Set a profile so recipients see a name
signal-cli -a +15555550123 updateProfile --given-name Claw

# Add owner as a contact (for UUID→phone resolution)
signal-cli -a +15555550123 updateContact <owner-number>
```

Get the captcha by visiting `https://signalcaptchas.org/registration/generate.html`, solving it, right-clicking "Open Signal", and copying the link.

## Configuration

Copy `.env.example` to `.env` and fill in:

```
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
SIGNAL_ACCOUNT=+15555550123
SIGNAL_OWNER_NUMBER=+19999999999
```

## Run

```bash
uv run claw
```

## Add a tool

1. Subclass `assistant.tools.base.Tool`
2. Set `name`, `description`, `parameters_schema`
3. Implement `async def run(**kwargs)`
4. Register in `assistant/main.py` via `tools.register(...)`

## Test

```bash
uv run pytest -q
```
