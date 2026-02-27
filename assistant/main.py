"""Application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from assistant.agent_runtime import AgentRuntime
from assistant.config import load_settings
from assistant.db import Database
from assistant.llm.openrouter import OpenRouterProvider
from assistant.models import Message
from assistant.scheduler import TaskScheduler
from assistant.signal_adapter import SignalAdapter
from assistant.tools.notes_tool import ListNotesTool, WriteNoteTool
from assistant.tools.registry import ToolRegistry
from assistant.tools.time_tool import GetCurrentTimeTool, WebSearchTool

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


async def run() -> None:
    """Initialize app layers and start processing loop."""

    settings = load_settings()

    db = Database(settings.database_path)
    db.initialize()

    provider = OpenRouterProvider(settings)
    tools = ToolRegistry(db)
    tools.register(GetCurrentTimeTool())
    tools.register(WebSearchTool())
    tools.register(WriteNoteTool(db))
    tools.register(ListNotesTool(db))

    runtime = AgentRuntime(
        db=db,
        llm=provider,
        tool_registry=tools,
        memory_window_messages=settings.memory_window_messages,
        summary_trigger_messages=settings.memory_summary_trigger_messages,
        request_timeout_seconds=settings.request_timeout_seconds,
    )

    signal_adapter = SignalAdapter(
        signal_cli_path=settings.signal_cli_path,
        account=settings.signal_account,
        poll_interval_seconds=settings.signal_poll_interval_seconds,
        owner_number=settings.signal_owner_number,
    )

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
        await signal_adapter.send_message(group_id, response, is_group=True)

    scheduler = TaskScheduler(db=db, handler=handle_scheduled_prompt)

    scheduler_task = asyncio.create_task(scheduler.run_forever(), name="task-scheduler")

    try:
        async for message in signal_adapter.poll_messages():
            reply = await runtime.handle_message(message)
            await signal_adapter.send_message(message.group_id, reply, is_group=message.is_group)
    except asyncio.CancelledError:
        raise
    finally:
        scheduler.stop()
        scheduler_task.cancel()
        LOGGER.info("Assistant shutdown complete")


def main() -> None:
    """Synchronous wrapper for asyncio entrypoint."""

    asyncio.run(run())


if __name__ == "__main__":
    main()
