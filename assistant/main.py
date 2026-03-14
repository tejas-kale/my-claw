"""Application entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from assistant.agent_runtime import AgentRuntime
from assistant.commands import CommandDispatcher
from assistant.config import allowed_telegram_senders, load_settings
from assistant.db import Database
from assistant.llm.openrouter import OpenRouterProvider
from assistant.models import Message
from assistant.scheduler import TaskScheduler
from assistant.telegram_adapter import TelegramAdapter
from assistant.tools.ddg_search_tool import DdgSearchTool
from assistant.tools.magazine_tool import MagazineTool
from assistant.tools.notes_tool import ListNotesTool, WriteNoteTool
from assistant.tools.podcast_tool import PodcastTool
from assistant.tools.price_tracker_tool import PriceTrackerTool
from assistant.tools.read_url_tool import ReadUrlTool
from assistant.tools.registry import ToolRegistry
from assistant.tools.time_tool import GetCurrentTimeTool
from assistant.tools.citation_tracker_tool import CitationTrackerTool
from assistant.tools.web_search_tool import KagiSearchTool

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


async def run() -> None:
    """Initialize app layers and start processing loop."""

    settings = load_settings()

    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

    db = Database(settings.database_path)
    db.initialize()

    provider = OpenRouterProvider(settings)
    tools = ToolRegistry(db)
    tools.register(GetCurrentTimeTool())
    tools.register(KagiSearchTool(api_key=settings.kagi_api_key))
    tools.register(ReadUrlTool(api_key=settings.jina_api_key))
    tools.register(WriteNoteTool(db))
    tools.register(ListNotesTool(db))

    telegram_adapter = TelegramAdapter(
        bot_token=settings.telegram_bot_token,
        poll_timeout=settings.telegram_poll_timeout,
        allowed_sender_ids=allowed_telegram_senders(settings),
    )

    podcast_tool = PodcastTool(signal_adapter=telegram_adapter, llm=provider)
    tools.register(podcast_tool)

    magazine_tool = MagazineTool(signal_adapter=telegram_adapter)

    price_tracker_tool: PriceTrackerTool | None = None
    if settings.bigquery_project_id:
        price_tracker_tool = PriceTrackerTool(
            llm=provider,
            bq_project=settings.bigquery_project_id,
            bq_dataset=settings.bigquery_dataset_id,
            bq_table=settings.bigquery_table_id,
        )

    command_dispatcher = CommandDispatcher(
        podcast_tool=podcast_tool,
        kagi_search_tool=KagiSearchTool(api_key=settings.kagi_api_key),
        ddg_search_tool=DdgSearchTool(),
        read_url_tool=ReadUrlTool(api_key=settings.jina_api_key),
        llm=provider,
        db=db,
        price_tracker_tool=price_tracker_tool,
        magazine_tool=magazine_tool,
        citation_tracker_tool=CitationTrackerTool(),
    )

    runtime = AgentRuntime(
        db=db,
        llm=provider,
        tool_registry=tools,
        memory_window_messages=settings.memory_window_messages,
        summary_trigger_messages=settings.memory_summary_trigger_messages,
        request_timeout_seconds=settings.request_timeout_seconds,
        command_dispatcher=command_dispatcher,
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
        await telegram_adapter.send_message(group_id, response, is_group=True)

    scheduler = TaskScheduler(db=db, handler=handle_scheduled_prompt)

    scheduler_task = asyncio.create_task(scheduler.run_forever(), name="task-scheduler")

    try:
        async for message in telegram_adapter.poll_messages():
            try:
                reply = await runtime.handle_message(message)
            except Exception:
                LOGGER.exception("Unhandled error processing message from %s", message.sender_id)
                reply = "Sorry, something went wrong on my end. Please try again."
            await telegram_adapter.send_message(message.group_id, reply, is_group=message.is_group)
            for attachment in message.attachments:
                try:
                    Path(attachment["local_path"]).unlink(missing_ok=True)
                except Exception:
                    LOGGER.warning("Failed to delete temp file %s", attachment["local_path"])
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
