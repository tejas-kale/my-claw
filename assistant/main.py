"""Application entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
import zoneinfo
from datetime import datetime, timedelta, timezone

from assistant.agent_runtime import AgentRuntime
from assistant.commands import CommandDispatcher
from assistant.config import Settings, allowed_telegram_senders, load_settings
from assistant.db import Database
from assistant.llm.openrouter import OpenRouterProvider
from assistant.models import Message
from assistant.scheduler import TaskScheduler
from assistant.telegram_adapter import TelegramAdapter
from assistant.tools.ddg_search_tool import DdgSearchTool
from assistant.tools.get_meal_summary_tool import GetMealSummaryTool
from assistant.tools.magazine_tool import MagazineTool
from assistant.tools.meal_tracker import MealTracker
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

_MEAL_SUMMARY_PROMPT = (
    "__meal_summary__ Generate today's meal nutrition summary. "
    "Flag any nutrients that are low or high (fiber <25g, sodium >2000mg, protein <50g). "
    "End with 3 concrete nutritional tips for tomorrow."
)


async def _fetch_location() -> str:
    """Return a city/region/country string from IP geolocation, or 'unknown'."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            data = (await client.get("https://ipinfo.io/json")).json()
        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country", "")
        return ", ".join(part for part in (city, region, country) if part) or "unknown"
    except Exception:
        LOGGER.warning("Could not determine location; defaulting to unknown")
        return "unknown"


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
        prompt=_MEAL_SUMMARY_PROMPT,
        run_at=target_utc,
    )
    LOGGER.info("Scheduled meal summary for %s UTC", target_utc.isoformat())


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
            prompt=_MEAL_SUMMARY_PROMPT,
            run_at=target_utc,
        )
        LOGGER.info("Rescheduled meal summary for %s UTC", target_utc.isoformat())


async def run() -> None:
    """Initialize app layers and start processing loop."""

    settings = load_settings()

    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

    db = Database(settings.database_path)
    db.initialize()
    _schedule_meal_summary_if_needed(db, settings)

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
            bq_dataset=settings.price_bigquery_dataset_id,
            bq_table=settings.price_bigquery_table_id,
        )

    if settings.bigquery_project_id:
        from google.cloud import bigquery as _bq  # type: ignore[import-untyped]
        _bq_client = _bq.Client(project=settings.bigquery_project_id)
    else:
        _bq_client = None

    # MealTracker gets its own KagiSearchTool instance (same API key, separate object).
    meal_tracker = MealTracker(
        config=settings,
        llm=provider,
        kagi=KagiSearchTool(api_key=settings.kagi_api_key),
        bq_client=_bq_client,
    )
    meal_summary_tool = GetMealSummaryTool(tracker=meal_tracker)
    tools.register(meal_summary_tool)

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
        meal_tracker=meal_tracker,
    )

    location = await _fetch_location()
    LOGGER.info("Location: %s", location)

    runtime = AgentRuntime(
        db=db,
        llm=provider,
        tool_registry=tools,
        memory_window_messages=settings.memory_window_messages,
        summary_trigger_messages=settings.memory_summary_trigger_messages,
        request_timeout_seconds=settings.request_timeout_seconds,
        command_dispatcher=command_dispatcher,
        location=location,
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
        # Reschedule daily meal summary for next day after it fires.
        if "__meal_summary__" in prompt:
            _reschedule_meal_summary(db, settings)

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
