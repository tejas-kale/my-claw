"""Async scheduler for delayed prompts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable

from assistant.db import Database


class TaskScheduler:
    """Polls due tasks and dispatches them via callback."""

    def __init__(
        self,
        db: Database,
        handler: Callable[[str, str], Awaitable[None]],
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self._db = db
        self._handler = handler
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()

    def schedule(self, group_id: str, prompt: str, run_at: datetime) -> int:
        """Persist a task to run in the future."""

        return self._db.create_scheduled_task(group_id=group_id, prompt=prompt, run_at=run_at)

    async def run_forever(self) -> None:
        """Run scheduler loop until stop() is called."""

        while not self._stop_event.is_set():
            due_tasks = self._db.get_due_tasks(datetime.now(timezone.utc))
            for task in due_tasks:
                task_id = int(task["id"])
                try:
                    self._db.mark_task_status(task_id, "running")
                    await self._handler(task["group_id"], task["prompt"])
                    self._db.mark_task_status(task_id, "completed")
                except Exception:  # noqa: BLE001
                    self._db.mark_task_status(task_id, "failed")
            await asyncio.sleep(self._poll_interval_seconds)

    def stop(self) -> None:
        """Signal the loop to stop."""

        self._stop_event.set()
