"""Magazine chapter narration via Gemini TTS (podcaster CLI)."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.signal_adapter import SignalAdapter

LOGGER = logging.getLogger(__name__)

_PODCASTER_TIMEOUT = 1800  # 30 minutes for full chapter TTS


async def _run_podcaster(*args: str, timeout: int = _PODCASTER_TIMEOUT) -> tuple[int, str, str]:
    """Run a podcaster CLI command, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "podcaster",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", "podcaster timed out"
    return proc.returncode, stdout_bytes.decode().strip(), stderr_bytes.decode().strip()


async def _generate_and_send(
    signal_adapter: SignalAdapter,
    group_id: str,
    is_group: bool,
    epub: str,
    chapter: str,
    output_path: str,
) -> None:
    """Background task: generate chapter audio, send to Signal, then clean up."""
    try:
        rc, stdout, stderr = await _run_podcaster("create", epub, chapter, output_path)
        if rc != 0:
            LOGGER.error(
                "podcaster create failed: rc=%d stdout=%r stderr=%r", rc, stdout, stderr
            )
            await signal_adapter.send_message(
                group_id,
                f"Chapter audio generation failed: {stderr or stdout}",
                is_group=is_group,
            )
            return

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            LOGGER.error("podcaster create: output file missing or empty at %s", output_path)
            await signal_adapter.send_message(
                group_id,
                "Chapter audio generation finished but the file is missing or empty.",
                is_group=is_group,
            )
            return

        await signal_adapter.send_message(
            group_id,
            f'Chapter "{chapter}" is ready!',
            is_group=is_group,
            attachment_path=output_path,
        )
        LOGGER.info("Magazine chapter sent: epub=%r chapter=%r", epub, chapter)
    except Exception:
        LOGGER.exception("Magazine generation failed for epub=%r chapter=%r", epub, chapter)
        await signal_adapter.send_message(
            group_id,
            "Chapter audio generation failed due to an unexpected error.",
            is_group=is_group,
        )
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass


class MagazineTool:
    """Narrate magazine/book chapters via the podcaster CLI.

    Provides chapter listing and non-blocking audio generation with delivery
    to Signal when complete.
    """

    def __init__(self, signal_adapter: SignalAdapter) -> None:
        self._signal_adapter = signal_adapter

    async def list_chapters(self, epub: str) -> str:
        """Return a chapter listing for the given epub source."""
        rc, stdout, stderr = await _run_podcaster("inspect", epub, timeout=30)
        if rc != 0:
            return f"Could not list chapters: {stderr or stdout}"
        return stdout

    async def start_generation(
        self,
        group_id: str,
        is_group: bool,
        epub: str,
        chapter: str,
    ) -> str:
        """Kick off background audio generation, returning immediately."""
        output_path = os.path.join(
            tempfile.gettempdir(), f"magazine_{uuid.uuid4().hex}.mp3"
        )
        asyncio.create_task(
            _generate_and_send(
                signal_adapter=self._signal_adapter,
                group_id=group_id,
                is_group=is_group,
                epub=epub,
                chapter=chapter,
                output_path=output_path,
            ),
            name=f"magazine-{epub}-{chapter}",
        )
        return (
            f'Generating audio for chapter "{chapter}"... '
            "I'll send the MP3 when it's ready."
        )
