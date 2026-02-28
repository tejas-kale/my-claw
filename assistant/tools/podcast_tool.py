"""Podcast generation tool via NotebookLM CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from assistant.tools.base import Tool

if TYPE_CHECKING:
    from assistant.signal_adapter import SignalAdapter

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Podcast type → focus prompt mapping.
# Fill in the prompt strings for each type below.
# ---------------------------------------------------------------------------
PODCAST_TYPES: dict[str, str] = {
    "econpod": "YOUR ECONPOD PROMPT HERE",
    "cspod": "YOUR CSPOD PROMPT HERE",
    "ddpod": "YOUR DDPOD PROMPT HERE",
}

_NLM_TIMEOUT = 60  # seconds for any single nlm CLI call
_POLL_INTERVAL = 30  # seconds between studio status polls
_MAX_POLLS = 20  # 20 × 30 s = 10 minutes total


async def _run_nlm(*args: str, timeout: int = _NLM_TIMEOUT) -> tuple[int, str, str]:
    """Run an nlm CLI command, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "nlm",
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
        return -1, "", "nlm command timed out"
    return proc.returncode, stdout_bytes.decode().strip(), stderr_bytes.decode().strip()


def _parse_notebook_id(stdout: str) -> str | None:
    """Extract notebook ID from `nlm notebook create --json` output."""
    try:
        data = json.loads(stdout)
        # Handles {"id": "..."} or [{"id": "..."}] shapes.
        if isinstance(data, list):
            data = data[0]
        return str(data.get("id") or data.get("notebook_id") or "")
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass
    # Fallback: if --quiet was also used the line might just be the raw ID.
    line = stdout.splitlines()[0].strip() if stdout else ""
    return line or None


def _parse_artifact_id(stdout: str) -> str | None:
    """Extract artifact ID from `nlm audio create --json` output."""
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            data = data[0]
        for key in ("id", "artifact_id", "artifactId"):
            if data.get(key):
                return str(data[key])
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass
    return None


def _find_completed_artifact(stdout: str, artifact_id: str) -> bool:
    """Return True if the target artifact is complete in studio status output."""
    try:
        data = json.loads(stdout)
        # Handles {"artifacts": [...]} or [{"id": ..., "status": ...}] shapes.
        artifacts: list[dict[str, Any]] = []
        if isinstance(data, dict):
            artifacts = data.get("artifacts") or data.get("items") or []
        elif isinstance(data, list):
            artifacts = data
        for artifact in artifacts:
            aid = str(artifact.get("id") or artifact.get("artifact_id") or "")
            status = str(artifact.get("status") or "").lower()
            if aid == artifact_id and status in ("complete", "done", "ready"):
                return True
    except (json.JSONDecodeError, AttributeError):
        pass
    return False


async def _poll_and_send(
    signal_adapter: SignalAdapter,
    group_id: str,
    is_group: bool,
    notebook_id: str,
    artifact_id: str,
    podcast_type: str,
    output_path: str,
) -> None:
    """Background task: poll generation status, download, send, then clean up."""
    success = False

    try:
        for attempt in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL)

            rc, stdout, stderr = await _run_nlm("studio", "status", notebook_id, "--json")
            if rc != 0:
                LOGGER.warning("studio status poll %d failed: %s", attempt + 1, stderr)
                continue

            if _find_completed_artifact(stdout, artifact_id):
                LOGGER.info("Podcast artifact %s is ready; downloading", artifact_id)
                rc, _, stderr = await _run_nlm(
                    "download", "audio", notebook_id, artifact_id,
                    "--output", output_path,
                    timeout=120,
                )
                if rc != 0:
                    LOGGER.error("Failed to download podcast: %s", stderr)
                    await signal_adapter.send_message(
                        group_id,
                        "Podcast generation finished but download failed. Sorry about that.",
                        is_group=is_group,
                    )
                else:
                    await signal_adapter.send_message(
                        group_id,
                        f"Your {podcast_type} podcast is ready!",
                        is_group=is_group,
                        attachment_path=output_path,
                    )
                    success = True
                break
        else:
            LOGGER.warning("Podcast generation timed out after %d polls", _MAX_POLLS)
            await signal_adapter.send_message(
                group_id,
                "Podcast generation timed out (over 10 minutes). Please try again.",
                is_group=is_group,
            )
    finally:
        # Always delete notebook and temp file regardless of outcome.
        rc, _, stderr = await _run_nlm("notebook", "delete", notebook_id, "--confirm")
        if rc != 0:
            LOGGER.warning("Failed to delete notebook %s: %s", notebook_id, stderr)
        else:
            LOGGER.info("Deleted notebook %s", notebook_id)

        try:
            os.remove(output_path)
        except OSError:
            pass

        if success:
            LOGGER.info("Podcast pipeline complete for %s", podcast_type)


class PodcastTool(Tool):
    """Generate a NotebookLM deep-dive podcast from a PDF source and send it to Signal.

    Accepts either a file attachment path (local path after signal-cli saves it)
    or a URL pointing to a PDF. The podcast type determines the focus prompt used
    during generation. Generation runs in the background; the audio is sent to the
    group automatically when ready.

    Supported types: econpod, cspod, ddpod.
    """

    name = "create_podcast"
    description = (
        "Generate a NotebookLM podcast from a PDF. "
        "Use when the user sends a message like 'podcast econpod' with a PDF attachment "
        "or a URL to a PDF. "
        "Pass attachment_path from message.attachments[0].local_path when a file is attached, "
        "or source_url when a URL is present in the message. "
        f"Valid podcast_type values: {', '.join(PODCAST_TYPES)}."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string"},
            "podcast_type": {
                "type": "string",
                "enum": list(PODCAST_TYPES),
                "description": "The podcast format type.",
            },
            "source_url": {
                "type": "string",
                "description": "URL of the PDF to use as source. Provide when no attachment.",
            },
            "attachment_path": {
                "type": "string",
                "description": "Local filesystem path to an attached PDF. Provide when a file was attached.",
            },
        },
        "required": ["group_id", "podcast_type"],
        "additionalProperties": False,
    }

    def __init__(self, signal_adapter: SignalAdapter) -> None:
        self._signal_adapter = signal_adapter

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        group_id: str = kwargs["group_id"]
        podcast_type: str = kwargs["podcast_type"]
        source_url: str | None = kwargs.get("source_url")
        attachment_path: str | None = kwargs.get("attachment_path")

        if podcast_type not in PODCAST_TYPES:
            return {"error": f"Unknown podcast type '{podcast_type}'. Valid types: {', '.join(PODCAST_TYPES)}."}
        if not source_url and not attachment_path:
            return {"error": "Either source_url or attachment_path must be provided."}

        focus_prompt = PODCAST_TYPES[podcast_type]

        # --- 1. Verify nlm is installed ---
        rc, _, _ = await _run_nlm("--version")
        if rc != 0:
            return {
                "error": (
                    "The NotebookLM CLI (nlm) is not installed or not on PATH. "
                    "Install it with: uv tool install notebooklm-mcp-cli"
                )
            }

        # --- 2. Create notebook ---
        title = f"Podcast {podcast_type} {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        rc, stdout, stderr = await _run_nlm("notebook", "create", title, "--json")
        if rc != 0:
            return {"error": f"Failed to create NotebookLM notebook: {stderr}"}
        notebook_id = _parse_notebook_id(stdout)
        if not notebook_id:
            return {"error": f"Could not parse notebook ID from: {stdout!r}"}
        LOGGER.info("Created notebook %s for %s podcast", notebook_id, podcast_type)

        # --- 3. Add source ---
        if attachment_path:
            rc, _, stderr = await _run_nlm(
                "source", "add", notebook_id, "--file", attachment_path, "--wait",
                timeout=120,
            )
        else:
            rc, _, stderr = await _run_nlm(
                "source", "add", notebook_id, "--url", source_url, "--wait",  # type: ignore[arg-type]
                timeout=120,
            )
        if rc != 0:
            await _run_nlm("notebook", "delete", notebook_id, "--confirm")
            return {"error": f"Failed to add source to notebook: {stderr}"}
        LOGGER.info("Source added to notebook %s", notebook_id)

        # --- 4. Create podcast ---
        rc, stdout, stderr = await _run_nlm(
            "audio", "create", notebook_id,
            "--format", "deep_dive",
            "--length", "long",
            "--focus", focus_prompt,
            "--confirm",
            "--json",
        )
        if rc != 0:
            await _run_nlm("notebook", "delete", notebook_id, "--confirm")
            return {"error": f"Failed to start podcast generation: {stderr}"}
        artifact_id = _parse_artifact_id(stdout)
        if not artifact_id:
            await _run_nlm("notebook", "delete", notebook_id, "--confirm")
            return {"error": f"Could not parse artifact ID from: {stdout!r}"}
        LOGGER.info("Podcast generation started, artifact %s", artifact_id)

        # --- 5. Spawn background polling task ---
        output_path = os.path.join(tempfile.gettempdir(), f"podcast_{notebook_id}.m4a")
        asyncio.create_task(
            _poll_and_send(
                signal_adapter=self._signal_adapter,
                group_id=group_id,
                is_group=True,
                notebook_id=notebook_id,
                artifact_id=artifact_id,
                podcast_type=podcast_type,
                output_path=output_path,
            ),
            name=f"podcast-{notebook_id}",
        )

        return {
            "status": "started",
            "message": (
                f"Podcast generation started (type: {podcast_type}). "
                "I'll send the audio file when it's ready — usually 2–5 minutes."
            ),
        }
