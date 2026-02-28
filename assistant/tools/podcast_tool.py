"""Podcast generation tool via NotebookLM CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    "econpod": """
        You are to generate a podcast script in the style of Planet Money by Planet Money.

        Using only the material provided in this notebook as your source, create a compelling 20–30 minute podcast episode script that:
            1.	Tells a clear economic story centred on one strong, curiosity-driven question.
            2.	Opens with a hook (an intriguing anecdote, paradox, or surprising fact drawn from the material).
            3.	Develops the narrative through:
            •	Concrete examples
            •	Characters (real individuals mentioned in the material, if available)
            •	Data explained in accessible terms
            •	Moments of tension, uncertainty, or discovery
            4.	Breaks down complex ideas using:
            •	Plain language
            •	Analogies
            •	Step-by-step reasoning
            5.	Includes:
            •	Host narration
            •	Short conversational exchanges between two hosts (natural, informal, but precise)
            •	Occasional “wait, what?” clarification moments
            6.	Avoids jargon unless clearly explained.
            7.	Ends with a satisfying takeaway that reframes the original question.

        Structure the output as:
            •	Episode title
            •	Cold open (1–2 minutes)
            •	Theme music cue
            •	Main narrative segments (with clear transitions)
            •	Short mid-episode recap
            •	Final insight / closing reflection

        Tone: Curious, sharp, lightly playful, but intellectually rigorous.
        Style: Story first, economics through narrative.

        If multiple angles are possible, choose the one with the strongest narrative tension.
    """,
    "cspod": """
        You are to generate a podcast script in the style of Planet Money by Planet Money — but focused on a computer science topic where the core audience is primarily interested in understanding the algorithm.

        Using only the material provided in this notebook as your source, create a compelling 20–30 minute podcast episode script that:

        Core Objective

        Tell the story of one central algorithm through a strong, curiosity-driven technical question.

        The episode should:
            1.	Open with a sharp hook:
            •	A surprising computational constraint
            •	A failure case
            •	A performance bottleneck
            •	Or a real-world problem that demanded this algorithm
            2.	Build narrative tension around:
            •	Why naïve solutions fail
            •	What constraints make the problem hard (time, space, scale, adversarial input, distribution, etc.)
            •	The key insight that unlocks the algorithm
            3.	Make the algorithm the protagonist:
            •	Explain the intuition first
            •	Then walk through the mechanics step by step
            •	Clearly articulate invariants, trade-offs, and complexity
            •	Highlight what makes it elegant, clever, or counterintuitive
            4.	Include:
            •	Host narration
            •	Conversational exchanges between two hosts
            •	“Hold on, why does that work?” clarification moments
            •	Occasional pseudo-code explanations in spoken form (clear but not overly formal)
            5.	Break down complexity with:
            •	Concrete examples
            •	Small input walkthroughs
            •	Visual mental models
            •	Comparisons to simpler baselines
            6.	Discuss:
            •	Time and space complexity (intuitively, then formally)
            •	Edge cases
            •	Where it breaks
            •	Why alternatives are worse
            •	Real-world applications
            7.	Avoid unnecessary jargon, but do not oversimplify. The audience is technically literate and cares about rigour.

        Structure the output as:
            •	Episode title
            •	Cold open (1–2 minutes)
            •	Theme music cue
            •	Segment 1: The problem
            •	Segment 2: Failed approaches
            •	Segment 3: The key insight
            •	Segment 4: The algorithm walkthrough
            •	Segment 5: Complexity and trade-offs
            •	Short recap
            •	Closing reflection (what this teaches us about computation)

        Tone: Curious, analytical, technically precise, lightly playful.
        Style: Story first, algorithm second — but with real depth.

        If multiple interpretations are possible, choose the version with the clearest algorithmic insight and strongest explanatory arc.
    """,
    "ddpod": """
        You are to generate a podcast episode script in the style of Planet Money, using only the content from the provided academic paper as your source.

        Your episode should:
            1.	Open with a compelling question or real-world problem that the paper addresses.
        Start with an engaging hook drawn from the paper’s motivation, surprising insight, paradox, or failure case.
            2.	Explain the core scientific or technical contribution of the paper in accessible language:
            •	Define key concepts introduced by the paper.
            •	Highlight what problem the authors are solving and why it matters.
            •	Clarify any foundational terms before referring to formal definitions or equations.
            3.	Structure around a narrative arc:
            •	What existing approaches failed or were insufficient?
            •	What key idea or insight the authors introduce?
            •	How the new approach works (intuitive explanation first, then technical mechanics).
            •	What results or evidence the authors present.
            4.	Illustrate complex ideas with examples:
            •	Simple everyday analogies.
            •	Concrete, small-scale examples to make abstract ideas tangible.
            •	Conversational clarifications between two hosts (e.g., “Why does this matter?” “How is this different?”).
            5.	Discuss evaluation and results:
            •	What methods did the authors use to validate their approach?
            •	What are the key findings?
            •	How do these findings support the central thesis of the paper?
            6.	Reflect on broader implications and limitations:
            •	Why the contribution matters beyond the paper.
            •	Where it could be applied.
            •	What limitations or open questions remain.

        Required Structure
            •	Episode Title
            •	Cold Open (1–2 minutes)
            •	Theme Music Cue
            •	Segment 1: The Big Question/Problem
            •	Segment 2: Background & Context
            •	Segment 3: What’s New — The Paper’s Contribution
            •	Segment 4: How It Works — Intuition + Mechanics
            •	Segment 5: Evidence & Results
            •	Segment 6: Broader Implications
            •	Short Recap
            •	Closing Reflection

        Tone & Style
            •	Story first, explanation second
            •	Accessible for technically literate audiences
            •	Minimal jargon; when used, always clearly explained
            •	Conversational but accurate

        Length

        2,500–3,500 words (approximately a 20–30 minute episode)
    """,
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
    """Extract notebook ID from `nlm notebook create` output."""
    try:
        data = json.loads(stdout)
        # Handles {"id": "..."} or [{"id": "..."}] shapes.
        if isinstance(data, list):
            data = data[0]
        return str(data.get("id") or data.get("notebook_id") or "")
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass
    # Fallback: scan output for a UUID.
    m = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', stdout, re.IGNORECASE)
    if m:
        return m.group(0)
    # Last-ditch: bare single-word line.
    line = stdout.splitlines()[0].strip() if stdout else ""
    return line or None


def _parse_artifact_id(stdout: str) -> str | None:
    """Extract artifact ID from `nlm audio create` output."""
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            data = data[0]
        for key in ("id", "artifact_id", "artifactId"):
            if data.get(key):
                return str(data[key])
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass
    # Fallback: scan output for a UUID.
    m = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', stdout, re.IGNORECASE)
    if m:
        return m.group(0)
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
                    "download", "audio", notebook_id,
                    "--id", artifact_id,
                    "--output", output_path,
                    "--no-progress",
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
            "is_group": {
                "type": "boolean",
                "description": "True if the message came from a group chat, False for a direct message.",
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
        is_group: bool = bool(kwargs.get("is_group", True))

        if podcast_type not in PODCAST_TYPES:
            return {"error": f"Unknown podcast type '{podcast_type}'. Valid types: {', '.join(PODCAST_TYPES)}."}
        if not source_url and not attachment_path:
            return {"error": "Either source_url or attachment_path must be provided."}

        focus_prompt = PODCAST_TYPES[podcast_type]

        # --- 1. Verify nlm is installed ---
        rc, stdout, stderr = await _run_nlm("--version")
        if rc != 0:
            msg = f"nlm not found or failed: rc={rc} stdout={stdout!r} stderr={stderr!r}"
            LOGGER.error(msg)
            return {
                "error": (
                    "The NotebookLM CLI (nlm) is not installed or not on PATH. "
                    "Install it with: uv tool install notebooklm-mcp-cli"
                )
            }

        # --- 2. Create notebook ---
        title = f"Podcast {podcast_type} {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        rc, stdout, stderr = await _run_nlm("notebook", "create", title)
        if rc != 0:
            LOGGER.error("notebook create failed: rc=%d stdout=%r stderr=%r", rc, stdout, stderr)
            return {"error": f"Failed to create NotebookLM notebook: rc={rc} {stderr or stdout}"}
        notebook_id = _parse_notebook_id(stdout)
        if not notebook_id:
            LOGGER.error("notebook create: could not parse ID from stdout=%r stderr=%r", stdout, stderr)
            return {"error": f"Could not parse notebook ID from: {stdout!r}"}
        LOGGER.info("Created notebook %s for %s podcast", notebook_id, podcast_type)

        # --- 3. Add source ---
        if attachment_path:
            rc, stdout, stderr = await _run_nlm(
                "source", "add", notebook_id, "--file", attachment_path, "--wait",
                timeout=120,
            )
        else:
            rc, stdout, stderr = await _run_nlm(
                "source", "add", notebook_id, "--url", source_url, "--wait",  # type: ignore[arg-type]
                timeout=120,
            )
        if rc != 0:
            LOGGER.error("source add failed: rc=%d stdout=%r stderr=%r", rc, stdout, stderr)
            await _run_nlm("notebook", "delete", notebook_id, "--confirm")
            return {"error": f"Failed to add source to notebook: rc={rc} {stderr or stdout}"}
        LOGGER.info("Source added to notebook %s", notebook_id)

        # --- 4. Create podcast ---
        rc, stdout, stderr = await _run_nlm(
            "audio", "create", notebook_id,
            "--format", "deep_dive",
            "--length", "long",
            "--focus", focus_prompt,
            "--confirm",
        )
        if rc != 0:
            LOGGER.error("audio create failed: rc=%d stdout=%r stderr=%r", rc, stdout, stderr)
            await _run_nlm("notebook", "delete", notebook_id, "--confirm")
            return {"error": f"Failed to start podcast generation: rc={rc} {stderr or stdout}"}
        artifact_id = _parse_artifact_id(stdout)
        if not artifact_id:
            LOGGER.error("audio create: could not parse artifact ID from stdout=%r stderr=%r", stdout, stderr)
            await _run_nlm("notebook", "delete", notebook_id, "--confirm")
            return {"error": f"Could not parse artifact ID from: {stdout!r}"}
        LOGGER.info("Podcast generation started, artifact %s", artifact_id)

        # --- 5. Spawn background polling task ---
        output_path = os.path.join(tempfile.gettempdir(), f"podcast_{notebook_id}.m4a")
        asyncio.create_task(
            _poll_and_send(
                signal_adapter=self._signal_adapter,
                group_id=group_id,
                is_group=is_group,
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
