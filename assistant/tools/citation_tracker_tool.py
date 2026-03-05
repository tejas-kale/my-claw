"""Async wrapper around the citation-tracker CLI."""

from __future__ import annotations

import asyncio
import logging

LOGGER = logging.getLogger(__name__)
_CT_TIMEOUT = 60  # seconds for quick commands (status, list, add, citations)


async def _run_ct(*args: str, timeout: int = _CT_TIMEOUT) -> tuple[int, str, str]:
    """Run `citation-tracker <args>`, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "citation-tracker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 1, "", f"citation-tracker timed out after {timeout}s"
    return proc.returncode, stdout_b.decode(), stderr_b.decode()


class CitationTrackerTool:
    """Wraps citation-tracker CLI subcommands for use in @cite command."""

    async def status(self) -> str:
        rc, out, err = await _run_ct("status")
        return out.strip() or err.strip() or "No output."

    async def list_papers(self) -> str:
        rc, out, err = await _run_ct("list")
        return out.strip() or err.strip() or "No tracked papers."

    async def add_paper(self, source: str) -> str:
        """source is a URL, DOI (10.xxx/xxx), or Semantic Scholar ID."""
        if source.startswith("10."):
            args = ("add", "--doi", source)
        elif source.startswith("http"):
            args = ("add", source)
        else:
            args = ("add", "--ss-id", source)
        rc, out, err = await _run_ct(*args)
        if rc != 0:
            return f"Failed to add paper: {err.strip() or out.strip()}"
        return out.strip() or "Paper added."

    async def citations(self, paper_id: str) -> str:
        rc, out, err = await _run_ct("citations", "--id", paper_id)
        if rc != 0:
            return f"Failed: {err.strip() or out.strip()}"
        return out.strip() or "No citations found."

    async def run(self, paper_id: str | None = None) -> str:
        """Fire the discovery pipeline in the background; return ack immediately."""
        args = ("run",) if paper_id is None else ("run", "--id", paper_id)
        asyncio.create_task(_run_ct(*args, timeout=3600))
        target = "all active papers" if paper_id is None else paper_id
        return f"Citation discovery started for {target}. Use @cite status to check progress."
