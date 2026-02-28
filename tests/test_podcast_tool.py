"""Tests for the podcast generation tool."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.tools.podcast_tool import (
    PodcastTool,
    _find_completed_artifact,
    _parse_artifact_id,
    _parse_notebook_id,
    _poll_and_send,
)


# ---------------------------------------------------------------------------
# Pure parser unit tests
# ---------------------------------------------------------------------------


def test_parse_notebook_id_from_json_object():
    assert _parse_notebook_id(json.dumps({"id": "nb-abc"})) == "nb-abc"


def test_parse_notebook_id_from_json_list():
    assert _parse_notebook_id(json.dumps([{"id": "nb-xyz"}])) == "nb-xyz"


def test_parse_notebook_id_fallback_plain_text():
    assert _parse_notebook_id("plain-id-123") == "plain-id-123"


def test_parse_notebook_id_empty_returns_none():
    assert _parse_notebook_id("") is None


def test_parse_artifact_id_from_json_object():
    assert _parse_artifact_id(json.dumps({"id": "art-abc"})) == "art-abc"


def test_parse_artifact_id_alias_fields():
    assert _parse_artifact_id(json.dumps({"artifact_id": "art-xyz"})) == "art-xyz"
    assert _parse_artifact_id(json.dumps({"artifactId": "art-ijk"})) == "art-ijk"


def test_parse_artifact_id_missing_returns_none():
    assert _parse_artifact_id(json.dumps({"type": "audio"})) is None


def test_find_completed_artifact_matching():
    data = {"artifacts": [{"id": "art-1", "status": "complete"}]}
    assert _find_completed_artifact(json.dumps(data), "art-1") is True


def test_find_completed_artifact_wrong_id():
    data = {"artifacts": [{"id": "art-2", "status": "complete"}]}
    assert _find_completed_artifact(json.dumps(data), "art-1") is False


def test_find_completed_artifact_not_ready():
    data = {"artifacts": [{"id": "art-1", "status": "generating"}]}
    assert _find_completed_artifact(json.dumps(data), "art-1") is False


def test_find_completed_artifact_list_shape():
    data = [{"id": "art-1", "status": "done"}]
    assert _find_completed_artifact(json.dumps(data), "art-1") is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_process(returncode: int, stdout: str = "", stderr: str = "") -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


def _make_signal_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    return adapter


# ---------------------------------------------------------------------------
# PodcastTool.run() — validation and error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_rejects_unknown_podcast_type():
    adapter = _make_signal_adapter()
    tool = PodcastTool(signal_adapter=adapter)
    result = await tool.run(group_id="g1", podcast_type="badtype", source_url="http://x.com/f.pdf")
    assert "error" in result
    assert "Unknown podcast type" in result["error"]


@pytest.mark.asyncio
async def test_run_rejects_missing_source():
    adapter = _make_signal_adapter()
    tool = PodcastTool(signal_adapter=adapter)
    result = await tool.run(group_id="g1", podcast_type="econpod")
    assert "error" in result
    assert "source_url or attachment_path" in result["error"]


@pytest.mark.asyncio
async def test_run_returns_error_when_nlm_not_installed():
    adapter = _make_signal_adapter()
    tool = PodcastTool(signal_adapter=adapter)

    nlm_not_found = _make_process(returncode=1, stderr="command not found")
    with patch("asyncio.create_subprocess_exec", return_value=nlm_not_found):
        result = await tool.run(group_id="g1", podcast_type="econpod", source_url="http://x.com/f.pdf")

    assert "error" in result
    assert "uv tool install" in result["error"]


# ---------------------------------------------------------------------------
# PodcastTool.run() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_happy_path_spawns_background_task():
    """Full success path: nlm installed, notebook created, source added, audio initiated."""
    adapter = _make_signal_adapter()
    tool = PodcastTool(signal_adapter=adapter)

    notebook_resp = json.dumps({"id": "nb-001"})
    artifact_resp = json.dumps({"id": "art-001", "status": "generating"})

    call_responses = [
        _make_process(0, "nlm 0.3.0"),       # nlm --version
        _make_process(0, notebook_resp),       # nlm notebook create
        _make_process(0, ""),                  # nlm source add --url --wait
        _make_process(0, artifact_resp),       # nlm audio create
    ]
    call_iter = iter(call_responses)

    created_tasks: list[asyncio.Task[None]] = []

    def fake_create_task(coro: object, **kwargs: object) -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.ensure_future(coro)  # type: ignore[arg-type]
        created_tasks.append(task)
        task.cancel()  # Don't let the background task run during this test
        return task

    with (
        patch("asyncio.create_subprocess_exec", side_effect=lambda *a, **kw: next(call_iter)),
        patch("asyncio.create_task", side_effect=fake_create_task),
    ):
        result = await tool.run(group_id="g1", podcast_type="cspod", source_url="http://x.com/f.pdf")

    assert result["status"] == "started"
    assert "cspod" in result["message"]
    assert len(created_tasks) == 1


# ---------------------------------------------------------------------------
# _poll_and_send — success case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_and_send_success(tmp_path: "os.PathLike[str]") -> None:
    """Podcast completes on 2nd poll: should download, send with attachment, delete notebook."""
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "podcast.m4a")

    # Create dummy m4a so os.remove doesn't fail
    with open(output_path, "w") as f:
        f.write("fake audio")

    status_generating = json.dumps({"artifacts": [{"id": "art-1", "status": "generating"}]})
    status_complete = json.dumps({"artifacts": [{"id": "art-1", "status": "complete"}]})

    call_responses = [
        _make_process(0, status_generating),   # poll 1
        _make_process(0, status_complete),     # poll 2
        _make_process(0, ""),                  # download
        _make_process(0, ""),                  # notebook delete
    ]
    call_iter = iter(call_responses)

    with (
        patch("asyncio.create_subprocess_exec", side_effect=lambda *a, **kw: next(call_iter)),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await _poll_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            notebook_id="nb-1",
            artifact_id="art-1",
            podcast_type="econpod",
            output_path=output_path,
        )

    adapter.send_message.assert_awaited_once()
    call_kwargs = adapter.send_message.call_args
    assert call_kwargs.kwargs.get("attachment_path") == output_path or (
        len(call_kwargs.args) > 2 and call_kwargs.args[2] == output_path
    )
    # Notebook should be deleted; temp file should be gone
    assert not os.path.exists(output_path)


# ---------------------------------------------------------------------------
# _poll_and_send — timeout case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_and_send_timeout_sends_failure_and_deletes_notebook(
    tmp_path: "os.PathLike[str]",
) -> None:
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "podcast.m4a")

    status_generating = json.dumps({"artifacts": [{"id": "art-1", "status": "generating"}]})

    from assistant.tools.podcast_tool import _MAX_POLLS

    poll_procs = [_make_process(0, status_generating) for _ in range(_MAX_POLLS)]
    delete_proc = _make_process(0, "")  # notebook delete after timeout
    call_iter = iter(poll_procs + [delete_proc])

    with (
        patch("asyncio.create_subprocess_exec", side_effect=lambda *a, **kw: next(call_iter)),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await _poll_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            notebook_id="nb-1",
            artifact_id="art-1",
            podcast_type="ddpod",
            output_path=output_path,
        )

    adapter.send_message.assert_awaited_once()
    failure_msg = adapter.send_message.call_args.args[1]
    assert "timed out" in failure_msg.lower()
