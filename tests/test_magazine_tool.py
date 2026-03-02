"""Tests for the magazine chapter narration tool."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.tools.magazine_tool import MagazineTool, _generate_and_send


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


def _make_tool() -> MagazineTool:
    return MagazineTool(signal_adapter=_make_signal_adapter())


# ---------------------------------------------------------------------------
# MagazineTool.list_chapters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_chapters_returns_stdout_on_success():
    tool = _make_tool()
    chapter_list = "1  Introduction\n2  Chapter Two\n3  Chapter Three"
    proc = _make_process(returncode=0, stdout=chapter_list)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await tool.list_chapters("blizzard")
    assert result == chapter_list


@pytest.mark.asyncio
async def test_list_chapters_returns_error_on_failure():
    tool = _make_tool()
    proc = _make_process(returncode=1, stderr="Source not found")
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await tool.list_chapters("unknown-epub")
    assert "Source not found" in result


@pytest.mark.asyncio
async def test_list_chapters_calls_podcaster_inspect():
    tool = _make_tool()
    proc = _make_process(returncode=0, stdout="chapters")
    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await tool.list_chapters("my-book")
    args = mock_exec.call_args.args
    assert args[0] == "podcaster"
    assert "inspect" in args
    assert "my-book" in args


# ---------------------------------------------------------------------------
# MagazineTool.start_generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_generation_returns_immediately_with_status():
    tool = _make_tool()
    created_tasks: list[asyncio.Task] = []

    def fake_create_task(coro, **kwargs):
        task = asyncio.ensure_future(coro)
        created_tasks.append(task)
        task.cancel()
        return task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        result = await tool.start_generation(
            group_id="g1", is_group=True, epub="blizzard", chapter="3"
        )

    assert "3" in result
    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_start_generation_spawns_task_with_correct_epub_and_chapter():
    tool = _make_tool()
    task_kwargs: dict = {}

    def fake_create_task(coro, **kwargs):
        task_kwargs["name"] = kwargs.get("name", "")
        task = asyncio.ensure_future(coro)
        task.cancel()
        return task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        await tool.start_generation(
            group_id="g1", is_group=True, epub="blizzard", chapter="Introduction"
        )

    assert "blizzard" in task_kwargs["name"]
    assert "Introduction" in task_kwargs["name"]


# ---------------------------------------------------------------------------
# _generate_and_send — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_and_send_sends_mp3_on_success(tmp_path):
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "chapter.mp3")

    def fake_exec(*args, **kwargs):
        # Create the output file when podcaster is called
        if "create" in args:
            with open(output_path, "wb") as f:
                f.write(b"fake mp3 data")
        return _make_process(returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await _generate_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            epub="blizzard",
            chapter="3",
            output_path=output_path,
        )

    adapter.send_message.assert_awaited_once()
    call_kwargs = adapter.send_message.call_args
    assert call_kwargs.kwargs.get("attachment_path") == output_path


@pytest.mark.asyncio
async def test_generate_and_send_message_mentions_chapter(tmp_path):
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "chapter.mp3")

    def fake_exec(*args, **kwargs):
        if "create" in args:
            with open(output_path, "wb") as f:
                f.write(b"audio")
        return _make_process(returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await _generate_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            epub="blizzard",
            chapter="Introduction",
            output_path=output_path,
        )

    msg = adapter.send_message.call_args.args[1]
    assert "Introduction" in msg


@pytest.mark.asyncio
async def test_generate_and_send_deletes_output_file_after_send(tmp_path):
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "chapter.mp3")

    def fake_exec(*args, **kwargs):
        if "create" in args:
            with open(output_path, "wb") as f:
                f.write(b"audio")
        return _make_process(returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await _generate_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            epub="blizzard",
            chapter="3",
            output_path=output_path,
        )

    assert not os.path.exists(output_path)


# ---------------------------------------------------------------------------
# _generate_and_send — failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_and_send_reports_podcaster_failure(tmp_path):
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "chapter.mp3")
    proc = _make_process(returncode=1, stderr="EPUB chapter not found")

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        await _generate_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            epub="blizzard",
            chapter="99",
            output_path=output_path,
        )

    adapter.send_message.assert_awaited_once()
    msg = adapter.send_message.call_args.args[1]
    assert "EPUB chapter not found" in msg


@pytest.mark.asyncio
async def test_generate_and_send_reports_missing_output_file(tmp_path):
    """podcaster exits 0 but doesn't write the output file."""
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "chapter.mp3")
    proc = _make_process(returncode=0)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        await _generate_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            epub="blizzard",
            chapter="3",
            output_path=output_path,
        )

    adapter.send_message.assert_awaited_once()
    msg = adapter.send_message.call_args.args[1]
    assert "missing" in msg.lower() or "empty" in msg.lower()


@pytest.mark.asyncio
async def test_generate_and_send_cleans_up_on_failure(tmp_path):
    """Output file is deleted even when podcaster fails."""
    adapter = _make_signal_adapter()
    output_path = str(tmp_path / "chapter.mp3")
    # Pre-create the file to verify it gets cleaned up
    with open(output_path, "wb") as f:
        f.write(b"partial")
    proc = _make_process(returncode=1, stderr="failure")

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        await _generate_and_send(
            signal_adapter=adapter,
            group_id="g1",
            is_group=True,
            epub="blizzard",
            chapter="3",
            output_path=output_path,
        )

    assert not os.path.exists(output_path)
