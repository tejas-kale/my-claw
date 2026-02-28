"""Tests for memory and search tools."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.tools.memory_tool import ReadNotesTool, SaveNoteTool, _slugify
from assistant.tools.search_tool import FuzzyFilterTool, RipgrepSearchTool


# ---------------------------------------------------------------------------
# SaveNoteTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_note_daily_appends_timestamped_entry(tmp_path):
    tool = SaveNoteTool(tmp_path)
    result = await tool.run(content="bought milk", note_type="daily")

    today = date.today().isoformat()
    daily_file = tmp_path / "daily" / f"{today}.md"
    assert daily_file.exists()
    file_content = daily_file.read_text()
    assert "bought milk" in file_content
    assert "Saved to daily notes" in result


@pytest.mark.asyncio
async def test_save_note_daily_appends_without_overwriting(tmp_path):
    tool = SaveNoteTool(tmp_path)
    await tool.run(content="first note", note_type="daily")
    await tool.run(content="second note", note_type="daily")

    today = date.today().isoformat()
    file_content = (tmp_path / "daily" / f"{today}.md").read_text()
    assert "first note" in file_content
    assert "second note" in file_content


@pytest.mark.asyncio
async def test_save_note_topic_creates_file_with_heading(tmp_path):
    tool = SaveNoteTool(tmp_path)
    result = await tool.run(content="use asyncio.gather for concurrency", note_type="topic", topic="Python Async")

    slug_file = tmp_path / "topics" / "python-async.md"
    assert slug_file.exists()
    file_content = slug_file.read_text()
    assert "# Python Async" in file_content
    assert "use asyncio.gather for concurrency" in file_content
    assert "Created" in result


@pytest.mark.asyncio
async def test_save_note_topic_appends_to_existing_file(tmp_path):
    tool = SaveNoteTool(tmp_path)
    await tool.run(content="first entry", note_type="topic", topic="cooking")
    result = await tool.run(content="second entry", note_type="topic", topic="cooking")

    file_content = (tmp_path / "topics" / "cooking.md").read_text()
    assert file_content.count("# cooking") == 1  # heading written only once
    assert "first entry" in file_content
    assert "second entry" in file_content
    assert "Appended to" in result


@pytest.mark.asyncio
async def test_save_note_missing_topic_returns_error(tmp_path):
    tool = SaveNoteTool(tmp_path)
    result = await tool.run(content="oops", note_type="topic")
    assert "Error" in result


# ---------------------------------------------------------------------------
# ReadNotesTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_notes_daily_returns_todays_content(tmp_path):
    save_tool = SaveNoteTool(tmp_path)
    await save_tool.run(content="stand-up done", note_type="daily")

    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="daily")
    assert "stand-up done" in result


@pytest.mark.asyncio
async def test_read_notes_daily_no_notes_returns_message(tmp_path):
    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="daily")
    assert "No daily notes found" in result


@pytest.mark.asyncio
async def test_read_notes_topic_returns_file_content(tmp_path):
    save_tool = SaveNoteTool(tmp_path)
    await save_tool.run(content="GIL is released during I/O", note_type="topic", topic="python-async")

    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="topic", topic="python-async")
    assert "GIL is released during I/O" in result


@pytest.mark.asyncio
async def test_read_notes_topic_fuzzy_fallback_lists_similar(tmp_path):
    save_tool = SaveNoteTool(tmp_path)
    await save_tool.run(content="notes", note_type="topic", topic="python-async")

    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="topic", topic="python")
    assert "python-async" in result


@pytest.mark.asyncio
async def test_read_notes_topic_not_found_returns_message(tmp_path):
    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="topic", topic="nonexistent-topic-xyz")
    assert "No topic notes found" in result


@pytest.mark.asyncio
async def test_read_notes_topics_list_returns_all_stems(tmp_path):
    save_tool = SaveNoteTool(tmp_path)
    await save_tool.run(content="a", note_type="topic", topic="alpha")
    await save_tool.run(content="b", note_type="topic", topic="beta topic")

    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="topics_list")
    assert "alpha" in result
    assert "beta-topic" in result


@pytest.mark.asyncio
async def test_read_notes_topics_list_empty_returns_message(tmp_path):
    read_tool = ReadNotesTool(tmp_path)
    result = await read_tool.run(note_type="topics_list")
    assert "No topic notes yet" in result


# ---------------------------------------------------------------------------
# RipgrepSearchTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ripgrep_search_rejects_path_outside_allowed_roots(tmp_path):
    tool = RipgrepSearchTool(tmp_path / "memory")
    result = await tool.run(pattern="test", path="/etc/passwd")
    assert "not in allowed roots" in result


@pytest.mark.asyncio
async def test_ripgrep_search_returns_matches(tmp_path):
    rg_json_output = (
        '{"type":"match","data":{"path":{"text":"foo.py"},"line_number":1,'
        '"lines":{"text":"def hello():"},"submatches":[]}}\n'
    )

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(rg_json_output.encode(), b""))

    tool = RipgrepSearchTool(tmp_path)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.run(pattern="hello", path=str(tmp_path))

    assert "foo.py" in result
    assert "def hello():" in result


@pytest.mark.asyncio
async def test_ripgrep_search_no_matches_returns_message(tmp_path):
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    tool = RipgrepSearchTool(tmp_path)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.run(pattern="zzznomatch", path=str(tmp_path))

    assert "No matches" in result


@pytest.mark.asyncio
async def test_ripgrep_search_timeout_returns_message(tmp_path):
    async def _slow_communicate():
        await asyncio.sleep(100)
        return b"", b""

    mock_proc = MagicMock()
    mock_proc.communicate = _slow_communicate
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    tool = RipgrepSearchTool(tmp_path)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with patch("assistant.tools.search_tool._RG_TIMEOUT", 0.01):
            result = await tool.run(pattern="x", path=str(tmp_path))

    assert "timed out" in result


# ---------------------------------------------------------------------------
# FuzzyFilterTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fuzzy_filter_returns_ranked_matches():
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"auth-handler.py\nauth-utils.py\n", b""))

    tool = FuzzyFilterTool()
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.run(query="auth", items=["auth-handler.py", "auth-utils.py", "main.py"])

    assert "auth-handler.py" in result
    assert "auth-utils.py" in result


@pytest.mark.asyncio
async def test_fuzzy_filter_no_items_returns_message():
    tool = FuzzyFilterTool()
    result = await tool.run(query="anything", items=[])
    assert "No items" in result


@pytest.mark.asyncio
async def test_fuzzy_filter_no_matches_returns_message():
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    tool = FuzzyFilterTool()
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.run(query="zzznomatch", items=["alpha", "beta"])

    assert "No fuzzy matches" in result


# ---------------------------------------------------------------------------
# _slugify helper
# ---------------------------------------------------------------------------


def test_slugify_lowercases_and_replaces_spaces():
    assert _slugify("Python Async") == "python-async"


def test_slugify_replaces_slashes():
    assert _slugify("work/project-x") == "work-project-x"


def test_slugify_truncates_at_60_chars():
    long_name = "a" * 100
    assert len(_slugify(long_name)) == 60
