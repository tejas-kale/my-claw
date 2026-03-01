"""Tests for the price tracker tool and @trackprice command dispatcher handler."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.commands import CommandDispatcher
from assistant.models import Message
from assistant.tools.price_tracker_tool import PriceTrackerTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_EXTRACTION = json.dumps(
    {
        "supermarket": "Rewe",
        "date": "2024-01-15",
        "total_price": 10.50,
        "items": [
            {"name_german": "Vollmilch", "name_english": "Whole Milk", "price": 1.19},
            {"name_german": "Brot", "name_english": "Bread", "price": 2.49},
        ],
    }
)


def _make_llm(content: str) -> AsyncMock:
    llm = AsyncMock()
    response = MagicMock()
    response.content = content
    llm.generate = AsyncMock(return_value=response)
    return llm


def _make_tool(llm: AsyncMock | None = None) -> PriceTrackerTool:
    return PriceTrackerTool(
        llm=llm or _make_llm("{}"),
        bq_project="proj",
        bq_dataset="ds",
        bq_table="tbl",
    )


def _make_message(text: str = "@trackprice", attachments: list | None = None) -> Message:
    return Message(
        group_id="g1",
        sender_id="s1",
        text=text,
        timestamp=datetime.now(timezone.utc),
        attachments=attachments or [],
    )


# ---------------------------------------------------------------------------
# CommandDispatcher: @trackprice handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_no_attachment_returns_error() -> None:
    dispatcher = CommandDispatcher()
    msg = _make_message(text="@trackprice", attachments=[])
    result = await dispatcher.dispatch(msg)
    assert result is not None
    assert "attach" in result.lower()


@pytest.mark.asyncio
async def test_run_tool_not_configured_returns_error() -> None:
    dispatcher = CommandDispatcher(price_tracker_tool=None)
    msg = _make_message(
        text="@trackprice",
        attachments=[{"local_path": "/tmp/r.jpg", "content_type": "image/jpeg"}],
    )
    result = await dispatcher.dispatch(msg)
    assert result is not None
    assert "not configured" in result.lower()


# ---------------------------------------------------------------------------
# PriceTrackerTool: LLM extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_valid_json_parsed_correctly() -> None:
    tool = _make_tool(llm=_make_llm(_VALID_EXTRACTION))
    with (
        patch.object(tool, "_encode_attachment", return_value=b"fake-png"),
        patch.object(tool, "_insert_rows", return_value=[]),
        patch.object(tool, "_query_preview", return_value=[]),
    ):
        result = await tool.run("/tmp/receipt.jpg", "image/jpeg")
    assert result["status"] == "ok"
    assert "Rewe" in result["message"]


@pytest.mark.asyncio
async def test_extraction_bad_json_returns_error() -> None:
    tool = _make_tool(llm=_make_llm("not valid json!!!"))
    with (
        patch.object(tool, "_encode_attachment", return_value=b"fake-png"),
        patch.object(tool, "_insert_rows", return_value=[]) as mock_insert,
    ):
        result = await tool.run("/tmp/receipt.jpg", "image/jpeg")
    assert "error" in result
    mock_insert.assert_not_called()


# ---------------------------------------------------------------------------
# PriceTrackerTool: BigQuery insert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bigquery_insert_called_with_correct_rows() -> None:
    tool = _make_tool(llm=_make_llm(_VALID_EXTRACTION))
    captured: list = []

    def fake_insert(rows: list) -> list:
        captured.extend(rows)
        return []

    with (
        patch.object(tool, "_encode_attachment", return_value=b"bytes"),
        patch.object(tool, "_insert_rows", side_effect=fake_insert),
        patch.object(tool, "_query_preview", return_value=[]),
    ):
        await tool.run("/tmp/r.jpg", "image/jpeg")

    assert len(captured) == 2
    assert captured[0]["supermarket"] == "Rewe"
    assert captured[0]["item_name_german"] == "Vollmilch"
    assert captured[1]["item_name_german"] == "Brot"
    for row in captured:
        assert "inserted_at" in row
        assert row["total_price"] == 10.50


# ---------------------------------------------------------------------------
# PriceTrackerTool: preview formatting
# ---------------------------------------------------------------------------


def test_format_preview_five_rows() -> None:
    tool = _make_tool()
    extraction = json.loads(_VALID_EXTRACTION)
    rows = [
        {"item_name_german": f"Item{i}", "item_name_english": f"Eng{i}", "price": float(i)}
        for i in range(5)
    ]
    text = tool._format_preview(extraction, rows)
    assert "Rewe" in text
    assert "Item0" in text
    assert "Eng4" in text
    assert "10.50" in text


# ---------------------------------------------------------------------------
# PriceTrackerTool: PDF attachment conversion
# ---------------------------------------------------------------------------


def test_pdf_attachment_converted_to_image() -> None:
    tool = _make_tool()
    fake_png = b"fake-png-from-pdf"

    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = fake_png
    mock_page.get_pixmap.return_value = mock_pixmap
    mock_doc.load_page.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    with patch.dict("sys.modules", {"fitz": mock_fitz}):
        result = tool._encode_attachment("/tmp/receipt.pdf", "application/pdf")

    assert result == fake_png
    mock_fitz.open.assert_called_once_with("/tmp/receipt.pdf")
    mock_doc.load_page.assert_called_once_with(0)
    mock_pixmap.tobytes.assert_called_once_with("png")
