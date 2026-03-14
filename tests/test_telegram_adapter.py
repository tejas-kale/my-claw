"""Tests for TelegramAdapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.telegram_adapter import TelegramAdapter


def _make_adapter(allowed_ids: frozenset[str] | None = None) -> TelegramAdapter:
    return TelegramAdapter(
        bot_token="test-token",
        poll_timeout=1,
        allowed_sender_ids=allowed_ids or frozenset({"123"}),
    )


def _mock_json_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _make_update(
    update_id: int = 1,
    sender_id: str = "123",
    text: str = "hello",
    chat_id: str = "456",
    chat_type: str = "private",
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender_id},
            "chat": {"id": int(chat_id), "type": chat_type},
            "text": text,
            "date": 1700000000,
        },
    }


# --- _drain_stale ---


@pytest.mark.asyncio
async def test_drain_stale_sets_offset_from_last_update():
    adapter = _make_adapter()
    results = [{"update_id": 10}, {"update_id": 11}]
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_json_response({"ok": True, "result": results}))

    await adapter._drain_stale(mock_client)

    assert adapter._offset == 12


@pytest.mark.asyncio
async def test_drain_stale_empty_result_leaves_offset_none():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_json_response({"ok": True, "result": []}))

    await adapter._drain_stale(mock_client)

    assert adapter._offset is None


# --- _to_message ---


@pytest.mark.asyncio
async def test_to_message_text_message():
    adapter = _make_adapter()
    update = _make_update(update_id=5, sender_id="123", text="hello world", chat_id="456")
    mock_client = AsyncMock()

    msg = await adapter._to_message(mock_client, update)

    assert msg is not None
    assert msg.sender_id == "123"
    assert msg.group_id == "456"
    assert msg.text == "hello world"
    assert msg.is_group is False
    assert msg.attachments == []


@pytest.mark.asyncio
async def test_to_message_supergroup_sets_is_group():
    adapter = _make_adapter()
    update = _make_update(chat_type="supergroup")
    mock_client = AsyncMock()

    msg = await adapter._to_message(mock_client, update)

    assert msg is not None
    assert msg.is_group is True


@pytest.mark.asyncio
async def test_to_message_no_text_no_attachment_returns_none():
    adapter = _make_adapter()
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "from": {"id": "123"},
            "chat": {"id": 456, "type": "private"},
            "date": 1700000000,
        },
    }
    mock_client = AsyncMock()

    msg = await adapter._to_message(mock_client, update)

    assert msg is None


@pytest.mark.asyncio
async def test_to_message_non_message_update_returns_none():
    adapter = _make_adapter()
    update = {"update_id": 1, "callback_query": {"data": "something"}}
    mock_client = AsyncMock()

    msg = await adapter._to_message(mock_client, update)

    assert msg is None


@pytest.mark.asyncio
async def test_to_message_photo_downloads_attachment():
    adapter = _make_adapter()
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "from": {"id": "123"},
            "chat": {"id": 456, "type": "private"},
            "photo": [{"file_id": "file123", "width": 100, "height": 100}],
            "date": 1700000000,
        },
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=_mock_json_response({"ok": True, "result": {"file_path": "photos/img.jpg"}})
    )

    stream_ctx = AsyncMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)
    stream_ctx.raise_for_status = MagicMock()

    async def aiter_bytes():
        yield b"fake-image-data"

    stream_ctx.aiter_bytes = aiter_bytes
    mock_client.stream = MagicMock(return_value=stream_ctx)

    msg = await adapter._to_message(mock_client, update)

    assert msg is not None
    assert len(msg.attachments) == 1
    attachment = msg.attachments[0]
    assert attachment["content_type"] == "image/jpeg"
    local_path = Path(attachment["local_path"])
    assert local_path.exists()
    assert local_path.read_bytes() == b"fake-image-data"
    local_path.unlink(missing_ok=True)


# --- poll_messages: sender authorization and offset tracking ---


@pytest.mark.asyncio
async def test_poll_messages_drops_unauthorized_sender():
    adapter = _make_adapter(allowed_ids=frozenset({"authorized"}))
    adapter._drain_stale = AsyncMock()

    call_count = 0

    async def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_json_response({
                "ok": True,
                "result": [
                    _make_update(update_id=1, sender_id="unauthorized", text="bad"),
                    _make_update(update_id=2, sender_id="authorized", text="good"),
                ],
            })
        raise asyncio.CancelledError()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = mock_get

    messages = []
    with patch("assistant.telegram_adapter.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(asyncio.CancelledError):
            async for msg in adapter.poll_messages():
                messages.append(msg)

    assert len(messages) == 1
    assert messages[0].sender_id == "authorized"


@pytest.mark.asyncio
async def test_poll_messages_advances_offset_per_update():
    adapter = _make_adapter()
    adapter._drain_stale = AsyncMock()

    call_count = 0

    async def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_json_response({
                "ok": True,
                "result": [_make_update(update_id=7, sender_id="123", text="hi")],
            })
        raise asyncio.CancelledError()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = mock_get

    with patch("assistant.telegram_adapter.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(asyncio.CancelledError):
            async for _ in adapter.poll_messages():
                pass

    assert adapter._offset == 8


# --- send_message routing ---


@pytest.mark.asyncio
async def test_send_message_text_calls_sendmessage():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_json_response({"ok": True}))

    with patch("assistant.telegram_adapter.httpx.AsyncClient", return_value=mock_client):
        await adapter.send_message("456", "hello")

    url = mock_client.post.call_args.args[0]
    assert url.endswith("/sendMessage")


@pytest.mark.asyncio
async def test_send_message_mp3_calls_sendaudio():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_json_response({"ok": True}))

    with patch("assistant.telegram_adapter.httpx.AsyncClient", return_value=mock_client):
        with patch("builtins.open", MagicMock()):
            await adapter.send_message("456", "listen", attachment_path="/tmp/clip.mp3")

    url = mock_client.post.call_args.args[0]
    assert url.endswith("/sendAudio")


@pytest.mark.asyncio
async def test_send_message_pdf_calls_senddocument():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_json_response({"ok": True}))

    with patch("assistant.telegram_adapter.httpx.AsyncClient", return_value=mock_client):
        with patch("builtins.open", MagicMock()):
            await adapter.send_message("456", "here's a doc", attachment_path="/tmp/report.pdf")

    url = mock_client.post.call_args.args[0]
    assert url.endswith("/sendDocument")
