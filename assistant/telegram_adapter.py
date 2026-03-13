"""Telegram Bot API adapter using long-polling."""

from __future__ import annotations

import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import AsyncIterator

import httpx

from assistant.models import Message

LOGGER = logging.getLogger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org"
_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".ogg"}


class TelegramAdapter:
    """Adapter for the Telegram Bot API using httpx long-polling."""

    def __init__(
        self,
        bot_token: str,
        poll_timeout: int,
        allowed_sender_ids: frozenset[str],
    ) -> None:
        self._bot_token = bot_token
        self._poll_timeout = poll_timeout
        self._allowed_sender_ids = allowed_sender_ids
        self._offset: int | None = None
        self._base_url = f"{_TELEGRAM_API_BASE}/bot{bot_token}"

    async def poll_messages(self) -> AsyncIterator[Message]:
        """Long-poll getUpdates and yield normalized Message objects."""

        async with httpx.AsyncClient(timeout=self._poll_timeout + 10) as client:
            # Drain stale messages on startup.
            await self._drain_stale(client)

            while True:
                params: dict = {"timeout": self._poll_timeout}
                if self._offset is not None:
                    params["offset"] = self._offset

                try:
                    resp = await client.get(f"{self._base_url}/getUpdates", params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    LOGGER.exception("getUpdates failed; retrying")
                    continue

                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1
                    message = await self._to_message(client, update)
                    if message is None:
                        continue
                    sender = message.sender_id
                    if sender not in self._allowed_sender_ids:
                        LOGGER.warning("Dropping message from unauthorized sender %s", sender)
                        continue
                    yield message

    async def _drain_stale(self, client: httpx.AsyncClient) -> None:
        """Discard messages that arrived before startup."""

        try:
            resp = await client.get(
                f"{self._base_url}/getUpdates",
                params={"timeout": 0, "offset": -1},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            if results:
                self._offset = results[-1]["update_id"] + 1
                LOGGER.info("Drained %d stale update(s); starting from offset %d", len(results), self._offset)
        except Exception:
            LOGGER.exception("Failed to drain stale messages; continuing from offset 0")

    async def _to_message(self, client: httpx.AsyncClient, update: dict) -> Message | None:
        """Normalize a Telegram update dict into a Message."""

        raw = update.get("message") or update.get("edited_message")
        if not isinstance(raw, dict):
            return None

        chat = raw.get("chat", {})
        chat_id = str(chat.get("id", ""))
        chat_type = chat.get("type", "private")
        is_group = chat_type in ("group", "supergroup", "channel")

        sender = raw.get("from", {})
        sender_id = str(sender.get("id", ""))

        text = raw.get("text") or raw.get("caption") or ""
        text = text.strip()

        attachments: list[dict[str, str]] = []

        # Download attachment if present.
        file_id: str | None = None
        content_type = "application/octet-stream"
        filename = ""

        if "document" in raw:
            doc = raw["document"]
            file_id = doc.get("file_id")
            content_type = doc.get("mime_type", "application/octet-stream")
            filename = doc.get("file_name", "")
        elif "photo" in raw:
            photos = raw["photo"]
            # Use the last (largest) photo.
            if photos:
                file_id = photos[-1].get("file_id")
                content_type = "image/jpeg"
                filename = "photo.jpg"
        elif "audio" in raw:
            audio = raw["audio"]
            file_id = audio.get("file_id")
            content_type = audio.get("mime_type", "audio/mpeg")
            filename = audio.get("file_name", "audio.mp3")
        elif "voice" in raw:
            voice = raw["voice"]
            file_id = voice.get("file_id")
            content_type = voice.get("mime_type", "audio/ogg")
            filename = "voice.ogg"

        if file_id:
            local_path = await self._download_file(client, file_id, filename)
            if local_path:
                if not content_type:
                    guessed, _ = mimetypes.guess_type(local_path)
                    content_type = guessed or "application/octet-stream"
                attachments.append({
                    "local_path": local_path,
                    "content_type": content_type,
                    "filename": filename,
                })

        if not text and not attachments:
            return None

        import datetime
        timestamp = datetime.datetime.fromtimestamp(
            raw.get("date", 0), tz=datetime.timezone.utc
        )

        return Message(
            group_id=chat_id,
            sender_id=sender_id,
            text=text,
            timestamp=timestamp,
            message_id=str(raw.get("message_id", "")),
            is_group=is_group,
            attachments=attachments,
        )

    async def _download_file(
        self, client: httpx.AsyncClient, file_id: str, hint_filename: str
    ) -> str | None:
        """Download a Telegram file and return the local temp path."""

        try:
            resp = await client.get(f"{self._base_url}/getFile", params={"file_id": file_id})
            resp.raise_for_status()
            result = resp.json().get("result", {})
            file_path = result.get("file_path")
            if not file_path:
                return None

            download_url = f"{_TELEGRAM_API_BASE}/file/bot{self._bot_token}/{file_path}"
            file_resp = await client.get(download_url)
            file_resp.raise_for_status()

            suffix = Path(hint_filename).suffix or Path(file_path).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_resp.content)
                return tmp.name
        except Exception:
            LOGGER.exception("Failed to download file_id=%s", file_id)
            return None

    async def send_message(
        self,
        chat_id: str,
        text: str,
        is_group: bool = True,
        attachment_path: str | None = None,
    ) -> None:
        """Send a text message or file to a Telegram chat."""

        async with httpx.AsyncClient(timeout=60) as client:
            if attachment_path is not None:
                path = Path(attachment_path)
                ext = path.suffix.lower()
                if ext in _AUDIO_EXTENSIONS:
                    endpoint = "sendAudio"
                    field = "audio"
                else:
                    endpoint = "sendDocument"
                    field = "document"

                with open(attachment_path, "rb") as f:
                    resp = await client.post(
                        f"{self._base_url}/{endpoint}",
                        data={"chat_id": chat_id, "caption": text},
                        files={field: (path.name, f)},
                    )
            else:
                resp = await client.post(
                    f"{self._base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                )

            if resp.status_code != 200:
                LOGGER.warning("Telegram send failed (%d): %s", resp.status_code, resp.text)
            else:
                resp.raise_for_status()
