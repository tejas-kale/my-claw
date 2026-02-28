"""Signal CLI adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from assistant.models import Message

LOGGER = logging.getLogger(__name__)


class SignalAdapter:
    """Adapter around signal-cli JSON commands."""

    def __init__(
        self,
        signal_cli_path: str,
        account: str,
        poll_interval_seconds: float,
        owner_number: str,
        allowed_senders: frozenset[str],
    ) -> None:
        self._signal_cli_path = signal_cli_path
        self._account = account
        self._poll_interval_seconds = poll_interval_seconds
        self._owner_number = owner_number
        self._allowed_senders = allowed_senders

    async def start_daemon(self) -> None:
        """Start signal-cli daemon process in background."""

        process = await asyncio.create_subprocess_exec(
            self._signal_cli_path,
            "-o",
            "json",
            "-a",
            self._account,
            "daemon",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        LOGGER.info("Started signal-cli daemon with pid %s", process.pid)

    async def poll_messages(self) -> AsyncIterator[Message]:
        """Poll receive endpoint and yield normalized message objects."""

        while True:
            process = await asyncio.create_subprocess_exec(
                self._signal_cli_path,
                "-o",
                "json",
                "-a",
                self._account,
                "receive",
                "-t",
                str(int(self._poll_interval_seconds)),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                LOGGER.warning("signal-cli receive failed: %s", stderr.decode().strip())
                await asyncio.sleep(self._poll_interval_seconds)
                continue

            for line in stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    message = _to_message(payload)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if message is not None:
                    sender = message.sender_id
                    if not sender.startswith("+"):
                        sender = await self.resolve_number(sender)
                        message.sender_id = sender
                    if sender not in self._allowed_senders:
                        LOGGER.warning(
                            "Dropping message from unauthorized sender %s", message.sender_id
                        )
                        continue
                    yield message

    async def resolve_number(self, uuid: str) -> str:
        """Return the phone number for a UUID by scanning the contacts list.

        Falls back to the original UUID if not found.
        """
        process = await asyncio.create_subprocess_exec(
            self._signal_cli_path,
            "-o",
            "json",
            "-a",
            self._account,
            "listContacts",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        raw = stdout.decode()
        for line in raw.splitlines():
            try:
                contact = json.loads(line)
                if contact.get("uuid") == uuid and contact.get("number"):
                    return contact["number"]
            except (json.JSONDecodeError, AttributeError):
                continue
        LOGGER.warning("Could not resolve UUID %s via contacts, falling back to owner number", uuid)
        return self._owner_number

    async def send_message(
        self,
        recipient: str,
        text: str,
        is_group: bool = True,
        attachment_path: str | None = None,
    ) -> None:
        """Send text message to Signal recipient, optionally with a file attachment."""

        if not is_group and not recipient.startswith("+"):
            recipient = await self.resolve_number(recipient)

        args = [
            self._signal_cli_path,
            "-a",
            self._account,
            "send",
            "-m",
            text,
        ]
        if is_group:
            args.extend(["-g", recipient])
        else:
            args.append(recipient)
        if attachment_path is not None:
            args.extend(["-a", attachment_path])

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"signal-cli send failed: {stderr.decode().strip()}")


_SIGNAL_ATTACHMENTS_DIR = "~/.local/share/signal-cli/attachments"


def _parse_attachments(raw: list[object]) -> list[dict[str, str]]:
    """Normalise signal-cli attachment dicts into a consistent internal shape.

    Each returned dict has at minimum a 'local_path' key constructed from the
    attachment id if no explicit file path is present in the signal-cli output.
    """
    import os

    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Newer signal-cli versions include the stored path directly.
        local_path: str = (
            item.get("file")  # type: ignore[assignment]
            or item.get("storedFilename")
            or os.path.expanduser(
                f"{_SIGNAL_ATTACHMENTS_DIR}/{item.get('id', '')}"
            )
        )
        result.append(
            {
                "local_path": str(local_path),
                "content_type": str(item.get("contentType", "application/octet-stream")),
                "filename": str(item.get("filename") or ""),
            }
        )
    return result


def _to_message(payload: dict[str, object]) -> Message | None:
    envelope = payload.get("envelope")
    if not isinstance(envelope, dict):
        return None
    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        return None

    text = data_message.get("message")
    text = text.strip() if isinstance(text, str) else ""

    raw_attachments = data_message.get("attachments")
    attachments = _parse_attachments(raw_attachments if isinstance(raw_attachments, list) else [])

    # Drop messages with no content at all.
    if not text and not attachments:
        return None

    source = str(envelope.get("source") or "unknown")
    timestamp_ms = int(envelope.get("timestamp") or 0)
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    group_info = data_message.get("groupInfo")
    if isinstance(group_info, dict) and isinstance(group_info.get("groupId"), str):
        group_id = group_info["groupId"]
        is_group = True
    else:
        group_id = source
        is_group = False

    return Message(
        group_id=group_id,
        sender_id=source,
        text=text,
        timestamp=timestamp,
        message_id=str(envelope.get("timestamp") or ""),
        is_group=is_group,
        attachments=attachments,
    )
