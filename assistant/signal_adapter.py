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
    ) -> None:
        self._signal_cli_path = signal_cli_path
        self._account = account
        self._poll_interval_seconds = poll_interval_seconds
        self._owner_number = owner_number

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

    async def send_message(self, recipient: str, text: str, is_group: bool = True) -> None:
        """Send text message to Signal recipient."""

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

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"signal-cli send failed: {stderr.decode().strip()}")


def _to_message(payload: dict[str, object]) -> Message | None:
    envelope = payload.get("envelope")
    if not isinstance(envelope, dict):
        return None
    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        return None
    text = data_message.get("message")
    if not isinstance(text, str) or not text.strip():
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
        text=text.strip(),
        timestamp=timestamp,
        message_id=str(envelope.get("timestamp") or ""),
        is_group=is_group,
    )
