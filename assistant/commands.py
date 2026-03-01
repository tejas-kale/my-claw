"""Command dispatcher for @-prefixed messages.

Commands bypass the LLM and invoke tools directly.
An unrecognised @command returns None, letting it fall through to the LLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from assistant.models import Message
from assistant.tools.podcast_tool import PODCAST_TYPES

if TYPE_CHECKING:
    from assistant.tools.podcast_tool import PodcastTool

LOGGER = logging.getLogger(__name__)

_PODCAST_USAGE = f"Usage: @podcast <type> [url]\nValid types: {', '.join(PODCAST_TYPES)}"


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Split an @-prefixed message into (command, args).

    Returns:
        A (command, args) tuple where command is lowercased, or None if text
        is not a valid @command.
    """
    text = text.strip()
    if not text.startswith("@"):
        return None
    parts = text[1:].split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


class CommandDispatcher:
    """Routes @-prefixed messages to tool handlers, bypassing the LLM.

    Returns None for unrecognised commands so the caller can fall through.
    """

    def __init__(self, podcast_tool: PodcastTool | None = None) -> None:
        self._podcast_tool = podcast_tool

    async def dispatch(self, message: Message) -> str | None:
        """Dispatch a message to a command handler.

        Returns:
            A reply string for recognised commands, or None for unknown ones.
        """
        parsed = parse_command(message.text)
        if parsed is None:
            return None
        command, args = parsed
        LOGGER.info("Command dispatch: command=%r args=%r", command, args)
        if command == "podcast":
            return await self._handle_podcast(args, message)
        return None

    async def _handle_podcast(self, args: list[str], message: Message) -> str:
        if self._podcast_tool is None:
            return "Podcast tool is not configured."
        if not args:
            return _PODCAST_USAGE

        podcast_type = args[0]
        if podcast_type not in PODCAST_TYPES:
            return f"Unknown podcast type '{podcast_type}'.\n{_PODCAST_USAGE}"

        # URL wins over attachment when both are present.
        source_url: str | None = next((a for a in args[1:] if a.startswith("http")), None)
        attachment_path: str | None = (
            None if source_url else (
                message.attachments[0]["local_path"] if message.attachments else None
            )
        )

        if not source_url and not attachment_path:
            return f"Attach a PDF or provide a URL.\n{_PODCAST_USAGE}"

        kwargs: dict[str, Any] = {
            "group_id": message.group_id,
            "is_group": message.is_group,
            "podcast_type": podcast_type,
        }
        if source_url:
            kwargs["source_url"] = source_url
        if attachment_path:
            kwargs["attachment_path"] = attachment_path

        result = await self._podcast_tool.run(**kwargs)
        if "error" in result:
            return f"Podcast failed: {result['error']}"
        return result.get("message", "Podcast generation started.")
