"""Core agent runtime."""

from __future__ import annotations

import asyncio
import json

from assistant.db import Database
from assistant.llm.base import LLMProvider
from assistant.models import Message
from assistant.tools.registry import ToolRegistry


class AgentRuntime:
    """Group-isolated runtime orchestrating memory, tools, and model calls."""

    def __init__(
        self,
        db: Database,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        memory_window_messages: int,
        summary_trigger_messages: int,
        request_timeout_seconds: float,
    ) -> None:
        self._db = db
        self._llm = llm
        self._tool_registry = tool_registry
        self._memory_window_messages = memory_window_messages
        self._summary_trigger_messages = summary_trigger_messages
        self._request_timeout_seconds = request_timeout_seconds

    async def handle_message(self, message: Message) -> str:
        """Handle one inbound user message and return assistant reply."""

        self._db.upsert_group(message.group_id)
        self._db.add_message(message.group_id, role="user", content=message.text, sender_id=message.sender_id)

        await self._maybe_summarize(message.group_id)
        context = self._build_context(message.group_id)

        response = await asyncio.wait_for(
            self._llm.generate(context, tools=self._tool_registry.list_tool_specs()),
            timeout=self._request_timeout_seconds,
        )

        if response.tool_calls:
            tool_messages: list[dict[str, str]] = []
            for tool_call in response.tool_calls:
                if "group_id" not in tool_call.arguments:
                    tool_call.arguments["group_id"] = message.group_id
                result = await self._tool_registry.execute(message.group_id, tool_call.name, tool_call.arguments)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )

            final_response = await asyncio.wait_for(
                self._llm.generate(context + [{"role": "assistant", "content": response.content}] + tool_messages),
                timeout=self._request_timeout_seconds,
            )
            reply = final_response.content
        else:
            reply = response.content

        self._db.add_message(message.group_id, role="assistant", content=reply)
        return reply

    def _build_context(self, group_id: str) -> list[dict[str, str]]:
        summary = self._db.get_summary(group_id)
        history = self._db.get_recent_messages(group_id, self._memory_window_messages)
        system_content = "You are a helpful personal AI assistant."
        if summary:
            system_content += f"\nConversation summary:\n{summary}"
        return [{"role": "system", "content": system_content}, *history]

    async def _maybe_summarize(self, group_id: str) -> None:
        messages = self._db.get_recent_messages(group_id, self._summary_trigger_messages)
        if len(messages) < self._summary_trigger_messages:
            return

        prompt = [
            {
                "role": "system",
                "content": "Summarize this conversation briefly for long-term memory.",
            },
            *messages,
        ]
        summary_response = await asyncio.wait_for(
            self._llm.generate(prompt), timeout=self._request_timeout_seconds
        )
        self._db.save_summary(group_id, summary_response.content)
