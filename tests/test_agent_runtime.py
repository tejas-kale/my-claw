from datetime import datetime, timezone

import pytest

from assistant.agent_runtime import AgentRuntime
from assistant.db import Database
from assistant.models import LLMResponse, Message
from assistant.tools.registry import ToolRegistry


class FakeProvider:
    async def generate(self, messages, tools=None, response_format=None):  # noqa: ANN001, ANN201
        return LLMResponse(content="hello")


@pytest.mark.asyncio
async def test_agent_runtime_returns_reply(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()

    registry = ToolRegistry(db)
    runtime = AgentRuntime(
        db=db,
        llm=FakeProvider(),
        tool_registry=registry,
        memory_window_messages=10,
        summary_trigger_messages=100,
        request_timeout_seconds=5,
    )

    reply = await runtime.handle_message(
        Message(
            group_id="group-1",
            sender_id="user-1",
            text="hi",
            timestamp=datetime.now(timezone.utc),
        )
    )
    assert reply == "hello"

    history = db.get_recent_messages("group-1", limit=10)
    assert [m["role"] for m in history] == ["user", "assistant"]
