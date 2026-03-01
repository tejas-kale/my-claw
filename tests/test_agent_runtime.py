from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.agent_runtime import AgentRuntime
from assistant.db import Database
from assistant.models import LLMResponse, LLMToolCall, Message
from assistant.tools.registry import ToolRegistry


def _msg(text: str) -> Message:
    return Message(
        group_id="group-1",
        sender_id="user-1",
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


def _runtime(db: object, llm: object) -> AgentRuntime:
    return AgentRuntime(
        db=db,
        llm=llm,
        tool_registry=ToolRegistry(db),
        memory_window_messages=10,
        summary_trigger_messages=100,
        request_timeout_seconds=5,
    )


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


class TestWebSearchPermission:
    @pytest.mark.asyncio
    async def test_web_search_tool_call_returns_permission_request(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[LLMToolCall(name="web_search", arguments={"query": "Howard Lutnick"})],
            )
        )
        runtime = _runtime(db, llm)
        reply = await runtime.handle_message(_msg("Who is Howard Lutnick?"))
        assert "Howard Lutnick" in reply
        assert "search" in reply.lower()
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_permission_request_says_reply_ok(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[LLMToolCall(name="web_search", arguments={"query": "test"})],
            )
        )
        runtime = _runtime(db, llm)
        reply = await runtime.handle_message(_msg("something"))
        assert "ok" in reply.lower()

    @pytest.mark.asyncio
    async def test_web_search_permission_shows_all_proposed_queries(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[
                    LLMToolCall(name="web_search", arguments={"query": "query one"}),
                    LLMToolCall(name="web_search", arguments={"query": "query two"}),
                ],
            )
        )
        runtime = _runtime(db, llm)
        reply = await runtime.handle_message(_msg("something complex"))
        assert "query one" in reply
        assert "query two" in reply

    @pytest.mark.asyncio
    async def test_approval_dispatches_pending_web_search(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[LLMToolCall(name="web_search", arguments={"query": "Howard Lutnick"})],
            )
        )
        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(return_value="search results")
        runtime = AgentRuntime(
            db=db, llm=llm, tool_registry=ToolRegistry(db),
            memory_window_messages=10, summary_trigger_messages=100,
            request_timeout_seconds=5, command_dispatcher=dispatcher,
        )
        await runtime.handle_message(_msg("Who is Howard Lutnick?"))
        dispatcher.dispatch.reset_mock()

        reply = await runtime.handle_message(_msg("ok"))

        assert "search results" in reply
        dispatched_msg = dispatcher.dispatch.call_args[0][0]
        assert "websearch" in dispatched_msg.text
        assert "Howard Lutnick" in dispatched_msg.text

    @pytest.mark.asyncio
    async def test_approval_words_are_case_insensitive(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[LLMToolCall(name="web_search", arguments={"query": "test"})],
            )
        )
        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(return_value="search results")
        runtime = AgentRuntime(
            db=db, llm=llm, tool_registry=ToolRegistry(db),
            memory_window_messages=10, summary_trigger_messages=100,
            request_timeout_seconds=5, command_dispatcher=dispatcher,
        )
        for word in ("OK", "Yes", "YES", "sure", "Yep"):
            db2 = Database(tmp_path / f"assistant_{word}.db")
            db2.initialize()
            runtime2 = AgentRuntime(
                db=db2, llm=llm, tool_registry=ToolRegistry(db2),
                memory_window_messages=10, summary_trigger_messages=100,
                request_timeout_seconds=5, command_dispatcher=dispatcher,
            )
            llm.generate.reset_mock()
            llm.generate.return_value = LLMResponse(
                content="",
                tool_calls=[LLMToolCall(name="web_search", arguments={"query": "test"})],
            )
            await runtime2.handle_message(_msg("something"))
            dispatcher.dispatch.reset_mock()
            await runtime2.handle_message(_msg(word))
            assert dispatcher.dispatch.called, f"Expected approval for {word!r}"

    @pytest.mark.asyncio
    async def test_ok_without_pending_search_goes_to_llm(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=LLMResponse(content="llm reply"))
        runtime = _runtime(db, llm)
        reply = await runtime.handle_message(_msg("ok"))
        assert reply == "llm reply"
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_web_search_tool_calls_execute_normally(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        tool = MagicMock()
        tool.name = "get_current_time"
        tool.run = AsyncMock(return_value={"utc_time": "2026-01-01T00:00:00"})
        tool.parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

        registry = ToolRegistry(db)
        registry.register(tool)

        llm = MagicMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(name="get_current_time", call_id="c1", arguments={})],
                ),
                LLMResponse(content="It is 2026-01-01."),
            ]
        )
        runtime = AgentRuntime(
            db=db,
            llm=llm,
            tool_registry=registry,
            memory_window_messages=10,
            summary_trigger_messages=100,
            request_timeout_seconds=5,
        )
        reply = await runtime.handle_message(_msg("What time is it?"))
        assert reply == "It is 2026-01-01."
        assert llm.generate.call_count == 2
        tool.run.assert_called_once()
