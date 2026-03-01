"""Tests for the @command dispatch system.

Written RED-first: all tests in this file must fail before implementation begins.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.commands import CommandDispatcher, parse_command
from assistant.db import Database
from assistant.models import LLMResponse, Message
from assistant.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(text: str, attachments: list[dict] | None = None) -> Message:
    return Message(
        group_id="group-1",
        sender_id="user-1",
        text=text,
        timestamp=datetime.now(timezone.utc),
        attachments=attachments or [],
    )


def _podcast_tool(return_value: dict[str, Any]) -> Any:
    tool = MagicMock()
    tool.run = AsyncMock(return_value=return_value)
    return tool


# ===========================================================================
# parse_command
# ===========================================================================


class TestParseCommand:
    def test_regular_text_returns_none(self):
        assert parse_command("hello world") is None

    def test_empty_string_returns_none(self):
        assert parse_command("") is None

    def test_at_sign_alone_returns_none(self):
        assert parse_command("@") is None

    def test_at_sign_with_whitespace_only_returns_none(self):
        assert parse_command("@   ") is None

    def test_command_with_no_args(self):
        assert parse_command("@podcast") == ("podcast", [])

    def test_command_with_one_arg(self):
        assert parse_command("@podcast econpod") == ("podcast", ["econpod"])

    def test_command_with_url_arg(self):
        result = parse_command("@podcast econpod https://example.com/paper.pdf")
        assert result == ("podcast", ["econpod", "https://example.com/paper.pdf"])

    def test_command_keyword_is_lowercased(self):
        assert parse_command("@PODCAST econpod") == ("podcast", ["econpod"])

    def test_leading_trailing_whitespace_stripped(self):
        assert parse_command("  @podcast econpod  ") == ("podcast", ["econpod"])

    def test_unknown_command_is_parsed(self):
        assert parse_command("@websearch kagi api") == ("websearch", ["kagi", "api"])

    def test_args_case_preserved(self):
        # args are NOT lowercased — only the command keyword is
        cmd, args = parse_command("@podcast EconPod")  # type: ignore[misc]
        assert args == ["EconPod"]


# ===========================================================================
# CommandDispatcher.dispatch — routing
# ===========================================================================


class TestCommandDispatcherRouting:
    @pytest.mark.asyncio
    async def test_non_at_message_returns_none(self):
        dispatcher = CommandDispatcher()
        assert await dispatcher.dispatch(_msg("hello world")) is None

    @pytest.mark.asyncio
    async def test_unknown_at_command_returns_none(self):
        dispatcher = CommandDispatcher()
        assert await dispatcher.dispatch(_msg("@foobar something")) is None

    @pytest.mark.asyncio
    async def test_at_sign_alone_returns_none(self):
        dispatcher = CommandDispatcher()
        assert await dispatcher.dispatch(_msg("@")) is None


# ===========================================================================
# CommandDispatcher.dispatch — @podcast argument validation
# ===========================================================================


class TestCommandDispatcherPodcast:
    @pytest.mark.asyncio
    async def test_missing_podcast_type_returns_usage(self):
        dispatcher = CommandDispatcher(podcast_tool=_podcast_tool({}))
        result = await dispatcher.dispatch(_msg("@podcast"))
        assert result is not None
        assert "usage" in result.lower() or "valid types" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_podcast_type_returns_error(self):
        dispatcher = CommandDispatcher(podcast_tool=_podcast_tool({}))
        result = await dispatcher.dispatch(_msg("@podcast nosuchtype"))
        assert result is not None
        assert "nosuchtype" in result

    @pytest.mark.asyncio
    async def test_valid_type_with_no_source_returns_error(self):
        dispatcher = CommandDispatcher(podcast_tool=_podcast_tool({}))
        result = await dispatcher.dispatch(_msg("@podcast econpod"))
        assert result is not None
        assert "pdf" in result.lower() or "url" in result.lower() or "attach" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_not_configured_returns_error(self):
        dispatcher = CommandDispatcher(podcast_tool=None)
        result = await dispatcher.dispatch(
            _msg("@podcast econpod https://example.com/paper.pdf")
        )
        assert result is not None
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_error_is_surfaced(self):
        tool = _podcast_tool({"error": "nlm not found on PATH"})
        dispatcher = CommandDispatcher(podcast_tool=tool)
        result = await dispatcher.dispatch(
            _msg("@podcast econpod https://example.com/paper.pdf")
        )
        assert result is not None
        assert "nlm not found on PATH" in result

    @pytest.mark.asyncio
    async def test_url_source_calls_tool_with_correct_kwargs(self):
        tool = _podcast_tool({"message": "Podcast generation started."})
        dispatcher = CommandDispatcher(podcast_tool=tool)
        result = await dispatcher.dispatch(
            _msg("@podcast econpod https://example.com/paper.pdf")
        )
        assert result == "Podcast generation started."
        kw = tool.run.call_args.kwargs
        assert kw["podcast_type"] == "econpod"
        assert kw["source_url"] == "https://example.com/paper.pdf"
        assert kw["group_id"] == "group-1"
        assert "attachment_path" not in kw

    @pytest.mark.asyncio
    async def test_attachment_source_calls_tool_with_correct_kwargs(self):
        tool = _podcast_tool({"message": "Podcast generation started."})
        dispatcher = CommandDispatcher(podcast_tool=tool)
        result = await dispatcher.dispatch(
            _msg(
                "@podcast cspod",
                attachments=[{"local_path": "/tmp/paper.pdf", "content_type": "application/pdf"}],
            )
        )
        assert result == "Podcast generation started."
        kw = tool.run.call_args.kwargs
        assert kw["podcast_type"] == "cspod"
        assert kw["attachment_path"] == "/tmp/paper.pdf"
        assert "source_url" not in kw

    @pytest.mark.asyncio
    async def test_url_arg_takes_priority_over_attachment(self):
        """When both URL arg and attachment are present, URL wins."""
        tool = _podcast_tool({"message": "ok"})
        dispatcher = CommandDispatcher(podcast_tool=tool)
        await dispatcher.dispatch(
            _msg(
                "@podcast econpod https://example.com/paper.pdf",
                attachments=[{"local_path": "/tmp/other.pdf", "content_type": "application/pdf"}],
            )
        )
        kw = tool.run.call_args.kwargs
        assert kw["source_url"] == "https://example.com/paper.pdf"
        assert "attachment_path" not in kw

    @pytest.mark.asyncio
    async def test_all_valid_podcast_types_are_accepted(self):
        from assistant.tools.podcast_tool import PODCAST_TYPES

        for podcast_type in PODCAST_TYPES:
            tool = _podcast_tool({"message": "ok"})
            dispatcher = CommandDispatcher(podcast_tool=tool)
            result = await dispatcher.dispatch(
                _msg(f"@podcast {podcast_type} https://example.com/p.pdf")
            )
            assert result == "ok", f"Expected success for type {podcast_type!r}, got {result!r}"


# ===========================================================================
# CommandDispatcher.dispatch — @websearch ddg
# ===========================================================================


def _search_tool(return_value: str) -> Any:
    tool = MagicMock()
    tool.run = AsyncMock(return_value=return_value)
    return tool


def _llm_search(
    sub_queries: list[str],
    ranked_urls: list[str],
    final: str = "synthesized answer",
) -> Any:
    """Mock LLM that handles the 3-call search flow: sub-queries → ranking → synthesis."""
    from assistant.models import LLMResponse

    llm = MagicMock()
    llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(content=json.dumps({"queries": sub_queries})),
            LLMResponse(content=json.dumps({"urls": ranked_urls})),
            LLMResponse(content=final),
        ]
    )
    return llm


class TestCommandDispatcherWebsearch:
    # --- Error/usage cases (no LLM needed) ---

    @pytest.mark.asyncio
    async def test_websearch_kagi_no_tool_returns_error(self):
        dispatcher = CommandDispatcher()
        result = await dispatcher.dispatch(_msg("@websearch something"))
        assert result is not None
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_websearch_ddg_no_tool_returns_error(self):
        dispatcher = CommandDispatcher()
        result = await dispatcher.dispatch(_msg("@websearch ddg something"))
        assert result is not None
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_websearch_no_args_returns_usage(self):
        dispatcher = CommandDispatcher()
        result = await dispatcher.dispatch(_msg("@websearch"))
        assert result is not None
        assert "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_websearch_ddg_missing_query_returns_usage(self):
        tool = _search_tool("should not be called")
        dispatcher = CommandDispatcher(ddg_search_tool=tool)
        result = await dispatcher.dispatch(_msg("@websearch ddg"))
        assert result is not None
        assert "usage" in result.lower()
        tool.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_websearch_no_llm_returns_raw_results(self):
        tool = _search_tool("raw results text")
        dispatcher = CommandDispatcher(kagi_search_tool=tool)
        result = await dispatcher.dispatch(_msg("@websearch something"))
        assert result == "raw results text"

    # --- Sub-query generation ---

    @pytest.mark.asyncio
    async def test_websearch_generates_sub_queries_from_original_query(self):
        llm = _llm_search(["elim chamber 2026"], [])
        search = _search_tool("raw results")
        dispatcher = CommandDispatcher(kagi_search_tool=search, llm=llm)
        await dispatcher.dispatch(_msg("@websearch elimination chamber 2026"))
        first_call_msgs = llm.generate.call_args_list[0][0][0]
        user_msg = next(m["content"] for m in first_call_msgs if m["role"] == "user")
        assert "elimination chamber 2026" in user_msg

    @pytest.mark.asyncio
    async def test_websearch_runs_all_sub_queries(self):
        llm = _llm_search(["query one", "query two"], [])
        search = _search_tool("raw results")
        dispatcher = CommandDispatcher(kagi_search_tool=search, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        assert search.run.call_count == 2
        called = {c.kwargs["query"] for c in search.run.call_args_list}
        assert called == {"query one", "query two"}

    @pytest.mark.asyncio
    async def test_websearch_falls_back_to_original_query_on_bad_json(self):
        from assistant.models import LLMResponse

        llm = MagicMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(content="not json at all"),
                LLMResponse(content=json.dumps({"urls": []})),
                LLMResponse(content="fallback answer"),
            ]
        )
        search = _search_tool("raw results")
        dispatcher = CommandDispatcher(kagi_search_tool=search, llm=llm)
        await dispatcher.dispatch(_msg("@websearch original query"))
        search.run.assert_called_once_with(query="original query")

    @pytest.mark.asyncio
    async def test_websearch_ddg_generates_sub_queries_without_ddg_prefix(self):
        llm = _llm_search(["generated sub-query"], [])
        tool = _search_tool("raw ddg results")
        dispatcher = CommandDispatcher(ddg_search_tool=tool, llm=llm)
        await dispatcher.dispatch(_msg("@websearch ddg elimination chamber"))
        first_call_msgs = llm.generate.call_args_list[0][0][0]
        user_msg = next(m["content"] for m in first_call_msgs if m["role"] == "user")
        assert "elimination chamber" in user_msg
        assert "ddg" not in user_msg.lower()

    # --- Ranking ---

    @pytest.mark.asyncio
    async def test_websearch_ranks_results_via_llm(self):
        llm = _llm_search(["q"], ["https://best.com", "https://second.com"])
        search = _search_tool("raw results")
        dispatcher = CommandDispatcher(kagi_search_tool=search, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        second_call_msgs = llm.generate.call_args_list[1][0][0]
        user_msg = next(m["content"] for m in second_call_msgs if m["role"] == "user")
        assert "something" in user_msg

    # --- Jina fetching ---

    @pytest.mark.asyncio
    async def test_websearch_fetches_jina_for_ranked_urls(self):
        llm = _llm_search(["q"], ["https://ranked-1.com", "https://ranked-2.com"])
        search = _search_tool("raw results")
        read_url = _search_tool("page content")
        dispatcher = CommandDispatcher(kagi_search_tool=search, read_url_tool=read_url, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        called_urls = [c.kwargs["url"] for c in read_url.run.call_args_list]
        assert "https://ranked-1.com" in called_urls
        assert "https://ranked-2.com" in called_urls

    @pytest.mark.asyncio
    async def test_websearch_stops_after_2_successful_jina_fetches(self):
        ranked = ["https://a.com", "https://b.com", "https://c.com"]
        llm = _llm_search(["q"], ranked)
        search = _search_tool("raw results")
        read_url = _search_tool("page content")
        dispatcher = CommandDispatcher(kagi_search_tool=search, read_url_tool=read_url, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        assert read_url.run.call_count == 2

    @pytest.mark.asyncio
    async def test_websearch_skips_failed_jina_and_tries_next(self):
        ranked = ["https://blocked.com", "https://works.com", "https://also-works.com"]
        llm = _llm_search(["q"], ranked, "final answer")
        search = _search_tool("raw results")
        read_url = MagicMock()
        read_url.run = AsyncMock(
            side_effect=[
                "Failed to read URL (HTTP 409): https://blocked.com",
                "page content 1",
                "page content 2",
            ]
        )
        dispatcher = CommandDispatcher(kagi_search_tool=search, read_url_tool=read_url, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        assert read_url.run.call_count == 3

    @pytest.mark.asyncio
    async def test_websearch_includes_jina_content_in_synthesis(self):
        llm = _llm_search(["q"], ["https://example.com"], "final answer")
        search = _search_tool("raw results")
        read_url = _search_tool("JINA PAGE CONTENT")
        dispatcher = CommandDispatcher(kagi_search_tool=search, read_url_tool=read_url, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        synthesis_msgs = llm.generate.call_args_list[2][0][0]
        user_msg = next(m["content"] for m in synthesis_msgs if m["role"] == "user")
        assert "JINA PAGE CONTENT" in user_msg

    @pytest.mark.asyncio
    async def test_websearch_skips_jina_when_not_configured(self):
        llm = _llm_search(["q"], ["https://example.com"], "answer without jina")
        search = _search_tool("raw results")
        dispatcher = CommandDispatcher(kagi_search_tool=search, llm=llm)
        result = await dispatcher.dispatch(_msg("@websearch something"))
        assert result == "answer without jina"

    @pytest.mark.asyncio
    async def test_websearch_handles_jina_exception_and_tries_next(self):
        import httpx

        ranked = ["https://slow.com", "https://fast.com"]
        llm = _llm_search(["q"], ranked, "final answer")
        search = _search_tool("raw results")
        read_url = MagicMock()
        read_url.run = AsyncMock(
            side_effect=[
                httpx.ReadTimeout("timed out"),
                "good page content",
            ]
        )
        dispatcher = CommandDispatcher(kagi_search_tool=search, read_url_tool=read_url, llm=llm)
        result = await dispatcher.dispatch(_msg("@websearch something"))
        assert result == "final answer"
        assert read_url.run.call_count == 2

    @pytest.mark.asyncio
    async def test_websearch_skips_jina_when_ranking_returns_no_urls(self):
        llm = _llm_search(["q"], [])
        search = _search_tool("raw results")
        read_url = _search_tool("should not be called")
        dispatcher = CommandDispatcher(kagi_search_tool=search, read_url_tool=read_url, llm=llm)
        await dispatcher.dispatch(_msg("@websearch something"))
        read_url.run.assert_not_called()

    # --- Final synthesis ---

    @pytest.mark.asyncio
    async def test_websearch_returns_llm_synthesis(self):
        llm = _llm_search(["q"], [], "THE FINAL ANSWER")
        search = _search_tool("raw results")
        dispatcher = CommandDispatcher(kagi_search_tool=search, llm=llm)
        result = await dispatcher.dispatch(_msg("@websearch something"))
        assert result == "THE FINAL ANSWER"


# ===========================================================================
# AgentRuntime integration: command interception
# ===========================================================================


class FakeLLM:
    """LLM stub that records calls and returns a fixed reply."""

    def __init__(self, reply: str = "llm reply") -> None:
        self.calls: list[list[dict]] = []
        self._reply = reply

    async def generate(self, messages: list[dict], tools=None, response_format=None) -> LLMResponse:  # noqa: ANN001
        self.calls.append(messages)
        return LLMResponse(content=self._reply)


class TestAgentRuntimeCommandIntegration:
    """AgentRuntime must route @commands via CommandDispatcher and skip the LLM."""

    def _make_runtime(self, db: Database, llm: FakeLLM, dispatcher: CommandDispatcher):
        from assistant.agent_runtime import AgentRuntime

        return AgentRuntime(
            db=db,
            llm=llm,
            tool_registry=ToolRegistry(db),
            memory_window_messages=10,
            summary_trigger_messages=100,
            request_timeout_seconds=5,
            command_dispatcher=dispatcher,
        )

    @pytest.mark.asyncio
    async def test_known_command_bypasses_llm(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = FakeLLM()

        tool = _podcast_tool({"message": "Podcast started."})
        dispatcher = CommandDispatcher(podcast_tool=tool)
        runtime = self._make_runtime(db, llm, dispatcher)

        reply = await runtime.handle_message(
            _msg("@podcast econpod https://example.com/p.pdf")
        )

        assert reply == "Podcast started."
        assert llm.calls == [], "LLM must NOT be called for a known @command"

    @pytest.mark.asyncio
    async def test_known_command_saves_both_turns_to_history(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = FakeLLM()

        tool = _podcast_tool({"message": "Podcast started."})
        dispatcher = CommandDispatcher(podcast_tool=tool)
        runtime = self._make_runtime(db, llm, dispatcher)

        await runtime.handle_message(_msg("@podcast econpod https://example.com/p.pdf"))

        history = db.get_recent_messages("group-1", limit=10)
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_unknown_command_falls_through_to_llm(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = FakeLLM(reply="llm reply")

        dispatcher = CommandDispatcher()  # no tools wired → unknown command
        runtime = self._make_runtime(db, llm, dispatcher)

        reply = await runtime.handle_message(_msg("@foobar some query"))

        assert reply == "llm reply"
        assert len(llm.calls) == 1, "LLM must be called for unknown @commands"

    @pytest.mark.asyncio
    async def test_regular_message_goes_to_llm(self, tmp_path):
        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = FakeLLM(reply="llm reply")

        dispatcher = CommandDispatcher()
        runtime = self._make_runtime(db, llm, dispatcher)

        reply = await runtime.handle_message(_msg("what's the weather?"))

        assert reply == "llm reply"
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_runtime_without_dispatcher_handles_at_message_via_llm(self, tmp_path):
        """When no dispatcher is configured, @ messages go straight to the LLM."""
        from assistant.agent_runtime import AgentRuntime

        db = Database(tmp_path / "assistant.db")
        db.initialize()
        llm = FakeLLM(reply="llm reply")

        runtime = AgentRuntime(
            db=db,
            llm=llm,
            tool_registry=ToolRegistry(db),
            memory_window_messages=10,
            summary_trigger_messages=100,
            request_timeout_seconds=5,
        )

        reply = await runtime.handle_message(_msg("@podcast anything"))
        assert reply == "llm reply"
        assert len(llm.calls) == 1
