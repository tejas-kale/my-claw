"""Core domain models used across layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Message:
    """Message normalized by adapters for runtime usage."""

    group_id: str
    sender_id: str
    text: str
    timestamp: datetime
    message_id: str | None = None
    is_group: bool = True


@dataclass(slots=True)
class LLMToolCall:
    """Tool invocation returned by an LLM provider."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(slots=True)
class LLMResponse:
    """Result from an LLM generation request."""

    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class ScheduledTask:
    """Represents a persisted scheduled task."""

    id: int
    group_id: str
    prompt: str
    run_at: datetime
    status: str
