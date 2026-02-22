"""LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from assistant.models import LLMResponse


class LLMProvider(ABC):
    """Abstract model provider used by the agent runtime."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a model response."""
