"""Tool contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all assistant tools."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute tool with validated arguments."""
