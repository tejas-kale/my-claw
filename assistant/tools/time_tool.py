"""Time utility tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from assistant.tools.base import Tool


class GetCurrentTimeTool(Tool):
    """Returns current UTC time."""

    name = "get_current_time"
    description = "Get the current UTC date/time in ISO-8601 format."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> dict[str, str]:
        return {"utc_time": datetime.now(timezone.utc).isoformat()}

