"""Simple notes tools."""

from __future__ import annotations

from typing import Any

from assistant.db import Database
from assistant.tools.base import Tool


class WriteNoteTool(Tool):
    """Persist a note in per-group namespace."""

    name = "write_note"
    description = "Save a short note for later retrieval."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["group_id", "note"],
        "additionalProperties": False,
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        note_id = self._db.write_note(group_id=kwargs["group_id"], note=kwargs["note"])
        return {"note_id": note_id}


class ListNotesTool(Tool):
    """List saved notes for a group."""

    name = "list_notes"
    description = "List recent saved notes."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["group_id"],
        "additionalProperties": False,
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def run(self, **kwargs: Any) -> list[dict[str, Any]]:
        limit = int(kwargs.get("limit", 20))
        return self._db.list_notes(group_id=kwargs["group_id"], limit=limit)
