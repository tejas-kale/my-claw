import pytest

from assistant.db import Database
from assistant.tools.notes_tool import ListNotesTool, WriteNoteTool
from assistant.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_tool_registry_validates_and_executes(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()
    registry = ToolRegistry(db)
    registry.register(WriteNoteTool(db))
    registry.register(ListNotesTool(db))

    await registry.execute("g1", "write_note", {"group_id": "g1", "note": "n1"})
    notes = await registry.execute("g1", "list_notes", {"group_id": "g1"})

    assert len(notes) == 1
    assert notes[0]["note"] == "n1"


@pytest.mark.asyncio
async def test_tool_registry_rejects_invalid_input(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()
    registry = ToolRegistry(db)
    registry.register(WriteNoteTool(db))

    with pytest.raises(ValueError):
        await registry.execute("g1", "write_note", {"group_id": "g1"})
