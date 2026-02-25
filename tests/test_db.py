from datetime import datetime, timedelta, timezone

from assistant.db import Database


def test_database_initialization_and_notes(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()
    db.upsert_group("group-1")

    note_id = db.write_note("group-1", "remember this")
    assert note_id > 0

    notes = db.list_notes("group-1", limit=5)
    assert len(notes) == 1
    assert notes[0]["note"] == "remember this"


def test_due_tasks(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()
    db.upsert_group("group-1")

    due_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.create_scheduled_task("group-1", "ping", due_at)

    due = db.get_due_tasks(datetime.now(timezone.utc))
    assert len(due) == 1
    assert due[0]["prompt"] == "ping"
