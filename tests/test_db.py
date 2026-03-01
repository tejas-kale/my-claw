from datetime import datetime, timedelta, timezone

import pytest

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


def test_clear_history_removes_messages_and_summary(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()
    db.upsert_group("group-1")
    db.add_message("group-1", "user", "hello")
    db.add_message("group-1", "assistant", "hi")
    db.save_summary("group-1", "a summary")

    db.clear_history("group-1")

    assert db.get_recent_messages("group-1", limit=10) == []
    assert db.get_summary("group-1") is None


def test_clear_history_does_not_affect_other_groups(tmp_path):
    db = Database(tmp_path / "assistant.db")
    db.initialize()
    db.upsert_group("group-1")
    db.upsert_group("group-2")
    db.add_message("group-1", "user", "hello")
    db.add_message("group-2", "user", "hey")
    db.save_summary("group-2", "group 2 summary")

    db.clear_history("group-1")

    assert db.get_recent_messages("group-2", limit=10) != []
    assert db.get_summary("group-2") == "group 2 summary"
