"""SQLite persistence layer."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


class Database:
    """Small SQLite wrapper with explicit schema management."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create or migrate schema."""

        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                self._create_schema(conn)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported schema version {row['version']} (expected {SCHEMA_VERSION})"
                )

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                name TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                summary TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(group_id) REFERENCES groups(group_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                role TEXT NOT NULL,
                sender_id TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(group_id) REFERENCES groups(group_id)
            );

            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                succeeded INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(group_id) REFERENCES groups(group_id)
            );

            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                run_at TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(group_id) REFERENCES groups(group_id)
            );

            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(group_id) REFERENCES groups(group_id)
            );
            """
        )

    def upsert_group(self, group_id: str, name: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        now = _utc_now_iso()
        metadata_json = json.dumps(metadata or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO groups(group_id, name, metadata_json, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    name=excluded.name,
                    metadata_json=excluded.metadata_json
                """,
                (group_id, name, metadata_json, now),
            )

    def add_message(self, group_id: str, role: str, content: str, sender_id: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(group_id, role, sender_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (group_id, role, sender_id, content, _utc_now_iso()),
            )

    def get_recent_messages(self, group_id: str, limit: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE group_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (group_id, limit),
            ).fetchall()
        ordered = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in ordered]

    def save_summary(self, group_id: str, summary: str) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE group_id = ? ORDER BY id DESC LIMIT 1", (group_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?",
                    (summary, now, row["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO conversations(group_id, summary, updated_at) VALUES (?, ?, ?)",
                    (group_id, summary, now),
                )

    def get_summary(self, group_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM conversations WHERE group_id = ? ORDER BY id DESC LIMIT 1",
                (group_id,),
            ).fetchone()
        return row["summary"] if row else None

    def clear_history(self, group_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE group_id = ?", (group_id,))
            conn.execute("DELETE FROM conversations WHERE group_id = ?", (group_id,))

    def log_tool_execution(
        self,
        group_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        succeeded: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_executions(group_id, tool_name, input_json, output_json, succeeded, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    tool_name,
                    json.dumps(tool_input),
                    json.dumps(tool_output),
                    int(succeeded),
                    _utc_now_iso(),
                ),
            )

    def create_scheduled_task(self, group_id: str, prompt: str, run_at: datetime) -> int:
        now = _utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scheduled_tasks(group_id, prompt, run_at, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (group_id, prompt, run_at.astimezone(timezone.utc).isoformat(), now, now),
            )
            return int(cur.lastrowid)

    def get_due_tasks(self, now: datetime) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, group_id, prompt, run_at, status
                FROM scheduled_tasks
                WHERE status = 'pending' AND run_at <= ?
                ORDER BY run_at ASC
                """,
                (now.astimezone(timezone.utc).isoformat(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_task_status(self, task_id: int, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE scheduled_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, _utc_now_iso(), task_id),
            )

    def write_note(self, group_id: str, note: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO notes(group_id, note, created_at) VALUES (?, ?, ?)",
                (group_id, note, _utc_now_iso()),
            )
            return int(cur.lastrowid)

    def list_notes(self, group_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, note, created_at FROM notes WHERE group_id = ? ORDER BY id DESC LIMIT ?",
                (group_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
