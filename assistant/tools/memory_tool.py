"""Markdown-file-based memory tools (save and read notes)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from assistant.tools.base import Tool

_MAX_TOPIC_SLUG_LENGTH = 60


def _ensure_dirs(memory_root: Path) -> None:
    (memory_root / "daily").mkdir(parents=True, exist_ok=True)
    (memory_root / "topics").mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")[:_MAX_TOPIC_SLUG_LENGTH]


class SaveNoteTool(Tool):
    """Append a note to the markdown-file memory store."""

    name = "save_note"
    description = (
        "Save a note to memory. Use note_type='daily' for the running daily log "
        "(timestamped, append-only). Use note_type='topic' with a topic name for "
        "subject-specific notes. Call this proactively when the user shares "
        "preferences, project context, or asks you to remember something."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The note content."},
            "note_type": {
                "type": "string",
                "enum": ["daily", "topic"],
                "description": "Where to save: 'daily' or 'topic'.",
            },
            "topic": {
                "type": "string",
                "description": "Topic name (required if note_type='topic').",
            },
        },
        "required": ["content", "note_type"],
        "additionalProperties": False,
    }

    def __init__(self, memory_root: Path) -> None:
        self._memory_root = memory_root

    async def run(self, **kwargs: Any) -> str:
        content: str = kwargs["content"]
        note_type: str = kwargs["note_type"]
        topic: str | None = kwargs.get("topic")

        _ensure_dirs(self._memory_root)

        if note_type == "daily":
            filepath = self._memory_root / "daily" / f"{date.today().isoformat()}.md"
            ts = datetime.now().strftime("%H:%M")
            with open(filepath, "a") as f:
                f.write(f"\n- [{ts}] {content}\n")
            return f"Saved to daily notes ({filepath.name})."

        if note_type == "topic" and topic:
            slug = _slugify(topic)
            filepath = self._memory_root / "topics" / f"{slug}.md"
            is_new = not filepath.exists()
            with open(filepath, "a") as f:
                if is_new:
                    f.write(f"# {topic}\n\n")
                f.write(f"{content}\n\n")
            action = "Created" if is_new else "Appended to"
            return f"{action} topic note: {slug}.md"

        return "Error: specify note_type='daily' or note_type='topic' with a topic name."


class ReadNotesTool(Tool):
    """Read notes from the markdown-file memory store."""

    name = "read_notes"
    description = (
        "Read from memory. Use note_type='daily' to read recent daily logs. "
        "Use note_type='topic' with a topic name to read subject-specific notes. "
        "Use note_type='topics_list' to see all available topics."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "note_type": {
                "type": "string",
                "enum": ["daily", "topic", "topics_list"],
                "description": "What to read.",
            },
            "topic": {
                "type": "string",
                "description": "Topic name (for note_type='topic').",
            },
            "days_back": {
                "type": "integer",
                "description": "How many days of daily notes to read (default 1).",
            },
        },
        "required": ["note_type"],
        "additionalProperties": False,
    }

    def __init__(self, memory_root: Path) -> None:
        self._memory_root = memory_root

    async def run(self, **kwargs: Any) -> str:
        note_type: str = kwargs["note_type"]
        topic: str | None = kwargs.get("topic")
        days_back: int = int(kwargs.get("days_back") or 1)

        _ensure_dirs(self._memory_root)

        if note_type == "daily":
            entries = []
            for i in range(days_back):
                d = date.today() - timedelta(days=i)
                filepath = self._memory_root / "daily" / f"{d.isoformat()}.md"
                if filepath.exists():
                    entries.append(f"## {d.isoformat()}\n{filepath.read_text()}")
            return "\n\n".join(entries) if entries else "No daily notes found."

        if note_type == "topic" and topic:
            slug = _slugify(topic)
            filepath = self._memory_root / "topics" / f"{slug}.md"
            if filepath.exists():
                return filepath.read_text()
            # Fuzzy fallback: list topics whose slug contains the query.
            matches = [
                f.stem
                for f in (self._memory_root / "topics").glob("*.md")
                if slug in f.stem or topic.lower() in f.stem
            ]
            if matches:
                return f"No exact match for '{topic}'. Similar topics: {', '.join(matches)}"
            return f"No topic notes found for '{topic}'."

        if note_type == "topics_list":
            topics_dir = self._memory_root / "topics"
            topics = sorted(f.stem for f in topics_dir.glob("*.md"))
            return (
                "Available topics:\n" + "\n".join(f"- {t}" for t in topics)
                if topics
                else "No topic notes yet."
            )

        return "Error: specify note_type='daily', 'topic', or 'topics_list'."
