"""Subprocess-backed search tools: ripgrep and fzf --filter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from assistant.tools.base import Tool

_MAX_OUTPUT_CHARS = 30_000
_RG_TIMEOUT = 10.0
_FZF_TIMEOUT = 5.0


class RipgrepSearchTool(Tool):
    """Search file contents using ripgrep."""

    name = "ripgrep_search"
    description = (
        "Search file contents using ripgrep (rg). Searches recursively in the "
        "project directory or memory notes. Returns matching lines with file paths "
        "and line numbers. Use this to find code patterns, function definitions, "
        "configuration values, or search through your notes."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {
                "type": "string",
                "description": (
                    "Directory to search. Defaults to current directory. "
                    "Use '~/.my-claw/memory' to search notes."
                ),
            },
            "glob": {"type": "string", "description": "File glob filter, e.g. '*.py'."},
            "file_type": {"type": "string", "description": "File type, e.g. 'py', 'json'."},
            "case_insensitive": {"type": "boolean", "description": "Case-insensitive search."},
            "fixed_strings": {"type": "boolean", "description": "Literal match (no regex)."},
            "context_lines": {
                "type": "integer",
                "description": "Context lines around matches (0-5).",
            },
            "max_results": {
                "type": "integer",
                "description": "Max matching lines (1-100).",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def __init__(self, memory_root: Path) -> None:
        self._allowed_roots = [Path.cwd().resolve(), memory_root.resolve()]

    def _validate_path(self, user_path: str) -> Path:
        target = Path(user_path).expanduser().resolve()
        for root in self._allowed_roots:
            if str(target).startswith(str(root)):
                return target
        raise ValueError(f"Path not in allowed roots: {user_path}")

    async def run(self, **kwargs: Any) -> str:
        pattern: str = kwargs["pattern"]
        path: str = kwargs.get("path") or "."
        glob: str | None = kwargs.get("glob")
        file_type: str | None = kwargs.get("file_type")
        case_insensitive: bool = bool(kwargs.get("case_insensitive", False))
        fixed_strings: bool = bool(kwargs.get("fixed_strings", False))
        context_lines: int = min(int(kwargs.get("context_lines") or 2), 5)
        max_results: int = min(int(kwargs.get("max_results") or 50), 100)

        try:
            search_path = self._validate_path(path)
        except ValueError as exc:
            return str(exc)

        args = ["rg", "-e", pattern, "--json"]
        if case_insensitive:
            args.append("-i")
        if fixed_strings:
            args.append("-F")
        if glob:
            args.extend(["--glob", glob])
        if file_type:
            args.extend(["-t", file_type])
        args.extend(["-C", str(context_lines)])
        args.extend(["-m", str(max_results)])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(search_path),
        )

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_RG_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Search timed out. Try a more specific pattern or path."

        matches: list[str] = []
        total_chars = 0

        for line in stdout.decode("utf-8", errors="replace").splitlines():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") not in ("match", "context"):
                continue

            data = msg["data"]
            file_path = data.get("path", {}).get("text", "")
            line_num = data.get("line_number", "?")
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            prefix = ">" if msg["type"] == "match" else " "

            entry = f"{prefix} {file_path}:{line_num}: {line_text}"
            total_chars += len(entry)
            if total_chars > _MAX_OUTPUT_CHARS:
                matches.append("... (results truncated)")
                break
            matches.append(entry)

        if not matches:
            return f"No matches for pattern: {pattern}"

        match_count = sum(1 for m in matches if m.startswith(">"))
        header = f"Found {match_count} matches for '{pattern}' in {search_path}:\n"
        return header + "\n".join(matches)


class FuzzyFilterTool(Tool):
    """Fuzzy-filter a list of strings using fzf --filter."""

    name = "fuzzy_filter"
    description = (
        "Fuzzy-filter a list of strings using fzf's matching algorithm. "
        "Useful for approximate/typo-tolerant matching on file names, "
        "function names, topic names, etc. Input is a list of strings; "
        "output is the subset that fuzzy-matches the query, ranked by relevance."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Fuzzy search query."},
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of strings to filter (max 10,000).",
            },
            "max_results": {"type": "integer", "description": "Max results (default 20)."},
        },
        "required": ["query", "items"],
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        items: list[str] = kwargs["items"][:10_000]
        max_results: int = int(kwargs.get("max_results") or 20)

        if not items:
            return "No items to filter."

        input_text = "\n".join(items)
        proc = await asyncio.create_subprocess_exec(
            "fzf",
            "--filter",
            query,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=input_text.encode("utf-8")),
                timeout=_FZF_TIMEOUT,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Fuzzy filter timed out."

        results = stdout.decode("utf-8").strip().splitlines()[:max_results]
        if not results:
            return f"No fuzzy matches for '{query}'."
        return f"Fuzzy matches for '{query}':\n" + "\n".join(f"- {r}" for r in results)
