# my-claw

This repository contains a lean OpenClaw workspace setup focused on:

- safer memory across compaction cycles
- better retrieval quality (hybrid search)
- explicit boot-time retrieval habits
- lighter always-loaded documentation

## Configuration

Use `/openclaw.config.json` as the baseline runtime configuration.

Key improvements included:

- `compaction.memoryFlush.enabled: true` to persist important context before compaction
- `contextPruning` with TTL to reduce long-session context bloat
- `memory.qmd.paths` configured for `MEMORY.md`, daily logs, and learnings

## Workspace layout

- `AGENTS.md`: boot sequence + retrieval/write/handover discipline
- `MEMORY.md`: curated long-term memory (kept intentionally small)
- `learnings/LEARNINGS.md`: one-line rules from mistakes
- `memory/`: daily append-only logs (`YYYY-MM-DD.md`)
- `docs/`: reference material moved out of always-loaded memory
