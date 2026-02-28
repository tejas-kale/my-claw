# Learnings

## signal-cli

- The `--json` flag moved to a global output flag in newer versions: use `-o json` before the subcommand, not after it.
- The daemon and standalone `send`/`receive` subprocesses use different Signal sessions/identity keys. Running both concurrently causes sends to arrive as "Unknown" message requests. Do not use the daemon — use standalone `receive -t N` for polling instead.
- `receive -t 0` returns immediately with no messages. Use a real timeout (e.g. `-t 5`) so the process blocks and fetches pending messages before returning.
- `sourceNumber` is null when the sender has Signal phone-number privacy enabled. Resolve UUID → phone number via `listContacts`, or fall back to a configured owner number.
- Even with the correct phone number, send replies go to a separate "Unknown" thread until the bot has a profile set. Run `signal-cli updateProfile --given-name <name>` once after registration.
- The recipient must accept the initial message request on their phone. After acceptance the thread is established and all future messages flow correctly.
- For groups, `groupInfo.groupId` is a string in the JSON envelope. For DMs there is no `groupInfo`; use `sourceUuid` as the conversation key.

## Signal protocol

- Signal does not render Markdown. Strip all markers (`**`, `*`, `_`, `#`, backticks) before sending.

## OpenRouter / LLM tool calls

- After executing tool calls, the follow-up `generate()` call must include: (1) the assistant message with a `tool_calls` array, and (2) each tool result message with a matching `tool_call_id`. Missing either causes models to return empty content.

## signal-cli setup (one-time)

1. Register or link the bot number with signal-cli.
2. Run `signal-cli updateProfile --given-name <name>` to set a profile so recipients see a name.
3. Add the owner's number as a contact: `signal-cli updateContact <owner_number>`.
4. Set `SIGNAL_OWNER_NUMBER` in `.env` as fallback for UUID resolution.

## NotebookLM MCP-CLI authentication

`nlm login` fails on headless VMs (e.g. GCP Debian) because Chrome DevTools port 9222 cannot drive Google OAuth. Workaround using cookie export:

1. Install Chrome on the VM: `sudo apt-get install google-chrome-stable`
2. On MacBook: install the **EditThisCookie** extension from the Chrome Web Store.
3. Log in to notebooklm.google.com fully in Chrome.
4. Use EditThisCookie → **Export** to get clean JSON cookies (avoids Unicode `✓` characters that break TSV copy-paste).
5. `scp cookies.json VM:~/.local/share/notebooklm-mcp-cli/profiles/default/`
6. `nlm login --manual --file cookies.json --force`
7. Verify: `nlm login --check`

This bypasses headless Chrome OAuth limitations entirely and gives persistent VM auth.
