# Claude quota isolation for MeetingMemory's headless automations

## Why

The transcription pipeline hands the LLM-only downstream (insight extraction,
Linear task creation, follow-up email drafts, ClientContext refresh, Telegram)
to a **fire-and-forget** headless session:

```
claude -p "Read .claude/commands/meeting-actions.md and process transcript: … --source-id N"
```

fired from `tools/transcribe_watcher.py::_trigger_claude`. It has **no retry**.

On **2026-07-10** every one of these sessions died on its first line —
`You've hit your session limit · resets …` (the tell-tale ~64-byte
`~/Documents/MeetingRecorder/logs/claude-*.log` files; healthy runs are ~3 KB).
Insight extraction for that day's Philipp meeting (source 463) had already
landed, but the session that owns Linear/email/ClientContext/Telegram never
ran, so every downstream action was silently dropped.

Root problem: the automation sessions share **one** Claude quota pool with
interactive usage. A burst of interactive work can exhaust it and starve the
meeting-actions sessions.

## The fix: a separate quota pool via `CLAUDE_CONFIG_DIR`

Claude Code keys its rate-limit/quota to `CLAUDE_CONFIG_DIR` — each config dir
is its own credential, its own login, and its own quota pool (Brain memory
`reference_claude_multiaccount_auth`, "Option A — full isolation"). Pointing the
headless automations at a dedicated config dir stops them competing with
interactive usage.

`_trigger_claude` sets `CLAUDE_CONFIG_DIR` on the child session's environment,
resolved in this order:

1. `claude_trigger.config_dir` in `~/Documents/MeetingRecorder/config.yaml`
2. an inherited `CLAUDE_CONFIG_DIR` (e.g. from the watcher's LaunchAgent plist)
3. neither → shares the default pool (legacy behaviour) and logs a warning

## One-time setup (must be done interactively — cannot be scripted headlessly)

A fresh config dir has no credentials, and Claude Code stores its OAuth token in
the macOS **keychain** under a service name hashed per config dir
(`Claude Code-credentials-<hash>`), so credentials **cannot** be copied from the
default dir — the isolated dir needs its own login.

```bash
# 1. Create the dir and log in (opens the normal browser OAuth flow).
#    In Claude Code you can run this with the `!` prefix so its output lands
#    in the session:
CLAUDE_CONFIG_DIR=/Users/Matthias/.claude-automation claude
#    → then run  /login  in that session and complete the browser flow.
#    Ideally log in as a DIFFERENT account (e.g. an automation seat) so the
#    pool is truly separate; even the same account in a separate config dir
#    helps, but a distinct seat is the strongest isolation.

# 2. Verify it's authenticated:
CLAUDE_CONFIG_DIR=/Users/Matthias/.claude-automation claude -p "say ok"
#    → should print "ok" (a ~real response), not a login prompt / session-limit line.
```

## Wire it up (after the login above succeeds)

Pick **one** of:

**A. config.yaml (simplest, per-repo):** add under `claude_trigger:`

```yaml
claude_trigger:
  enabled: true
  claude_path: /Users/Matthias/.local/bin/claude
  brain_repo: /Users/Matthias/Repos/Brain
  command: meeting-actions
  config_dir: /Users/Matthias/.claude-automation   # ← quota isolation
```

**B. LaunchAgent plist (covers everything the daemon spawns):** the repo
template `tools/launchagents/com.user.transcribewatcher.plist` already sets

```xml
<key>CLAUDE_CONFIG_DIR</key>
<string>/Users/Matthias/.claude-automation</string>
```

in `EnvironmentVariables`. Copy it to the live agent and reload:

```bash
cp tools/launchagents/com.user.transcribewatcher.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.user.transcribewatcher 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.transcribewatcher.plist
```

> The live plist currently pins repo/venv paths under `~/Repos/MeetingMemory`
> while the checked-in template still says `~/Desktop/Repos/…`. Reconcile the
> paths before copying, or just add the two `CLAUDE_CONFIG_DIR` lines to the
> live plist in place.

The reconciliation sweep (`reconcile_meeting_tasks.py`) is the deterministic
backstop that **does not need any of this** — it never starts a Claude session,
so it can never hit a session limit. Quota isolation reduces how often the
primary (Claude) path fails; the sweep guarantees tasks land even when it does.
