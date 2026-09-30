# Follow-up hand-off (`followup.mode`)

After a transcript lands, the watcher hands the meeting to whoever runs the follow-up agent (`/meeting-actions` in the tenant repo). Two modes, set in the live `config.yaml` (this repo ships no live config; add the keys there):

```yaml
followup:
  mode: watcher          # watcher (default) | automations

# Already existing keys, now with documented defaults (current values):
claude_trigger:
  enabled: true          # still the kill switch for the follow-up in BOTH modes
  claude_path: ~/.local/bin/claude
  brain_repo: ~/Repos/Brain       # cwd of the headless session (watcher mode only)

notifications:
  telegram_notify_script: null    # else $TELEGRAM_NOTIFY_SCRIPT, else <brain_repo>/.claude/scripts/telegram_notify.py
```

| | `watcher` (default) | `automations` |
|---|---|---|
| Marks `sources.metadata.pipeline_status = 'ready_for_followup'` (+ `pipeline_status_at`, `followup`) | yes, `triggered_by: watcher` | yes, `triggered_by: automations` |
| Starts headless Claude (`/meeting-actions`) | yes, unchanged | no |
| Sends the "Meeting captured" Telegram ping | yes | no |
| Alerts if the hand-off could not be recorded | logs only (Claude still runs) | Telegram alert (nobody else will run it) |

`metadata.followup` carries what the job needs to rebuild the prompt: `transcript_path`, `triggered_by`, and `prompt_suffix` (the calendar resolve-then-extract text, when the attribution gate asks for it). The lailix-automations `meeting-followup` job treats a `watcher` source's run as attempt 1 and only retries after its grace period; an `automations` source it runs immediately.

A mistyped `followup.mode` stops the watcher at startup. Flip to `automations` only after the job is installed and `meeting-followup doctor` passes; switching back is the same one-line change.

Failure alerts for the watcher's own work (transcription failed, partial, coherence) still go through the configured notifier script above.
