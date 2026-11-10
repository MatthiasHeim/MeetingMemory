#!/usr/bin/env python3
"""Flag recordings stuck at "Sending to Gemini API..." with no transcript.

The transcribe watcher has frozen mid-Gemini-call at least three times
(2026-07-28, 2026-09-17, 2026-11-09): the process stays alive, the log's last
line is "Sending to Gemini API...", and nothing alerts, because the Telegram
ping only fires on success. Silence looked exactly like "still working", and
meetings went missing until a human noticed the next morning.

The root cause (google-genai requests with no timeout) is fixed in
gemini_processor.py. This check is the backstop for anything else that leaves
a recording in that state: a new hang class, a dead watcher, a crash between
the call and the save.

Rule: a recording is STUCK when watcher.log's most recent processing attempt
for it logged "Sending to Gemini API..." more than --max-hours ago (default 2),
with no "Gemini processing complete" after it, Transcripts/<stem>.json does
not exist, and Recordings/<stem>.wav still exists. (A recording whose WAV is
gone cannot be re-queued by a watcher restart, so it is not "stuck in the
pipeline"; prune_recordings.py only deletes WAVs that already have a JSON.)

Exit codes: 0 = nothing stuck, 1 = at least one stuck recording, 2 = the check
itself could not run (log missing or unreadable). Never 0 on a failure to look.

Usage:
  python3 tools/check_stuck_transcriptions.py            # report, exit code
  python3 tools/check_stuck_transcriptions.py --json
  python3 tools/check_stuck_transcriptions.py --notify   # also Telegram via Brain's notify policy
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE = Path.home() / "Documents" / "MeetingRecorder"
DEFAULT_LOG = BASE / "logs" / "watcher.log"
DEFAULT_TRANSCRIPTS = BASE / "Transcripts"
DEFAULT_RECORDINGS = BASE / "Recordings"
BRAIN_SCRIPTS_DIR = Path("/Users/Matthias/Repos/Brain/.claude/scripts")
# Routed to Telegram in Brain's .claude/config/notify-policy.yaml. Only a human
# (or a session they start) can unstick it, and a meeting with no transcript
# blocks its actions and InsightBase ingest, so it cannot wait for the portal.
NOTIFY_CATEGORY = "Transcription stuck"
# Only the tail matters: the newest attempt per recording. 4 MB is months of log.
TAIL_BYTES = 4 * 1024 * 1024

LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - \w+ - (.*)$")
START_RE = re.compile(r"^Starting Gemini processing: (.+?)\.wav\s*$")
SENDING = "Sending to Gemini API..."
COMPLETE_PREFIX = "Gemini processing complete"


@dataclass
class Stuck:
    stem: str
    sending_since: str  # "YYYY-MM-DD HH:MM:SS", from watcher.log
    hours: float


def _read_tail(path: Path, nbytes: int = TAIL_BYTES) -> list[str]:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > nbytes and lines:
        lines = lines[1:]  # first line is probably cut mid-way
    return lines


def pending_sends(lines: list[str]) -> dict[str, datetime]:
    """Recording stem -> timestamp of a "Sending to Gemini API..." that never completed.

    The watcher processes one file at a time, so "Sending..." belongs to the
    most recent "Starting Gemini processing: <stem>.wav". A new start for the
    same stem (re-queue after a restart) resets that stem's clock.
    """
    current: Optional[str] = None
    pending: dict[str, datetime] = {}
    for line in lines:
        m = LINE_RE.match(line)
        if not m:
            continue
        ts_str, msg = m.groups()
        start = START_RE.match(msg)
        if start:
            current = start.group(1)
            pending.pop(current, None)
            continue
        if current is None:
            continue
        if msg.startswith(SENDING):
            pending[current] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        elif msg.startswith(COMPLETE_PREFIX):
            pending.pop(current, None)
    return pending


def find_stuck(log_path: Path, transcripts_dir: Path, recordings_dir: Path,
               now: datetime, max_hours: float) -> list[Stuck]:
    lines = _read_tail(log_path)
    stuck = []
    for stem, since in sorted(pending_sends(lines).items(), key=lambda kv: kv[1]):
        age = now - since
        if age <= timedelta(hours=max_hours):
            continue
        if (transcripts_dir / f"{stem}.json").exists():
            continue
        if not (recordings_dir / f"{stem}.wav").exists():
            continue
        stuck.append(Stuck(stem=stem, sending_since=since.strftime("%Y-%m-%d %H:%M:%S"),
                           hours=round(age.total_seconds() / 3600, 1)))
    return stuck


def format_message(stuck: list[Stuck]) -> str:
    # Deliberately no age in the text: telegram_notify dedupes on identical
    # category+message for 12h, so one incident is one ping, not one per run.
    names = ", ".join(f"{s.stem} (sent {s.sending_since})" for s in stuck)
    return (
        f"Meeting transcription stuck at 'Sending to Gemini API' with no transcript: "
        f"{names}. Restart the watcher: kill $(pgrep -f tools/transcribe_watcher.py) "
        f"(launchd respawns it and re-queues), then check watcher.log."
    )


def _telegram_notify_script() -> Optional[str]:
    for c in (os.environ.get("TELEGRAM_NOTIFY_SCRIPT", ""),
              str(BRAIN_SCRIPTS_DIR / "telegram_notify.py"),
              os.path.expanduser("~/.claude/scripts/telegram_notify.py")):
        if c and os.path.exists(c):
            return c
    return None


def send_notification(message: str) -> None:
    script = _telegram_notify_script()
    if not script:
        raise RuntimeError("telegram_notify.py not found in any known location")
    proc = subprocess.run(
        [sys.executable, script, "--category", NOTIFY_CATEGORY, message],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"telegram_notify exited {proc.returncode}: "
                           f"{proc.stderr.strip() or proc.stdout.strip()}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    p.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--now", help="override current time (YYYY-MM-DD HH:MM:SS), for tests")
    p.add_argument("--json", action="store_true")
    p.add_argument("--notify", action="store_true",
                   help="send a Telegram alert via Brain's telegram_notify.py when stuck")
    args = p.parse_args(argv)

    now = (datetime.strptime(args.now, "%Y-%m-%d %H:%M:%S") if args.now
           else datetime.now())
    try:
        stuck = find_stuck(args.log, args.transcripts, args.recordings, now, args.max_hours)
    except OSError as e:
        print(f"ERROR: cannot read {args.log}: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"checked_at": now.isoformat(timespec="seconds"),
                          "stuck": [asdict(s) for s in stuck]}))
    elif stuck:
        print(format_message(stuck))
    else:
        print(f"OK: no recording stuck at '{SENDING}' for more than {args.max_hours:g}h")

    if stuck and args.notify:
        try:
            send_notification(format_message(stuck))
        except Exception as e:  # a lost alert must not look like a clean run
            print(f"ERROR: alert not delivered: {e}", file=sys.stderr)
            return 2
    return 1 if stuck else 0


if __name__ == "__main__":
    sys.exit(main())
