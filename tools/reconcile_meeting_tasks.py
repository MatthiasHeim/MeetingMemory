#!/usr/bin/env python3
"""
reconcile_meeting_tasks: decision-record check for MeetingMemory (backstop).

WHY THIS EXISTS (2026-07-10 incident, reshaped 2026-09-30)
──────────────────────────────────────────────────────────
The transcription→insight→action pipeline splits work between a deterministic
watcher (tools/transcribe_watcher.py) and a fire-and-forget headless Claude
session (`claude -p /meeting-actions`). The watcher seeds the InsightBase
`sources` row; the Claude session owns the LLM-only downstream, including every
Linear ticket consequence of the meeting.

On 2026-07-10 every headless Claude session died on its first line ("You've hit
your session limit") and the tickets of that meeting were silently dropped. This
job was built as the backstop and, until 2026-09-30, created a Linear ticket for
every `type='action'` insight of a meeting that had none.

That design failed the other way round on 2026-09-29 (source 1121): a
commitment stored as an opportunity insight never became a ticket, and meetings
whose decisions are only updates/closes of existing tickets have zero
`transcript:<sid>:*` tickets, so the sweep would have created duplicates
(LAI-770 to 772). Insights are memory, tickets are work: deriving one from the
other turns every labelling mistake into a missed or an extra task. See
Brain `Areas/lailix-internal/docs/specs/2026-09-30-meeting-pipeline-boundaries.md`.

The /meeting-actions agent now decides every ticket consequence and records it
in `sources.metadata.task_decisions` (Brain `meeting_task_decisions.py`),
including an explicit "none". A missing record is the mechanical signal that the
agent never ran.

WHAT IT DOES
────────────
PURE PYTHON, read-only: it never starts a Claude/LLM session and NEVER creates,
updates or closes a Linear issue. For every `sources` row with
source_type='meeting' and origin='noscribe' (the watcher's meetings) from the
last ~48h (``--window-hours``), whether or not it has any insights:

  * `metadata ? 'task_decisions'`  → skipped, nothing reported.
  * no record, but the meeting already has ≥1 `transcript:<sid>:*` Linear
    ticket → skipped. That is a meeting from before the record existed whose
    agent run succeeded; reporting it would page for every pre-cutover meeting.
  * no record and no ticket → REPORTED: logged, and ONE Telegram line per sweep
    lists every such meeting with the command to re-run.

Reporting is the whole job. Re-running the agent is a human (or, later, the
lailix-automations `meeting-followup` job) action.

SCOPE (what it intentionally does NOT do)
─────────────────────────────────────────
No ticket writes of any kind. It does not extract insights, draft emails or
refresh ClientContext. It does not read insights at all. The Linear gate read (`linear_client.query`) stays
read-only; a failed gate read aborts the sweep instead of reporting meetings
that may in fact have tickets.

Environment:
  INSIGHTBASE_DATABASE_URL: Postgres DSN for InsightBase (via neon_insert).
  LINEAR_API_KEY: read by Brain's linear_client (env or Brain/.env), reads only.
  TELEGRAM_NOTIFY_SCRIPT: optional override for Brain's telegram_notify.py.

CLI:
    python3 reconcile_meeting_tasks.py                 # sweep last 48h, report
    python3 reconcile_meeting_tasks.py --window-hours 72
    python3 reconcile_meeting_tasks.py --dry-run       # print, send nothing
    python3 reconcile_meeting_tasks.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ── Paths / constants ─────────────────────────────────────────────────

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Brain owns the Linear client (read-only here) and the audit log. Add its scripts dir to the path so we can import them.
BRAIN_SCRIPTS_DIR = Path("/Users/Matthias/Repos/Brain/.claude/scripts")
if str(BRAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_SCRIPTS_DIR))

DEFAULT_WINDOW_HOURS = 48
DEFAULT_STATE_FILE = (
    Path.home() / "Documents" / "MeetingRecorder" / "reconcile_meeting_tasks_state.json"
)

# Skill name used for brain_audit events.
AUDIT_SKILL = "meeting-actions-reconcile"

# The gate read looks for meeting tickets by this label (read-only).
TASK_SOURCE = "meeting-action"

# How many recent meeting-action issues to pull when building the "already has
# tasks" gate snapshot. We only reconcile meetings from the last ~48h, so any
# existing task for one of them was created recently and lands in this window.
# 250 is Linear's hard per-page `first:` cap — asking for more is a GraphQL
# "Argument Validation Error".
DEFAULT_LINEAR_FETCH_LIMIT = 250

logger = logging.getLogger("reconcile_meeting_tasks")


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class Meeting:
    source_id: int
    title: Optional[str]
    company: Optional[str]
    started_at: Optional[datetime]
    # True when sources.metadata carries a `task_decisions` record.
    has_decision_record: bool = False


# ── InsightBase reads ─────────────────────────────────────────────────

def fetch_meetings(conn, window_hours: int) -> list[Meeting]:
    """Return every recorded meeting from the last `window_hours`, with whether
    it already carries a `task_decisions` record.

    A meeting is a `sources` row with source_type='meeting' and origin='noscribe'
    (what the watcher seeds; Teams/Gmail/Gemini-origin meeting rows never go
    through /meeting-actions from the watcher). Insights are deliberately not
    consulted: a meeting where the agent died before extracting anything, or
    stored its commitments under other insight types, must still be reported.
    Window is measured on `started_at`, falling back to `created_at`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.title, s.company, s.started_at,
                   COALESCE(s.metadata ? 'task_decisions', false)
            FROM sources s
            WHERE s.source_type = 'meeting'
              AND s.origin = 'noscribe'
              AND COALESCE(s.started_at, s.created_at)
                  >= now() - make_interval(hours => %s)
            ORDER BY s.id;
            """,
            (window_hours,),
        )
        return [
            Meeting(source_id=sid, title=title, company=company,
                    started_at=started_at, has_decision_record=bool(has_record))
            for sid, title, company, started_at, has_record in cur.fetchall()
        ]


# ── Pure helpers ──────────────────────────────────────────────────────

def meeting_task_prefix(source_id: int | str) -> str:
    """The Linear source_id prefix shared by every task of one meeting."""
    return f"transcript:{source_id}:"


def parse_meeting_source_id(task_source_id: str) -> Optional[str]:
    """Extract the InsightBase source id from a task source_id.

    ``transcript:463:sql-export`` → ``"463"``. Returns None if the string is
    not a meeting-task source_id.
    """
    if not task_source_id:
        return None
    parts = task_source_id.split(":")
    if len(parts) >= 3 and parts[0] == "transcript" and parts[1]:
        return parts[1]
    return None


# ── Linear gateway ────────────────────────────────────────────────────

class LinearGateway:
    """Reads the "already has tickets" gate from Linear. Read-only: this class
    has no write path, so the sweep cannot create tickets even by accident.

    Reads go through the imported `linear_client` module (reuses its meta
    parser).
    """

    def __init__(self, fetch_limit: int = DEFAULT_LINEAR_FETCH_LIMIT):
        self.fetch_limit = fetch_limit

    def source_ids_with_tasks(self) -> set[str]:
        """Set of InsightBase source ids that already have ≥1 Linear task.

        Raises on failure — a failed gate read must ABORT the sweep rather than
        report meetings that may in fact have tickets.
        """
        import linear_client as lc

        rows = lc.query(label=TASK_SOURCE, limit=self.fetch_limit)
        if len(rows) >= self.fetch_limit:
            logger.warning(
                "Linear gate fetched the full %d-issue limit — older meeting "
                "tasks may be missing from the gate snapshot; raise "
                "--linear-fetch-limit if duplicates appear.",
                self.fetch_limit,
            )
        found: set[str] = set()
        for r in rows:
            meta = lc.parse_meta(r.get("description"))
            sid = parse_meeting_source_id(str(meta.get("source_id", "")))
            if sid:
                found.add(sid)
        return found


# ── State file ────────────────────────────────────────────────────────

def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write state file %s: %s", path, e)


# ── Telegram ──────────────────────────────────────────────────────────

def _telegram_notify_script() -> Optional[str]:
    """Resolve Brain's telegram_notify.py (same lookup order as the watcher).

    Phase 3 of the meeting-pipeline spec removes this dependency.
    """
    candidates = [
        os.environ.get("TELEGRAM_NOTIFY_SCRIPT", ""),
        str(BRAIN_SCRIPTS_DIR / "telegram_notify.py"),
        os.path.expanduser("~/.claude/scripts/telegram_notify.py"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def send_telegram(message: str) -> None:
    """Send one Telegram line through Brain's telegram_notify.py.

    Raises if the script is missing or exits non-zero, so the sweep counts a
    lost alert as an error instead of pretending it was delivered. Tests inject
    a fake `notify` instead of calling this.
    """
    script = _telegram_notify_script()
    if not script:
        raise RuntimeError("telegram_notify.py not found in any known location")
    proc = subprocess.run(
        [sys.executable, script, "--category", "Meeting", message],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"telegram_notify exited {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def format_report(unrecorded: list[Meeting]) -> str:
    """One line naming every meeting without a decision record."""
    if len(unrecorded) == 1:
        m = unrecorded[0]
        return (
            f"Meeting {m.title or 'untitled'} (source {m.source_id}) has no ticket "
            f"decisions — re-run /meeting-actions --source-id {m.source_id}"
        )
    listing = ", ".join(f"{m.title or 'untitled'} (source {m.source_id})" for m in unrecorded)
    return (
        f"{len(unrecorded)} meetings have no ticket decisions: {listing} — "
        "re-run /meeting-actions --source-id N for each"
    )


# ── State file ────────────────────────────────────────────────────────

def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write state file %s: %s", path, e)


# ── Orchestration ─────────────────────────────────────────────────────

def reconcile(
    conn,
    gateway: LinearGateway,
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    dry_run: bool = False,
    audit=None,
    state_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    notify: Optional[Callable[[str], None]] = None,
) -> dict:
    """Core sweep. Returns a summary dict. Injected `conn`/`gateway`/`audit`/
    `notify` keep this unit-testable without a real DB, Linear or Telegram.
    Nothing here writes to Linear."""
    notify = notify or send_telegram
    now = now or datetime.now(timezone.utc)
    meetings = fetch_meetings(conn, window_hours)
    logger.info(
        "Found %d meeting(s) in the last %dh",
        len(meetings), window_hours,
    )

    # Snapshot the legacy gate (meetings from before the record existed).
    existing = gateway.source_ids_with_tasks()

    summary = {
        "window_hours": window_hours,
        "dry_run": dry_run,
        "meetings_in_window": len(meetings),
        "meetings_skipped_decision_record": 0,
        "meetings_skipped_existing_tasks": 0,
        "meetings_unrecorded": 0,
        "telegram_sent": False,
        "errors": 0,
        "unrecorded": [],  # per-meeting detail
    }

    unrecorded: list[Meeting] = []
    for m in meetings:
        if m.has_decision_record:
            summary["meetings_skipped_decision_record"] += 1
            logger.info("Skip meeting %s (%s): has task_decisions record",
                        m.source_id, m.title or "untitled")
            continue
        if str(m.source_id) in existing:
            summary["meetings_skipped_existing_tasks"] += 1
            logger.info(
                "Skip meeting %s (%s): no record but already has ≥1 Linear ticket "
                "(pre-record meeting)", m.source_id, m.title or "untitled",
            )
            continue
        logger.warning(
            "Meeting %s (%s) has no ticket decisions — re-run /meeting-actions "
            "--source-id %s", m.source_id, m.title or "untitled", m.source_id,
        )
        unrecorded.append(m)
        summary["unrecorded"].append(
            {"source_id": m.source_id, "title": m.title, "company": m.company,
             "started_at": m.started_at}
        )
    summary["meetings_unrecorded"] = len(unrecorded)

    if unrecorded:
        line = format_report(unrecorded)
        if dry_run:
            logger.info("[dry-run] would send Telegram: %s", line)
        else:
            try:
                notify(line)
                summary["telegram_sent"] = True
            except Exception as e:  # noqa: BLE001 — a lost alert is an error, not a crash
                summary["errors"] += 1
                logger.error("Telegram report failed: %s", e)

    _finalize(summary, now, dry_run, audit, state_path)
    return summary


def _finalize(summary, now, dry_run, audit, state_path):
    """Persist state + emit a brain_audit event. Best-effort — never raises."""
    if state_path is not None and not dry_run:
        state = _load_state(state_path)
        state["last_run_utc"] = now.isoformat()
        state["last_summary"] = {k: v for k, v in summary.items() if k != "unrecorded"}
        reported = state.setdefault("reported_sources", {})
        for entry in summary["unrecorded"]:
            reported[str(entry["source_id"])] = {
                "reported_utc": now.isoformat(),
                "title": entry["title"],
            }
        _save_state(state_path, state)

    if audit is not None:
        details = (
            f"window={summary['window_hours']}h "
            f"meetings={summary['meetings_in_window']} "
            f"unrecorded={summary['meetings_unrecorded']} "
            f"skipped_record={summary['meetings_skipped_decision_record']} "
            f"skipped_tickets={summary['meetings_skipped_existing_tasks']} "
            f"errors={summary['errors']}"
            + (" [dry-run]" if dry_run else "")
        )
        status = "error" if summary["errors"] else "ok"
        try:
            audit.log(
                AUDIT_SKILL,
                "run",
                details=details,
                status=status,
                context=json.dumps(
                    {k: v for k, v in summary.items() if k != "unrecorded"}
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("brain_audit logging failed: %s", e)


# ── CLI ───────────────────────────────────────────────────────────────

def _build_audit():
    """Best-effort AuditLog. Returns None if brain_audit is unavailable."""
    try:
        import brain_audit
        return brain_audit.AuditLog()
    except Exception as e:  # noqa: BLE001
        logger.warning("brain_audit unavailable (%s); run will not be audited", e)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report recently-transcribed meetings that have no "
        "task_decisions record (the /meeting-actions agent never ran). "
        "Read-only: never creates Linear tickets."
    )
    parser.add_argument(
        "--window-hours", type=int, default=DEFAULT_WINDOW_HOURS,
        help=f"Look back this many hours (default: {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be reported; send no Telegram, write no state.",
    )
    parser.add_argument(
        "--linear-fetch-limit", type=int, default=DEFAULT_LINEAR_FETCH_LIMIT,
        help="How many recent meeting-action issues to scan for the legacy gate.",
    )
    parser.add_argument(
        "--state-file", type=Path, default=DEFAULT_STATE_FILE,
        help=f"State file path (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Lazy import so `--help` and unit tests don't require psycopg2/DSN.
    from neon_insert import _get_conn

    gateway = LinearGateway(fetch_limit=args.linear_fetch_limit)
    audit = None if args.dry_run else _build_audit()  # dry-run writes nothing

    conn = _get_conn()
    try:
        summary = reconcile(
            conn,
            gateway,
            window_hours=args.window_hours,
            dry_run=args.dry_run,
            audit=audit,
            state_path=args.state_file,
        )
    finally:
        conn.close()

    print(json.dumps({k: v for k, v in summary.items()}, indent=2, ensure_ascii=False, default=str))
    # Non-zero exit if the report could not be delivered, so launchd notices.
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
