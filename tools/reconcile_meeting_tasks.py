#!/usr/bin/env python3
"""
reconcile_meeting_tasks — deterministic Linear-task gap-fill for MeetingMemory.

WHY THIS EXISTS (2026-07-10 incident)
─────────────────────────────────────
The transcription→insight→action pipeline splits work between a deterministic
watcher (tools/transcribe_watcher.py) and a fire-and-forget headless Claude
session (`claude -p /meeting-actions`). The watcher seeds the InsightBase
`sources` row and Gemini-derived fields; the Claude session owns the LLM-only
downstream — insight extraction, **Linear task creation**, follow-up email
drafts, ClientContext refresh, Telegram.

On 2026-07-10 every headless Claude session died on its first line
("You've hit your session limit"). For the Philipp meeting (source 463) insight
extraction had already landed, but the Claude session that owns task creation
never ran — and because that trigger is fire-and-forget with NO retry, every
Linear task for that meeting was silently dropped.

This job is the deterministic backstop for exactly that failure mode. It is
PURE PYTHON — it never starts a Claude/LLM session, so it can never hit a
session limit. It reads InsightBase directly and creates any missing Linear
tasks through Brain's existing `linear_client.py`.

WHAT IT DOES
────────────
For each `sources` row from the last ~48h (``--window-hours``) that has at
least one active ``insights`` row of ``type='action'`` but ZERO Linear issues
referencing that meeting (source_id pattern ``transcript:<source_id>:*``),
create one Linear task per action insight.

Gate is at the MEETING level, not per-insight: if the meeting already has ≥1
task, we assume /meeting-actions succeeded (or was manually backfilled) and
skip the whole meeting. This deliberately avoids duplicating a
partially-successful run — the tradeoff the design accepts is that a meeting
whose first sweep crashed mid-way is not re-topped-up on a later run (a single
sweep creates all of a meeting's tasks in one pass, so this only bites on a
crash between two insights of the same meeting).

IDEMPOTENCY
───────────
Two independent layers:
  1. Meeting-level gate — a meeting with any ``transcript:<sid>:*`` task is
     skipped. A prior sweep's own tasks (``transcript:<sid>:recon-<insight_id>``)
     satisfy this, so a second run is a no-op.
  2. linear_client upserts on ``source_id`` (``upsert_by_source_id``), so even
     if the gate is bypassed the same insight never spawns a duplicate issue.

SCOPE (what it intentionally does NOT do)
─────────────────────────────────────────
Only Linear tasks. It does NOT extract insights (that still needs the LLM
path), draft emails, refresh ClientContext, or send Telegram. Its single job
is to guarantee that action insights which already exist in InsightBase are
reflected as Linear tasks.

Environment:
  INSIGHTBASE_DATABASE_URL — Postgres DSN for InsightBase (via neon_insert).
  LINEAR_API_KEY           — read by Brain's linear_client (env or Brain/.env).

CLI:
    python3 reconcile_meeting_tasks.py                 # reconcile last 48h
    python3 reconcile_meeting_tasks.py --window-hours 72
    python3 reconcile_meeting_tasks.py --dry-run       # show, create nothing
    python3 reconcile_meeting_tasks.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths / constants ─────────────────────────────────────────────────

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Brain owns the Linear client (single source of truth for Linear writes) and
# the audit log. Add its scripts dir to the path so we can import them.
BRAIN_SCRIPTS_DIR = Path("/Users/Matthias/Repos/Brain/.claude/scripts")
if str(BRAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_SCRIPTS_DIR))

LINEAR_CLIENT_PATH = BRAIN_SCRIPTS_DIR / "linear_client.py"

DEFAULT_WINDOW_HOURS = 48
DEFAULT_STATE_FILE = (
    Path.home() / "Documents" / "MeetingRecorder" / "reconcile_meeting_tasks_state.json"
)

# Skill name used for brain_audit events.
AUDIT_SKILL = "meeting-actions-reconcile"

# Linear task defaults — mirror /meeting-actions Step 5 conventions.
TASK_SOURCE = "meeting-action"
TASK_STATE = "Triage"          # automation-created → Triage inbox for human accept/decline
TASK_PRIORITY = "medium"       # /meeting-actions default priority
EXTERNAL_OWNER_LABEL = "external-owner"

# How many recent meeting-action issues to pull when building the "already has
# tasks" gate snapshot. We only reconcile meetings from the last ~48h, so any
# existing task for one of them was created recently and lands in this window.
# 250 is Linear's hard per-page `first:` cap — asking for more is a GraphQL
# "Argument Validation Error".
DEFAULT_LINEAR_FETCH_LIMIT = 250

# Assignees treated as "Matthias" (no external-owner label). Lowercased.
_SELF_NAMES = {"matthias", "matthias heim", "matthi", "mättu"}

logger = logging.getLogger("reconcile_meeting_tasks")


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class ActionInsight:
    insight_id: int
    title: str
    content: str
    assignee: Optional[str]
    due_date: Optional[date]
    confidence: str
    source_quote: Optional[str]


@dataclass
class Meeting:
    source_id: int
    title: Optional[str]
    company: Optional[str]
    started_at: Optional[datetime]
    actions: list[ActionInsight] = field(default_factory=list)


# ── InsightBase reads ─────────────────────────────────────────────────

def fetch_meetings_with_actions(conn, window_hours: int) -> list[Meeting]:
    """Return meetings from the last `window_hours` that have ≥1 active action
    insight, each with its action insights attached.

    Window is measured on the meeting time (`started_at`, falling back to the
    row's `created_at` when a recording has no parsed start time).
    """
    meetings: dict[int, Meeting] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                s.id, s.title, s.company, s.started_at,
                i.id, i.title, i.content, i.assignee, i.due_date,
                i.confidence, i.source_quote
            FROM sources s
            JOIN insights i ON i.source_id = s.id
            WHERE COALESCE(s.started_at, s.created_at)
                  >= now() - make_interval(hours => %s)
              AND i.type = 'action'
              AND COALESCE(i.lifecycle_status, 'active') = 'active'
            ORDER BY s.id, i.id;
            """,
            (window_hours,),
        )
        for row in cur.fetchall():
            (
                sid, s_title, company, started_at,
                iid, i_title, content, assignee, due_date, confidence, quote,
            ) = row
            m = meetings.get(sid)
            if m is None:
                m = Meeting(
                    source_id=sid, title=s_title, company=company,
                    started_at=started_at,
                )
                meetings[sid] = m
            m.actions.append(
                ActionInsight(
                    insight_id=iid,
                    title=i_title,
                    content=content,
                    assignee=assignee,
                    due_date=due_date,
                    confidence=confidence,
                    source_quote=quote,
                )
            )
    return list(meetings.values())


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


def _is_external_owner(assignee: Optional[str]) -> bool:
    """True if the action's owner is someone other than Matthias."""
    if not assignee:
        return False
    return assignee.strip().lower() not in _SELF_NAMES


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_task_spec(meeting: Meeting, action: ActionInsight) -> dict:
    """Build the linear_client `--create-from-json` spec for one action insight.

    Pure function — no I/O — so it is trivially unit-testable.
    """
    due = action.due_date.isoformat() if action.due_date else None
    started = (
        meeting.started_at.date().isoformat() if meeting.started_at else "unknown date"
    )

    body_lines = [action.content.strip() if action.content else ""]
    body_lines.append("")
    body_lines.append(f"**Owner:** {action.assignee or 'unassigned'}")
    if action.source_quote:
        body_lines.append(f"**Source quote:** “{action.source_quote.strip()}”")
    body_lines.append("")
    body_lines.append(
        "_Auto-created by the deterministic reconciliation sweep "
        "(reconcile_meeting_tasks.py) — the meeting's /meeting-actions Claude "
        "session did not create Linear tasks._"
    )
    body_lines.append(
        f"_Meeting: {meeting.title or 'untitled'} "
        f"(InsightBase source {meeting.source_id}, {started}). "
        f"Insight {action.insight_id}, confidence {action.confidence}._"
    )

    spec: dict = {
        "source": TASK_SOURCE,
        "source_id": f"{meeting_task_prefix(meeting.source_id)}recon-{action.insight_id}",
        "title": _truncate(action.title, 250),
        "body": "\n".join(body_lines).strip() + "\n",
        "priority": TASK_PRIORITY,
        "state": TASK_STATE,
    }
    if meeting.company:
        spec["client"] = meeting.company
    if due:
        spec["due"] = due
    if _is_external_owner(action.assignee):
        # Mirror /meeting-actions convention: still create the task, but flag
        # it so it never auto-closes on Matthias's behalf.
        spec["labels"] = [EXTERNAL_OWNER_LABEL]
    return spec


# ── Linear gateway ────────────────────────────────────────────────────

class LinearGateway:
    """Reads the "already has tasks" gate from Linear and creates tasks.

    Reads go through the imported `linear_client` module (reuses its meta
    parser); writes go through the `linear_client.py --create-from-json -`
    CLI, matching the contract every other writer in this system uses.
    """

    def __init__(
        self,
        client_path: Path = LINEAR_CLIENT_PATH,
        fetch_limit: int = DEFAULT_LINEAR_FETCH_LIMIT,
    ):
        self.client_path = Path(client_path)
        self.fetch_limit = fetch_limit

    def source_ids_with_tasks(self) -> set[str]:
        """Set of InsightBase source ids that already have ≥1 Linear task.

        Raises on failure — a failed gate read must ABORT the sweep rather than
        silently create duplicate tasks.
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

    def create_task(self, spec: dict) -> dict:
        """Create/upsert one Linear task; return {issue_id, identifier, url, created}."""
        proc = subprocess.run(
            [sys.executable, str(self.client_path), "--create-from-json", "-"],
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"linear_client exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        out = proc.stdout.strip()
        # The client prints the JSON result on the last line.
        last = out.splitlines()[-1] if out else "{}"
        return json.loads(last)


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
) -> dict:
    """Core sweep. Returns a summary dict. Injected `conn`/`gateway`/`audit`
    keep this unit-testable without a real DB or Linear."""
    now = now or datetime.now(timezone.utc)
    meetings = fetch_meetings_with_actions(conn, window_hours)
    logger.info(
        "Found %d meeting(s) in the last %dh with action insights",
        len(meetings), window_hours,
    )

    # Snapshot the gate BEFORE creating anything.
    existing = gateway.source_ids_with_tasks()

    summary = {
        "window_hours": window_hours,
        "dry_run": dry_run,
        "meetings_with_actions": len(meetings),
        "meetings_skipped_existing_tasks": 0,
        "meetings_reconciled": 0,
        "action_insights_seen": 0,
        "tasks_created": 0,
        "tasks_upserted_existing": 0,
        "errors": 0,
        "reconciled": [],  # per-meeting detail
    }

    for m in meetings:
        summary["action_insights_seen"] += len(m.actions)
        if str(m.source_id) in existing:
            summary["meetings_skipped_existing_tasks"] += 1
            logger.info(
                "Skip meeting %s (%s): already has ≥1 Linear task",
                m.source_id, m.title or "untitled",
            )
            continue

        logger.info(
            "Reconciling meeting %s (%s): %d action insight(s)%s",
            m.source_id, m.title or "untitled", len(m.actions),
            " [dry-run]" if dry_run else "",
        )
        created_here: list[dict] = []
        for a in m.actions:
            spec = build_task_spec(m, a)
            if dry_run:
                logger.info("  would create: %s → %s", spec["source_id"], spec["title"])
                created_here.append({"source_id": spec["source_id"], "title": spec["title"]})
                continue
            try:
                res = gateway.create_task(spec)
                if res.get("created"):
                    summary["tasks_created"] += 1
                else:
                    summary["tasks_upserted_existing"] += 1
                logger.info(
                    "  %s %s → %s",
                    "created" if res.get("created") else "updated",
                    res.get("identifier", "?"), spec["title"],
                )
                created_here.append(
                    {
                        "source_id": spec["source_id"],
                        "identifier": res.get("identifier"),
                        "url": res.get("url"),
                        "created": res.get("created"),
                    }
                )
            except Exception as e:  # noqa: BLE001 — one bad insight must not abort the run
                summary["errors"] += 1
                logger.error("  failed to create task for insight %s: %s", a.insight_id, e)

        summary["meetings_reconciled"] += 1
        summary["reconciled"].append(
            {
                "source_id": m.source_id,
                "title": m.title,
                "company": m.company,
                "tasks": created_here,
            }
        )

    _finalize(summary, now, dry_run, audit, state_path)
    return summary


def _finalize(summary, now, dry_run, audit, state_path):
    """Persist state + emit a brain_audit event. Best-effort — never raises."""
    if state_path is not None and not dry_run:
        state = _load_state(state_path)
        state["last_run_utc"] = now.isoformat()
        state["last_summary"] = {
            k: v for k, v in summary.items() if k != "reconciled"
        }
        reconciled_map = state.setdefault("reconciled_sources", {})
        for entry in summary["reconciled"]:
            reconciled_map[str(entry["source_id"])] = {
                "reconciled_utc": now.isoformat(),
                "title": entry["title"],
                "tasks": [t.get("identifier") for t in entry["tasks"] if t.get("identifier")],
            }
        _save_state(state_path, state)

    if audit is not None:
        details = (
            f"window={summary['window_hours']}h "
            f"meetings={summary['meetings_with_actions']} "
            f"reconciled={summary['meetings_reconciled']} "
            f"skipped={summary['meetings_skipped_existing_tasks']} "
            f"tasks_created={summary['tasks_created']} "
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
                    {k: v for k, v in summary.items() if k != "reconciled"}
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
        description="Deterministically gap-fill missing Linear tasks for "
        "recently-transcribed meetings whose /meeting-actions session failed."
    )
    parser.add_argument(
        "--window-hours", type=int, default=DEFAULT_WINDOW_HOURS,
        help=f"Look back this many hours (default: {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be created without touching Linear or state.",
    )
    parser.add_argument(
        "--linear-fetch-limit", type=int, default=DEFAULT_LINEAR_FETCH_LIMIT,
        help="How many recent meeting-action issues to scan for the gate.",
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
    audit = _build_audit()

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
    # Non-zero exit if any task failed, so launchd/monitoring notices.
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
