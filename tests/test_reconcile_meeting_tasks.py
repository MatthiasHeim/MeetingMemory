"""Tests for the meeting decision-record backstop.

The sweep (tools/reconcile_meeting_tasks.py) began as the 2026-07-10 backstop
that created Linear tickets from action insights. Since 2026-09-30 it only
checks `sources.metadata.task_decisions` (written by the /meeting-actions
agent): a meeting with a record is skipped, a meeting without one is reported
by one Telegram line, and the sweep never writes to Linear.

These tests exercise the orchestration with injected fakes for the DB
connection, the Linear gate and the notifier: no real DB, Linear or Telegram.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import reconcile_meeting_tasks as rc  # noqa: E402


# ── fakes ──────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.last_cursor = None

    def cursor(self):
        self.last_cursor = _FakeCursor(self._rows)
        return self.last_cursor

    def close(self):
        pass


class _FakeGateway:
    """Read-only fake. It deliberately has NO create_task: any attempt by the
    sweep to create a ticket raises AttributeError and fails the test."""

    def __init__(self, existing=None):
        self._existing = {str(x) for x in (existing or [])}
        self.gate_calls = 0
        self.raise_on_gate = False

    def source_ids_with_tasks(self):
        self.gate_calls += 1
        if self.raise_on_gate:
            raise RuntimeError("linear gate unreachable")
        return set(self._existing)


class _FakeNotify:
    def __init__(self, fail=False):
        self.lines: list[str] = []
        self.fail = fail

    def __call__(self, line):
        if self.fail:
            raise RuntimeError("telegram down")
        self.lines.append(line)


class _FakeAudit:
    def __init__(self):
        self.events = []

    def log(self, skill, event_type, details=None, status="ok", context=None):
        self.events.append(
            {"skill": skill, "event_type": event_type, "details": details,
             "status": status, "context": context}
        )
        return len(self.events)


def _row(sid, _iid=None, *, title="Meeting", company="BlueCare",
         started=datetime(2026, 7, 10, 6, 0, tzinfo=timezone.utc), record=False, **_ignored):
    """One meeting row in the exact column order the sweep's SELECT emits.
    `_iid` and extra kwargs are accepted so a meeting can be written with or
    without insights: the sweep no longer reads insights at all."""
    return (sid, title, company, started, record)


# ── pure helpers ───────────────────────────────────────────────────────

def test_parse_meeting_source_id():
    assert rc.parse_meeting_source_id("transcript:463:sql-export") == "463"
    assert rc.parse_meeting_source_id("transcript:466:recon-4501") == "466"
    assert rc.parse_meeting_source_id("email-triage:abc") is None
    assert rc.parse_meeting_source_id("") is None
    assert rc.parse_meeting_source_id("transcript:") is None


# ── orchestration ──────────────────────────────────────────────────────

NOW = datetime(2026, 7, 11, 18, 30, tzinfo=timezone.utc)


def test_meeting_with_decision_record_is_skipped_and_not_reported():
    conn = _FakeConn([_row(466, record=True)])
    notify = _FakeNotify()
    summary = rc.reconcile(conn, _FakeGateway(), window_hours=48, notify=notify, now=NOW)
    assert summary["meetings_in_window"] == 1
    assert summary["meetings_skipped_decision_record"] == 1
    assert summary["meetings_unrecorded"] == 0
    assert notify.lines == []
    assert summary["telegram_sent"] is False


def test_meeting_without_record_is_reported_and_creates_no_ticket():
    """Record absent → reported. _FakeGateway has no create_task, so a single
    Linear create call from the sweep would raise and fail this test."""
    conn = _FakeConn([_row(466, title="BlueCare Retainer")])
    gw = _FakeGateway()
    notify = _FakeNotify()
    summary = rc.reconcile(conn, gw, window_hours=48, notify=notify, now=NOW)
    assert summary["meetings_unrecorded"] == 1
    assert summary["unrecorded"][0]["source_id"] == 466
    assert notify.lines == [
        "Meeting BlueCare Retainer (source 466) has no ticket decisions — "
        "re-run /meeting-actions --source-id 466"
    ]
    assert summary["telegram_sent"] is True
    assert not hasattr(gw, "create_task")
    assert not hasattr(rc, "build_task_spec")


def test_two_sided_mixed_batch_sends_one_line_listing_only_unrecorded():
    conn = _FakeConn([
        _row(466, 1, title="Has record", record=True),
        _row(467, 2, title="No record A"),
        _row(468, 3, title="No record B"),
    ])
    notify = _FakeNotify()
    summary = rc.reconcile(conn, _FakeGateway(), window_hours=48, notify=notify, now=NOW)
    assert summary["meetings_skipped_decision_record"] == 1
    assert summary["meetings_unrecorded"] == 2
    assert len(notify.lines) == 1  # ONE line per sweep
    line = notify.lines[0]
    assert "No record A (source 467)" in line and "No record B (source 468)" in line
    assert "Has record" not in line
    assert "\n" not in line


def test_legacy_meeting_with_tickets_but_no_record_is_not_reported():
    """Meetings from before the record existed: agent ran, tickets exist."""
    conn = _FakeConn([_row(463, 4329)])
    notify = _FakeNotify()
    summary = rc.reconcile(conn, _FakeGateway(existing=["463"]), window_hours=48,
                           notify=notify, now=NOW)
    assert summary["meetings_skipped_existing_tasks"] == 1
    assert summary["meetings_unrecorded"] == 0
    assert notify.lines == []


def test_record_wins_over_gate_even_without_tickets():
    """update/close/none-only meetings have a record and zero tickets: skipped."""
    conn = _FakeConn([_row(1121, 8387, record=True)])
    notify = _FakeNotify()
    summary = rc.reconcile(conn, _FakeGateway(existing=[]), window_hours=48,
                           notify=notify, now=NOW)
    assert summary["meetings_skipped_decision_record"] == 1
    assert notify.lines == []


def test_no_meetings_sends_nothing():
    notify = _FakeNotify()
    summary = rc.reconcile(_FakeConn([]), _FakeGateway(), window_hours=48,
                           notify=notify, now=NOW)
    assert summary["meetings_in_window"] == 0
    assert notify.lines == []


def test_dry_run_reports_in_summary_but_sends_and_persists_nothing(tmp_path):
    state = tmp_path / "state.json"
    notify = _FakeNotify()
    summary = rc.reconcile(_FakeConn([_row(466, 4501)]), _FakeGateway(),
                           window_hours=48, dry_run=True, state_path=state,
                           notify=notify, now=NOW)
    assert summary["dry_run"] is True
    assert summary["meetings_unrecorded"] == 1
    assert summary["telegram_sent"] is False
    assert notify.lines == []
    assert not state.exists()


def test_notify_failure_counts_as_error_not_crash():
    notify = _FakeNotify(fail=True)
    summary = rc.reconcile(_FakeConn([_row(466, 4501)]), _FakeGateway(),
                           window_hours=48, notify=notify, now=NOW)
    assert summary["errors"] == 1
    assert summary["telegram_sent"] is False
    assert summary["meetings_unrecorded"] == 1


def test_aborts_when_gate_read_fails_and_reports_nothing():
    gw = _FakeGateway()
    gw.raise_on_gate = True
    notify = _FakeNotify()
    with pytest.raises(RuntimeError):
        rc.reconcile(_FakeConn([_row(466, 4501)]), gw, window_hours=48, notify=notify)
    assert notify.lines == []


def test_writes_state_file_with_reported_sources(tmp_path):
    state = tmp_path / "reconcile_state.json"
    rc.reconcile(_FakeConn([_row(466, 4501)]), _FakeGateway(), window_hours=48,
                 state_path=state, notify=_FakeNotify(), now=NOW)
    data = json.loads(state.read_text())
    assert data["last_run_utc"] == NOW.isoformat()
    assert "466" in data["reported_sources"]
    assert data["last_summary"]["meetings_unrecorded"] == 1


def test_audits_run_with_error_status_when_notify_fails():
    audit = _FakeAudit()
    rc.reconcile(_FakeConn([_row(466, 4501)]), _FakeGateway(), window_hours=48,
                 audit=audit, notify=_FakeNotify(fail=True), now=NOW)
    assert len(audit.events) == 1
    ev = audit.events[0]
    assert ev["skill"] == rc.AUDIT_SKILL
    assert ev["event_type"] == "run"
    assert ev["status"] == "error"
    assert "unrecorded=1" in ev["details"]


def test_send_telegram_raises_when_script_missing(monkeypatch):
    monkeypatch.setattr(rc, "_telegram_notify_script", lambda: None)
    with pytest.raises(RuntimeError):
        rc.send_telegram("x")


def test_fetch_meetings_returns_one_row_per_meeting_with_record_flag():
    conn = _FakeConn([_row(466, title="A"), _row(467, title="B", record=True)])
    meetings = rc.fetch_meetings(conn, 48)
    by_id = {m.source_id: m for m in meetings}
    assert set(by_id) == {466, 467}
    assert by_id[466].has_decision_record is False
    assert by_id[467].has_decision_record is True
    sql, params = conn.last_cursor.executed[0]
    assert params == (48,)
    assert "task_decisions" in sql
    assert "source_type = 'meeting'" in sql and "origin = 'noscribe'" in sql
    assert "insights" not in sql  # action insights are no longer consulted


def test_meeting_with_zero_insights_and_no_record_is_reported():
    """The agent died before extracting anything: no insight rows exist."""
    notify = _FakeNotify()
    summary = rc.reconcile(_FakeConn([_row(1130, title="Died early")]), _FakeGateway(),
                           window_hours=48, notify=notify, now=NOW)
    assert summary["meetings_unrecorded"] == 1
    assert notify.lines == [
        "Meeting Died early (source 1130) has no ticket decisions — "
        "re-run /meeting-actions --source-id 1130"
    ]


def test_meeting_with_zero_insights_but_a_record_is_skipped():
    """An explicit `none` decision on a meeting with no insights is fine."""
    notify = _FakeNotify()
    summary = rc.reconcile(_FakeConn([_row(1131, record=True)]), _FakeGateway(),
                           window_hours=48, notify=notify, now=NOW)
    assert summary["meetings_skipped_decision_record"] == 1
    assert notify.lines == []


# ── gate read (source_ids_with_tasks) with a fake linear_client ─────────

def test_gateway_source_ids_with_tasks_parses_meta(monkeypatch):
    import types

    fake_lc = types.SimpleNamespace()
    issues = [
        {"description": "meta1"},
        {"description": "meta2"},
        {"description": "meta3"},
    ]
    meta_map = {
        "meta1": {"source_id": "transcript:463:sql-export"},
        "meta2": {"source_id": "transcript:466:recon-4501"},
        "meta3": {"source_id": "email-triage:xyz"},  # not a meeting task
    }
    fake_lc.query = lambda label, limit: issues
    fake_lc.parse_meta = lambda desc: meta_map[desc]
    monkeypatch.setitem(sys.modules, "linear_client", fake_lc)

    gw = rc.LinearGateway(fetch_limit=250)
    assert gw.source_ids_with_tasks() == {"463", "466"}


def test_fetch_meetings_reads_decision_record_flag():
    meetings = rc.fetch_meetings(
        _FakeConn([_row(466, 1, record=True), _row(467, 2, record=False)]), 48)
    by_id = {m.source_id: m for m in meetings}
    assert by_id[466].has_decision_record is True
    assert by_id[467].has_decision_record is False
