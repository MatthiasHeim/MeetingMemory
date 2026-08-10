"""Tests for the deterministic Linear-task reconciliation sweep.

The sweep (tools/reconcile_meeting_tasks.py) is the backstop for the 2026-07-10
incident: every fire-and-forget `claude -p /meeting-actions` session died on a
session limit, so meetings that had action insights in InsightBase never got
their Linear tasks. This sweep gap-fills them deterministically (pure Python,
no Claude session).

These tests exercise the pure spec-builder and the orchestration with injected
fakes for the DB connection and the Linear gateway — no real DB or Linear.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
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
    """Stateful fake: source_ids_with_tasks() reflects both preset existing
    tasks AND anything created_task() has produced so far, so a second
    reconcile() over the same gateway exercises real idempotency."""

    def __init__(self, existing=None, fail_insight_ids=None):
        self._existing = {str(x) for x in (existing or [])}
        self.fail_insight_ids = {str(x) for x in (fail_insight_ids or [])}
        self.created: list[dict] = []
        self.gate_calls = 0
        self.raise_on_gate = False

    def source_ids_with_tasks(self):
        self.gate_calls += 1
        if self.raise_on_gate:
            raise RuntimeError("linear gate unreachable")
        derived = set(self._existing)
        for spec in self.created:
            sid = rc.parse_meeting_source_id(spec["source_id"])
            if sid:
                derived.add(sid)
        return derived

    def create_task(self, spec):
        iid = spec["source_id"].rsplit("recon-", 1)[-1]
        if iid in self.fail_insight_ids:
            raise RuntimeError("linear create failed")
        self.created.append(spec)
        return {
            "identifier": f"LAI-{len(self.created)}",
            "url": f"https://linear.app/x/{len(self.created)}",
            "created": True,
        }


class _FakeAudit:
    def __init__(self):
        self.events = []

    def log(self, skill, event_type, details=None, status="ok", context=None):
        self.events.append(
            {"skill": skill, "event_type": event_type, "details": details,
             "status": status, "context": context}
        )
        return len(self.events)


def _row(sid, iid, *, title="Meeting", company="BlueCare",
         started=datetime(2026, 7, 10, 6, 0, tzinfo=timezone.utc),
         i_title="Do the thing", content="Context here", assignee="Matthias Heim",
         due=None, confidence="high", quote="I'll do it"):
    """Build one JOIN row in the exact column order the sweep's SELECT emits."""
    return (sid, title, company, started, iid, i_title, content, assignee, due,
            confidence, quote)


# ── pure helpers ───────────────────────────────────────────────────────

def test_parse_meeting_source_id():
    assert rc.parse_meeting_source_id("transcript:463:sql-export") == "463"
    assert rc.parse_meeting_source_id("transcript:466:recon-4501") == "466"
    assert rc.parse_meeting_source_id("email-triage:abc") is None
    assert rc.parse_meeting_source_id("") is None
    assert rc.parse_meeting_source_id("transcript:") is None


def test_build_task_spec_basic():
    m = rc.Meeting(source_id=466, title="Confluence rollout", company="BlueCare",
                   started_at=datetime(2026, 7, 10, 16, 34, tzinfo=timezone.utc))
    a = rc.ActionInsight(insight_id=4501, title="Send the plugin note",
                         content="Matthias sends the note to Stefan.",
                         assignee="Matthias Heim", due_date=date(2026, 7, 11),
                         confidence="high", source_quote="I'll send it")
    spec = rc.build_task_spec(m, a)
    assert spec["source"] == "meeting-action"
    assert spec["source_id"] == "transcript:466:recon-4501"
    assert spec["title"] == "Send the plugin note"
    assert spec["state"] == "Triage"
    assert spec["priority"] == "medium"
    assert spec["client"] == "BlueCare"
    assert spec["due"] == "2026-07-11"
    assert "labels" not in spec  # Matthias-owned → no external-owner label
    assert "source 466" in spec["body"]
    assert "4501" in spec["body"]


def test_build_task_spec_external_owner_labeled():
    m = rc.Meeting(source_id=466, title="x", company=None, started_at=None)
    a = rc.ActionInsight(insight_id=9, title="Stefan rolls out plugins",
                         content="", assignee="Stefan", due_date=None,
                         confidence="medium", source_quote=None)
    spec = rc.build_task_spec(m, a)
    assert spec["labels"] == ["external-owner"]
    assert "client" not in spec   # no company
    assert "due" not in spec      # no due date


def test_build_task_spec_truncates_title():
    m = rc.Meeting(source_id=1, title=None, company=None, started_at=None)
    long = "x" * 400
    a = rc.ActionInsight(insight_id=1, title=long, content="", assignee=None,
                         due_date=None, confidence="low", source_quote=None)
    spec = rc.build_task_spec(m, a)
    assert len(spec["title"]) <= 250


def test_is_external_owner():
    assert rc._is_external_owner("Stefan") is True
    assert rc._is_external_owner("Matthias") is False
    assert rc._is_external_owner("Matthias Heim") is False
    assert rc._is_external_owner(None) is False
    assert rc._is_external_owner("") is False


# ── orchestration ──────────────────────────────────────────────────────

def test_reconcile_creates_one_task_per_action_for_ungated_meeting():
    rows = [_row(466, 4501, i_title="A"), _row(466, 4502, i_title="B")]
    conn = _FakeConn(rows)
    gw = _FakeGateway(existing=[])
    summary = rc.reconcile(conn, gw, window_hours=48, now=datetime(2026, 7, 11, tzinfo=timezone.utc))
    assert summary["meetings_with_actions"] == 1
    assert summary["meetings_reconciled"] == 1
    assert summary["meetings_skipped_existing_tasks"] == 0
    assert summary["tasks_created"] == 2
    assert summary["errors"] == 0
    assert {s["source_id"] for s in gw.created} == {
        "transcript:466:recon-4501", "transcript:466:recon-4502"}


def test_reconcile_skips_meeting_that_already_has_tasks():
    """The meeting-level gate: source 463 already has LAI-376..379."""
    rows = [_row(463, 4329)]
    conn = _FakeConn(rows)
    gw = _FakeGateway(existing=["463"])
    summary = rc.reconcile(conn, gw, window_hours=48)
    assert summary["meetings_skipped_existing_tasks"] == 1
    assert summary["meetings_reconciled"] == 0
    assert summary["tasks_created"] == 0
    assert gw.created == []


def test_reconcile_is_idempotent_across_runs():
    rows = [_row(466, 4501), _row(466, 4502)]
    gw = _FakeGateway(existing=[])
    first = rc.reconcile(_FakeConn(rows), gw, window_hours=48)
    assert first["tasks_created"] == 2
    # Second run over the same gateway: its gate now reports 466 has tasks.
    second = rc.reconcile(_FakeConn(rows), gw, window_hours=48)
    assert second["meetings_skipped_existing_tasks"] == 1
    assert second["tasks_created"] == 0
    assert len(gw.created) == 2  # no new tasks


def test_reconcile_dry_run_creates_nothing(tmp_path):
    rows = [_row(466, 4501)]
    gw = _FakeGateway(existing=[])
    state = tmp_path / "state.json"
    summary = rc.reconcile(_FakeConn(rows), gw, window_hours=48,
                           dry_run=True, state_path=state)
    assert summary["dry_run"] is True
    assert summary["meetings_reconciled"] == 1
    assert summary["tasks_created"] == 0
    assert gw.created == []
    assert not state.exists()  # dry-run must not persist state


def test_reconcile_one_failing_insight_counts_error_and_continues():
    rows = [_row(466, 4501), _row(466, 4502)]
    gw = _FakeGateway(existing=[], fail_insight_ids=[4501])
    summary = rc.reconcile(_FakeConn(rows), gw, window_hours=48)
    assert summary["errors"] == 1
    assert summary["tasks_created"] == 1  # 4502 still created
    assert summary["meetings_reconciled"] == 1


def test_reconcile_aborts_when_gate_read_fails():
    rows = [_row(466, 4501)]
    gw = _FakeGateway(existing=[])
    gw.raise_on_gate = True
    with pytest.raises(RuntimeError):
        rc.reconcile(_FakeConn(rows), gw, window_hours=48)
    assert gw.created == []  # never created anything on a failed gate read


def test_reconcile_writes_state_file(tmp_path):
    rows = [_row(466, 4501)]
    gw = _FakeGateway(existing=[])
    state = tmp_path / "reconcile_state.json"
    now = datetime(2026, 7, 11, 18, 30, tzinfo=timezone.utc)
    rc.reconcile(_FakeConn(rows), gw, window_hours=48, state_path=state, now=now)
    import json
    data = json.loads(state.read_text())
    assert data["last_run_utc"] == now.isoformat()
    assert "466" in data["reconciled_sources"]
    assert data["last_summary"]["tasks_created"] == 1


def test_reconcile_audits_run(tmp_path):
    rows = [_row(466, 4501), _row(466, 4502)]
    gw = _FakeGateway(existing=[], fail_insight_ids=[4502])
    audit = _FakeAudit()
    rc.reconcile(_FakeConn(rows), gw, window_hours=48, audit=audit)
    assert len(audit.events) == 1
    ev = audit.events[0]
    assert ev["skill"] == rc.AUDIT_SKILL
    assert ev["event_type"] == "run"
    assert ev["status"] == "error"  # one insight failed
    assert "tasks_created=1" in ev["details"]


def test_fetch_meetings_groups_actions_by_source():
    rows = [_row(466, 1), _row(466, 2), _row(467, 3)]
    conn = _FakeConn(rows)
    meetings = rc.fetch_meetings_with_actions(conn, 48)
    by_id = {m.source_id: m for m in meetings}
    assert set(by_id) == {466, 467}
    assert len(by_id[466].actions) == 2
    assert len(by_id[467].actions) == 1
    # window param is threaded into the query
    assert conn.last_cursor.executed[0][1] == (48,)


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
