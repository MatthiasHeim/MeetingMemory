"""Tests for tools/update_source_content_text.py.

docs/BACKFILL-speaker-attribution-2026-08-07.md documented a gap: once a
transcript JSON is repaired on disk, nothing pushes the fix into
InsightBase — re-extraction reads whatever sources.content_text already
says. This tool closes that gap by reusing neon_insert's own formatting
(_read_transcript) and fingerprint (_compute_content_revision_id) so a
repaired row matches byte-for-byte what a fresh insert_source would have
written.

Everything here runs against a fake connection/cursor — never a real
database. The real-DB wiring is proven separately by a --dry-run smoke
test against actual source 767 (see this branch's final report); that one
is not repeatable as an automated test since it depends on a live local
Postgres and a specific meeting having been recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import update_source_content_text as usct  # noqa: E402


class _FakeCursor:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result
        self.executed: list[tuple[str, tuple]] = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetch_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    """Models psycopg2's context-manager semantics closely enough for
    these tests: `with conn:` commits on clean exit, does NOT commit (real
    psycopg2 rolls back) when an exception propagates out of the block."""

    def __init__(self, fetch_result=None):
        self.cursor_obj = _FakeCursor(fetch_result)
        self.closed = False
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.committed = True
        return False

    def close(self):
        self.closed = True


def _json_file(tmp_path, transcript: str) -> Path:
    p = tmp_path / "2026-08-07_14-11-24.json"
    p.write_text(json.dumps({"transcript": transcript}), encoding="utf-8")
    return p


# ── _first_diff_line: pure-function unit tests ─────────────────────────


def test_first_diff_line_identical_returns_none():
    assert usct._first_diff_line("a\nb\nc", "a\nb\nc") is None


def test_first_diff_line_finds_first_divergent_line():
    old = "line1\nline2\nline3"
    new = "line1\nCHANGED\nline3"
    assert usct._first_diff_line(old, new) == (2, "line2", "CHANGED")


def test_first_diff_line_handles_appended_lines():
    """New text is a strict extension of old (lines added at the end)."""
    old = "line1\nline2"
    new = "line1\nline2\nline3"
    line_no, old_line, new_line = usct._first_diff_line(old, new)
    assert line_no == 3
    assert old_line == ""
    assert new_line == "line3"


def test_first_diff_line_handles_truncated_lines():
    """SIDE B of the appended-lines case: old text is longer (repair
    removed trailing content) — the diff must point at old's tail, not
    new's."""
    old = "line1\nline2\nline3"
    new = "line1\nline2"
    line_no, old_line, new_line = usct._first_diff_line(old, new)
    assert line_no == 3
    assert old_line == "line3"
    assert new_line == ""


# ── run(): refuse-to-run guards ──────────────────────────────────────────


def test_refuses_when_source_row_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(usct, "_get_conn", lambda: _FakeConn(fetch_result=None))
    json_path = _json_file(tmp_path, "[00:00] Matthias: Hi.\n")
    rc = usct.run(source_id=999999999, json_path=str(json_path), dry_run=True)
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_refuses_when_json_missing(tmp_path, capsys):
    rc = usct.run(source_id=1, json_path=str(tmp_path / "nope.json"), dry_run=True)
    assert rc == 1
    assert "not found" in capsys.readouterr().err


# ── run(): dry-run reports the diff but never writes ────────────────────


def test_dry_run_reports_diff_but_issues_no_update(monkeypatch, tmp_path, capsys):
    fake_conn = _FakeConn(fetch_result=("old text", "oldrev"))
    monkeypatch.setattr(usct, "_get_conn", lambda: fake_conn)
    json_path = _json_file(tmp_path, "new text")
    rc = usct.run(source_id=767, json_path=str(json_path), dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "CHANGED" in out
    assert "dry-run" in out
    # Exactly one DB call happened -- the SELECT. No UPDATE was ever issued.
    assert len(fake_conn.cursor_obj.executed) == 1
    assert fake_conn.cursor_obj.executed[0][0].strip().upper().startswith("SELECT")


def test_unchanged_content_reports_unchanged(monkeypatch, tmp_path, capsys):
    """Mirrors the real --dry-run run against source 767 in the final
    report: an untouched transcript must report 'unchanged', not a
    spurious diff."""
    fake_conn = _FakeConn(fetch_result=("same text", "somerev"))
    monkeypatch.setattr(usct, "_get_conn", lambda: fake_conn)
    json_path = _json_file(tmp_path, "same text")
    rc = usct.run(source_id=767, json_path=str(json_path), dry_run=True)
    assert rc == 0
    assert "unchanged" in capsys.readouterr().out
    assert len(fake_conn.cursor_obj.executed) == 1  # still just the SELECT


# ── run(): a real update issues exactly one UPDATE with the right values ──


def test_apply_writes_new_content_and_revision(monkeypatch, tmp_path):
    fake_conn = _FakeConn(fetch_result=("old text", "oldrev"))
    monkeypatch.setattr(usct, "_get_conn", lambda: fake_conn)
    json_path = _json_file(tmp_path, "repaired text")
    rc = usct.run(source_id=767, json_path=str(json_path), dry_run=False)
    assert rc == 0
    executed = fake_conn.cursor_obj.executed
    assert len(executed) == 2  # SELECT, then UPDATE
    update_sql, update_params = executed[1]
    assert "UPDATE sources" in update_sql
    content_arg, revision_json_arg, source_id_arg = update_params
    assert content_arg == "repaired text"
    assert json.loads(revision_json_arg) == usct._compute_content_revision_id("repaired text")
    assert source_id_arg == 767
    assert fake_conn.committed is True


def test_apply_raises_if_update_affects_no_rows(monkeypatch, tmp_path):
    """A vanished row between the existence check and the UPDATE (race)
    must surface loudly, not silently report success."""
    fake_conn = _FakeConn(fetch_result=("old text", "oldrev"))
    fake_conn.cursor_obj.rowcount = 0
    monkeypatch.setattr(usct, "_get_conn", lambda: fake_conn)
    json_path = _json_file(tmp_path, "repaired text")
    raised = False
    try:
        usct.run(source_id=767, json_path=str(json_path), dry_run=False)
    except RuntimeError as e:
        raised = True
        assert "767" in str(e)
    assert raised, "expected a RuntimeError when the UPDATE affects 0 rows"
    assert fake_conn.committed is False
