"""Follow-up hand-off: `followup.mode: watcher | automations` and the ready_for_followup marker.

Both modes mark `sources.metadata.pipeline_status = 'ready_for_followup'`. Only
`watcher` mode (the default) also starts headless Claude and sends the Brain
"meeting captured" ping; `automations` mode leaves the run to the
lailix-automations `meeting-followup` job. No test starts Claude, sends
Telegram or touches a database.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import neon_insert  # noqa: E402
import transcribe_watcher as tw  # noqa: E402


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def __getattr__(self, level):
        return lambda msg, *a: self.lines.append((level, msg))


class _FakeProc:
    pid = 4242

    def poll(self):
        return None

    def wait(self, timeout=None):
        raise tw.subprocess.TimeoutExpired(cmd="claude", timeout=timeout)


class Env:
    """A bare watcher with Popen, Telegram and the InsightBase marker faked out."""

    def __init__(self, monkeypatch, tmp_path, mode=None, enabled=True, mark_fails=False, **claude_cfg):
        cfg = {"claude_trigger": {"enabled": enabled, "claude_path": "/bin/claude", "brain_repo": "/repo/brain", **claude_cfg},
               "paths": {"logs": str(tmp_path / "logs")}}
        if mode is not None:
            cfg["followup"] = {"mode": mode}
        self.w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
        self.w.logger = _Log()
        self.w.config = cfg
        self.popen: list[tuple] = []
        self.telegram: list[list[str]] = []
        self.marks: list[dict] = []

        def mark(source_id, transcript_path, triggered_by, prompt_suffix=None):
            if mark_fails:
                raise RuntimeError("db down")
            self.marks.append({"source_id": source_id, "transcript_path": str(transcript_path),
                               "triggered_by": triggered_by, "prompt_suffix": prompt_suffix})

        monkeypatch.setattr(tw, "_insightbase_mark_ready", mark, raising=False)
        monkeypatch.setattr(tw, "NEON_INSERT_AVAILABLE", True)
        monkeypatch.setattr(tw.subprocess, "Popen", lambda args, cwd=None, env=None, stdout=None, stderr=None:
                            self.popen.append((args, cwd)) or _FakeProc())
        monkeypatch.setattr(tw.subprocess, "run", lambda argv, **k: self.telegram.append(argv))
        monkeypatch.setattr(tw.TranscribeWatcher, "_telegram_notify_script", staticmethod(lambda: "/notify.py"))
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def _json(tmp_path):
    return tmp_path / "2026-09-30_10-00-00.json"


# ── watcher mode (default): today's trigger unchanged, plus the marker ───────

@pytest.mark.parametrize("mode", [None, "watcher"])
def test_watcher_mode_marks_ready_and_keeps_todays_trigger_unchanged(monkeypatch, tmp_path, mode):
    env = Env(monkeypatch, tmp_path, mode=mode)
    env.w._trigger_claude(_json(tmp_path), source_id=466)
    assert env.marks == [{"source_id": 466, "transcript_path": str(_json(tmp_path)),
                          "triggered_by": "watcher", "prompt_suffix": None}]
    (args, cwd), = env.popen
    assert args == ["/bin/claude", "-p",
                    f"Read .claude/commands/meeting-actions.md and process transcript: {_json(tmp_path)} --source-id 466",
                    "--dangerously-skip-permissions"]
    assert cwd == "/repo/brain"


def test_watcher_mode_still_triggers_when_the_marker_write_fails(monkeypatch, tmp_path):
    env = Env(monkeypatch, tmp_path, mode="watcher", mark_fails=True)
    env.w._trigger_claude(_json(tmp_path), source_id=466)
    assert len(env.popen) == 1 and env.telegram == []
    assert any(level == "error" and "ready_for_followup" in msg for level, msg in env.w.logger.lines)


def test_watcher_mode_captured_ping_is_still_sent(monkeypatch, tmp_path):
    env = Env(monkeypatch, tmp_path, mode="watcher")
    env.w._notify_telegram_meeting_captured(466, None, None, 600)
    assert len(env.telegram) == 1 and "Meeting captured (#466)" in env.telegram[0][-1]


# ── automations mode: marker only ────────────────────────────────────────────

def test_automations_mode_marks_ready_and_does_not_start_claude_or_ping(monkeypatch, tmp_path):
    env = Env(monkeypatch, tmp_path, mode="automations")
    env.w._trigger_claude(_json(tmp_path), source_id=466)
    assert env.marks == [{"source_id": 466, "transcript_path": str(_json(tmp_path)),
                          "triggered_by": "automations", "prompt_suffix": None}]
    assert env.popen == []           # no headless Claude
    assert env.telegram == []        # no Brain telegram call either
    env.w._notify_telegram_meeting_captured(466, None, None, 600)
    assert env.telegram == []        # the "captured" ping is dropped in this mode


def test_automations_mode_alerts_when_the_handoff_fails_because_nobody_else_will_run(monkeypatch, tmp_path):
    env = Env(monkeypatch, tmp_path, mode="automations", mark_fails=True)
    env.w._trigger_claude(_json(tmp_path), source_id=466)
    assert env.popen == []
    assert len(env.telegram) == 1 and "follow-up NOT queued (source 466)" in env.telegram[0][-1]


def test_automations_mode_without_a_source_row_alerts_instead_of_dropping_silently(monkeypatch, tmp_path):
    env = Env(monkeypatch, tmp_path, mode="automations")
    env.w._trigger_claude(_json(tmp_path), source_id=None)
    assert env.marks == [] and env.popen == [] and len(env.telegram) == 1


# ── shared behaviour ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["watcher", "automations"])
def test_resolve_calendar_suffix_is_recorded_for_the_job_and_sent_by_the_watcher(monkeypatch, tmp_path, mode):
    env = Env(monkeypatch, tmp_path, mode=mode)
    env.w._trigger_claude(_json(tmp_path), source_id=7, resolve_calendar=True)
    suffix = env.marks[0]["prompt_suffix"]
    assert "CALENDAR IDENTITY IS AMBIGUOUS" in suffix and "--source-id 7" in suffix
    if mode == "watcher":
        assert env.popen[0][0][2].endswith("\n" + suffix)   # exactly today's prompt
    else:
        assert env.popen == []


@pytest.mark.parametrize("mode", ["watcher", "automations"])
def test_disabled_claude_trigger_stays_a_kill_switch_in_both_modes(monkeypatch, tmp_path, mode):
    env = Env(monkeypatch, tmp_path, mode=mode, enabled=False)
    env.w._trigger_claude(_json(tmp_path), source_id=466)
    assert env.marks == [] and env.popen == [] and env.telegram == []


def test_unknown_mode_is_rejected_not_guessed():
    assert tw.followup_mode({}) == "watcher"
    assert tw.followup_mode({"followup": {"mode": "automations"}}) == "automations"
    with pytest.raises(ValueError, match="followup.mode"):
        tw.followup_mode({"followup": {"mode": "automation"}})
    with pytest.raises(ValueError, match="followup.mode"):
        tw.TranscribeWatcher({"followup": {"mode": "nope"}}, _Log())


# ── notifier / brain_repo now come from config, current values are the defaults ──

@pytest.fixture
def clean_notifier():
    before = dict(tw._notifier_settings)
    yield
    tw._notifier_settings.update(before)


def test_telegram_script_prefers_configured_path_then_configured_brain_repo(monkeypatch, tmp_path, clean_notifier):
    monkeypatch.delenv("TELEGRAM_NOTIFY_SCRIPT", raising=False)
    configured = tmp_path / "custom_notify.py"
    configured.write_text("#")
    tw.configure_notifier({"notifications": {"telegram_notify_script": str(configured)}})
    assert tw.TranscribeWatcher._telegram_notify_script() == str(configured)
    brain = tmp_path / "brain"
    (brain / ".claude" / "scripts").mkdir(parents=True)
    (brain / ".claude" / "scripts" / "telegram_notify.py").write_text("#")
    tw.configure_notifier({"claude_trigger": {"brain_repo": str(brain)}})
    monkeypatch.setattr(os.path, "exists", lambda p: p == str(brain / ".claude/scripts/telegram_notify.py"))
    assert tw.TranscribeWatcher._telegram_notify_script() == str(brain / ".claude/scripts/telegram_notify.py")


def test_defaults_equal_the_previously_hardcoded_values():
    assert tw.DEFAULT_BRAIN_REPO == os.path.expanduser("~/Repos/Brain")
    assert tw.DEFAULT_CLAUDE_PATH == os.path.expanduser("~/.local/bin/claude")


# ── the marker itself (neon_insert.mark_ready_for_followup) ──────────────────────

def _conn(rowcount=1):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = rowcount
    return conn, cur


def test_mark_ready_writes_status_timestamp_and_followup_and_merges_metadata(monkeypatch):
    conn, cur = _conn()
    monkeypatch.setattr(neon_insert, "_get_conn", lambda: conn)
    at = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)
    neon_insert.mark_ready_for_followup(1121, "/rec/x.json", "automations", "SUFFIX", now=at)
    sql, params = cur.execute.call_args.args
    assert "COALESCE(metadata" in sql and "|| jsonb_build_object" in sql and "WHERE id = %s" in sql
    assert params[0] == "ready_for_followup" and params[1] == "2026-09-30T10:00:00+00:00" and params[3] == 1121
    assert json.loads(params[2]) == {"transcript_path": "/rec/x.json", "triggered_by": "automations", "prompt_suffix": "SUFFIX"}
    conn.close.assert_called_once()


def test_mark_ready_omits_empty_suffix_and_raises_for_a_missing_row(monkeypatch):
    conn, cur = _conn()
    monkeypatch.setattr(neon_insert, "_get_conn", lambda: conn)
    neon_insert.mark_ready_for_followup(1, "/x.json", "watcher")
    assert "prompt_suffix" not in json.loads(cur.execute.call_args.args[1][2])
    conn, cur = _conn(rowcount=0)
    monkeypatch.setattr(neon_insert, "_get_conn", lambda: conn)
    with pytest.raises(RuntimeError, match="not found"):
        neon_insert.mark_ready_for_followup(2, "/x.json", "watcher")
