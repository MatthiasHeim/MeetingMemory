"""Tests for the watcher's fail-fast Claude-startup monitor.

P1 fix (docs/../meeting-pipeline-investigation-2026-08-10.md §6): the
Telegram ping in _notify_telegram_meeting_captured fires the moment the
fire-and-forget `claude -p /meeting-actions` trigger STARTS ("...running…"),
implying success. A session that dies on its first line (expired OAuth, org
spend limit, session limit, untrusted workspace) left that ping standing as
the only — and misleading — signal; three meetings sat with zero insights
for days before anyone noticed (07.08 incident). _monitor_claude_startup
waits up to CLAUDE_STARTUP_FAILFAST_SEC for the child to exit and, if it
does, checks the log tail against FATAL_STARTUP_SIGNATURES before alerting.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402


class _StubLogger:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))

    def error(self, msg):
        self.lines.append(("error", msg))

    def debug(self, msg):
        self.lines.append(("debug", msg))


class _FakeProc:
    """Stand-in for subprocess.Popen. `still_alive` controls whether
    .wait() raises TimeoutExpired (still running past the window) or
    returns immediately (already exited) — the two branches the monitor
    has to tell apart without ever actually sleeping 90s in a test."""

    def __init__(self, still_alive: bool, returncode: int = 1):
        self.still_alive = still_alive
        self.pid = 9999
        self.returncode = None if still_alive else returncode

    def wait(self, timeout=None):
        if self.still_alive:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
        return self.returncode


def _watcher():
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    return w


def _run_monitor(w, monkeypatch, tmp_path, log_text, still_alive=False):
    """Run _monitor_claude_startup against a fake proc + a real log file on
    disk; return whatever args were passed to the mocked Telegram sender
    (empty dict if nothing was sent)."""
    log_file = tmp_path / "claude-test.log"
    log_file.write_text(log_text)
    monkeypatch.setattr(w, "_telegram_notify_script",
                         lambda: str(tmp_path / "telegram_notify.py"))
    sent: dict = {}

    def fake_run(args, **kwargs):
        sent["args"] = args
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(tw.subprocess, "run", fake_run)
    proc = _FakeProc(still_alive=still_alive)
    w._monitor_claude_startup(proc, log_file, source_id=778)
    return sent


# ── SIDE A: every known fatal-startup signature fires an alert ────────

FATAL_LOG_SAMPLES = {
    "OAuth session expired": "Error: OAuth session expired, please /login\n",
    "spend limit": "You've hit your org's monthly spend limit\n",
    "session limit": "You've hit your session limit for this account\n",
    "not been trusted": "this workspace has not been trusted\n",
    "Failed to authenticate": "Failed to authenticate with the API\n",
    # Real 15-byte body observed overnight 2026-08-11 (three reconciler
    # re-triggers, no other content in the log at all) — the failure that
    # motivated adding this signature.
    "Execution error": "Execution error\n",
}


def test_each_fatal_signature_fires_an_alert(monkeypatch, tmp_path):
    for signature, log_text in FATAL_LOG_SAMPLES.items():
        w = _watcher()
        sent = _run_monitor(w, monkeypatch, tmp_path, log_text)
        assert sent, f"no Telegram alert sent for signature {signature!r}"
        msg = sent["args"][-1]
        assert signature in msg, (signature, msg)
        assert "778" in msg  # source_id named in the alert
        assert any(lvl == "error" for lvl, _ in w.logger.lines), signature


# ── SIDE B: a healthy log — or a still-running session — never fires ──


def test_still_running_after_window_does_not_fire(monkeypatch, tmp_path):
    """The common case: the session is doing real work past the fail-fast
    window. Must not alert, and must not block on the (mocked) 90s wait."""
    w = _watcher()
    sent = _run_monitor(
        w, monkeypatch, tmp_path,
        log_text="",  # nothing written yet — session still starting up
        still_alive=True,
    )
    assert sent == {}
    assert not any(lvl == "error" for lvl, _ in w.logger.lines)


def test_early_exit_without_fatal_signature_does_not_fire(monkeypatch, tmp_path):
    """Died within the window, but the log doesn't match any known fatal
    signature — e.g. a real (if unusually fast) successful run. Silence,
    not a false alarm."""
    w = _watcher()
    sent = _run_monitor(
        w, monkeypatch, tmp_path,
        log_text="Reading .claude/commands/meeting-actions.md...\nDone.\n",
        still_alive=False,
    )
    assert sent == {}
    assert not any(lvl == "error" for lvl, _ in w.logger.lines)


def test_waits_up_to_the_configured_window(monkeypatch, tmp_path):
    """The monitor must ask proc.wait() for exactly CLAUDE_STARTUP_FAILFAST_SEC,
    not some other value — a regression here would silently widen or shrink
    the fail-fast window."""
    w = _watcher()
    log_file = tmp_path / "claude-test.log"
    log_file.write_text("")
    seen = {}

    class _Proc:
        pid = 1
        returncode = None

        def wait(self, timeout=None):
            seen["timeout"] = timeout
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)

    w._monitor_claude_startup(_Proc(), log_file, source_id=1)
    assert seen["timeout"] == tw.CLAUDE_STARTUP_FAILFAST_SEC


# ── Exception safety: a monitoring bug must never propagate ───────────


def test_missing_log_file_is_handled_without_raising(monkeypatch, tmp_path):
    w = _watcher()
    monkeypatch.setattr(w, "_telegram_notify_script", lambda: None)
    proc = _FakeProc(still_alive=False)
    missing_log = tmp_path / "does-not-exist.log"
    w._monitor_claude_startup(proc, missing_log, source_id=1)  # must not raise
    assert any(lvl == "warning" for lvl, _ in w.logger.lines)


def test_notify_script_missing_does_not_raise(monkeypatch, tmp_path):
    """No telegram_notify.py on disk -> warn and return, never raise, even
    though a fatal signature WAS matched."""
    w = _watcher()
    monkeypatch.setattr(w, "_telegram_notify_script", lambda: None)
    log_file = tmp_path / "claude-test.log"
    log_file.write_text("spend limit hit\n")
    proc = _FakeProc(still_alive=False)
    w._monitor_claude_startup(proc, log_file, source_id=1)  # must not raise
