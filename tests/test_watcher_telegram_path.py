"""Tests for the watcher's telegram_notify.py path resolver.

Regression guard: the path was hardcoded to ~/.claude/scripts, but the script
actually lives in the Brain repo. The wrong path silently disabled ALL pings
(captured + failure alerts) via an os.path.exists guard, so a failed recording
dropped with no notification at all (observed 2026-06-23: a 102-byte empty
recording failed and the user was never told). The resolver now checks several
known locations and warns (not debug) when none exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402


def test_resolver_prefers_env_override(monkeypatch, tmp_path):
    script = tmp_path / "telegram_notify.py"
    script.write_text("# stub")
    monkeypatch.setenv("TELEGRAM_NOTIFY_SCRIPT", str(script))
    assert tw.TranscribeWatcher._telegram_notify_script() == str(script)


def test_resolver_finds_brain_repo_path(monkeypatch):
    """With no env override, the Brain-repo location is used when present."""
    monkeypatch.delenv("TELEGRAM_NOTIFY_SCRIPT", raising=False)
    brain = os.path.expanduser("~/Repos/Brain/.claude/scripts/telegram_notify.py")
    monkeypatch.setattr(os.path, "exists", lambda p: p == brain)
    assert tw.TranscribeWatcher._telegram_notify_script() == brain


def test_resolver_returns_none_when_nothing_exists(monkeypatch):
    monkeypatch.delenv("TELEGRAM_NOTIFY_SCRIPT", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    assert tw.TranscribeWatcher._telegram_notify_script() is None
