"""Tests for the watcher's Claude trigger — specifically the CLAUDE_CONFIG_DIR
quota isolation added after the 2026-07-10 incident.

The fire-and-forget `claude -p /meeting-actions` sessions used to share one
Claude quota pool with interactive usage, which starved every session into a
"session limit" death and silently dropped all downstream actions.
_trigger_claude now forwards a dedicated CLAUDE_CONFIG_DIR (its own quota pool)
to the child session, resolved from config or an inherited env var.
"""

from __future__ import annotations

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
    pid = 4242

    def poll(self):
        return None


def _watcher(tmp_path, claude_cfg):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    w.config = {
        "claude_trigger": claude_cfg,
        "paths": {"logs": str(tmp_path / "logs")},
    }
    return w


def _run_trigger(w, monkeypatch, tmp_path, env_config_dir=None):
    """Fire _trigger_claude with subprocess.Popen mocked; return the captured env."""
    captured = {}

    def fake_popen(args, cwd=None, env=None, stdout=None, stderr=None):
        captured["args"] = args
        captured["env"] = env
        captured["cwd"] = cwd
        return _FakeProc()

    monkeypatch.setattr(tw.subprocess, "Popen", fake_popen)
    if env_config_dir is None:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", env_config_dir)

    w._trigger_claude(tmp_path / "2026-07-10_16-34-26.json", source_id=466)
    return captured


def test_trigger_disabled_is_noop(tmp_path, monkeypatch):
    w = _watcher(tmp_path, {"enabled": False})
    called = {"popen": False}
    monkeypatch.setattr(tw.subprocess, "Popen",
                        lambda *a, **k: called.__setitem__("popen", True))
    w._trigger_claude(tmp_path / "x.json", source_id=1)
    assert called["popen"] is False


def test_config_dir_sets_claude_config_dir_env(tmp_path, monkeypatch):
    cfg = {"enabled": True, "config_dir": str(tmp_path / "auto-cfg")}
    w = _watcher(tmp_path, cfg)
    captured = _run_trigger(w, monkeypatch, tmp_path)
    assert captured["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "auto-cfg")
    assert any("quota-isolated" in msg for lvl, msg in w.logger.lines)


def test_env_claude_config_dir_is_honored_when_config_absent(tmp_path, monkeypatch):
    w = _watcher(tmp_path, {"enabled": True})
    captured = _run_trigger(w, monkeypatch, tmp_path,
                            env_config_dir="/Users/Matthias/.claude-automation")
    assert captured["env"]["CLAUDE_CONFIG_DIR"] == "/Users/Matthias/.claude-automation"


def test_config_dir_takes_precedence_over_env(tmp_path, monkeypatch):
    cfg = {"enabled": True, "config_dir": str(tmp_path / "from-config")}
    w = _watcher(tmp_path, cfg)
    captured = _run_trigger(w, monkeypatch, tmp_path,
                            env_config_dir="/from/env")
    assert captured["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "from-config")


def test_no_isolation_warns(tmp_path, monkeypatch):
    w = _watcher(tmp_path, {"enabled": True})
    captured = _run_trigger(w, monkeypatch, tmp_path, env_config_dir=None)
    # No CLAUDE_CONFIG_DIR forced into the child env beyond what os.environ had.
    assert "CLAUDE_CONFIG_DIR" not in captured["env"]
    assert any(
        lvl == "warning" and "shares the default Claude quota pool" in msg
        for lvl, msg in w.logger.lines
    )


def test_source_id_passed_in_prompt(tmp_path, monkeypatch):
    w = _watcher(tmp_path, {"enabled": True, "config_dir": str(tmp_path / "c")})
    captured = _run_trigger(w, monkeypatch, tmp_path)
    prompt = captured["args"][2]  # [claude_path, "-p", prompt, ...]
    assert "--source-id 466" in prompt
