"""Tests for the watcher's stale-code detector.

Operational regression guard for the failure mode behind the Stefan-mislabel
incident: the long-running watcher daemon held pre-fix bytecode in memory
for 23 days while the fixes sat on disk untouched. The detector logs a
WARNING the next time a transcript is processed after any tools/*.py is
modified, giving the operator a single clear signal to restart the daemon.
"""

from __future__ import annotations

import logging
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_reported():
    """Each test gets a clean dedup-set; reset state after to not leak."""
    tw._STALE_FILES_REPORTED.clear()
    yield
    tw._STALE_FILES_REPORTED.clear()


def test_no_warning_when_files_predate_process(caplog):
    """If no tools/*.py is newer than process start, no warning fires."""
    # Pretend the process just started — anything older than "now+1s" is
    # not-stale.
    tw._PROCESS_START_WALL = time.time() + 1
    with caplog.at_level(logging.WARNING, logger="test"):
        logger = logging.getLogger("test")
        tw._warn_if_code_is_stale(logger)
    assert not any("Stale watcher" in r.message for r in caplog.records)


def test_warns_once_per_file(caplog, tmp_path, monkeypatch):
    """A newer tools/*.py file triggers the warning exactly once per file —
    subsequent calls don't re-warn the same file (anti-spam)."""
    # Point the helper at a temp tools-like dir so the test is hermetic.
    fake_tools = tmp_path / "tools"
    fake_tools.mkdir()
    (fake_tools / "transcribe_watcher.py").write_text("# placeholder\n")
    (fake_tools / "speaker_reconcile.py").write_text("# placeholder\n")
    # Mark process as having started in the deep past.
    tw._PROCESS_START_WALL = 1.0
    # Redirect _tool_file_mtimes to inspect our fake dir.
    monkeypatch.setattr(tw, "_tool_file_mtimes", lambda: {
        p.name: p.stat().st_mtime for p in fake_tools.glob("*.py")
    })

    logger = logging.getLogger("test_stale")
    with caplog.at_level(logging.WARNING, logger="test_stale"):
        tw._warn_if_code_is_stale(logger)
        first_count = len([r for r in caplog.records if "Stale watcher" in r.message])
        tw._warn_if_code_is_stale(logger)
        second_count = len([r for r in caplog.records if "Stale watcher" in r.message])

    assert first_count == 1, "first call must warn about the stale files"
    assert second_count == 1, "second call must NOT re-warn (dedup)"


def test_git_sha_returns_string_or_none():
    """_git_sha must not raise; it returns a short SHA or None on failure."""
    out = tw._git_sha()
    assert out is None or (isinstance(out, str) and len(out) >= 4)


def test_remote_topology_diarization_fusion_default_off():
    topology = SimpleNamespace(topology=tw.TOPOLOGY_MULTI_SOURCE_GENUINE)
    assert tw.TranscribeWatcher._should_run_diarization_prior(
        topology, channel_fusion=False
    ) is False
    assert tw.TranscribeWatcher._should_run_diarization_prior(
        topology, channel_fusion=True
    ) is True


def test_single_source_runs_diarization_prior_by_default():
    topology = SimpleNamespace(topology=tw.TOPOLOGY_SINGLE_SOURCE)
    assert tw.TranscribeWatcher._should_run_diarization_prior(
        topology, channel_fusion=False
    ) is True
