"""A local "start if idle" request starts a recording once and never touches a running one."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
pytest.importorskip("rumps")
pytest.importorskip("soundfile")

import meeting_recorder
from meeting_recorder import START_REQUEST_NAME, MeetingRecorderApp, consume_start_request


def _request(control: Path, age: float = 0.0) -> Path:
    request = control / START_REQUEST_NAME
    request.write_text("")
    stamp = time.time() - age
    os.utime(request, (stamp, stamp))
    return request


def _app(control: Path, recording: bool):
    app = MeetingRecorderApp.__new__(MeetingRecorderApp)
    app.control_dir = control
    app.recorder = SimpleNamespace(is_recording=recording)
    app._start_stop_item = SimpleNamespace(title="Start Recording")
    started = []
    app._start_recording = started.append
    return app, started


def test_no_request_means_no_start(tmp_path):
    assert consume_start_request(tmp_path) is False
    app, started = _app(tmp_path, recording=False)
    app._poll_start_request(None)
    assert started == []


def test_fresh_request_starts_once_when_idle(tmp_path):
    request = _request(tmp_path)
    app, started = _app(tmp_path, recording=False)
    app._poll_start_request(None)
    app._poll_start_request(None)
    assert started == [app._start_stop_item]
    assert not request.exists()


def test_request_while_recording_changes_nothing(tmp_path):
    request = _request(tmp_path)
    app, started = _app(tmp_path, recording=True)
    app._poll_start_request(None)
    assert started == []
    assert not request.exists()


def test_stale_request_is_removed_without_starting(tmp_path):
    request = _request(tmp_path, age=meeting_recorder.START_REQUEST_MAX_AGE_SECONDS + 60)
    app, started = _app(tmp_path, recording=False)
    app._poll_start_request(None)
    assert started == []
    assert not request.exists()


def test_launch_discards_a_leftover_request(tmp_path, monkeypatch):
    recordings = tmp_path / "Recordings"
    recordings.mkdir()
    request = _request(tmp_path / "control") if (tmp_path / "control").mkdir() is None else None
    timers = []

    class Timer:
        def __init__(self, callback, interval):
            timers.append((callback, interval))

        def start(self):
            timers.append("started")

    monkeypatch.setattr(meeting_recorder.rumps, "Timer", Timer)
    app = MeetingRecorderApp.__new__(MeetingRecorderApp)
    app.recordings_dir = recordings
    app._watch_start_requests()
    assert not request.exists()
    assert timers[-1] == "started"
