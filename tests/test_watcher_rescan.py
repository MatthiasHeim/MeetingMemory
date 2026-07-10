"""Tests for the F3 periodic rescan (docs/SPEC-error-path-escalation-2026-07-10.md
RC3): _process_existing_files used to run only at watcher startup, so a
meeting that failed during a Gemini outage (and, per F1's junk guard, left
no JSON) sat unprocessed until someone manually restarted the watcher.
_rescan_unprocessed_wavs re-runs that logic on a timer, with a per-WAV
attempt cap + min spacing (persisted to a state-file sidecar so a watcher
restart doesn't reset the count) and alert de-duplication (only the first
failure and the final give-up fire Telegram).
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


def _bare_watcher(tmp_path, state=None):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    w.recordings_dir = tmp_path
    w.transcripts_dir = tmp_path
    w.processing_mode = "gemini"
    w.queue = tw.TranscriptionQueue(w.logger)
    w._rescan_state = state if state is not None else {}
    w._rescan_state_path = tmp_path / ".rescan_state.json"
    w._giveups: list[tuple[Path, int]] = []
    w._notify_telegram_giveup = (
        lambda audio_file, attempts: w._giveups.append((audio_file, attempts))
    )
    return w


def _make_wav(tmp_path: Path, name: str = "rec.wav") -> Path:
    wav = tmp_path / name
    wav.write_bytes(b"x" * (tw.MIN_RECORDING_BYTES + 1))
    return wav


# ── attempt cap + spacing ──────────────────────────────────────────────


def test_rescan_requeues_unprocessed_wav():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        wav = _make_wav(tmp_path)
        w = _bare_watcher(tmp_path)
        w._rescan_unprocessed_wavs(now=1_000_000.0)
        assert w.queue.get() == wav
        assert w._rescan_state["rec"]["attempts"] == 1
        assert w._rescan_state["rec"]["last_attempt_ts"] == 1_000_000.0


def test_rescan_skips_wav_that_already_has_json():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _make_wav(tmp_path)
        (tmp_path / "rec.json").write_text("{}")
        w = _bare_watcher(tmp_path)
        w._rescan_unprocessed_wavs(now=1_000_000.0)
        assert w.queue.is_empty()
        assert "rec" not in w._rescan_state


def test_rescan_skips_corrupt_too_small_file():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        wav = tmp_path / "tiny.wav"
        wav.write_bytes(b"x" * 10)  # far below MIN_RECORDING_BYTES
        w = _bare_watcher(tmp_path)
        w._rescan_unprocessed_wavs(now=1_000_000.0)
        assert w.queue.is_empty()


def test_rescan_does_not_requeue_before_spacing_elapses():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _make_wav(tmp_path)
        w = _bare_watcher(tmp_path)
        t0 = 1_000_000.0
        w._rescan_unprocessed_wavs(now=t0)
        assert w.queue.get() is not None  # drain the first queue entry

        w._rescan_unprocessed_wavs(now=t0 + 60)  # only 1 min later
        assert w.queue.is_empty()
        assert w._rescan_state["rec"]["attempts"] == 1


def test_rescan_requeues_again_after_spacing_elapses():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        wav = _make_wav(tmp_path)
        w = _bare_watcher(tmp_path)
        t0 = 1_000_000.0
        w._rescan_unprocessed_wavs(now=t0)
        w.queue.get()

        t1 = t0 + tw.RESCAN_MIN_SPACING_SEC + 1
        w._rescan_unprocessed_wavs(now=t1)
        assert w.queue.get() == wav
        assert w._rescan_state["rec"]["attempts"] == 2
        assert w._rescan_state["rec"]["last_attempt_ts"] == t1


def test_rescan_gives_up_after_max_attempts_and_stops_requeueing():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _make_wav(tmp_path)
        w = _bare_watcher(tmp_path)
        t = 1_000_000.0
        for _ in range(tw.RESCAN_MAX_ATTEMPTS):
            w._rescan_unprocessed_wavs(now=t)
            w.queue.get()  # drain -- still no JSON appears (simulates continued failure)
            t += tw.RESCAN_MIN_SPACING_SEC + 1

        assert w._rescan_state["rec"]["attempts"] == tw.RESCAN_MAX_ATTEMPTS
        assert w._giveups == []  # not yet -- give-up fires on the NEXT tick

        w._rescan_unprocessed_wavs(now=t)
        assert w.queue.is_empty()
        assert len(w._giveups) == 1
        assert w._rescan_state["rec"]["gave_up"] is True

        # Further ticks must not requeue or re-alert.
        t += tw.RESCAN_MIN_SPACING_SEC + 1
        w._rescan_unprocessed_wavs(now=t)
        assert w.queue.is_empty()
        assert len(w._giveups) == 1


def test_rescan_stops_once_json_appears():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _make_wav(tmp_path)
        w = _bare_watcher(tmp_path)
        t0 = 1_000_000.0
        w._rescan_unprocessed_wavs(now=t0)
        w.queue.get()

        (tmp_path / "rec.json").write_text("{}")  # processing succeeded meanwhile
        w._rescan_unprocessed_wavs(now=t0 + tw.RESCAN_MIN_SPACING_SEC + 1)
        assert w.queue.is_empty()


# ── state survives a simulated restart ────────────────────────────────


def test_rescan_state_survives_restart():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        _make_wav(tmp_path)

        w1 = _bare_watcher(tmp_path)
        t0 = 1_000_000.0
        w1._rescan_unprocessed_wavs(now=t0)
        w1.queue.get()
        assert w1._rescan_state["rec"]["attempts"] == 1

        # Simulate a watcher restart: fresh instance, state reloaded from disk.
        w2 = _bare_watcher(tmp_path, state=tw._load_rescan_state(tmp_path / ".rescan_state.json"))
        assert w2._rescan_state["rec"]["attempts"] == 1

        t1 = t0 + tw.RESCAN_MIN_SPACING_SEC + 1
        w2._rescan_unprocessed_wavs(now=t1)
        assert w2.queue.get() is not None
        assert w2._rescan_state["rec"]["attempts"] == 2  # continued, not reset


def test_load_rescan_state_tolerates_missing_file(tmp_path):
    assert tw._load_rescan_state(tmp_path / "does_not_exist.json") == {}


def test_load_rescan_state_tolerates_corrupt_file(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not valid json")
    assert tw._load_rescan_state(p) == {}


def test_save_and_load_rescan_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    state = {"rec": {"attempts": 2, "last_attempt_ts": 123.0, "gave_up": False}}
    tw._save_rescan_state(p, state)
    assert tw._load_rescan_state(p) == state


# ── alert suppression for rescan-triggered retries ────────────────────


def test_notify_telegram_failure_suppressed_for_rescan_retry(monkeypatch, tmp_path):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    w._rescan_state = {"rec": {"attempts": 2, "last_attempt_ts": 0.0, "gave_up": False}}

    def _must_not_be_reached():
        raise AssertionError("must not be reached -- suppression should short-circuit first")

    monkeypatch.setattr(
        tw.TranscribeWatcher, "_telegram_notify_script", staticmethod(_must_not_be_reached),
    )
    # If suppression didn't fire, this would raise via the monkeypatched
    # notify-script resolver above.
    w._notify_telegram_failure(Path("/tmp/rec.wav"), "boom")
    assert any("suppressing" in msg.lower() for level, msg in w.logger.lines if level == "info")


def test_notify_telegram_failure_not_suppressed_for_first_attempt(monkeypatch):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    w._rescan_state = {}  # no rescan bookkeeping yet -- this is the original live attempt

    monkeypatch.setattr(tw.TranscribeWatcher, "_telegram_notify_script", staticmethod(lambda: None))
    w._notify_telegram_failure(Path("/tmp/rec.wav"), "boom")
    assert any("telegram_notify.py not found" in msg for level, msg in w.logger.lines if level == "warning")
