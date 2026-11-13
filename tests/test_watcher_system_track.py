"""Watcher wiring for the system-track attribution stage."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402
from gemini_processor import GeminiResult  # noqa: E402

TEXT = "Das ist ein ausreichend langer Satz fuer die Messung hier."


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, level):
        return lambda msg: self.lines.append((level, msg))


class _Vad:
    """Bleeding-mic VAD stub: only segments + duration (mic is inadmissible)."""
    def __init__(self, segments):
        self.segments = segments
        self.duration_sec = 240.0

    def separation_report(self):
        return {"admissible": False, "reason": "mic_hears_remote"}


def _watcher():
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _Log()
    return w


def _inverted():
    lines, segs = [], []
    for i in range(12):
        t = i * 20
        remote = i % 2 == 0
        if remote:
            segs.append((t, t + 19, "both"))   # bleeding: mic also active
        else:
            segs.append((t, t + 19, "host"))
        label = "Matthias" if remote else "Speaker 1"   # inverted
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {label}: {TEXT}")
    return "\n".join(lines), segs


def test_stage_runs_and_fixes_when_mic_is_inadmissible():
    text, segs = _inverted()
    result = GeminiResult(text, "de")
    w = _watcher()
    w._system_track_attribution_inplace(result, _Vad(segs), None, 240.0)
    assert result.system_track_attribution["ran"] is True
    assert len(result.system_track_attribution["swaps"]) == 1
    assert result.transcript.startswith("[00:00] Speaker 1:")
    assert result.parsed_response["_meta"]["system_track_attribution"]["ran"] is True


def test_stage_skips_without_channel_vad_and_leaves_text():
    text, _ = _inverted()
    result = GeminiResult(text, "de")
    _watcher()._system_track_attribution_inplace(result, None, None, 240.0)
    assert result.transcript == text
    assert result.system_track_attribution["ran"] is False
    assert result.system_track_attribution["reason"].startswith("no_three_channel")


def test_stage_skips_on_remote_host_namesake():
    text, segs = _inverted()
    result = GeminiResult(text, "de")
    cal = {"participant_details": [{"name": "Matthias Müller", "role": "participant"}]}
    _watcher()._system_track_attribution_inplace(result, _Vad(segs), cal, 240.0)
    assert result.transcript == text
    assert result.system_track_attribution["reason"] == "host_namesake_remote_attendee"


def test_stage_never_raises(monkeypatch):
    text, segs = _inverted()
    result = GeminiResult(text, "de")
    monkeypatch.setattr(tw, "_system_track_run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    _watcher()._system_track_attribution_inplace(result, _Vad(segs), None, 240.0)
    assert result.transcript == text


def test_flow_uses_system_track_stage_not_mic_verify():
    src = inspect.getsource(tw.TranscribeWatcher._process_with_gemini)
    assert "_system_track_attribution_inplace(" in src
    assert "_verify_speakers_inplace(" not in src
    # the stage must not be gated on channel admissibility
    stage = src.split("_system_track_attribution_inplace(")[1].split(")")[0]
    assert "channel_admissible" not in stage
