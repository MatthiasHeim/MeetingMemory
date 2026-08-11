"""Tests for TranscribeWatcher._convert_for_gemini_routed — the branch point
that picks the ducked pre-mix (bleeding mic) vs. the legacy equal-weight mix
(clean/headphone recordings, or no channel VAD at all).

This is a routing test only: it spies on which module-level converter
function got called with which arguments, not on the audio bytes each
produces (that's covered by tests/test_audio_converter_ducked.py). The
critical invariant under test: an admissible channel_separation (or no
channel_vad) must take the untouched legacy path byte-for-byte; only a
channel_vad present AND flagged inadmissible may take the ducked path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

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


@dataclass
class _FakeChannelVAD:
    """Stand-in for channel_vad.ChannelVAD — only the bits the routing
    decision and its logging touch."""
    segments: list = field(default_factory=list)
    _bleed_rate: Optional[float] = 0.6

    def host_bleed_rate(self):
        return self._bleed_rate


def _watcher(config=None):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    w.transcripts_dir = Path("/tmp/fake-transcripts")
    w.config = config or {}
    return w


def _spy(monkeypatch, name):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return Path(f"/tmp/fake-transcripts/{name}-result.mp3")

    monkeypatch.setattr(tw, name, fake)
    return calls


# ── admissible / no-VAD cases must take the untouched legacy path ──────


def test_no_channel_vad_takes_legacy_path(monkeypatch):
    w = _watcher()
    legacy_calls = _spy(monkeypatch, "convert_for_gemini")
    ducked_calls = _spy(monkeypatch, "convert_for_gemini_ducked")

    result = w._convert_for_gemini_routed(Path("rec.wav"), None, False)

    assert legacy_calls, "no channel_vad must fall back to the legacy mix"
    assert not ducked_calls
    assert result == Path("/tmp/fake-transcripts/convert_for_gemini-result.mp3")


def test_admissible_channel_vad_takes_legacy_path(monkeypatch):
    """Clean/headphone recording: channel_vad exists but the mic is not
    bleeding — must behave exactly as before this change."""
    w = _watcher()
    legacy_calls = _spy(monkeypatch, "convert_for_gemini")
    ducked_calls = _spy(monkeypatch, "convert_for_gemini_ducked")
    vad = _FakeChannelVAD()

    result = w._convert_for_gemini_routed(Path("rec.wav"), vad, True)

    assert legacy_calls
    assert not ducked_calls
    args, kwargs = legacy_calls[0]
    assert args[0] == Path("rec.wav")
    assert kwargs.get("output_dir") == w.transcripts_dir


# ── inadmissible + channel_vad present is the ONLY ducked-path trigger ──


def test_inadmissible_channel_vad_takes_ducked_path(monkeypatch):
    w = _watcher()
    legacy_calls = _spy(monkeypatch, "convert_for_gemini")
    ducked_calls = _spy(monkeypatch, "convert_for_gemini_ducked")
    vad = _FakeChannelVAD()

    result = w._convert_for_gemini_routed(Path("rec.wav"), vad, False)

    assert ducked_calls, "bleeding mic must route to the ducked pre-mix"
    assert not legacy_calls
    args, kwargs = ducked_calls[0]
    assert args[0] == Path("rec.wav")
    assert args[1] is vad
    assert kwargs.get("output_dir") == w.transcripts_dir


def test_ducked_path_uses_configured_duck_db(monkeypatch):
    w = _watcher(config={"audio_ducking": {"duck_db": 24.0}})
    ducked_calls = _spy(monkeypatch, "convert_for_gemini_ducked")
    _spy(monkeypatch, "convert_for_gemini")
    vad = _FakeChannelVAD()

    w._convert_for_gemini_routed(Path("rec.wav"), vad, False)

    _, kwargs = ducked_calls[0]
    assert kwargs.get("duck_db") == 24.0


def test_ducked_path_defaults_duck_db_when_unconfigured(monkeypatch):
    w = _watcher(config={})
    ducked_calls = _spy(monkeypatch, "convert_for_gemini_ducked")
    _spy(monkeypatch, "convert_for_gemini")
    vad = _FakeChannelVAD()

    w._convert_for_gemini_routed(Path("rec.wav"), vad, False)

    _, kwargs = ducked_calls[0]
    assert kwargs.get("duck_db") == 18.0


def test_ducked_path_logs_bleed_rate_and_db_at_info(monkeypatch):
    w = _watcher()
    _spy(monkeypatch, "convert_for_gemini_ducked")
    _spy(monkeypatch, "convert_for_gemini")
    vad = _FakeChannelVAD(_bleed_rate=0.42)

    w._convert_for_gemini_routed(Path("rec.wav"), vad, False)

    info_lines = [msg for level, msg in w.logger.lines if level == "info"]
    assert any("0.42" in m and "18" in m for m in info_lines), info_lines
