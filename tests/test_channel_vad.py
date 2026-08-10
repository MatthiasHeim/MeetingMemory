"""Tests for channel_vad — channel-based host/remote voice activity detection.

Synthetic 3-channel WAVs with known tone placement exercise the full
ffmpeg-decode → band-pass → RMS → adaptive-threshold → segments pipeline.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from channel_vad import (  # noqa: E402
    FFMPEG_PATH,
    channel_separation_report,
    compute_channel_vad,
    host_bleed_rate,
    render_map_text,
    slice_segments,
)
from audio_converter import (  # noqa: E402
    TOPOLOGY_MULTI_SOURCE_GENUINE,
    TOPOLOGY_SINGLE_SOURCE,
    classify_source_topology,
)
from speaker_verify import verify  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    not os.path.exists(FFMPEG_PATH), reason="ffmpeg not installed"
)

SR = 8000


def _write_wav(path: Path, channels: list[list[tuple[float, float]]],
               duration_sec: float, sr: int = SR, freq: float = 1000.0,
               amplitude: float = 0.3) -> Path:
    """Write an n-channel WAV with `freq` tone bursts at given spans.

    `channels` is a list (one entry per channel) of (t0, t1) spans during
    which that channel carries the tone; silence elsewhere.
    """
    n_samples = int(duration_sec * sr)
    n_ch = len(channels)
    frames = bytearray()
    for i in range(n_samples):
        t = i / sr
        for spans in channels:
            on = any(t0 <= t < t1 for t0, t1 in spans)
            v = amplitude * math.sin(2 * math.pi * freq * t) if on else 0.0
            frames += struct.pack('<h', int(v * 32767))
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return path


def _label_at(segments, t: float) -> str:
    for t0, t1, label in segments:
        if t0 <= t < t1:
            return label
    return ''


# ── full pipeline on synthetic 3-channel audio ────────────────────────


@needs_ffmpeg
def test_host_remote_both_segments(tmp_path):
    """Tone on ch0 → 'host'; tone on ch1/ch2 → 'remote'; both → 'both'."""
    wav = _write_wav(
        tmp_path / "synth.wav",
        channels=[
            [(0, 8), (20, 26)],    # ch0 = mic (host)
            [(10, 18)],            # ch1 = system L
            [(20, 26)],            # ch2 = system R
        ],
        duration_sec=30,
    )
    vad = compute_channel_vad(wav)
    assert vad is not None
    # Mid-span probes (segment edges are quantised/smoothed by ±~0.75 s).
    assert _label_at(vad.segments, 4.0) == 'host'
    assert _label_at(vad.segments, 14.0) == 'remote'
    assert _label_at(vad.segments, 23.0) == 'both'
    assert _label_at(vad.segments, 28.5) == ''  # trailing silence


@needs_ffmpeg
def test_host_share_helper(tmp_path):
    wav = _write_wav(
        tmp_path / "synth.wav",
        channels=[[(0, 8)], [(10, 18)], []],
        duration_sec=20,
    )
    vad = compute_channel_vad(wav)
    assert vad is not None
    assert vad.host_share(1, 7) > 0.9
    assert vad.host_share(11, 17) < 0.1
    # No speech at all in the trailing second → None.
    assert vad.host_share(19.0, 20.0) is None
    s = vad.shares(0, 20)
    assert s["host_only"] == pytest.approx(8.0, abs=1.5)
    assert s["remote_only"] == pytest.approx(8.0, abs=1.5)
    assert s["both"] == pytest.approx(0.0, abs=1.0)


@needs_ffmpeg
def test_map_text_contains_turn_spans(tmp_path):
    wav = _write_wav(
        tmp_path / "synth.wav",
        channels=[[(0, 8)], [(10, 18)], []],
        duration_sec=20,
    )
    vad = compute_channel_vad(wav)
    text = vad.map_text()
    lines = text.splitlines()
    assert any(line.endswith(" host") for line in lines)
    assert any(line.endswith(" remote") for line in lines)
    # Span format MM:SS-MM:SS.
    assert all(
        len(line.split()) == 2 and "-" in line.split()[0] for line in lines
    )


@needs_ffmpeg
def test_mono_wav_returns_none(tmp_path):
    """Mic-only fallback recordings have no channel separation — skip."""
    wav = _write_wav(tmp_path / "mono.wav", channels=[[(0, 5)]],
                     duration_sec=6)
    assert compute_channel_vad(wav) is None


@needs_ffmpeg
def test_stereo_wav_returns_none(tmp_path):
    """Plain stereo isn't the hybrid layout either."""
    wav = _write_wav(tmp_path / "stereo.wav",
                     channels=[[(0, 5)], [(0, 5)]], duration_sec=6)
    assert compute_channel_vad(wav) is None


@needs_ffmpeg
def test_silent_system_three_channel_returns_none(tmp_path):
    """Regression: 3ch in-room capture with only ch0 active is single-source,
    so it must not produce a host ground-truth map."""
    wav = _write_wav(
        tmp_path / "room_mic_only.wav",
        channels=[[(0, 5)], [], []],
        duration_sec=6,
    )
    assert compute_channel_vad(wav) is None


@needs_ffmpeg
def test_silent_system_three_channel_does_not_flip_to_matthias(tmp_path):
    wav = _write_wav(
        tmp_path / "room_mic_only.wav",
        channels=[[(0, 10)], [], []],
        duration_sec=12,
    )
    vad = compute_channel_vad(wav)
    assert vad is None
    gemini = {
        "transcript": (
            "[00:00] Speaker B: Das ist ein langer Beitrag aus dem Raum.\n"
            "[00:10] Speaker C: Ende.\n"
        ),
        "participants": [{"name": "Speaker B"}, {"name": "Matthias"}],
    }
    log = verify(gemini, vad)
    assert log["skipped_no_vad"] is True
    assert "Speaker B: Das ist ein langer Beitrag" in gemini["transcript"]
    assert "Matthias: Das ist ein langer Beitrag" not in gemini["transcript"]


@needs_ffmpeg
def test_missing_file_returns_none(tmp_path):
    assert compute_channel_vad(tmp_path / "nope.wav") is None


@needs_ffmpeg
def test_topology_classifier_mono(tmp_path):
    wav = _write_wav(tmp_path / "mono.wav", channels=[[(0, 5)]],
                     duration_sec=6)
    info = classify_source_topology(wav)
    assert info.total_channels == 1
    assert info.active_channels == [0]
    assert info.topology == TOPOLOGY_SINGLE_SOURCE


@needs_ffmpeg
def test_topology_classifier_silent_system_3ch(tmp_path):
    wav = _write_wav(tmp_path / "room.wav",
                     channels=[[(0, 5)], [], []], duration_sec=6)
    info = classify_source_topology(wav)
    assert info.total_channels == 3
    assert info.active_channels == [0]
    assert info.topology == TOPOLOGY_SINGLE_SOURCE


@needs_ffmpeg
def test_topology_classifier_genuine_multichannel(tmp_path):
    wav = _write_wav(tmp_path / "remote.wav",
                     channels=[[(0, 5)], [(2, 6)], []], duration_sec=7)
    info = classify_source_topology(wav)
    assert info.total_channels == 3
    assert set(info.active_channels) == {0, 1}
    assert info.topology == TOPOLOGY_MULTI_SOURCE_GENUINE


@needs_ffmpeg
def test_constant_level_channel_stays_inactive(tmp_path):
    """A channel with constant-level content (fan hum, line noise) must not
    register as speech: the adaptive threshold sits 10 dB above the
    20th-percentile floor, which IS the constant level."""
    wav = _write_wav(
        tmp_path / "hum.wav",
        channels=[
            [(0, 30)],   # ch0: constant tone for the whole file
            [(10, 18)],  # ch1: real remote burst
            [],
        ],
        duration_sec=30,
    )
    vad = compute_channel_vad(wav)
    assert vad is not None
    assert not bool(vad.host_active.any()), (
        "constant-level host channel must never trigger host activity"
    )
    assert _label_at(vad.segments, 14.0) == 'remote'


# ── pure helpers (no ffmpeg needed) ───────────────────────────────────


def test_slice_segments_clips_and_rebases():
    segments = [(0.0, 50.0, 'host'), (50.0, 130.0, 'remote'),
                (130.0, 200.0, 'both')]
    out = slice_segments(segments, 40.0, 140.0)
    assert out == [(0.0, 10.0, 'host'), (10.0, 90.0, 'remote'),
                   (90.0, 100.0, 'both')]
    # Fully outside the window → dropped.
    assert slice_segments(segments, 500.0, 600.0) == []
    assert slice_segments([], 0.0, 10.0) == []


def test_render_map_text_merges_short_spans():
    segments = [
        (0.0, 42.0, 'remote'),
        (42.0, 43.0, 'both'),      # 1 s blip — absorbed into prior span
        (43.0, 130.0, 'host'),
        (130.0, 131.0, 'host'),    # adjacent same label — merged
    ]
    text = render_map_text(segments)
    assert text == "00:00-00:43 remote\n00:43-02:11 host"


def test_render_map_text_short_leading_span_folds_forward():
    segments = [(0.0, 1.0, 'both'), (1.0, 60.0, 'host')]
    assert render_map_text(segments) == "00:00-01:00 host"


def test_render_map_text_preserves_short_exclusive_spans():
    """A 1.5 s remote interjection between host spans must SURVIVE: the
    prompt declares host spans exclusive ('only the host is speaking'), so
    absorbing the remote blip would instruct Gemini to attribute those
    words to the host — the exact error the map exists to prevent."""
    segments = [
        (0.0, 30.0, 'host'),
        (30.0, 31.5, 'remote'),
        (31.5, 60.0, 'host'),
    ]
    text = render_map_text(segments)
    assert "00:30-00:32 remote" in text  # 31.5 rounds up to :32
    lines = text.splitlines()
    assert [line.split()[-1] for line in lines] == ['host', 'remote', 'host']


def test_render_map_text_hour_format():
    segments = [(3600.0, 3725.0, 'host')]
    assert render_map_text(segments) == "01:00:00-01:02:05 host"


def test_render_map_text_empty():
    assert render_map_text([]) == ""


# ── bleed self-test: is the mic channel an oracle at all? (2026-08-07) ────


def test_host_bleed_rate_low_when_channels_are_isolated():
    """Headphones: the mic only overlaps the remote during real backchannel."""
    segments = [
        (0.0, 300.0, 'remote'),
        (300.0, 330.0, 'both'),     # 30 s of "mhm" over 330 s of remote speech
        (330.0, 600.0, 'host'),
    ]
    rate = host_bleed_rate(segments)
    assert rate is not None and abs(rate - 30.0 / 330.0) < 1e-9
    report = channel_separation_report(segments)
    assert report["admissible"] is True
    assert report["reason"] == "ok"


def test_host_bleed_rate_high_when_mic_hears_the_speakers():
    """Open speakers: the mic reads active through most of the remote's turns.

    This is source 767 (2026-08-07, measured 0.71): `host_share` then sits
    above the flip threshold for every turn regardless of who spoke.
    """
    segments = [
        (0.0, 100.0, 'remote'),
        (100.0, 400.0, 'both'),
        (400.0, 600.0, 'host'),
    ]
    rate = host_bleed_rate(segments)
    assert rate is not None and rate > 0.7
    report = channel_separation_report(segments)
    assert report["admissible"] is False
    assert report["reason"] == "mic_hears_remote"
    assert report["host_bleed_rate"] == 0.75


def test_bleed_rate_unmeasurable_without_enough_remote_speech():
    """Under a minute of remote speech cannot support the ratio — report it
    as unmeasured, but do NOT withdraw the oracle: the recordings this
    describes have almost no remote speech to misattribute, and the
    everyone-on-ch0 case is caught upstream by the topology probe."""
    segments = [(0.0, 20.0, 'remote'), (20.0, 600.0, 'host')]
    assert host_bleed_rate(segments) is None
    report = channel_separation_report(segments)
    assert report["reason"] == "insufficient_remote_speech"
    assert report["admissible"] is True


def test_separation_report_accounts_for_all_speech():
    segments = [(0.0, 100.0, 'host'), (100.0, 300.0, 'remote'),
                (300.0, 360.0, 'both')]
    report = channel_separation_report(segments)
    assert report["host_only_sec"] == 100.0
    assert report["remote_only_sec"] == 200.0
    assert report["both_sec"] == 60.0
