"""Tests for the ducked pre-mix (convert_for_gemini_ducked) that replaces
the equal-weight mix for bleeding-mic recordings.

Background: on recordings made without headphones, ch0 (host mic) also
picks up ch1/ch2 (system audio / remote participants) acoustically. The
legacy `pan=mono|c0=0.333*c0+0.333*c1+0.333*c2` mix therefore sends every
remote utterance to Gemini TWICE, which a controlled A/B (2026-08-11
gating spike, Brain scratchpad `gating-spike/`) showed produces duplicated
transcript lines and phantom speakers. Ducking ch0 whenever ch1/ch2 is
active collapses those to single, correctly-attributed turns.

Two groups of tests:
  - Pure numpy tests on `build_duck_gain_envelope` / `_debounce_min_dwell`
    (no ffmpeg needed) covering the gate shape, duck depth, and the
    min-dwell debounce added as insurance against gate chatter.
  - ffmpeg-backed tests on `convert_for_gemini_ducked` itself, using a
    synthetic 3-channel WAV with known tone placement.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from audio_converter import (  # noqa: E402
    DUCK_DB_DEFAULT,
    FFMPEG_PATH,
    FFPROBE_PATH,
    _debounce_min_dwell,
    _decode_raw_pcm,
    _dilate_forward,
    build_duck_gain_envelope,
    convert_for_gemini_ducked,
    get_audio_duration,
)

needs_ffmpeg = pytest.mark.skipif(
    not os.path.exists(FFMPEG_PATH) or not os.path.exists(FFPROBE_PATH),
    reason="ffmpeg/ffprobe not installed",
)

WINDOW_SEC = 0.25
SR = 8000  # keep synthetic-audio tests cheap; production runs at 48kHz


@dataclass
class _FakeChannelVAD:
    """Minimal stand-in for channel_vad.ChannelVAD — just the two fields
    convert_for_gemini_ducked actually reads."""
    remote_active: "np.ndarray"
    window_sec: float


def _write_wav(path: Path, channels: list[list[tuple[float, float]]],
               duration_sec: float, sr: int = SR, freq: float = 1000.0,
               amplitude: float = 0.5) -> Path:
    """Write an n-channel WAV with `freq` tone bursts at given spans (same
    helper shape as test_channel_vad.py's, duplicated to keep this file
    self-contained)."""
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


# ── gain envelope shape (pure numpy, no ffmpeg) ────────────────────────


def test_gain_is_full_during_host_only_and_ducked_during_remote():
    n_windows = 40  # 10s at 0.25s/window
    remote_active = np.zeros(n_windows, dtype=bool)
    remote_active[20:30] = True  # remote speaks windows 20-29 (5.0-7.5s)
    n_samples = int(n_windows * WINDOW_SEC * SR)

    gain = build_duck_gain_envelope(
        remote_active, WINDOW_SEC, n_samples, sr=SR,
        duck_db=18.0, attack_ms=50, release_ms=120, hangover_ms=0,
        min_dwell_ms=0,
    )
    expected_floor = 10 ** (-18.0 / 20.0)

    # Sample well inside each region, away from the ramp edges.
    def at(t_sec):
        return gain[int(t_sec * SR)]

    assert at(2.0) == pytest.approx(1.0, abs=1e-6)   # host-only, before duck
    assert at(6.0) == pytest.approx(expected_floor, abs=1e-3)  # mid-duck
    assert at(9.0) == pytest.approx(1.0, abs=1e-6)   # host-only, after duck


def test_duck_depth_default_matches_expected_linear_floor():
    """18 dB attenuation is ~0.126 linear gain — the number cited in the
    controlled A/B writeup."""
    n_windows = 20
    remote_active = np.ones(n_windows, dtype=bool)  # remote active throughout
    n_samples = int(n_windows * WINDOW_SEC * SR)
    gain = build_duck_gain_envelope(
        remote_active, WINDOW_SEC, n_samples, sr=SR,
        duck_db=DUCK_DB_DEFAULT, attack_ms=10, release_ms=10,
        hangover_ms=0, min_dwell_ms=0,
    )
    floor = float(gain[-1])  # past the attack ramp, fully settled
    assert floor == pytest.approx(10 ** (-18.0 / 20.0), abs=1e-4)
    assert floor == pytest.approx(0.1259, abs=1e-3)


@pytest.mark.parametrize("duck_db,expected", [(12.0, 0.2512), (18.0, 0.1259), (24.0, 0.0631)])
def test_duck_depth_parametrized(duck_db, expected):
    n_windows = 20
    remote_active = np.ones(n_windows, dtype=bool)
    n_samples = int(n_windows * WINDOW_SEC * SR)
    gain = build_duck_gain_envelope(
        remote_active, WINDOW_SEC, n_samples, sr=SR,
        duck_db=duck_db, attack_ms=10, release_ms=10,
        hangover_ms=0, min_dwell_ms=0,
    )
    assert float(gain[-1]) == pytest.approx(expected, abs=1e-3)


def test_ramps_are_monotonic():
    n_windows = 40
    remote_active = np.zeros(n_windows, dtype=bool)
    remote_active[20:30] = True
    n_samples = int(n_windows * WINDOW_SEC * SR)
    attack_ms, release_ms = 50.0, 120.0
    gain = build_duck_gain_envelope(
        remote_active, WINDOW_SEC, n_samples, sr=SR,
        duck_db=18.0, attack_ms=attack_ms, release_ms=release_ms,
        hangover_ms=0, min_dwell_ms=0,
    )
    attack_start = int(20 * WINDOW_SEC * SR)
    attack_n = int(round(attack_ms / 1000.0 * SR))
    attack_seg = gain[attack_start:attack_start + attack_n]
    assert np.all(np.diff(attack_seg) <= 1e-9), "attack ramp must be non-increasing"

    release_start = int(30 * WINDOW_SEC * SR)
    release_n = int(round(release_ms / 1000.0 * SR))
    release_seg = gain[release_start:release_start + release_n]
    assert np.all(np.diff(release_seg) >= -1e-9), "release ramp must be non-decreasing"


def test_hangover_extends_duck_past_remote_end():
    n_windows = 40
    remote_active = np.zeros(n_windows, dtype=bool)
    remote_active[10:12] = True  # remote active 2.5-3.0s only
    n_samples = int(n_windows * WINDOW_SEC * SR)
    gain = build_duck_gain_envelope(
        remote_active, WINDOW_SEC, n_samples, sr=SR,
        duck_db=18.0, attack_ms=10, release_ms=10,
        hangover_ms=1000.0,  # 1s hangover
        min_dwell_ms=0,
    )
    expected_floor = 10 ** (-18.0 / 20.0)
    # 0.5s after remote speech ends (still within the 1s hangover window).
    assert gain[int(3.5 * SR)] == pytest.approx(expected_floor, abs=1e-3)


# ── min-dwell debounce: prevents gate chatter (two-sided) ──────────────


def test_debounce_absorbs_reopen_shorter_than_min_dwell():
    """A brief reopen right after a close must be swallowed back into the
    closed run — this is the insurance the spike recommended but did not
    validate against a real runaway-repetition file."""
    # windows: 10 closed, 2 open (too short), 10 closed
    binary = np.array([True] * 10 + [False] * 2 + [True] * 10)
    out = _debounce_min_dwell(binary, min_dwell_windows=5)
    assert bool(out.all()), "short reopen must be fully absorbed into one closed run"


def test_debounce_preserves_reopen_at_or_above_min_dwell():
    """Two-sided pairing: a reopen long enough to be legitimate must survive
    untouched — the debounce must not just always merge everything."""
    binary = np.array([True] * 10 + [False] * 6 + [True] * 10)
    out = _debounce_min_dwell(binary, min_dwell_windows=5)
    assert not bool(out[10:16].any()), "reopen >= min_dwell must survive"
    assert bool(out[:10].all()) and bool(out[16:].all())


def test_debounce_no_subthreshold_open_segment_survives_rapid_toggling():
    """Feed a synthetic rapidly-toggling VAD array (sub-min-dwell flips)
    after an initial real closure, and assert no open run shorter than
    min_dwell survives anywhere in the output."""
    rng = np.random.default_rng(42)
    min_dwell = 6
    # 20 windows of a real closure, then 200 windows of noisy 1-3-window
    # flips (well under min_dwell), then a real 20-window reopen.
    chatter = []
    state = True
    while sum(len(run) for run in chatter) < 200:
        run_len = int(rng.integers(1, 4))  # 1-3 windows, always < min_dwell
        chatter.append([state] * run_len)
        state = not state
    chatter_flat = [v for run in chatter for v in run]
    binary = np.array([True] * 20 + chatter_flat + [False] * 20)

    out = _debounce_min_dwell(binary, min_dwell_windows=min_dwell)

    # Walk the output and measure every run's length.
    change_idx = np.flatnonzero(np.diff(out.astype(np.int8)) != 0) + 1
    bounds = [0] + change_idx.tolist() + [len(out)]
    runs = [(bounds[i], bounds[i + 1], bool(out[bounds[i]])) for i in range(len(bounds) - 1)]
    # Skip the very first run — debounce only constrains REOPENS, i.e.
    # False-runs that follow a True-run; a leading run has no such
    # predecessor and legitimately falls outside the guarantee.
    open_runs_after_first_close = [
        (s, e) for i, (s, e, v) in enumerate(runs) if i > 0 and not v
    ]
    assert open_runs_after_first_close, "test setup should produce some open runs"
    for s, e in open_runs_after_first_close:
        assert (e - s) >= min_dwell, f"sub-threshold open run survived: [{s}, {e})"


def test_debounce_noop_when_min_dwell_is_zero():
    binary = np.array([True, False, True, False, True])
    out = _debounce_min_dwell(binary, min_dwell_windows=0)
    assert list(out) == list(binary)


def test_dilate_forward_extends_true_runs_only_forward():
    active = np.array([False, False, True, False, False, False])
    out = _dilate_forward(active, n_windows=2)
    assert list(out) == [False, False, True, True, True, False]


# ── end-to-end conversion (ffmpeg-backed) ──────────────────────────────


@needs_ffmpeg
def test_convert_for_gemini_ducked_produces_valid_mp3(tmp_path):
    wav = _write_wav(
        tmp_path / "synth3ch.wav",
        channels=[
            [(0, 10)],   # ch0: host mic, tone throughout
            [(4, 6)],    # ch1: remote burst
            [(4, 6)],    # ch2: remote burst
        ],
        duration_sec=10,
    )
    n_windows = int(round(10 / WINDOW_SEC))
    remote_active = np.zeros(n_windows, dtype=bool)
    remote_active[int(4 / WINDOW_SEC):int(6 / WINDOW_SEC)] = True
    fake_vad = _FakeChannelVAD(remote_active=remote_active, window_sec=WINDOW_SEC)

    out_dir = tmp_path / "out"
    mp3_path = convert_for_gemini_ducked(wav, fake_vad, output_dir=out_dir)

    assert mp3_path.exists()
    assert mp3_path.suffix == ".mp3"
    assert mp3_path.stat().st_size > 0
    duration = get_audio_duration(mp3_path)
    assert duration == pytest.approx(10.0, abs=0.5)
    # No leftover intermediate WAV.
    assert not any(p.suffix == ".wav" for p in out_dir.iterdir())


@needs_ffmpeg
def test_convert_for_gemini_ducked_rejects_non_hybrid_input(tmp_path):
    wav = _write_wav(tmp_path / "mono.wav", channels=[[(0, 2)]], duration_sec=3)
    fake_vad = _FakeChannelVAD(remote_active=np.zeros(12, dtype=bool), window_sec=WINDOW_SEC)
    with pytest.raises(ValueError):
        convert_for_gemini_ducked(wav, fake_vad, output_dir=tmp_path / "out")


@needs_ffmpeg
def test_duck_attenuates_ch0_in_gated_region_not_outside(tmp_path):
    """Duck depth on real (decoded) audio: mix RMS in the gated region
    should be reduced by ~duck_db relative to the ungated region, measured
    on the pre-loudnorm mix so EBU normalization doesn't obscure the ratio.
    """
    duration = 12.0
    wav = _write_wav(
        tmp_path / "synth3ch.wav",
        channels=[
            [(0, duration)],   # ch0: constant tone for the whole file
            [(5, 8)],          # ch1: remote burst
            [],
        ],
        duration_sec=duration,
        amplitude=0.5,
    )
    channels = 3
    frames = _decode_raw_pcm(wav, channels=channels, sr=SR)
    n = len(frames)
    c0, c1, c2 = frames[:, 0], frames[:, 1], frames[:, 2]

    n_windows = int(round(duration / WINDOW_SEC))
    remote_active = np.zeros(n_windows, dtype=bool)
    remote_active[int(5 / WINDOW_SEC):int(8 / WINDOW_SEC)] = True

    duck_db = 18.0
    gain = build_duck_gain_envelope(
        remote_active, WINDOW_SEC, n, sr=SR, duck_db=duck_db,
        attack_ms=50, release_ms=120, hangover_ms=0, min_dwell_ms=0,
    )
    mix = (gain * c0 + c1 + c2) / 3.0

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    # ch0-only contribution, isolated by re-deriving it from the mix minus
    # the (silent, in this test) c1/c2 contribution outside the burst.
    ungated_slice = slice(int(1 * SR), int(4 * SR))    # before the burst
    gated_slice = slice(int(6 * SR), int(7 * SR))      # mid-burst (past ramps)

    ungated_rms = rms(mix[ungated_slice])
    gated_ch0_rms = rms(gain[gated_slice] * c0[gated_slice] / 3.0)
    ungated_ch0_rms = rms(c0[ungated_slice] / 3.0)

    assert ungated_rms == pytest.approx(ungated_ch0_rms, rel=0.05)

    measured_db = 20 * math.log10(ungated_ch0_rms / gated_ch0_rms)
    assert measured_db == pytest.approx(duck_db, abs=0.5)
