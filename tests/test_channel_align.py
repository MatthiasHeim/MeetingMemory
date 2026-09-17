#!/usr/bin/env python3
"""Tests for channel_align — the 2026-09-17 constant mic/system offset fix.

Two-sided by construction: every detection test has a matching negative case,
so gutting the estimator fails the suite instead of quietly passing it.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from channel_align import (  # noqa: E402
    ENVELOPE_FPS,
    MIN_PROMINENCE_Z,
    estimate_channel_offset,
    write_aligned_wav,
)

SR = 16000

# Fixture length. `_speech_like` is bursty noise, whose correlation surface is
# flatter than real speech, so prominence grows with the amount of audio:
# a true 8s offset scores z≈5.0 over 40s but z≈10.7 over 180s, while an
# independent (headphone) mic stays at z≈2.3-2.8 regardless.
#
# 180s is used so the tests clear the SHIPPED MIN_PROMINENCE_Z rather than a
# relaxed test-only threshold. Tuning production down to suit a short fixture
# is exactly the mistake this guards against. For reference, the real
# 2026-09-17 recording with its known 15.46s offset scores z≈29.7.
FIXTURE_SEC = 180.0


def _speech_like(duration_s: float, seed: int = 0):
    """Bursty noise — a crude but adequate stand-in for speech energy."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    x = rng.normal(0, 0.05, n)
    # Impose an on/off envelope so cross-correlation has something to lock onto.
    env = np.zeros(n)
    t = 0
    while t < n:
        on = int(rng.uniform(0.4, 1.2) * SR)
        off = int(rng.uniform(0.3, 1.0) * SR)
        env[t:t + on] = 1.0
        t += on + off
    return (x * env).astype(np.float32)


def _write_wav(path: Path, ch0, ch1, ch2):
    import wave

    n = min(len(ch0), len(ch1), len(ch2))
    inter = np.stack([ch0[:n], ch1[:n], ch2[:n]], axis=1)
    pcm = np.clip(inter, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(3)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def _make_recording(path: Path, offset_s: float, bleed: float = 0.6,
                    duration_s: float = FIXTURE_SEC):
    """3-channel recording where ch0 carries a bleed copy of the system audio,
    delayed by `offset_s`.
    """
    system = _speech_like(duration_s + abs(offset_s) + 5, seed=1)
    host = _speech_like(duration_s + abs(offset_s) + 5, seed=2) * 0.8

    shift = int(offset_s * SR)
    bled = np.zeros_like(system)
    if shift >= 0:
        bled[shift:] = system[: len(system) - shift] * bleed
    else:
        bled[: len(bled) + shift] = system[-shift:] * bleed

    ch0 = host + bled
    return _write_wav(path, ch0, system, system)


# ---------------------------------------------------------------- detection

@pytest.mark.parametrize("offset", [15.46, 8.0, 2.5])
def test_detects_known_positive_offset(tmp_path, offset):
    """POSITIVE SIDE: a real offset must be found, to within one frame."""
    wav = _make_recording(tmp_path / "off.wav", offset_s=offset)
    r = estimate_channel_offset(wav)
    assert r.correctable, f"{offset}s offset not flagged correctable: {r.reason}"
    assert abs(r.lag_seconds - offset) <= 2.0 / ENVELOPE_FPS, (
        f"estimated {r.lag_seconds}, expected {offset}"
    )
    assert r.correlation > 0.2


def test_aligned_recording_is_left_alone(tmp_path):
    """NEGATIVE SIDE: zero offset must NOT be flagged correctable.

    Without this, an estimator that always returns 'correctable' passes.
    """
    wav = _make_recording(tmp_path / "aligned.wav", offset_s=0.0)
    r = estimate_channel_offset(wav)
    assert not r.correctable, f"aligned recording wrongly flagged: {r.reason}"
    assert abs(r.lag_seconds) < 0.5


def test_headphone_recording_is_left_alone(tmp_path):
    """NEGATIVE SIDE: no bleed means no shared content, so no offset claim.

    A headphone recording has an independent mic channel. Correlating it
    against system audio must fall below the confidence floor rather than
    inventing a lag from noise.
    """
    system = _speech_like(FIXTURE_SEC, seed=1)
    host = _speech_like(FIXTURE_SEC, seed=2)
    wav = _write_wav(tmp_path / "clean.wav", host, system, system)
    r = estimate_channel_offset(wav)
    assert not r.correctable, f"clean recording wrongly flagged: {r.reason}"
    assert r.prominence_z < MIN_PROMINENCE_Z, (
        f"independent mic scored z={r.prominence_z:.1f}, indistinguishable "
        "from a real offset"
    )


def test_negative_offset_detected(tmp_path):
    """Mic EARLY rather than late — sign must be preserved, not absolute."""
    wav = _make_recording(tmp_path / "early.wav", offset_s=-4.0)
    r = estimate_channel_offset(wav)
    assert r.correctable
    assert r.lag_seconds < 0, f"expected negative lag, got {r.lag_seconds}"


# ---------------------------------------------------------------- correction

def _channel_envelope(path: Path, ch: int):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-filter_complex", f"[0:a]pan=mono|c0=c{ch},aresample=8000",
         "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(out, dtype=np.float32).astype(np.float64)
    n = 8000 // ENVELOPE_FPS
    x = x[: len(x) // n * n].reshape(-1, n)
    return np.sqrt((x ** 2).mean(axis=1))


def test_alignment_removes_the_offset(tmp_path):
    """END TO END: after correction the residual offset is below threshold.

    This is the property that actually matters — detection alone is useless if
    the rewrite does not close the gap.
    """
    wav = _make_recording(tmp_path / "off.wav", offset_s=12.0)
    before = estimate_channel_offset(wav)
    assert before.correctable

    fixed = write_aligned_wav(wav, before.lag_seconds,
                              output_path=tmp_path / "fixed.wav")
    after = estimate_channel_offset(fixed)
    assert not after.correctable, (
        f"offset survived alignment: {after.lag_seconds:+.2f}s ({after.reason})"
    )
    assert abs(after.lag_seconds) < 0.5


def test_alignment_preserves_channel_count_and_system_audio(tmp_path):
    """The system channels must pass through untouched — we only move the mic."""
    wav = _make_recording(tmp_path / "off.wav", offset_s=6.0)
    ref_before = _channel_envelope(wav, 1)
    fixed = write_aligned_wav(wav, 6.0, output_path=tmp_path / "fixed.wav")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=channels",
         "-of", "default=nw=1:nk=1", str(fixed)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe.startswith("3"), f"channel count changed: {probe}"

    ref_after = _channel_envelope(fixed, 1)
    n = min(len(ref_before), len(ref_after))
    assert n > 0
    corr = np.corrcoef(ref_before[:n], ref_after[:n])[0, 1]
    assert corr > 0.99, f"system channel altered by alignment (r={corr:.3f})"


def test_rejects_non_hybrid_layout(tmp_path):
    """Mono/stereo recordings have no mic/system split to align."""
    mono = tmp_path / "mono.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=2",
         "-ac", "1", str(mono), "-y"], check=True,
    )
    with pytest.raises(ValueError, match="3-channel"):
        write_aligned_wav(mono, 1.0, output_path=tmp_path / "x.wav", channels=1)


def test_production_threshold_rejects_an_independent_mic(tmp_path):
    """The shipped default must reject the no-bleed case, not just the test one.

    Guards against someone lowering MIN_PROMINENCE_Z to chase a stubborn
    recording and silently turning on alignment for headphone recordings.
    """
    system = _speech_like(FIXTURE_SEC, seed=1)
    host = _speech_like(FIXTURE_SEC, seed=2)
    wav = _write_wav(tmp_path / "clean.wav", host, system, system)
    r = estimate_channel_offset(wav)  # production defaults
    assert not r.correctable
    assert r.prominence_z < MIN_PROMINENCE_Z


def test_no_audio_is_lost_to_alignment(tmp_path):
    """Duration must survive the rewrite.

    The first implementation trimmed the mic head without padding its tail, so
    `amerge` stopped at the shortest input and silently cut `lag` seconds of
    meeting audio off the end. That is data loss, and it looked like success.
    """
    wav = _make_recording(tmp_path / "off.wav", offset_s=6.0)
    before = _probe_duration(wav)
    fixed = write_aligned_wav(wav, 6.0, output_path=tmp_path / "fixed.wav")
    after = _probe_duration(fixed)
    assert after >= before - 0.1, (
        f"alignment lost audio: {before:.2f}s -> {after:.2f}s"
    )


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


# ------------------------------------------------- watcher integration hazards

def test_aligned_copy_stays_out_of_recordings_and_keeps_its_name(tmp_path,
                                                                 monkeypatch):
    """The aligned copy must not be queued as a new recording, and must keep
    the original filename.

    Two ways this integration goes wrong, both silent:
      - written into recordings_dir, where `glob("*.wav")` re-queues it as a
        fresh meeting on the next watcher restart;
      - renamed to `<stem>.aligned.wav`, which changes the MP3 name, the
        transcript JSON name and the timestamp the calendar lookup parses.
    """
    import tempfile as _tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import transcribe_watcher as tw

    recordings = tmp_path / "Recordings"
    recordings.mkdir()
    src = _make_recording(recordings / "2026-09-17_14-43-50.wav", offset_s=8.0)

    fake_tmp = tmp_path / "tmp"
    fake_tmp.mkdir()
    # The helper does a function-local `import tempfile`, so patching the
    # stdlib module itself is what reaches it.
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(fake_tmp))

    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = _NullLogger()

    out = watcher._align_channels_safe(src, topology=None)

    assert out != src, "offset recording was not aligned"
    assert out.name == src.name, f"filename changed: {src.name} -> {out.name}"
    assert recordings not in out.parents, (
        f"aligned copy landed in recordings_dir ({out}) — it will be re-queued"
    )
    assert list(recordings.glob("*.wav")) == [src], (
        "extra .wav appeared in recordings_dir"
    )


def test_alignment_failure_falls_back_to_the_original(tmp_path, monkeypatch):
    """A broken alignment must not break transcription.

    Alignment is a quality improvement, not a precondition — if it throws, the
    watcher has to carry on with the unaligned file exactly as it did before.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import transcribe_watcher as tw
    import channel_align

    src = _make_recording(tmp_path / "2026-09-17_14-43-50.wav", offset_s=8.0)
    monkeypatch.setattr(
        channel_align, "write_aligned_wav",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")),
    )
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = _NullLogger()

    out = watcher._align_channels_safe(src, topology=None)
    assert out == src, "failure did not fall back to the original recording"


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass
