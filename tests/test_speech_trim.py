"""speech_trim: skip near-empty recordings, cut long no-speech stretches,
and map Gemini's timestamps back to the original recording.

Two-sided by design: every "cuts/skips" case has a matching "must NOT
cut/skip" case, because dropping real meeting audio is far worse than paying
for a few minutes of silence.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import speech_trim as st  # noqa: E402
from channel_vad import compute_speech_activity  # noqa: E402

W = 0.25  # channel_vad window


def _active(*spans, total):
    """Boolean window array: True inside the given (start, end) second spans."""
    a = np.zeros(int(total / W), dtype=bool)
    for s, e in spans:
        a[int(s / W):int(e / W)] = True
    return a


# ── plan_trim: skip ────────────────────────────────────────────────────

def test_near_empty_quiet_recording_is_skipped():
    # 2026-11-24_16-00-15 shape: 11.5 min, ~24 s of speech, p95 -62 dBFS.
    plan = st.plan_trim(_active((60, 72), (300, 312), total=690), W, p95_db=-62)
    assert plan.skip
    assert "almost no speech" in plan.reason
    assert plan.keep == []


def test_no_detected_speech_but_loud_audio_is_sent_whole():
    # Wall-to-wall talk / a noisy room: nothing 10 dB above the floor, so
    # the detector sees "no speech". Loud audio vetoes the skip AND the cut.
    plan = st.plan_trim(_active(total=1800), W, p95_db=-20)
    assert not plan.skip and not plan.trimmed
    assert plan.keep == [(0.0, 1800.0)]
    assert "inconclusive" in plan.reason


def test_unknown_level_never_skips():
    plan = st.plan_trim(_active((60, 72), total=690), W, p95_db=None)
    assert not plan.skip and not plan.trimmed


def test_short_real_meeting_is_not_skipped():
    # A 2-minute call that is mostly talk: little absolute speech, high share.
    plan = st.plan_trim(_active((0, 25), total=40), W)
    assert not plan.skip


def test_forty_seconds_of_speech_in_a_long_recording_is_not_skipped():
    # Above the absolute floor: kept (and trimmed), never dropped.
    plan = st.plan_trim(_active((100, 140), total=3600), W, p95_db=-62)
    assert not plan.skip
    assert plan.trimmed
    assert plan.keep == [(0.0, 150.0)]  # 100 s lead-in is under the gap threshold


# ── plan_trim: cut ─────────────────────────────────────────────────────

def test_forgotten_stop_tail_is_cut_with_padding():
    # 30 min of meeting, then 40 min of nothing (2026-09-14_15-59-56 shape).
    plan = st.plan_trim(_active((0, 1800), total=4200), W)
    assert plan.trimmed
    assert plan.keep == [(0.0, 1800 + st.PAD_SEC)]
    assert plan.effective_end == pytest.approx(1800 + st.PAD_SEC)
    assert plan.cut_sec == pytest.approx(4200 - 1810)


def test_long_mid_meeting_gap_is_cut_but_both_sides_kept():
    plan = st.plan_trim(_active((0, 600), (1200, 1800), total=1800), W)
    assert plan.keep == [(0.0, 610.0), (1190.0, 1800.0)]


def test_gap_shorter_than_threshold_is_kept():
    # A 2-minute pause (screen share, coffee) stays in.
    plan = st.plan_trim(_active((0, 600), (720, 1800), total=1800), W)
    assert not plan.trimmed
    assert plan.reason == "nothing to cut"


def test_normal_dense_meeting_is_untouched():
    spans = [(t, t + 20) for t in range(0, 3600, 30)]  # talk with 10 s pauses
    plan = st.plan_trim(_active(*spans, total=3600), W)
    assert not plan.skip and not plan.trimmed
    assert plan.keep == [(0.0, 3600.0)]


def test_leading_silence_is_cut_without_pad_on_the_outer_side():
    plan = st.plan_trim(_active((900, 1800), total=1800), W)
    assert plan.keep == [(900 - st.PAD_SEC, 1800.0)]


# ── time mapping ───────────────────────────────────────────────────────

def _two_island_plan():
    return st.TrimPlan(duration_sec=3600, speech_sec=1000,
                       keep=[(0.0, 600.0), (1800.0, 2400.0)])


def test_to_original_maps_across_the_cut():
    plan = _two_island_plan()
    assert plan.to_original(0) == 0
    assert plan.to_original(599) == 599
    assert plan.to_original(600) == 1800       # first second after the cut
    assert plan.to_original(900) == 2100


def test_segments_move_into_trimmed_time_and_cut_parts_vanish():
    plan = _two_island_plan()
    segs = [(500, 700, "host"), (1000, 1100, "remote"), (1850, 1900, "both")]
    assert plan.to_trimmed_segments(segs) == [(500, 600, "host"), (650, 700, "both")]
    dicts = [{"start": 1850, "end": 1900, "label": "S1"}]
    assert plan.to_trimmed_segments(dicts) == [{"start": 650, "end": 700, "label": "S1"}]


def test_untrimmed_plan_is_identity():
    plan = st.TrimPlan(duration_sec=100, speech_sec=80, keep=[(0.0, 100.0)])
    assert plan.to_original(42) == 42
    segs = [(1, 2, "host")]
    assert plan.to_trimmed_segments(segs) is segs
    assert st.remap_text("[00:42] A: hi", plan) == "[00:42] A: hi"


class _Result:
    def __init__(self):
        self.transcript = "[00:05] A: start\n[10:00] B: after the cut\n[15:00] A: later"
        self.speaker_emotions = [{"speaker": "A", "arc": [{"time": "[10:30]", "emotion": "calm"}]}]
        self.interruptions = [{"time": "[11:00]", "interrupter": "B"}]
        self.energy_levels = {"A": {"avg": 3, "arc": [{"time": "[12:00]", "level": 2}]}}
        self.missing_time_ranges = [(600.0, 660.0)]
        self.audio_duration_seconds = 1200


def test_remap_result_moves_every_timestamp_to_original_time():
    r = _Result()
    st.remap_result(r, _two_island_plan())
    assert r.transcript.splitlines() == [
        "[00:05] A: start", "[30:00] B: after the cut", "[35:00] A: later"]
    assert r.speaker_emotions[0]["arc"][0]["time"] == "[30:30]"
    assert r.interruptions[0]["time"] == "[31:00]"
    assert r.energy_levels["A"]["arc"][0]["time"] == "[32:00]"
    assert r.missing_time_ranges == [(1800.0, 1860.0)]
    assert r.audio_duration_seconds == 3600


def test_remap_uses_hour_format_past_sixty_minutes():
    plan = st.TrimPlan(duration_sec=7200, speech_sec=900,
                       keep=[(0.0, 300.0), (3900.0, 4500.0)])
    assert st.remap_text("[06:00] A: x", plan) == "[01:06:00] A: x"


class _Inner:
    CHUNK_DURATION_SEC = 900
    model = "gemini-3.8-flash"

    def __init__(self):
        self.calls = []

    def process_audio(self, path, prompt=None, **kw):
        self.calls.append(("process_audio", Path(path), kw))
        return _Result()

    def _process_single_shot(self, path, prompt, duration, **kw):
        self.calls.append(("single", Path(path), duration, kw))
        return _Result()

    def _process_chunked(self, path, prompt, duration, **kw):
        self.calls.append(("chunked", Path(path), duration, kw))
        return _Result()


def test_time_mapped_processor_sends_trimmed_audio_and_returns_original_time():
    inner, plan = _Inner(), _two_island_plan()
    proc = st.TimeMappedProcessor(inner, plan, Path("/tmp/x.speech.mp3"))
    r = proc.process_audio(Path("/tmp/x.mp3"), channel_segments=[(1850, 1900, "host")],
                           known_attendees=[{"name": "A"}])
    name, path, kw = inner.calls[0]
    assert path == Path("/tmp/x.speech.mp3")
    assert kw["channel_segments"] == [(650, 700, "host")]
    assert kw["known_attendees"] == [{"name": "A"}]
    assert "[30:00] B: after the cut" in r.transcript
    proc._process_single_shot(Path("/tmp/x.mp3"), None, 3600)
    assert inner.calls[1][2] == plan.kept_sec   # trimmed duration, not 3600
    assert proc.CHUNK_DURATION_SEC == 900       # attributes pass through


# ── real audio: detector + ffmpeg ──────────────────────────────────────

SR = 16000


def _write_wav(path: Path, channels: list[np.ndarray]):
    data = np.stack(channels, axis=1)
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(len(channels))
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _speechlike(seconds: float, rng) -> np.ndarray:
    """1 kHz tone gated at syllable rate, over the room-noise floor."""
    t = np.arange(int(seconds * SR)) / SR
    gate = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(float)
    return 0.3 * np.sin(2 * np.pi * 1000 * t) * gate + _room(seconds, rng)


def _room(seconds: float, rng) -> np.ndarray:
    return 0.002 * rng.standard_normal(int(seconds * SR))


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_detector_finds_forgotten_tail_on_three_channel_wav(tmp_path, rng):
    # Remote talk on the system channel for 3 min, then 6 min of room tone.
    mic = np.concatenate([_room(180, rng), _room(360, rng)])
    sysch = np.concatenate([_speechlike(180, rng), np.zeros(360 * SR)])
    wav = tmp_path / "rec.wav"
    _write_wav(wav, [mic, sysch, sysch])
    act = compute_speech_activity(wav)
    assert act is not None
    assert 150 < act.speech_sec < 200
    plan = st.plan_from_activity(act)
    assert plan.trimmed
    assert plan.effective_end == pytest.approx(180 + st.PAD_SEC, abs=2)


def test_detector_keeps_room_tone_out_on_mono(tmp_path, rng):
    # Mono in-room capture: 60 s talk, 5 min room tone, 60 s talk.
    mono = np.concatenate([_speechlike(60, rng), _room(300, rng), _speechlike(60, rng)])
    wav = tmp_path / "mono.wav"
    _write_wav(wav, [mono])
    plan = st.plan_from_activity(compute_speech_activity(wav))
    assert plan.trimmed and len(plan.keep) == 2
    assert plan.cut_sec == pytest.approx(300 - 2 * st.PAD_SEC, abs=3)


def test_detector_never_skips_or_cuts_wall_to_wall_talk(tmp_path, rng):
    # Gated faster than one 0.25 s window: every window looks alike, the
    # floor sits at speech level and the detector finds "no speech". The
    # level witness must stop that from dropping the recording.
    wav = tmp_path / "talk.wav"
    _write_wav(wav, [_speechlike(420, rng)])
    plan = st.plan_from_activity(compute_speech_activity(wav))
    assert not plan.trimmed and not plan.skip


def test_detector_skips_a_quiet_empty_room(tmp_path, rng):
    wav = tmp_path / "empty.wav"
    quiet = 0.0003 * rng.standard_normal(600 * SR)  # about -70 dBFS
    _write_wav(wav, [quiet, np.zeros(600 * SR), np.zeros(600 * SR)])
    plan = st.plan_from_activity(compute_speech_activity(wav))
    assert plan.skip


def test_render_trimmed_mp3_has_the_kept_duration(tmp_path, rng):
    wav = tmp_path / "src.wav"
    _write_wav(wav, [_speechlike(120, rng)])
    mp3 = tmp_path / "src.mp3"
    subprocess.run([st.FFMPEG_PATH, "-v", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    plan = st.TrimPlan(duration_sec=120, speech_sec=60, keep=[(0.0, 20.0), (80.0, 110.0)])
    out = st.render_trimmed_mp3(mp3, plan, tmp_path / "src.speech.mp3")
    dur = float(subprocess.run(
        ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    assert dur == pytest.approx(50.0, abs=0.3)


def test_plan_serialises_for_meta():
    d = st.TrimPlan(duration_sec=100, speech_sec=10, keep=[(0.0, 50.0)]).to_dict()
    assert json.loads(json.dumps(d))["cut_sec"] == 50.0
