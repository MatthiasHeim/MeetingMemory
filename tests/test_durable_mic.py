"""Durable mic capture: incremental WAV, crash recovery, non-blocking callback."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
pytest.importorskip("rumps")
pytest.importorskip("soundfile")

from meeting_recorder import AudioRecorder
from mic_writer import recover_wav, wav_duration_seconds
from sidecar.live import snapshot_recording_tails


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def info(self):
        return type("Info", (), {"inputBufferAdcTime": self.t, "currentTime": self.t})()


def _recorder(tmp_path: Path, rate: int = 16000, ring: float = 120.0, checkpoint: float = 1.0):
    recorder = AudioRecorder({
        "audio": {
            "mic_sample_rate": rate,
            "mic_ring_seconds": ring,
            "mic_checkpoint_seconds": checkpoint,
            "archive_dir": str(tmp_path / "archive"),
            "compress_archives": False,
        }
    })
    recorder.sample_rate = rate
    recorder.tmp_dir = tmp_path
    recorder._mic_wav = tmp_path / "mic.wav"
    recorder._sys_wav = tmp_path / "sys.wav"
    recorder._capture_meta = {"discontinuities": [], "mic_timing_samples": []}
    recorder._mic_frames = 0
    recorder._previous_adc_end = None
    recorder._next_timing_sample = 10**9
    recorder.recording = True
    return recorder


def _feed(recorder, seconds: float, value: int = 1000, block_seconds: float = 0.1):
    rate = recorder.sample_rate
    block_n = int(rate * block_seconds)
    clock = _Clock()
    clock.t = recorder._mic_frames / float(rate)
    block = np.full((block_n, 1), value, dtype=np.int16)
    steps = int(round(seconds / block_seconds))
    for _ in range(steps):
        recorder._audio_callback(block, block_n, clock.info(), None)
        clock.t += block_seconds
    return steps * block_n


def _wait_until_flushed(recorder, samples: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while recorder._mic_writer.samples_written < samples and time.monotonic() < deadline:
        time.sleep(0.01)


def test_callback_source_has_no_blocking_io():
    source = inspect.getsource(AudioRecorder._audio_callback)
    assert "put_nowait" in source
    assert "indata.copy()" in source
    for banned in ("open(", "fsync", "soundfile", "sf.write", "write("):
        assert banned not in source


def test_full_queue_does_not_block_the_callback(tmp_path):
    recorder = _recorder(tmp_path, checkpoint=1.0)
    recorder._start_mic_writer()
    original = recorder._mic_writer.write

    def slow(block):
        time.sleep(0.3)
        original(block)

    recorder._mic_writer.write = slow
    try:
        _feed(recorder, 4.0)
        started = time.perf_counter()
        _feed(recorder, 2.0)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.25
        assert recorder._mic_dropped_blocks > 0
    finally:
        recorder.abandon_mic_writer()


def test_clean_stop_duration_matches_wall_clock(tmp_path):
    recorder = _recorder(tmp_path, checkpoint=0.5)
    recorder._start_mic_writer()
    rate = recorder.sample_rate
    block_n = rate // 10
    block = np.full((block_n, 1), 400, dtype=np.int16)
    clock = _Clock()
    started = time.monotonic()
    for _ in range(20):
        recorder._audio_callback(block, block_n, clock.info(), None)
        clock.t += 0.1
        time.sleep(0.1)
    elapsed = time.monotonic() - started
    recorder._finish_mic_writer()
    duration = wav_duration_seconds(recorder._mic_wav)
    assert duration == pytest.approx(2.0, abs=0.001)
    assert duration == pytest.approx(elapsed, abs=0.35)
    assert duration == pytest.approx(recorder._mic_frames / rate, abs=0.001)
    pcm, _rate = recover_wav(recorder._mic_wav)
    assert int(pcm[0, 0]) == 400
    assert pcm.shape[0] == 2 * rate


def test_abandoned_writer_keeps_flushed_audio_and_drops_the_tail(tmp_path):
    recorder = _recorder(tmp_path, checkpoint=1.0)
    recorder._start_mic_writer()
    # One second at a time stays inside the bounded queue, the way a live
    # callback does. Dumping the whole meeting at once would drop blocks.
    for second in range(8):
        _feed(recorder, 1.0, value=11)
        _wait_until_flushed(recorder, (second + 1) * recorder.sample_rate)
    assert recorder._mic_writer.samples_written >= 8 * recorder.sample_rate
    recorder._mic_writer_abandon.set()
    _feed(recorder, 2.0, value=11)
    recorder.abandon_mic_writer()
    duration = wav_duration_seconds(recorder._mic_wav)
    assert duration == pytest.approx(8.0, abs=1.0)
    assert duration < 9.0


def test_killed_process_recovers_all_but_the_last_seconds(tmp_path):
    wav = tmp_path / "killed.wav"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--kill-child", str(wav)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 99, proc.stderr[-500:]
    duration = wav_duration_seconds(wav)
    assert 7.0 <= duration <= 8.5
    assert duration < 9.5
    pcm, _rate = recover_wav(wav)
    assert int(pcm[0, 0]) == 7


def test_ring_is_bounded_while_the_file_keeps_the_meeting(tmp_path):
    recorder = _recorder(tmp_path, ring=1.0, checkpoint=0.5)
    recorder._start_mic_writer()
    try:
        _feed(recorder, 3.0, value=9)
        deadline = time.monotonic() + 5
        while recorder._mic_writer.samples_written < 3 * recorder.sample_rate and time.monotonic() < deadline:
            time.sleep(0.01)
        ring_samples = sum(int(np.asarray(block).shape[0]) for block in recorder.audio_data)
        limit = int(recorder.sample_rate * recorder._mic_ring_seconds) + int(recorder.sample_rate * 0.1)
        assert ring_samples <= limit
        assert recorder._mic_frames == 3 * recorder.sample_rate
        snap = snapshot_recording_tails(recorder, tail_seconds=0.5)
        assert snap is not None
        assert snap.window_end == pytest.approx(3.0, abs=0.02)
        assert snap.mic.shape[0] == pytest.approx(int(0.5 * recorder.sample_rate), abs=recorder.sample_rate * 0.1)
        assert snap.window_start == pytest.approx(snap.window_end - snap.mic.shape[0] / recorder.sample_rate, abs=0.02)
    finally:
        recorder._finish_mic_writer()
    assert wav_duration_seconds(recorder._mic_wav) == pytest.approx(3.0, abs=0.001)


def test_stop_still_writes_the_merged_three_channel_wav(tmp_path):
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        pytest.skip("ffmpeg is not installed")
    import soundfile as sf

    recorder = _recorder(tmp_path, checkpoint=0.25)
    recorder.output_file = tmp_path / "meeting.wav"
    recorder._sys_active = False
    rate = recorder.sample_rate
    sys_pcm = np.zeros((rate, 2), dtype=np.int16)
    sys_pcm[:, 0] = 200
    sys_pcm[:, 1] = 800
    sf.write(recorder._sys_wav, sys_pcm, rate, subtype="PCM_16")
    recorder._start_mic_writer()
    _feed(recorder, 1.0, value=440)
    output = recorder.stop()
    assert output == recorder.output_file
    data, file_rate = sf.read(output, dtype="int16")
    assert file_rate == rate
    assert data.ndim == 2 and data.shape[1] == 3
    assert data.shape[0] == pytest.approx(rate, abs=int(rate * 0.05))
    assert int(np.median(data[:, 0])) == 440


def test_stale_header_still_recovers_bytes_past_the_data_chunk(tmp_path):
    from mic_writer import IncrementalWavWriter

    path = tmp_path / "partial.wav"
    writer = IncrementalWavWriter(path, 16000, channels=1, checkpoint_seconds=1.0)
    block = np.full((16000, 1), 3, dtype=np.int16)
    writer.write(block)
    writer.flush()
    extra = np.full(8000, 5, dtype="<i2").tobytes()
    with path.open("ab") as handle:
        handle.write(extra)
    writer.abort()
    pcm, rate = recover_wav(path)
    assert rate == 16000
    assert pcm.shape[0] == 24000
    assert int(pcm[0, 0]) == 3
    assert int(pcm[-1, 0]) == 5


def _kill_child(path: str) -> None:
    dest = Path(path)
    recorder = _recorder(dest.parent, checkpoint=1.0)
    recorder._mic_wav = dest
    recorder._start_mic_writer()
    for second in range(8):
        _feed(recorder, 1.0, value=7)
        _wait_until_flushed(recorder, (second + 1) * recorder.sample_rate)
    recorder._mic_writer_abandon.set()
    _feed(recorder, 2.0, value=7)
    os._exit(99)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--kill-child":
        _kill_child(sys.argv[2])
