"""Tests for lossless source extraction and full-file channel activity."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audio_converter import (
    FFMPEG_PATH,
    FFPROBE_PATH,
    TOPOLOGY_MULTI_SOURCE_GENUINE,
    classify_source_topology,
    extract_source_tracks,
    get_audio_info,
)

needs_ffmpeg = pytest.mark.skipif(
    not os.path.exists(FFMPEG_PATH) or not os.path.exists(FFPROBE_PATH),
    reason="ffmpeg/ffprobe not installed",
)


def _write_pcm_wav(path: Path, samples: np.ndarray, sample_rate: int = 1000) -> Path:
    samples = np.asarray(samples, dtype=np.int16)
    if samples.ndim == 1:
        samples = samples[:, None]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(samples.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.astype("<i2", copy=False).tobytes())
    return path


def _decode_s16(path: Path, channels: int) -> np.ndarray:
    result = subprocess.run(
        [
            FFMPEG_PATH, "-v", "error", "-i", str(path),
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    data = np.frombuffer(result.stdout, dtype="<i2")
    assert len(data) % channels == 0
    return data.reshape(-1, channels)


@needs_ffmpeg
def test_legacy_three_channel_export_preserves_mic_and_stereo_system(tmp_path):
    frames = 2000
    original = np.column_stack([
        np.arange(frames, dtype=np.int16) - 1000,
        9000 - np.arange(frames, dtype=np.int16),
        np.full(frames, -7000, dtype=np.int16),
    ])
    wav = _write_pcm_wav(tmp_path / "legacy.wav", original)

    sources = extract_source_tracks(wav, tmp_path / "sources")

    assert [source["source_id"] for source in sources] == ["mic", "system"]
    assert all(source["start_seconds"] == 0.0 for source in sources)
    assert all(source["timing_basis"] == "unverified_sample_zero" for source in sources)
    mic_path, system_path = (Path(source["path"]) for source in sources)
    assert mic_path.suffix == system_path.suffix == ".flac"
    assert get_audio_info(mic_path).channels == 1
    assert get_audio_info(system_path).channels == 2
    np.testing.assert_array_equal(_decode_s16(mic_path, 1), original[:, :1])
    np.testing.assert_array_equal(_decode_s16(system_path, 2), original[:, 1:3])


@needs_ffmpeg
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_opaque_layouts_keep_declared_channels_without_inventing_roles(tmp_path, channels):
    frames = 400
    samples = np.column_stack([
        np.full(frames, 1000 * (index + 1), dtype=np.int16)
        for index in range(channels)
    ])
    wav = _write_pcm_wav(tmp_path / f"{channels}ch.wav", samples)

    sources = extract_source_tracks(wav, tmp_path / "sources")

    assert [source["source_id"] for source in sources] == ["audio"]
    output = Path(sources[0]["path"])
    assert get_audio_info(output).channels == channels
    np.testing.assert_array_equal(_decode_s16(output, channels), samples)


@needs_ffmpeg
def test_activity_classifier_reads_after_the_legacy_first_minute(tmp_path):
    """A source that begins after 60 seconds must still be classified active."""
    sample_rate = 1000
    duration_seconds = 66
    frames = sample_rate * duration_seconds
    time = np.arange(frames) / sample_rate
    mic = (10_000 * np.sin(2 * np.pi * 20 * time)).astype(np.int16)
    system_late = np.zeros(frames, dtype=np.int16)
    system_late[62 * sample_rate:] = (
        10_000 * np.sin(2 * np.pi * 35 * time[62 * sample_rate:])
    ).astype(np.int16)
    silent_system_right = np.zeros(frames, dtype=np.int16)
    wav = _write_pcm_wav(
        tmp_path / "late-system.wav",
        np.column_stack([mic, system_late, silent_system_right]),
        sample_rate,
    )

    # The legacy argument is intentionally ignored: activity is full-file.
    topology = classify_source_topology(wav, probe_seconds=60)

    assert set(topology.active_channels) == {0, 1}
    assert topology.topology == TOPOLOGY_MULTI_SOURCE_GENUINE
