"""Mic and system-audio tails for the live sidecar.

The microphone callback only appends copied blocks. This module reads those
blocks from another thread by copying the list of references. It never takes a
lock the callback would have to acquire, and it never writes audio.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def copy_chunk_references(chunks: list) -> list:
    """Copy the list of mic blocks. Each block is already an immutable copy."""
    for _ in range(3):
        try:
            return list(chunks)
        except RuntimeError:
            continue
    return list(chunks)


def tail_from_chunks(
    chunks: list, sample_rate: int, tail_seconds: float
) -> tuple[np.ndarray, float, float]:
    """Return the last ``tail_seconds`` of mono int16 audio and its time span.

    Times are seconds from the start of the concatenated buffer, which is the
    microphone clock from the first captured block.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not chunks:
        return np.zeros(0, dtype=np.int16), 0.0, 0.0
    arrays: list[np.ndarray] = []
    for chunk in chunks:
        array = np.asarray(chunk)
        if array.size == 0:
            continue
        if array.ndim == 2:
            array = array[:, 0]
        arrays.append(np.ascontiguousarray(array).reshape(-1))
    if not arrays:
        return np.zeros(0, dtype=np.int16), 0.0, 0.0
    pcm = np.concatenate(arrays)
    total = int(pcm.shape[0])
    keep = min(total, max(0, int(float(tail_seconds) * sample_rate)))
    start = total - keep
    return (
        np.ascontiguousarray(pcm[start:]),
        start / float(sample_rate),
        total / float(sample_rate),
    )


def _parse_pcm16_header(header: bytes) -> tuple[int, int, int]:
    """Return ``(data_offset, channels, sample_rate)`` for a PCM16 WAV header."""
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("system audio is not a WAV file yet")
    offset = 12
    channels = 0
    rate = 0
    bits = 0
    audio_format = 0
    data_offset = None
    while offset + 8 <= len(header):
        chunk_id = header[offset : offset + 4]
        chunk_size = int.from_bytes(header[offset + 4 : offset + 8], "little")
        payload = offset + 8
        if chunk_id == b"fmt " and payload + 16 <= len(header):
            audio_format = int.from_bytes(header[payload : payload + 2], "little")
            channels = int.from_bytes(header[payload + 2 : payload + 4], "little")
            rate = int.from_bytes(header[payload + 4 : payload + 8], "little")
            bits = int.from_bytes(header[payload + 14 : payload + 16], "little")
        elif chunk_id == b"data":
            data_offset = payload
            break
        step = 8 + chunk_size + (chunk_size & 1)
        if step <= 0:
            break
        offset += step
    if data_offset is None or channels < 1 or rate <= 0 or bits != 16 or audio_format != 1:
        raise ValueError("system WAV header is not finished PCM16")
    return data_offset, channels, rate


def read_wav_pcm_tail(path: Path, tail_seconds: float) -> tuple[np.ndarray, int]:
    """Read the last seconds of a WAV, including one still being written.

    ExtAudioFile may leave the data-chunk size at zero or at 0xFFFFFFFF until
    the tap closes the file. The readable tail is the bytes actually on disk
    after the data offset, aligned to a whole frame.
    """
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(min(file_size, 65536))
        data_offset, channels, rate = _parse_pcm16_header(header)
        frame = channels * 2
        available = max(0, file_size - data_offset)
        available -= available % frame
        tail_frames = max(0, int(float(tail_seconds) * rate))
        tail_bytes = min(available, tail_frames * frame)
        start = data_offset + available - tail_bytes
        handle.seek(start)
        raw = handle.read(tail_bytes)
    if len(raw) < frame:
        return np.zeros((0, channels), dtype=np.int16), rate
    usable = len(raw) - (len(raw) % frame)
    pcm = np.frombuffer(raw[:usable], dtype="<i2").copy()
    return pcm.reshape(-1, channels), rate


def as_mono_int16(pcm: np.ndarray) -> np.ndarray:
    """Average channels. An empty or 1-D array is returned unchanged."""
    if pcm.size == 0:
        return np.zeros(0, dtype=np.int16)
    if pcm.ndim == 1:
        return np.ascontiguousarray(pcm, dtype=np.int16)
    mixed = pcm.astype(np.float32).mean(axis=1)
    return np.clip(np.rint(mixed), -32768, 32767).astype(np.int16)


def resample_int16(samples: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Linear resample to the rate sent to Gemini. Empty input stays empty."""
    mono = as_mono_int16(samples)
    if mono.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.int16)
    if sample_rate == target_rate:
        return mono
    duration = mono.shape[0] / float(sample_rate)
    target_n = int(round(duration * target_rate))
    if target_n <= 1:
        return mono[:1]
    positions = np.linspace(0, mono.shape[0] - 1, target_n)
    resampled = np.interp(positions, np.arange(mono.shape[0]), mono.astype(np.float32))
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono int16 samples as a WAV payload for an inline Gemini part."""
    mono = as_mono_int16(samples)
    return _wav_bytes(mono.reshape(-1, 1), sample_rate)


def stereo_tail_wav(
    mic: np.ndarray,
    mic_rate: int,
    system: np.ndarray | None,
    system_rate: int,
    target_rate: int = 16000,
) -> bytes:
    """One stereo WAV: left is the mic tail, right is the system tail.

    Both channels end at the same moment. The shorter one is padded at the
    start. A missing system tail is silence on the right.
    """
    left = resample_int16(mic, mic_rate, target_rate)
    if system is None or getattr(system, "size", 0) == 0:
        right = np.zeros(0, dtype=np.int16)
    else:
        right = resample_int16(system, system_rate, target_rate)
    length = max(int(left.shape[0]), int(right.shape[0]))
    if length == 0:
        return b""
    stereo = np.zeros((length, 2), dtype=np.int16)
    if left.size:
        stereo[-left.shape[0] :, 0] = left
    if right.size:
        stereo[-right.shape[0] :, 1] = right
    return _wav_bytes(stereo, target_rate)


def _wav_bytes(frames: np.ndarray, sample_rate: int) -> bytes:
    """Encode ``(samples, channels)`` int16 frames."""
    channels = 1 if frames.ndim == 1 else int(frames.shape[1])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(np.ascontiguousarray(frames, dtype="<i2").tobytes())
    return buffer.getvalue()
