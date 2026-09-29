"""Incremental microphone WAV writer.

The PortAudio callback only copies a block into a bounded queue. A writer
thread appends PCM and checkpoints the WAV header, so a crash keeps every
flushed second and loses at most the open checkpoint plus the queued tail.
"""

from __future__ import annotations

import os
import queue
import struct
import threading
from collections import deque
from pathlib import Path

import numpy as np

MIC_RING_SECONDS = 120.0
MIC_QUEUE_SECONDS = 3.0
MIC_CHECKPOINT_SECONDS = 1.0


def sample_count(block) -> int:
    array = np.asarray(block)
    if array.size == 0:
        return 0
    if array.ndim == 1:
        return int(array.shape[0])
    return int(array.shape[0])


def _pcm_bytes(block, channels: int) -> bytes:
    array = np.asarray(block)
    if array.size == 0:
        return b""
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[1] > channels:
        array = array[:, :channels]
    if array.shape[1] < channels:
        padded = np.zeros((array.shape[0], channels), dtype=np.int16)
        padded[:, : array.shape[1]] = array
        array = padded
    return np.ascontiguousarray(array, dtype="<i2").tobytes()


class SampleQueue:
    """Bounded by sample count. ``put_nowait`` never waits for the consumer."""

    def __init__(self, max_samples: int):
        self.max_samples = max(1, int(max_samples))
        self._items: deque = deque()
        self._samples = 0
        self._condition = threading.Condition()

    def put_nowait(self, block) -> None:
        count = sample_count(block)
        with self._condition:
            if self._samples > 0 and self._samples + count > self.max_samples:
                raise queue.Full
            self._items.append(block)
            self._samples += count
            self._condition.notify()

    def get(self, timeout: float):
        with self._condition:
            if not self._items:
                if timeout <= 0 or not self._condition.wait(timeout):
                    if not self._items:
                        raise queue.Empty
            if not self._items:
                raise queue.Empty
            block = self._items.popleft()
            self._samples -= sample_count(block)
            return block

    def empty(self) -> bool:
        with self._condition:
            return not self._items


class IncrementalWavWriter:
    """PCM16 WAV that stays readable after every checkpoint."""

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        channels: int = 1,
        checkpoint_seconds: float = MIC_CHECKPOINT_SECONDS,
    ):
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.samples_written = 0
        self._pending = bytearray()
        self._pending_samples = 0
        self._fh = self.path.open("wb")
        self._write_header(0)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def write(self, block) -> None:
        pcm = _pcm_bytes(block, self.channels)
        if not pcm:
            return
        self._pending.extend(pcm)
        self._pending_samples += sample_count(block)
        limit = max(1, int(self.checkpoint_seconds * self.sample_rate))
        if self._pending_samples >= limit:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self._fh.write(self._pending)
        self.samples_written += self._pending_samples
        self._pending.clear()
        self._pending_samples = 0
        self._patch_header()
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self.flush()
        if not self._fh.closed:
            self._fh.close()

    def abort(self) -> None:
        """Close without flushing the open checkpoint. The last fsync remains."""
        self._pending.clear()
        self._pending_samples = 0
        if not self._fh.closed:
            self._fh.close()

    def _patch_header(self) -> None:
        data_bytes = self.samples_written * self.channels * 2
        self._fh.seek(0)
        self._fh.write(self._header(data_bytes))
        self._fh.seek(0, os.SEEK_END)

    def _write_header(self, data_bytes: int) -> None:
        self._fh.seek(0)
        self._fh.write(self._header(data_bytes))

    def _header(self, data_bytes: int) -> bytes:
        bits = 16
        block_align = self.channels * bits // 8
        byte_rate = self.sample_rate * block_align
        data_bytes -= data_bytes % block_align
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_bytes,
            b"WAVE",
            b"fmt ",
            16,
            1,
            self.channels,
            self.sample_rate,
            byte_rate,
            block_align,
            bits,
            b"data",
            data_bytes,
        )


def recover_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a checkpointed WAV, including PCM written past a stale data size."""
    raw = Path(path).read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    channels = struct.unpack_from("<H", raw, 22)[0]
    rate = struct.unpack_from("<I", raw, 24)[0]
    bits = struct.unpack_from("<H", raw, 34)[0]
    if bits != 16 or channels < 1 or rate <= 0:
        raise ValueError("unsupported WAV encoding")
    data_at = raw.find(b"data")
    if data_at < 0 or data_at + 8 > len(raw):
        raise ValueError("WAV data chunk is missing")
    frame = channels * 2
    available = len(raw) - (data_at + 8)
    available -= available % frame
    if available <= 0:
        return np.zeros((0, channels), dtype=np.int16), rate
    pcm = np.frombuffer(raw[data_at + 8 : data_at + 8 + available], dtype="<i2").copy()
    return pcm.reshape(-1, channels), rate


def wav_duration_seconds(path: str | Path) -> float:
    pcm, rate = recover_wav(path)
    if rate <= 0:
        return 0.0
    return pcm.shape[0] / float(rate)
