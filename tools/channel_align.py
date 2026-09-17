#!/usr/bin/env python3
"""channel_align — detect and correct a constant time offset between the host
mic channel and the system-audio channels of a hybrid multi-channel recording.

Why this exists (2026-09-17 incident, recording 2026-09-17_14-43-50):

The Aggregate Device does not guarantee that its sub-devices start capturing at
the same instant. On that recording ch0 (host mic) ran a constant **15.46 s**
behind ch1/ch2 (system audio) — measured by envelope cross-correlation in six
independent 5.5-minute windows: +15.44, +15.46, +15.46, +15.46, +15.50, +15.50.
A fixed offset, not clock drift (drift would grow; this moved 0.06 s over 33
minutes, i.e. ppm-level).

Consequence: the host mic also picks up the loudspeakers, so every remote
utterance reaches the mix twice — once clean from the system tap at t, once
through the mic at t + 15.46 s. Gemini faithfully transcribed both. 13 of 230
turns (5.7%) were verbatim duplicates, every single one at +15 or +16 s.

Why the existing duck does not catch it: `audio_converter.convert_for_gemini_ducked`
attenuates ch0 *while* ch1/ch2 are active (plus a ~200 ms hangover). That is
correct for acoustic echo, which is near-simultaneous — the 2026-08-11 spike
that motivated ducking saw duplicate pairs ~2 s apart. A 15 s-late copy arrives
long after the gate has released, so it passes through undecked. Ducking and
alignment are complementary, not alternatives: alignment puts the echo back
under the gate.

Correcting the offset on the 2026-09-17 recording took duplicate turns from
13 to 0 and let the model resolve real speaker names (Christian/Matthias/Roger/
Timo) instead of falling back to `Speaker 1/2/3`.

This module is a *legacy-capture* mitigation. The real fix is source-preserving
native capture with a verified shared clock (`docs/native-capture.md` on
`codex/meeting-pipeline-rebuild`), which records the true start delay instead of
assuming every source begins at zero. Until that ships, align here.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FFMPEG_PATH = 'ffmpeg'

# Analysis resolution. 8 kHz is ample for a speech-energy envelope and keeps a
# 60-minute recording well inside memory.
ALIGN_SAMPLE_RATE = 8000
ENVELOPE_FPS = 50  # 20 ms frames

# Search window. Startup offsets observed so far are ~15 s; 60 s gives headroom
# without letting a spurious far-out peak win.
MAX_LAG_SEC = 60.0

# Below this the offset is not worth a rewrite: the duck's hangover already
# covers sub-second acoustic echo, and re-encoding costs minutes on a long WAV.
MIN_CORRECTABLE_LAG_SEC = 0.5

# The mic must actually share content with the system channels for a lag to be
# meaningful. A headphone recording has near-zero bleed and will score low —
# which is the correct outcome: nothing to align.
MIN_CORRELATION = 0.20

# Peak prominence floor, in standard deviations above the rest of the search
# window. This is the real detector; MIN_CORRELATION only rejects silence.
# See `_best_lag` for the measured separation between true and spurious peaks.
MIN_PROMINENCE_Z = 6.0

# Half-width of the guard band excluded around the correlation peak when
# estimating the background. A genuine peak is not a spike: adjacent lags
# correlate nearly as well, and counting that shoulder as "background" hides
# the peak from its own prominence test.
PEAK_GUARD_SEC = 1.0


@dataclass
class ChannelOffset:
    """Result of an offset estimate.

    Attributes:
        lag_seconds: how far ch0 lags the reference channel. Positive means the
            mic is LATE — ch0[t + lag] carries what ch1[t] carries.
        correlation: peak normalised cross-correlation (-1..1) of the two
            energy envelopes at `lag_seconds`.
        prominence_z: how far the peak stands above the rest of the search
            window, in standard deviations. This, not `correlation`, is what
            separates a real offset from coincidence.
        correctable: True when the lag is both large enough to matter and
            supported by enough correlation to trust.
        reason: human-readable verdict for the log.
    """

    lag_seconds: float
    correlation: float
    prominence_z: float
    correctable: bool
    reason: str


def _decode_channel_envelope(path: Path, channel: int):
    """Energy envelope of one channel at ENVELOPE_FPS frames/sec."""
    import numpy as np

    cmd = [
        FFMPEG_PATH, '-v', 'error', '-i', str(path),
        '-filter_complex', f'[0:a]pan=mono|c0=c{channel},aresample={ALIGN_SAMPLE_RATE}',
        '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1',
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg channel decode failed: {result.stderr.decode(errors='replace')[:400]}"
        )
    x = np.frombuffer(result.stdout, dtype=np.float32)
    n = ALIGN_SAMPLE_RATE // ENVELOPE_FPS
    if len(x) < n:
        return np.zeros(0, dtype=np.float64)
    x = x[: len(x) // n * n].reshape(-1, n).astype(np.float64)
    return np.sqrt((x ** 2).mean(axis=1))


def _best_lag(a, b, max_lag_frames: int):
    """Peak normalised cross-correlation of two envelopes.

    Returns (lag_frames, correlation, prominence_z) where a positive lag means
    `a` is late relative to `b`, and prominence_z is how many standard
    deviations the peak stands above the rest of the searched window.
    """
    import numpy as np

    n = min(len(a), len(b))
    if n < 4:
        return 0, 0.0, 0.0
    # Never search further than half the signal: beyond that the overlap is too
    # short for the correlation to mean anything. Clamping (rather than bailing)
    # keeps short recordings analysable instead of silently returning "no
    # offset", which is indistinguishable from a clean recording.
    max_lag_frames = max(1, min(max_lag_frames, n // 2))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    sa, sb = a.std(), b.std()
    if sa <= 1e-12 or sb <= 1e-12:
        return 0, 0.0, 0.0
    a /= sa
    b /= sb

    # Full cross-correlation via FFT, then keep only the searched window.
    size = 1
    while size < 2 * n:
        size *= 2
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    corr = np.fft.irfft(fa * np.conj(fb), size)
    # corr[k] is the correlation with `a` advanced by k (i.e. `a` late by k).
    # Negative lags live at the tail: corr[size-k] is lag -k, so the last
    # `max_lag_frames` entries are already in ascending lag order (-m .. -1).
    pos = corr[: max_lag_frames + 1]                 # lags 0 .. +m
    neg = corr[-max_lag_frames:]                     # lags -m .. -1
    window = np.concatenate([neg, pos])              # lags -m .. +m
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    idx = int(np.argmax(window))
    best_val = float(window[idx] / n)
    best_lag = int(lags[idx])

    # Peak PROMINENCE, not absolute height. Two unrelated speech-like signals
    # correlate at r≈0.25-0.5 purely from shared burst statistics — enough to
    # clear any fixed correlation threshold low enough to catch a real offset.
    # What distinguishes a true offset is that its peak towers over the rest of
    # the search window.
    #
    # Two details matter for that to work:
    #  - A real peak has a SHOULDER: neighbouring lags correlate almost as well.
    #    Including it in the background inflates the spread and suppresses the
    #    very peak we are testing, so exclude a guard band around the peak.
    #  - Use median/MAD rather than mean/std. Any residual structure left in the
    #    window is an outlier to the background, and the mean chases outliers.
    scaled = window / n
    guard = max(1, int(PEAK_GUARD_SEC * ENVELOPE_FPS))
    lo, hi = max(0, idx - guard), min(len(scaled), idx + guard + 1)
    bg = np.concatenate([scaled[:lo], scaled[hi:]])
    if bg.size < 8:
        return best_lag, best_val, 0.0
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    sigma = 1.4826 * mad  # MAD -> standard-deviation equivalent for normal data
    z = (best_val - med) / sigma if sigma > 1e-12 else 0.0
    return best_lag, best_val, z


def estimate_channel_offset(
    audio_path: Path,
    reference_channel: int = 1,
    mic_channel: int = 0,
    max_lag_sec: float = MAX_LAG_SEC,
    min_lag_sec: float = MIN_CORRECTABLE_LAG_SEC,
    min_correlation: float = MIN_CORRELATION,
    min_prominence_z: float = MIN_PROMINENCE_Z,
) -> ChannelOffset:
    """Estimate how far the mic channel lags the system channel.

    Never raises on audio content; only on a genuinely broken decode.
    """
    mic = _decode_channel_envelope(Path(audio_path), mic_channel)
    ref = _decode_channel_envelope(Path(audio_path), reference_channel)
    if len(mic) == 0 or len(ref) == 0:
        return ChannelOffset(0.0, 0.0, 0.0, False, "empty channel envelope")

    max_frames = int(max_lag_sec * ENVELOPE_FPS)
    lag_frames, corr, z = _best_lag(mic, ref, max_frames)
    lag = lag_frames / ENVELOPE_FPS

    if corr < min_correlation:
        return ChannelOffset(
            lag, corr, z, False,
            f"correlation {corr:.3f} < {min_correlation} — channels share too "
            "little content to establish an offset (clean/headphone recording)",
        )
    if z < min_prominence_z:
        return ChannelOffset(
            lag, corr, z, False,
            f"peak prominence {z:.1f}sd < {min_prominence_z}sd — correlation "
            f"{corr:.3f} is not distinct from the rest of the search window, "
            "so no offset is claimed",
        )
    if abs(lag) < min_lag_sec:
        return ChannelOffset(
            lag, corr, z, False,
            f"offset {lag:+.2f}s below {min_lag_sec}s — within the duck's reach",
        )
    return ChannelOffset(
        lag, corr, z, True,
        f"mic channel lags system audio by {lag:+.2f}s "
        f"(r={corr:.3f}, prominence {z:.1f}sd)",
    )


def write_aligned_wav(
    audio_path: Path,
    lag_seconds: float,
    output_path: Optional[Path] = None,
    channels: int = 3,
) -> Path:
    """Write a copy of `audio_path` with the mic channel shifted into alignment.

    A positive `lag_seconds` means ch0 is late, so ch0 is advanced by trimming
    that much from its head; the system channels are passed through untouched
    and the result is re-interleaved at the original channel count. Output is
    PCM (lossless) — this feeds VAD and the pre-mix, so it must not be lossy.
    """
    audio_path = Path(audio_path)
    if channels < 3:
        raise ValueError(
            f"{audio_path.name} has {channels} channel(s); alignment needs the "
            "3-channel host-mic/system-audio hybrid layout"
        )
    if output_path is None:
        output_path = audio_path.with_name(f"{audio_path.stem}.aligned.wav")
    output_path = Path(output_path)

    if lag_seconds >= 0:
        # Mic is late: drop `lag` from its head, then pad the same amount back
        # onto its tail. Without the pad the mic track is shorter than the
        # system track and `amerge` stops at the shortest input — silently
        # truncating `lag` seconds of meeting audio off the end.
        mic_filter = (
            f"[0:a]pan=mono|c0=c0,atrim=start={lag_seconds:.4f},"
            f"asetpts=PTS-STARTPTS,apad=pad_dur={lag_seconds:.4f}[m]"
        )
    else:
        # Mic is early: delay it. It then runs longer than the system track,
        # so `amerge`'s shortest-input rule trims the mic tail, not the system
        # audio — which is the behaviour we want.
        delay_ms = int(round(abs(lag_seconds) * 1000))
        mic_filter = f"[0:a]pan=mono|c0=c0,adelay={delay_ms}[m]"
    sys_filter = "[0:a]pan=stereo|c0=c1|c1=c2[s]"

    # No `-ac`: forcing a channel count makes ffmpeg re-matrix the merged
    # stream and attenuate every channel (measured: system audio came out at
    # 0.900x). `amerge` already emits exactly `channels` channels.
    cmd = [
        FFMPEG_PATH, '-v', 'error', '-i', str(audio_path),
        '-filter_complex',
        f"{mic_filter};{sys_filter};[m][s]amerge=inputs=2[out]",
        '-map', '[out]',
        '-c:a', 'pcm_s16le', str(output_path), '-y',
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg alignment failed: {result.stderr.decode(errors='replace')[:400]}"
        )
    return output_path
