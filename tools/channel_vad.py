#!/usr/bin/env python3
"""channel_vad — channel-based voice activity detection for hybrid recordings.

MeetingRecorder writes 3-channel WAVs: channel 0 is the host's (Matthias's)
microphone, channels 1+2 are system audio (remote participants). With
headphones/AEC the channels are cleanly separated (measured sample-level
correlation ≈ 0.000), so the mic channel is a physical-layer oracle for
"is the host speaking right now" — far more reliable than Gemini's
voice-based diarization, which flips speakers on similar voices.

This module computes per-channel voice activity:

  - 250 ms-window RMS on ch0 (host) and max(|ch1|,|ch2|) (remote), after a
    300–3400 Hz band-pass that suppresses keyboard thuds / breathing on the
    mic channel and low-frequency rumble on system audio.
  - Asymmetric adaptive thresholds. The system channel is digitally clean
    (noise floor ≈ -100 dB on real recordings), so a 20th-percentile noise
    floor + 10 dB works. The mic channel is acoustically open — breathing,
    key clicks, and backchannel bursts sit 10-20 dB above its noise floor
    and would mark nearly every remote-speech window 'both' (measured on
    2026-06-11: 86% of remote-active time also host-active at floor+10 dB).
    So the host threshold is additionally anchored to the host's own speech
    level: median level of host-active windows while remote is silent
    (those are guaranteed host speech), minus 5 dB. Hysteresis (4 dB
    release) and 3-window median smoothing on top.
  - Output: labeled segments [(t0, t1, 'host'|'remote'|'both')], a
    `host_share(t0, t1)` helper for turn-level verification, and a compact
    textual host-speaking map for prompt injection (turn-level spans,
    merged: "00:00-00:42 remote / 00:42-02:10 host / ...").

`compute_channel_vad` returns None for recordings without the 3-channel
hybrid layout (mono mic-only fallback WAVs, plain stereo) — the channel
guarantee only holds for the hybrid capture format.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Absolute paths for launchd compatibility (minimal PATH under LaunchAgents).
FFMPEG_PATH = '/opt/homebrew/bin/ffmpeg'
FFPROBE_PATH = '/opt/homebrew/bin/ffprobe'

TARGET_SR = 8000            # speech-band analysis doesn't need more
WINDOW_SEC = 0.25
BANDPASS_LOW_HZ = 300       # below: keyboard thuds, breathing, HVAC rumble
BANDPASS_HIGH_HZ = 3400     # telephony speech band upper edge
NOISE_PERCENTILE = 20       # window-RMS percentile taken as the noise floor
ONSET_MARGIN_DB = 10.0      # activity starts at floor + this
HYSTERESIS_DB = 4.0         # ...and ends this far below the onset threshold
ABS_MIN_THRESHOLD_DB = -55.0  # never trigger below this (digital-silence guard)
SPEECH_ANCHOR_MARGIN_DB = 5.0  # host threshold: speech-level anchor minus this
MIN_ANCHOR_WINDOWS = 40     # ≥10 s of host-only speech needed to trust anchor
BRIDGE_GAP_SEC = 1.0        # same-label runs separated by less than this merge
MIN_MAP_SPAN_SEC = 2.0      # textual map: spans shorter than this get absorbed


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS, or HH:MM:SS past the hour."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def slice_segments(segments: list[tuple[float, float, str]],
                   start: float, end: float) -> list[tuple[float, float, str]]:
    """Clip segments to [start, end] and shift to chunk-relative time.

    Used by the chunked Gemini path: each audio chunk gets the slice of the
    meeting-global VAD map that covers it, re-based so its timestamps match
    the chunk-local [MM:SS] timestamps Gemini produces.
    """
    out: list[tuple[float, float, str]] = []
    for t0, t1, label in segments or []:
        s = max(t0, start)
        e = min(t1, end)
        if e - s > 0:
            out.append((s - start, e - start, label))
    return out


def render_map_text(segments: list[tuple[float, float, str]],
                    min_span_sec: float = MIN_MAP_SPAN_SEC) -> str:
    """Render segments as a compact textual host-speaking map.

    Adjacent same-label spans merge, and short (< `min_span_sec`) 'both'
    spans are absorbed into their neighbour — the result is turn-level, not
    backchannel-level. Short HOST and REMOTE spans are NOT absorbed: the
    prompt declares those labels exclusive ("only the host" / "not the
    host"), so folding a 1-second remote interjection into a host span
    would instruct Gemini to misattribute it to the host — exactly the
    error this map exists to prevent. Only 'both' spans (overlap, where the
    prompt already says "attribute by voice") are safe to absorb.
    """
    spans = [(t0, t1, label) for t0, t1, label in (segments or []) if t1 > t0]
    if not spans:
        return ""

    # Absorb short 'both' spans, then merge same-label neighbours.
    simplified: list[list] = []
    for t0, t1, label in spans:
        if (simplified and (t1 - t0) < min_span_sec and label == 'both'):
            simplified[-1][1] = max(simplified[-1][1], t1)
            continue
        if simplified and simplified[-1][2] == label:
            simplified[-1][1] = max(simplified[-1][1], t1)
            continue
        simplified.append([t0, t1, label])
    # A short leading 'both' span couldn't absorb backwards — fold forward.
    while (len(simplified) > 1
           and (simplified[0][1] - simplified[0][0]) < min_span_sec
           and simplified[0][2] == 'both'):
        simplified[1][0] = simplified[0][0]
        simplified.pop(0)
    # Folding can create new same-label neighbours; merge once more.
    merged: list[list] = []
    for t0, t1, label in simplified:
        if merged and merged[-1][2] == label:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1, label])

    return "\n".join(
        f"{_fmt_ts(t0)}-{_fmt_ts(t1)} {label}" for t0, t1, label in merged
    )


@dataclass
class ChannelVAD:
    """Window-level channel activity plus derived turn-level segments."""

    window_sec: float
    duration_sec: float
    host_active: "object"     # np.ndarray[bool], one entry per window
    remote_active: "object"   # np.ndarray[bool]
    segments: list[tuple[float, float, str]] = field(default_factory=list)

    def shares(self, t0: float, t1: float) -> dict:
        """Speech-time breakdown (seconds) for the interval [t0, t1].

        Returns {'host_only', 'remote_only', 'both', 'speech'} — `speech`
        is the sum of the other three. Windows are counted whole; with
        250 ms windows the quantisation error is negligible against the
        ±few-second accuracy of Gemini timestamps.
        """
        n = len(self.host_active)
        w0 = max(0, int(t0 / self.window_sec))
        w1 = min(n, max(w0, int(round(t1 / self.window_sec))))
        h = self.host_active[w0:w1]
        r = self.remote_active[w0:w1]
        host_only = float((h & ~r).sum()) * self.window_sec
        remote_only = float((r & ~h).sum()) * self.window_sec
        both = float((h & r).sum()) * self.window_sec
        return {
            "host_only": host_only,
            "remote_only": remote_only,
            "both": both,
            "speech": host_only + remote_only + both,
        }

    def host_share(self, t0: float, t1: float) -> Optional[float]:
        """Fraction of speech time in [t0, t1] with the host mic active.

        Overlap ('both') windows count toward the host — the mic channel
        was carrying speech. Returns None if the interval has no speech.
        """
        s = self.shares(t0, t1)
        if s["speech"] <= 0:
            return None
        return (s["host_only"] + s["both"]) / s["speech"]

    def map_text(self) -> str:
        """Compact textual host-speaking map for prompt injection."""
        return render_map_text(self.segments)


def _channel_count(wav_path: Path) -> int:
    r = subprocess.run(
        [FFPROBE_PATH, '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=channels', '-of', 'csv=p=0', str(wav_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:200]}")
    return int(r.stdout.strip())


def _decode_bandpassed(wav_path: Path, channels: int):
    """Decode to band-passed float32 frames (n_samples, channels) at TARGET_SR."""
    import numpy as np
    cmd = [
        FFMPEG_PATH, '-v', 'error', '-i', str(wav_path),
        # highpass/lowpass are per-channel biquads — channel separation survives.
        '-af', f'highpass=f={BANDPASS_LOW_HZ},lowpass=f={BANDPASS_HIGH_HZ}',
        '-ar', str(TARGET_SR),
        '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1',
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode()[:200]}")
    data = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n_frames = len(data) // channels
    return data[: n_frames * channels].reshape(n_frames, channels)


def _window_rms_db(signal, win_samples: int):
    """Per-window RMS in dBFS. Drops the trailing partial window."""
    import numpy as np
    n_win = len(signal) // win_samples
    if n_win == 0:
        return np.empty(0, dtype=np.float32)
    trimmed = signal[: n_win * win_samples].reshape(n_win, win_samples)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    return (20.0 * np.log10(rms + 1e-7)).astype(np.float32)


def _noise_floor_threshold(db) -> float:
    """Onset threshold from the channel's own noise floor."""
    import numpy as np
    floor = float(np.percentile(db, NOISE_PERCENTILE))
    return max(floor + ONSET_MARGIN_DB, ABS_MIN_THRESHOLD_DB)


def _host_threshold(host_db, remote_active) -> float:
    """Onset threshold for the acoustically-open host mic channel.

    Floor+10 dB over-triggers on breathing / key clicks / backchannel (they
    sit well above the mic's noise floor). Host-active windows while the
    remote channel is silent are guaranteed host speech — their median level
    anchors where host speech actually sits, and SPEECH_ANCHOR_MARGIN_DB
    below that separates speech from the contamination band. Falls back to
    the plain floor threshold when there isn't enough one-sided host speech
    to trust the anchor (short or host-silent recordings).
    """
    import numpy as np
    base_thr = _noise_floor_threshold(host_db)
    candidates = host_db[(~remote_active) & (host_db > base_thr)]
    if len(candidates) < MIN_ANCHOR_WINDOWS:
        logger.debug(
            f"channel_vad[host]: only {len(candidates)} one-sided speech "
            f"windows — keeping floor threshold {base_thr:.1f} dB"
        )
        return base_thr
    anchor = float(np.median(candidates))
    thr = max(base_thr, anchor - SPEECH_ANCHOR_MARGIN_DB)
    logger.debug(
        f"channel_vad[host]: speech anchor {anchor:.1f} dB → "
        f"threshold {thr:.1f} dB (floor threshold {base_thr:.1f} dB)"
    )
    return thr


def _activity(db, on_thr: float, label: str):
    """Threshold activity with hysteresis + 3-window median smoothing."""
    import numpy as np
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    off_thr = on_thr - HYSTERESIS_DB
    logger.debug(
        f"channel_vad[{label}]: on {on_thr:.1f} dB / off {off_thr:.1f} dB"
    )
    active = np.zeros(len(db), dtype=bool)
    on = False
    for i, v in enumerate(db):
        on = (v > off_thr) if on else (v > on_thr)
        active[i] = on
    # 3-window median smoothing: kill single-window blips and gaps.
    if len(active) >= 3:
        padded = np.concatenate(([active[0]], active, [active[-1]]))
        stacked = np.stack([padded[:-2], padded[1:-1], padded[2:]])
        active = stacked.sum(axis=0) >= 2
    return active


def _label_windows(host_active, remote_active):
    """Per-window labels: '' (silence) | 'host' | 'remote' | 'both'."""
    labels = []
    for h, r in zip(host_active, remote_active):
        if h and r:
            labels.append('both')
        elif h:
            labels.append('host')
        elif r:
            labels.append('remote')
        else:
            labels.append('')
    return labels


def _segments_from_labels(labels: list[str], window_sec: float
                          ) -> list[tuple[float, float, str]]:
    """Contiguous same-label runs → segments; short silence gaps bridged."""
    bridge_windows = int(BRIDGE_GAP_SEC / window_sec)
    # Bridge: silence runs shorter than BRIDGE_GAP_SEC between identical
    # labels become that label (a breath pause inside one speaker's turn).
    filled = list(labels)
    i = 0
    while i < len(filled):
        if filled[i] == '':
            j = i
            while j < len(filled) and filled[j] == '':
                j += 1
            if (0 < i and j < len(filled) and filled[i - 1] == filled[j]
                    and (j - i) <= bridge_windows):
                for k in range(i, j):
                    filled[k] = filled[j]
            i = j
        else:
            i += 1

    segments: list[tuple[float, float, str]] = []
    run_start = 0
    for i in range(1, len(filled) + 1):
        if i == len(filled) or filled[i] != filled[run_start]:
            label = filled[run_start]
            if label:
                segments.append(
                    (run_start * window_sec, i * window_sec, label)
                )
            run_start = i
    return segments


def compute_channel_vad(wav_path: Path,
                        window_sec: float = WINDOW_SEC) -> Optional[ChannelVAD]:
    """Compute channel-based VAD for a hybrid 3-channel recording.

    Returns None (caller skips channel-based attribution) when:
      - the file has fewer than 3 channels (mono mic-only fallback, stereo),
      - decoding fails, or
      - the audio is shorter than one analysis window.
    Never raises — this feeds a best-effort pipeline stage.
    """
    wav_path = Path(wav_path)
    try:
        channels = _channel_count(wav_path)
    except Exception as e:
        logger.warning(f"channel_vad: ffprobe failed for {wav_path.name}: {e}")
        return None
    if channels < 3:
        logger.info(
            f"channel_vad: {wav_path.name} has {channels} channel(s) — "
            f"not the hybrid mic+system layout, skipping"
        )
        return None

    try:
        import numpy as np
        frames = _decode_bandpassed(wav_path, channels)
        win_samples = int(window_sec * TARGET_SR)
        if len(frames) < win_samples:
            logger.info(f"channel_vad: {wav_path.name} shorter than one window")
            return None

        mic = frames[:, 0]
        # Remote envelope: strongest system channel per sample, so speech on
        # either system channel registers at full level (no dilution by a
        # quieter sibling channel).
        remote = np.max(np.abs(frames[:, 1:]), axis=1)

        host_db = _window_rms_db(mic, win_samples)
        remote_db = _window_rms_db(remote, win_samples)
        # Remote first — the host threshold anchor needs to know which
        # windows carry remote speech.
        remote_active = _activity(
            remote_db, _noise_floor_threshold(remote_db), "remote"
        )
        host_active = _activity(
            host_db, _host_threshold(host_db, remote_active), "host"
        )

        labels = _label_windows(host_active, remote_active)
        segments = _segments_from_labels(labels, window_sec)
        duration = len(host_active) * window_sec
        logger.info(
            f"channel_vad: {wav_path.name} — {len(segments)} segments over "
            f"{duration / 60:.1f} min (host {float(host_active.mean()) * 100:.0f}%, "
            f"remote {float(remote_active.mean()) * 100:.0f}% of windows active)"
        )
        return ChannelVAD(
            window_sec=window_sec,
            duration_sec=duration,
            host_active=host_active,
            remote_active=remote_active,
            segments=segments,
        )
    except Exception as e:
        logger.warning(f"channel_vad: failed for {wav_path.name}: {e}")
        return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python channel_vad.py <recording.wav>")
        sys.exit(1)
    vad = compute_channel_vad(Path(sys.argv[1]))
    if vad is None:
        print("No channel VAD (not a 3-channel hybrid recording).")
        sys.exit(0)
    print(f"Duration: {vad.duration_sec / 60:.1f} min, "
          f"{len(vad.segments)} segments")
    print(vad.map_text())
