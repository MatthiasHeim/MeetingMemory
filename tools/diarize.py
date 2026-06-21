#!/usr/bin/env python3
"""Best-effort local pyannote diarization for Gemini speaker priors.

This module intentionally never raises into callers. The watcher can ask for
an acoustic speaker map and proceed with the legacy Gemini path when pyannote,
torch, torchaudio, MPS, ffmpeg, or the worker process is unavailable.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as pyqueue
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"


def _deps_available() -> tuple[bool, Optional[str]]:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import pyannote.audio  # noqa: F401
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, None


PYANNOTE_AVAILABLE, PYANNOTE_IMPORT_ERROR = _deps_available()


def _fmt_ts(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def slice_segments(segments: list[dict], start: float, end: float) -> list[dict]:
    """Clip diarization segments to [start, end] and shift to chunk time."""
    out: list[dict] = []
    for seg in segments or []:
        try:
            s = max(float(seg.get("start", 0.0)), start)
            e = min(float(seg.get("end", 0.0)), end)
        except Exception:
            continue
        if e <= s:
            continue
        clipped = dict(seg)
        clipped["start"] = s - start
        clipped["end"] = e - start
        clipped["duration"] = e - s
        out.append(clipped)
    return out


def render_map_text(segments: list[dict]) -> str:
    """Render a compact timestamped pyannote prior for the Gemini prompt."""
    lines: list[str] = []
    for seg in segments or []:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            continue
        if end <= start:
            continue
        label = str(seg.get("label") or "SPEAKER_UNKNOWN")
        level = str(seg.get("level") or "unknown")
        confidence = seg.get("confidence")
        conf_text = ""
        if isinstance(confidence, (int, float)):
            conf_text = f" ({float(confidence):.2f})"
        overlap = " overlapped" if seg.get("overlapped") else ""
        lines.append(
            f"{_fmt_ts(start)}-{_fmt_ts(end)} {label} "
            f"confidence={level}{conf_text}{overlap}"
        )
    return "\n".join(lines)


def _decode_to_mono_wav(audio_path: Path, work_dir: Path) -> Path:
    """Decode the exact Gemini input audio to mono WAV for pyannote."""
    wav_path = work_dir / f"{audio_path.stem}_pyannote.wav"
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(audio_path),
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr[:200]}")
    return wav_path


def run_pyannote_diarization(
    audio_path: Path,
    num_speakers: Optional[int] = None,
    timeout_seconds: int = 3600,
    device: str = "",
) -> Optional[list[dict]]:
    """Run local pyannote in a spawned worker and return speaker segments.

    `audio_path` should be the same mono audio Gemini receives. If that is an
    MP3, it is decoded to mono WAV first so pyannote timestamps align with the
    uploaded audio content.
    """
    if not PYANNOTE_AVAILABLE:
        logger.warning(
            f"pyannote unavailable; no diarization prior "
            f"({PYANNOTE_IMPORT_ERROR})"
        )
        return None

    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.warning(f"pyannote: audio file not found: {audio_path}")
        return None

    work_dir = Path(tempfile.mkdtemp(
        prefix=f"meetingmemory_pyannote_{audio_path.stem}_"
    ))
    proc = None
    q = None
    try:
        wav_path = _decode_to_mono_wav(audio_path, work_dir)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        from pyannote_mp_worker import pyannote_proc_entrypoint

        args = {
            "audio_path": str(wav_path),
            "num_speakers": int(num_speakers) if num_speakers else None,
            "device": device,
        }
        proc = ctx.Process(target=pyannote_proc_entrypoint, args=(args, q))
        proc.start()

        deadline = time.time() + timeout_seconds
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"pyannote exceeded {timeout_seconds}s")
            try:
                msg = q.get(timeout=0.25)
            except pyqueue.Empty:
                if proc is not None and not proc.is_alive():
                    raise RuntimeError(
                        f"pyannote worker exited unexpectedly "
                        f"(code {proc.exitcode})"
                    )
                continue

            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            if mtype == "log":
                level = str(msg.get("level") or "info").lower()
                text = str(msg.get("msg") or "")
                log_level = (
                    level
                    if level in ("debug", "info", "warning", "error")
                    else "info"
                )
                getattr(logger, log_level)(f"pyannote: {text}")
            elif mtype == "progress":
                logger.debug(
                    f"pyannote {msg.get('step', '')}: {msg.get('pct', 0)}%"
                )
            elif mtype == "result":
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error") or "pyannote failed")
                segments = msg.get("segments") or []
                normalized: list[dict] = []
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    try:
                        start = float(seg.get("start", 0)) / 1000.0
                        end = float(seg.get("end", 0)) / 1000.0
                    except Exception:
                        continue
                    if end <= start:
                        continue
                    normalized.append({
                        **seg,
                        "start": start,
                        "end": end,
                        "duration": float(seg.get("duration", end - start)),
                    })
                logger.info(
                    f"pyannote: {len(normalized)} diarization segments "
                    f"for {audio_path.name}"
                )
                return normalized or None
    except Exception as e:
        logger.warning(f"pyannote diarization failed for {audio_path.name}: {e}")
        return None
    finally:
        if proc is not None:
            try:
                proc.join(timeout=0.2)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.close()
            except Exception:
                pass
        if q is not None:
            try:
                q.close()
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


def fuse_host_cluster_with_channel_vad(
    diarization_segments: Optional[list[dict]],
    channel_vad,
) -> Optional[list[dict]]:
    """Mark the pyannote cluster that overlaps host-mic windows the most.

    This is intentionally additive for the Phase-2 experiment. It does not
    rewrite transcript speakers; it only annotates the prior labels so Gemini
    can see which anonymous cluster most likely corresponds to the host.
    """
    if not diarization_segments or channel_vad is None:
        return diarization_segments
    try:
        scores: dict[str, float] = {}
        totals: dict[str, float] = {}
        for seg in diarization_segments:
            label = str(seg.get("label") or "")
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if not label or end <= start:
                continue
            shares = channel_vad.shares(start, end)
            host = shares.get("host_only", 0.0) + shares.get("both", 0.0)
            speech = shares.get("speech", 0.0)
            scores[label] = scores.get(label, 0.0) + host
            totals[label] = totals.get(label, 0.0) + speech
        ratios = {
            label: (scores.get(label, 0.0) / total)
            for label, total in totals.items()
            if total > 0
        }
        if not ratios:
            return diarization_segments
        host_label = max(ratios, key=ratios.get)
        if ratios[host_label] < 0.60:
            return diarization_segments
        fused = []
        for seg in diarization_segments:
            out = dict(seg)
            if out.get("label") == host_label:
                out["channel_role"] = "host"
                out["label"] = "HOST_MATTHIAS"
            fused.append(out)
        return fused
    except Exception as e:
        logger.warning(f"pyannote/channel fusion failed: {e}")
        return diarization_segments
