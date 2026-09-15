#!/usr/bin/env python3
"""Durable, source-separated, bounded transcription jobs.

This module deliberately has no watcher, database, calendar, webhook, or
speaker-name dependency.  It turns immutable source tracks into short,
lossless PCM clips and stores enough state to resume only the regions that
did not reach a durable terminal outcome.

The public entry point is :class:`SourceTranscriptionPipeline`.  Its output
is a draft transcript with anonymous, source-local speaker labels.  It does
not claim that tracks share a clock unless the caller supplied a verified
timing basis.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1
PROMPT_VERSION = "transcript-only-v1"
# This is distinct from the response schema version. It covers how that
# schema is submitted to Gemini, so a pre-structured-output response is never
# silently reused by a job that now requires SDK-enforced JSON.
REQUEST_CONTRACT_VERSION = "gemini-structured-output-v1"
DEFAULT_CHUNK_SECONDS = 180.0
MIN_CHUNK_SECONDS = 120.0
MAX_CHUNK_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_JOB_TIMEOUT_SECONDS = 600.0


# The prompt and schema are intentionally small.  Asking for sentiment,
# calendar names, turn percentages, or any semantic analysis makes an ASR
# failure needlessly capable of dropping otherwise useful words.
TRANSCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["outcome", "segments"],
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["transcribed", "no_speech", "uncertain", "failed"],
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start_seconds", "end_seconds", "text", "speaker"],
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "text": {"type": "string"},
                    "speaker": {"type": "string"},
                },
            },
        },
        "uncertain_ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start_seconds", "end_seconds"],
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                },
            },
        },
        "language": {"type": "string"},
    },
}

# Keep the wire-level response contract in one immutable, hashable value. It
# is included in the durable job identity below. ``response_schema`` is a
# supported python-google-genai GenerateContentConfig field (the SDK accepts
# the snake_case alias for its ``responseSchema`` field).
GEMINI_RESPONSE_MIME_TYPE = "application/json"
GEMINI_REQUEST_CONTRACT: dict[str, Any] = {
    "version": REQUEST_CONTRACT_VERSION,
    "response_mime_type": GEMINI_RESPONSE_MIME_TYPE,
    "response_schema": TRANSCRIPTION_SCHEMA,
}

TRANSCRIPT_ONLY_PROMPT = """You transcribe one bounded audio clip from an anonymous source.

Return raw JSON only, matching this exact shape:
{
  "outcome": "transcribed | no_speech | uncertain | failed",
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 0.0,
      "text": "verbatim words in the language spoken",
      "speaker": "speaker_01"
    }
  ],
  "uncertain_ranges": [{"start_seconds": 0.0, "end_seconds": 0.0}],
  "language": "optional language label when clear"
}

Times are relative to the beginning of this clip.  Preserve Swiss German and
mixed languages as spoken.  Use only anonymous labels such as speaker_01;
never infer or return a person's identity.  Do not use attendee information,
calendar information, emotions, sentiment, summaries, actions, or analysis.

Use outcome "no_speech" only when there is no audible human speech.  If there
is speech you cannot make out, return outcome "uncertain" and retain any
useful words or an uncertain range.  Do not omit audio merely because it is
quiet, overlapping, or hard to understand.
"""


class AdapterProtocol(Protocol):
    """Small adapter seam used by the durable worker and by tests."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        start_seconds: float,
        end_seconds: float,
        prompt: str,
        request_timeout_seconds: float,
    ) -> Any:
        """Return a JSON-compatible result or raw JSON text."""


@dataclass(frozen=True)
class SourceInput:
    """A source track supplied by the capture/derivation boundary.

    ``start_seconds`` is a declared source offset.  It is never treated as
    proof that two sources share a clock; that is represented separately by
    ``timing_basis``.
    """

    source_id: str
    path: Path
    start_seconds: float
    timing_basis: str
    input_sha256: str
    duration_seconds: float
    capture_gaps: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SpeechEvidence:
    """An auditable detector result, intentionally without a made-up score."""

    status: str  # speech_detected | no_speech_detected | digital_silence | unknown
    detector: str
    intervals: tuple[tuple[float, float], ...] = ()
    detail: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "detector": self.detector,
            "intervals": [list(interval) for interval in self.intervals],
        }
        if self.detail:
            data["detail"] = self.detail
        return data


class ConservativeSpeechDetector:
    """Use WebRTC VAD when available; otherwise report unknown.

    Energy is not silently substituted for speech detection.  A missing VAD
    dependency is evidence we do not have, not evidence that a region was
    quiet.  The pipeline therefore cannot turn a model-only ``no_speech``
    answer into a terminal quiet result when this detector is unavailable.
    """

    def __init__(self, aggressiveness: int = 2):
        self.aggressiveness = aggressiveness

    def detect(
        self,
        audio_path: Path,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> SpeechEvidence:
        try:
            if self._is_exact_digital_zero(audio_path):
                return SpeechEvidence(
                    status="digital_silence",
                    detector="pcm_exact_digital_zero",
                )
        except Exception:
            # A custom/non-PCM clip simply cannot establish the stronger
            # digital-zero condition.  Continue to VAD or unknown evidence.
            pass
        try:
            import webrtcvad  # type: ignore[import-not-found]
        except ImportError:
            return SpeechEvidence(
                status="unknown",
                detector="webrtcvad_unavailable",
                detail="No independent speech detector is installed.",
            )

        try:
            return self._detect_with_webrtcvad(audio_path, webrtcvad)
        except Exception as exc:  # detector failure must never become quiet
            return SpeechEvidence(
                status="unknown",
                detector="webrtcvad_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _is_exact_digital_zero(audio_path: Path) -> bool:
        try:
            import soundfile as sf  # type: ignore[import-not-found]

            saw_frames = False
            with sf.SoundFile(str(audio_path)) as stream:
                for block in stream.blocks(blocksize=65_536, dtype="float64", always_2d=True):
                    saw_frames = True
                    if block.any():
                        return False
            return saw_frames
        except Exception:
            # The WAV fallback covers a minimal dependency environment.  A
            # parse failure is intentionally handled by caller as "not proven
            # digital silence", never as a quiet result.
            with wave.open(str(audio_path), "rb") as stream:
                frames = stream.readframes(stream.getnframes())
            return bool(frames) and not any(frames)

    def _detect_with_webrtcvad(self, audio_path: Path, webrtcvad: Any) -> SpeechEvidence:
        """Run the actual VAD over 30 ms mono 16 kHz PCM frames.

        The durable clipper writes PCM WAV.  If a caller supplies a custom
        clipper with another format, we conservatively report unknown instead
        of confusing arbitrary bytes for quiet speech.
        """
        try:
            import numpy as np  # type: ignore[import-not-found]
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency environments vary
            raise RuntimeError("numpy and soundfile are required for VAD conversion") from exc

        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if samples.ndim != 2 or samples.shape[0] == 0 or sample_rate not in {8000, 16000, 32000, 48000}:
            raise RuntimeError("unsupported source format/rate for WebRTC VAD")
        # This downmix is used only for independent VAD evidence.  The audio
        # submitted to ASR stays source-separated and retains its channels.
        mono = samples.mean(axis=1)
        frames = np.clip(mono * 32767.0, -32768.0, 32767.0).astype("<i2").tobytes()
        sample_width = 2

        frame_ms = 30
        frame_bytes = int(sample_rate * frame_ms / 1000) * sample_width
        if frame_bytes <= 0 or len(frames) < frame_bytes:
            return SpeechEvidence(
                status="unknown",
                detector="webrtcvad",
                detail="Clip is shorter than one VAD frame.",
            )
        vad = webrtcvad.Vad(self.aggressiveness)
        active: list[tuple[float, float]] = []
        for offset in range(0, len(frames) - frame_bytes + 1, frame_bytes):
            frame = frames[offset:offset + frame_bytes]
            if vad.is_speech(frame, sample_rate):
                start = offset / (sample_rate * sample_width)
                active.append((start, start + frame_ms / 1000))
        # A single 30 ms positive can be a click.  Three frames is a small,
        # explicit and reproducible threshold, not a confidence score.
        if len(active) >= 3:
            return SpeechEvidence(
                status="speech_detected",
                detector=f"webrtcvad_aggressiveness_{self.aggressiveness}",
                intervals=tuple(active),
            )
        return SpeechEvidence(
            status="no_speech_detected",
            detector=f"webrtcvad_aggressiveness_{self.aggressiveness}",
        )


class GeminiTranscriptionAdapter:
    """Optional Google SDK adapter for the transcript-only contract.

    It is created lazily by the pipeline and reads only ``GEMINI_API_KEY``.
    No network action occurs when this module is imported or when a pipeline
    is constructed; an API call happens only when ``run`` reaches a pending
    clip.
    """

    def __init__(self, model: str, request_timeout_seconds: float):
        self.model = model
        self.request_timeout_seconds = request_timeout_seconds

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_id: str,
        start_seconds: float,
        end_seconds: float,
        prompt: str,
        request_timeout_seconds: float,
    ) -> dict[str, str]:
        del source_id, start_seconds, end_seconds  # source identity is not prompt context
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini transcription")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("google-genai is required for Gemini transcription") from exc

        # google-genai's HttpOptions.timeout is milliseconds (verified
        # against installed google-genai 1.60.0), unlike this API's seconds.
        timeout_ms = max(1, int(round(request_timeout_seconds * 1000)))
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        suffix = audio_path.suffix.lower()
        mime_type = {
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".mp3": "audio/mpeg",
        }.get(suffix, "audio/wav")
        uploaded = None
        deadline = time.monotonic() + request_timeout_seconds
        try:
            uploaded = client.files.upload(
                file=str(audio_path), config={"mime_type": mime_type}
            )
            while getattr(getattr(uploaded, "state", None), "name", None) == "PROCESSING":
                if time.monotonic() >= deadline:
                    raise TimeoutError("Gemini file preparation exceeded request deadline")
                time.sleep(min(0.5, max(0.01, deadline - time.monotonic())))
                uploaded = client.files.get(name=uploaded.name)
            state = getattr(getattr(uploaded, "state", None), "name", None)
            if state != "ACTIVE":
                raise RuntimeError(f"Gemini upload did not become ACTIVE: {state}")
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, uploaded],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type=GEMINI_RESPONSE_MIME_TYPE,
                    response_schema=TRANSCRIPTION_SCHEMA,
                ),
            )
            return {"raw_text": response.text or ""}
        finally:
            if uploaded is not None:
                with contextlib.suppress(Exception):
                    client.files.delete(name=uploaded.name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(
    path: Path,
    *,
    deadline: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if deadline is not None and clock() >= deadline:
                raise JobDeadlineExceeded("overall_job_deadline_exceeded during source hashing")
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Keep raw attempts durable even if an adapter returned an odd object."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return {"non_json_response_repr": repr(value)}


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace a JSON file and fsync the file when supported."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        with contextlib.suppress(OSError):
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


@contextlib.contextmanager
def _exclusive_job_lock(
    path: Path,
    *,
    deadline: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
):
    """Serialize same-hash runs from a watcher and a manual recovery command."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            import fcntl  # macOS/Linux durable worker hosts
        except ImportError as exc:  # pragma: no cover - Windows is unsupported here
            raise RuntimeError("durable transcription jobs require an advisory file lock") from exc
        if deadline is None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        else:
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if clock() >= deadline:
                        raise JobDeadlineExceeded("overall_job_deadline_exceeded waiting for job lock")
                    time.sleep(min(0.05, max(0.001, deadline - clock())))
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _anonymous_source_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ValueError(
            "source_id must be an anonymous lowercase opaque token "
            "([a-z0-9_-]); do not put a person's name in it"
        )
    return value


def _timing_is_verified(timing_basis: str) -> bool:
    # Do not infer synchronization from a numeric offset.  Only an explicit
    # capture/common clock claim qualifies; legacy sample-zero offsets remain
    # unverified even when their declared offset happens to be zero.
    return timing_basis.strip().lower() in {
        "host_clock",
        "common_clock",
        "verified_common_clock",
        "capture_clock_verified",
    }


def _normalise_range(item: Any, *, source_id: str, label: str) -> dict[str, Any]:
    if isinstance(item, dict):
        item_source = item.get("source_id", source_id)
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        reason = item.get("reason", label)
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        item_source, start, end, reason = source_id, item[0], item[1], label
    else:
        raise ValueError(f"{label} must contain start_seconds and end_seconds")
    if item_source != source_id:
        raise ValueError(f"{label} source_id does not match source {source_id}")
    start_number = _finite_nonnegative(start, f"{label} start_seconds")
    end_number = _finite_nonnegative(end, f"{label} end_seconds")
    if end_number <= start_number:
        raise ValueError(f"{label} end_seconds must be after start_seconds")
    return {
        "source_id": source_id,
        "start_seconds": start_number,
        "end_seconds": end_number,
        "reason": str(reason or label),
    }


def _range_overlap(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> Optional[tuple[float, float]]:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return (start, end) if end > start else None


def _adapter_worker(connection: Any, adapter: Any, kwargs: dict[str, Any]) -> None:
    """Top-level process target so a timed-out request can be reaped safely."""
    try:
        # A dedicated process group lets the parent terminate an adapter's
        # descendants too, instead of leaving an SDK/helper child behind.
        with contextlib.suppress(OSError):
            os.setsid()
        if hasattr(adapter, "transcribe"):
            result = adapter.transcribe(**kwargs)
        else:
            result = adapter(**kwargs)
        connection.send({"ok": True, "result": result})
    except BaseException as exc:  # send a safe error across process boundary
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12),
            }
        )
    finally:
        connection.close()


class AdapterCallError(RuntimeError):
    def __init__(self, message: str, *, timeout: bool = False):
        super().__init__(message)
        self.timeout = timeout


class JobDeadlineExceeded(TimeoutError):
    """The bounded run spent its total budget before the next safe step."""


class SourceTranscriptionPipeline:
    """Run source-local transcription clips with durable, bounded recovery.

    ``request_timeout_seconds`` applies to a single adapter call.  It is
    passed to the Gemini SDK as an HTTP timeout and independently enforced by
    a child process.  ``job_timeout_seconds`` is a wall-clock deadline for
    the whole :meth:`run` invocation, across every source, clip, and retry.
    A later invocation resumes only chunks left failed by that deadline.
    """

    def __init__(
        self,
        state_dir: Path | str,
        model: str = "gemini-2.5-flash",
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        *,
        adapter: Optional[AdapterProtocol | Callable[..., Any]] = None,
        max_attempts: int = 2,
        clipper: Optional[Callable[[Path, Path, float, float], Path | None]] = None,
        duration_probe: Optional[Callable[[Path], float]] = None,
        speech_detector: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.state_dir = Path(state_dir)
        self.model = str(model)
        if not self.model:
            raise ValueError("model is required")
        self.request_timeout_seconds = _finite_nonnegative(
            request_timeout_seconds, "request_timeout_seconds"
        )
        self.job_timeout_seconds = _finite_nonnegative(
            job_timeout_seconds, "job_timeout_seconds"
        )
        if self.request_timeout_seconds <= 0 or self.job_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds and job_timeout_seconds must be positive")
        self.chunk_seconds = _finite_nonnegative(chunk_seconds, "chunk_seconds")
        if not MIN_CHUNK_SECONDS <= self.chunk_seconds <= MAX_CHUNK_SECONDS:
            raise ValueError(
                f"chunk_seconds must stay between {MIN_CHUNK_SECONDS:g} and "
                f"{MAX_CHUNK_SECONDS:g} seconds"
            )
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = max_attempts
        self.adapter = adapter
        self.clipper = clipper
        self.duration_probe = duration_probe
        self.speech_detector = speech_detector or ConservativeSpeechDetector()
        self.clock = clock

    def run(
        self,
        sources: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        session_id: str,
        retry_failed: bool = True,
        capture_gaps: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Process ``sources`` and return a source-separated transcript payload.

        Every source must supply ``source_id``, ``path``, ``start_seconds``,
        and ``timing_basis``.  ``capture_gaps`` optionally carries source-local
        gap evidence from the capture boundary.  It is recorded as a known
        gap rather than silently treated as tail silence.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required")
        # The run budget starts before hashing/probing input.  Preparation is
        # local work, but it can still be expensive on a damaged or very large
        # recording and must not silently sit outside an advertised deadline.
        deadline = self.clock() + self.job_timeout_seconds
        specs, absent_source_gaps = self._prepare_sources(
            sources, capture_gaps or [], deadline=deadline
        )
        identity = self._job_identity(specs, absent_source_gaps)
        job_key = _sha256_text(_stable_json(identity))
        job_dir = self.state_dir / "jobs" / job_key
        manifest_path = job_dir / "manifest.json"
        with _exclusive_job_lock(job_dir / ".lock", deadline=deadline, clock=self.clock):
            return self._run_locked(
                specs=specs,
                absent_source_gaps=absent_source_gaps,
                identity=identity,
                job_key=job_key,
                job_dir=job_dir,
                manifest_path=manifest_path,
                session_id=session_id,
                retry_failed=retry_failed,
                deadline=deadline,
            )

    def _run_locked(
        self,
        *,
        specs: list[SourceInput],
        absent_source_gaps: list[dict[str, Any]],
        identity: dict[str, Any],
        job_key: str,
        job_dir: Path,
        manifest_path: Path,
        session_id: str,
        retry_failed: bool,
        deadline: float,
    ) -> dict[str, Any]:
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if manifest.get("identity") != identity:
                raise RuntimeError("durable job-key collision with different input identity")
            sessions = manifest.setdefault("session_ids", [])
            if session_id not in sessions:
                sessions.append(session_id)
        else:
            manifest = self._new_manifest(
                identity, specs, absent_source_gaps, job_key, session_id
            )

        self._recover_interrupted_chunks(manifest, job_dir)
        self._revalidate_terminal_results(manifest, job_dir)
        manifest["status"] = "running"
        manifest["updated_at"] = _utc_now()
        atomic_write_json(manifest_path, manifest)

        deadline_hit = False
        source_paths = {spec.source_id: spec.path for spec in specs}
        # Keep both tracks useful when a finite job budget expires.  A
        # source-local round robin yields mic0/system0/mic1/system1 rather
        # than spending a whole 10-minute window on one track first.  It does
        # not assert a cross-source clock or run requests concurrently.
        for source, chunk in self._round_robin_chunks(manifest, retry_failed):
            if self.clock() >= deadline:
                deadline_hit = True
                self._mark_deadline_chunk(chunk)
                continue
            self._process_chunk(
                manifest=manifest,
                manifest_path=manifest_path,
                job_dir=job_dir,
                source=source,
                source_path=source_paths[source["source_id"]],
                chunk=chunk,
                deadline=deadline,
            )
            if self.clock() >= deadline:
                deadline_hit = True

        if deadline_hit:
            for source in manifest["sources"]:
                for chunk in source["chunks"]:
                    if chunk["status"] in {"pending", "running"}:
                        self._mark_deadline_chunk(chunk)
        manifest["status"] = self._manifest_status(manifest)
        manifest["updated_at"] = _utc_now()
        atomic_write_json(manifest_path, manifest)
        return self._build_payload(manifest, job_dir)

    @staticmethod
    def _round_robin_chunks(
        manifest: dict[str, Any], retry_failed: bool
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Return remaining chunks in source-local, fair round-robin order."""
        queues: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for source in manifest["sources"]:
            eligible = []
            for chunk in sorted(
                source["chunks"], key=lambda item: (item["source_start_seconds"], item["index"])
            ):
                status = chunk["status"]
                if status in {"transcribed", "no_speech", "uncertain"}:
                    continue
                if status == "failed" and not retry_failed:
                    continue
                eligible.append(chunk)
            queues.append((source, eligible))
        scheduled: list[tuple[dict[str, Any], dict[str, Any]]] = []
        position = 0
        while True:
            added = False
            for source, queue in queues:
                if position < len(queue):
                    scheduled.append((source, queue[position]))
                    added = True
            if not added:
                break
            position += 1
        return scheduled

    def _prepare_sources(
        self,
        sources: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        global_capture_gaps: list[dict[str, Any]],
        *,
        deadline: Optional[float] = None,
    ) -> tuple[list[SourceInput], list[dict[str, Any]]]:
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("sources must be a non-empty list")
        by_source_gaps: dict[str, list[dict[str, Any]]] = {}
        for gap in global_capture_gaps:
            if not isinstance(gap, dict) or "source_id" not in gap:
                raise ValueError("capture_gaps entries must include source_id")
            source_id = _anonymous_source_id(gap["source_id"])
            by_source_gaps.setdefault(source_id, []).append(gap)

        prepared: list[SourceInput] = []
        seen: set[str] = set()
        for raw in sources:
            self._require_before_deadline(deadline, "source preparation")
            if not isinstance(raw, dict):
                raise ValueError("each source must be an object")
            missing = {"source_id", "path", "start_seconds", "timing_basis"} - set(raw)
            if missing:
                raise ValueError(f"source missing required fields: {', '.join(sorted(missing))}")
            source_id = _anonymous_source_id(raw["source_id"])
            if source_id in seen:
                raise ValueError(f"duplicate source_id: {source_id}")
            seen.add(source_id)
            path = Path(raw["path"])
            if not path.is_file():
                raise FileNotFoundError(f"source path does not exist: {path}")
            start_seconds = _finite_nonnegative(raw["start_seconds"], "source start_seconds")
            timing_basis = raw["timing_basis"]
            if not isinstance(timing_basis, str) or not timing_basis.strip():
                raise ValueError("timing_basis must be a non-empty string")
            source_gaps = list(raw.get("capture_gaps") or raw.get("known_capture_gaps") or [])
            source_gaps.extend(by_source_gaps.get(source_id, []))
            normalised_gaps = tuple(
                _normalise_range(gap, source_id=source_id, label="capture_gap")
                for gap in source_gaps
            )
            duration = self._get_duration(path, deadline=deadline)
            if duration <= 0:
                raise ValueError(f"source has no measurable audio duration: {path}")
            prepared.append(
                SourceInput(
                    source_id=source_id,
                    path=path,
                    start_seconds=start_seconds,
                    timing_basis=timing_basis,
                    input_sha256=_sha256_file(path, deadline=deadline, clock=self.clock),
                    duration_seconds=duration,
                    capture_gaps=normalised_gaps,
                )
            )
        # A native capture can genuinely contain only one recoverable track.
        # Preserve a declared gap for the absent track as evidence of partial
        # capture instead of rejecting the usable source or inventing silence.
        absent_source_gaps: list[dict[str, Any]] = []
        known_duration = max((spec.duration_seconds for spec in prepared), default=0.0)
        for source_id in sorted(set(by_source_gaps) - seen):
            for gap in by_source_gaps[source_id]:
                if (
                    "start_seconds" not in gap
                    and "end_seconds" not in gap
                    and str(gap.get("reason") or "") == "source_missing"
                    and known_duration > 0
                ):
                    # A capture manifest may know a track is entirely absent
                    # without spelling out its range.  The supplied source
                    # timeline bounds the absence; retain that derivation as
                    # explicit source-missing evidence, never as silence.
                    gap = {
                        **gap,
                        "start_seconds": 0.0,
                        "end_seconds": known_duration,
                    }
                absent_source_gaps.append(
                    _normalise_range(gap, source_id=source_id, label="missing_source_capture_gap")
                )
        return prepared, absent_source_gaps

    def _require_before_deadline(self, deadline: Optional[float], stage: str) -> None:
        if deadline is not None and self.clock() >= deadline:
            raise JobDeadlineExceeded(f"overall_job_deadline_exceeded during {stage}")

    def _get_duration(self, path: Path, *, deadline: Optional[float] = None) -> float:
        self._require_before_deadline(deadline, "source duration probe")
        if self.duration_probe is not None:
            duration = _finite_nonnegative(self.duration_probe(path), "audio duration")
            self._require_before_deadline(deadline, "source duration probe")
            return duration
        try:
            import soundfile as sf  # type: ignore[import-not-found]

            info = sf.info(str(path))
            if info.samplerate > 0 and info.frames >= 0:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass
        try:
            with wave.open(str(path), "rb") as stream:
                if stream.getframerate() > 0:
                    return stream.getnframes() / stream.getframerate()
        except Exception:
            pass
        ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=(
                    min(30, max(0.01, deadline - self.clock()))
                    if deadline is not None
                    else 30
                ),
            )
            return _finite_nonnegative(result.stdout.strip(), "audio duration")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise RuntimeError(f"could not determine duration for {path}") from exc

    def _job_identity(
        self, specs: list[SourceInput], absent_source_gaps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        identity = {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _sha256_text(TRANSCRIPT_ONLY_PROMPT),
            "schema_sha256": _sha256_text(_stable_json(TRANSCRIPTION_SCHEMA)),
            "request_contract_version": REQUEST_CONTRACT_VERSION,
            "request_contract_sha256": _sha256_text(
                _stable_json(GEMINI_REQUEST_CONTRACT)
            ),
            "model": self.model,
            "chunk_seconds": self.chunk_seconds,
            "sources": [
                {
                    "source_id": spec.source_id,
                    "input_sha256": spec.input_sha256,
                    "start_seconds": spec.start_seconds,
                    "timing_basis": spec.timing_basis,
                    "duration_seconds": spec.duration_seconds,
                    "capture_gaps": list(spec.capture_gaps),
                }
                for spec in specs
            ],
        }
        # Keep the initial v1 identity stable for an ordinary all-source job;
        # only a real absent-source declaration needs a distinct job key.
        if absent_source_gaps:
            identity["absent_source_capture_gaps"] = absent_source_gaps
        return identity

    def _new_manifest(
        self,
        identity: dict[str, Any],
        specs: list[SourceInput],
        absent_source_gaps: list[dict[str, Any]],
        job_key: str,
        session_id: str,
    ) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        for spec in specs:
            chunks: list[dict[str, Any]] = []
            start = 0.0
            index = 0
            while start < spec.duration_seconds - 1e-9:
                end = min(spec.duration_seconds, start + self.chunk_seconds)
                overlaps = [
                    gap
                    for gap in spec.capture_gaps
                    if _range_overlap(start, end, gap["start_seconds"], gap["end_seconds"])
                ]
                chunks.append(
                    {
                        "chunk_id": f"{spec.source_id}-{index:04d}",
                        "index": index,
                        "source_start_seconds": start,
                        "source_end_seconds": end,
                        "duration_seconds": end - start,
                        "clip_path": f"clips/{spec.source_id}/chunk-{index:04d}.wav",
                        "status": "pending",
                        "attempts": [],
                        "result_path": None,
                        "speech_evidence": None,
                        "review_flags": [],
                        "retryable": False,
                        "error": None,
                        "known_capture_gaps": overlaps,
                    }
                )
                start = end
                index += 1
            sources.append(
                {
                    "source_id": spec.source_id,
                    "input_sha256": spec.input_sha256,
                    "duration_seconds": spec.duration_seconds,
                    "declared_start_seconds": spec.start_seconds,
                    "timing_basis": spec.timing_basis,
                    "timing_verified": _timing_is_verified(spec.timing_basis),
                    "capture_gaps": list(spec.capture_gaps),
                    "chunks": chunks,
                }
            )
        now = _utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "job_key": job_key,
            "identity": identity,
            "session_ids": [session_id],
            "created_at": now,
            "updated_at": now,
            "status": "pending",
            "sources": sources,
            "absent_source_capture_gaps": absent_source_gaps,
        }

    def _recover_interrupted_chunks(self, manifest: dict[str, Any], job_dir: Path) -> None:
        changed = False
        for source in manifest.get("sources", []):
            for chunk in source.get("chunks", []):
                if chunk.get("status") == "running":
                    # The manifest records a result path before writing the
                    # result file.  If a process died after that atomic file
                    # write but before its final manifest update, promote the
                    # durable result rather than spend another model request.
                    result_path = chunk.get("result_path")
                    if result_path:
                        try:
                            result = _read_json(job_dir / result_path)
                            outcome = result.get("outcome")
                            if outcome in {"transcribed", "no_speech", "uncertain"}:
                                chunk["status"] = outcome
                                chunk["retryable"] = False
                                chunk["error"] = result.get("reason")
                                chunk["review_flags"] = list(result.get("review_flags") or [])
                                changed = True
                                continue
                        except Exception:
                            pass
                    chunk["status"] = "failed"
                    chunk["retryable"] = True
                    chunk["error"] = "interrupted before durable terminal result"
                    flags = chunk.setdefault("review_flags", [])
                    if "interrupted" not in flags:
                        flags.append("interrupted")
                    changed = True
        if changed:
            manifest["status"] = "interrupted"

    def _revalidate_terminal_results(self, manifest: dict[str, Any], job_dir: Path) -> None:
        """Turn an unreadable terminal cache entry back into durable retry work.

        This happens before scheduling while the job lock is held.  Mutating
        only the rendered payload would leave a manifest marked transcribed,
        causing every later resume to skip the lost region forever.
        """
        for source in manifest.get("sources", []):
            for chunk in source.get("chunks", []):
                if chunk.get("status") not in {"transcribed", "no_speech", "uncertain"}:
                    continue
                result_path = chunk.get("result_path")
                try:
                    result = _read_json(job_dir / result_path) if result_path else None
                    if not isinstance(result, dict) or result.get("outcome") != chunk.get("status"):
                        raise ValueError("terminal result outcome does not match manifest")
                except Exception:
                    chunk["status"] = "failed"
                    chunk["retryable"] = True
                    chunk["error"] = "durable_segment_result_unreadable"
                    self._add_flag(chunk, "durable_segment_result_unreadable")

    def _process_chunk(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
        job_dir: Path,
        source: dict[str, Any],
        source_path: Path,
        chunk: dict[str, Any],
        deadline: float,
    ) -> None:
        # A fully known missing-capture region is not source silence.  Keep it
        # as a terminal uncertain outcome without asking a model to certify it.
        if self._chunk_fully_known_gap(chunk):
            chunk["status"] = "running"
            chunk["retryable"] = False
            chunk["error"] = "known_capture_gap"
            self._add_flag(chunk, "known_capture_gap")
            result_path = self._result_path(job_dir, source["source_id"], chunk)
            result = {
                "outcome": "uncertain",
                "segments": [],
                "speech_evidence": SpeechEvidence(
                    status="unknown", detector="capture_gap_metadata"
                ).as_dict(),
                "review_flags": list(chunk["review_flags"]),
                "reason": "known_capture_gap",
            }
            chunk["result_path"] = str(result_path.relative_to(job_dir))
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(result_path, result)
            chunk["status"] = "uncertain"
            atomic_write_json(manifest_path, manifest)
            return

        chunk["status"] = "running"
        chunk["retryable"] = False
        chunk["error"] = None
        atomic_write_json(manifest_path, manifest)
        try:
            clip_path = self._ensure_clip(job_dir, source_path, chunk, deadline=deadline)
        except Exception as exc:
            self._record_failed_attempt(
                manifest, manifest_path, job_dir, source, chunk, exc, retryable=True
            )
            if "overall_job_deadline_exceeded" in str(exc):
                self._add_flag(chunk, "overall_job_deadline_exceeded")
                atomic_write_json(manifest_path, manifest)
            return

        evidence = self._detect_speech(
            clip_path,
            start_seconds=float(chunk["source_start_seconds"]),
            end_seconds=float(chunk["source_end_seconds"]),
        )
        chunk["speech_evidence"] = evidence.as_dict()
        atomic_write_json(manifest_path, manifest)

        attempts_this_run = 0
        while attempts_this_run < self.max_attempts:
            if self.clock() >= deadline:
                self._mark_deadline_chunk(chunk)
                atomic_write_json(manifest_path, manifest)
                return
            attempts_this_run += 1
            attempt_number = len(chunk["attempts"]) + 1
            attempt_metadata: dict[str, Any] = {
                "attempt": attempt_number,
                "started_at": _utc_now(),
                "model": self.model,
                "input_sha256": source["input_sha256"],
                "source_range_seconds": [
                    float(chunk["source_start_seconds"]),
                    float(chunk["source_end_seconds"]),
                ],
                "request_timeout_seconds": self.request_timeout_seconds,
                "request_contract_version": manifest["identity"][
                    "request_contract_version"
                ],
                "request_contract_sha256": manifest["identity"][
                    "request_contract_sha256"
                ],
            }
            attempt_started = self.clock()
            try:
                raw_response = self._call_adapter_bounded(
                    clip_path=clip_path,
                    source_id=source["source_id"],
                    start_seconds=float(chunk["source_start_seconds"]),
                    end_seconds=float(chunk["source_end_seconds"]),
                    deadline=deadline,
                )
                attempt_metadata["raw_response"] = _json_safe(raw_response)
                attempt_metadata["completed_at"] = _utc_now()
                attempt_metadata["elapsed_seconds"] = max(0.0, self.clock() - attempt_started)
                attempt_metadata["raw_response_bytes"] = len(
                    _stable_json(attempt_metadata["raw_response"]).encode("utf-8")
                )
                attempt_path = self._attempt_path(job_dir, source["source_id"], chunk, attempt_number)
                atomic_write_json(attempt_path, attempt_metadata)
                chunk["attempts"].append(
                    {
                        "attempt": attempt_number,
                        "status": "received",
                        "raw_attempt_path": str(attempt_path.relative_to(job_dir)),
                        "at": attempt_metadata["completed_at"],
                    }
                )
                parsed = self._normalise_response(
                    raw_response,
                    source_id=source["source_id"],
                    declared_start_seconds=float(source["declared_start_seconds"]),
                    timing_basis=source["timing_basis"],
                    timing_verified=bool(source["timing_verified"]),
                    chunk=chunk,
                    evidence=evidence,
                )
            except AdapterCallError as exc:
                attempt_metadata["elapsed_seconds"] = max(0.0, self.clock() - attempt_started)
                self._record_failed_attempt(
                    manifest,
                    manifest_path,
                    job_dir,
                    source,
                    chunk,
                    exc,
                    retryable=True,
                    attempt_number=attempt_number,
                    attempt_metadata=attempt_metadata,
                )
                if attempts_this_run < self.max_attempts and self.clock() < deadline:
                    continue
                return
            except Exception as exc:
                attempt_metadata["elapsed_seconds"] = max(0.0, self.clock() - attempt_started)
                # Invalid JSON/schema has no locatable useful words.  Retry
                # this region, retaining the raw response for inspection.
                self._record_failed_attempt(
                    manifest,
                    manifest_path,
                    job_dir,
                    source,
                    chunk,
                    exc,
                    retryable=True,
                    attempt_number=attempt_number,
                    attempt_metadata=attempt_metadata,
                )
                if attempts_this_run < self.max_attempts and self.clock() < deadline:
                    continue
                return

            result_path = self._result_path(job_dir, source["source_id"], chunk)
            # Make the result discoverable before writing it. If the process
            # dies after the atomic result write, _recover_interrupted_chunks
            # can promote it without another API request.
            chunk["result_path"] = str(result_path.relative_to(job_dir))
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(result_path, parsed)
            chunk["status"] = parsed["outcome"]
            chunk["retryable"] = False
            chunk["error"] = parsed.get("reason")
            chunk["review_flags"] = parsed["review_flags"]
            atomic_write_json(manifest_path, manifest)
            return

    def _chunk_fully_known_gap(self, chunk: dict[str, Any]) -> bool:
        start = float(chunk["source_start_seconds"])
        end = float(chunk["source_end_seconds"])
        for gap in chunk.get("known_capture_gaps", []):
            if gap["start_seconds"] <= start and gap["end_seconds"] >= end:
                return True
        return False

    def _ensure_clip(
        self,
        job_dir: Path,
        source_path: Path,
        chunk: dict[str, Any],
        *,
        deadline: Optional[float] = None,
    ) -> Path:
        clip_path = job_dir / chunk["clip_path"]
        if clip_path.is_file() and clip_path.stat().st_size > 0:
            return clip_path
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        start = float(chunk["source_start_seconds"])
        end = float(chunk["source_end_seconds"])
        if deadline is not None and self.clock() >= deadline:
            raise RuntimeError("overall_job_deadline_exceeded before clip extraction")
        if self.clipper is not None:
            result = self.clipper(source_path, clip_path, start, end)
            candidate = Path(result) if result is not None else clip_path
            if candidate != clip_path:
                # Test/custom clippers may write elsewhere.  Copy through a
                # temporary file so the durable path is still atomic.
                temporary = clip_path.with_suffix(".copying.wav")
                shutil.copyfile(candidate, temporary)
                os.replace(temporary, clip_path)
        else:
            remaining = None if deadline is None else max(0.01, deadline - self.clock())
            self._extract_lossless_clip(
                source_path, clip_path, start, end, timeout_seconds=remaining
            )
        if deadline is not None and self.clock() >= deadline:
            raise RuntimeError("overall_job_deadline_exceeded during clip extraction")
        if not clip_path.is_file() or clip_path.stat().st_size == 0:
            raise RuntimeError("clipper did not produce a non-empty lossless clip")
        return clip_path

    @staticmethod
    def _extract_lossless_clip(
        source_path: Path,
        clip_path: Path,
        start_seconds: float,
        end_seconds: float,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Create a bounded lossless source clip without channel mixing.

        For the native FLOAT WAV and legacy FLAC sources, soundfile preserves
        the original sample representation while slicing exact frame ranges.
        A generic ffmpeg fallback still emits uncompressed PCM rather than a
        lossy MP3/AAC derivative.  The original source is never modified.
        """
        try:
            SourceTranscriptionPipeline._extract_soundfile_lossless_clip(
                source_path, clip_path, start_seconds, end_seconds
            )
            return
        except Exception as exc:
            logger.warning("soundfile lossless clip extraction failed (%s); using PCM fallback", exc)
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        temporary = clip_path.with_name(f".{clip_path.stem}.partial.wav")
        duration = end_seconds - start_seconds
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            f"{start_seconds:.6f}",
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            str(temporary),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=(
                    min(max(60, int(math.ceil(duration)) + 30), timeout_seconds)
                    if timeout_seconds is not None
                    else max(60, int(math.ceil(duration)) + 30)
                ),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"lossless clip extraction failed: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(f"lossless clip extraction failed: {result.stderr[:500]}")
        os.replace(temporary, clip_path)

    @staticmethod
    def _extract_soundfile_lossless_clip(
        source_path: Path,
        clip_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        import soundfile as sf  # type: ignore[import-not-found]

        temporary = clip_path.with_name(f".{clip_path.stem}.partial.wav")
        with sf.SoundFile(str(source_path), "r") as source:
            if source.samplerate <= 0 or source.channels <= 0:
                raise RuntimeError("source has invalid sample format")
            subtype = source.subtype
            dtype_by_subtype = {
                "PCM_16": "int16",
                "PCM_24": "int32",
                "PCM_32": "int32",
                "FLOAT": "float32",
                "DOUBLE": "float64",
            }
            if subtype not in dtype_by_subtype:
                raise RuntimeError(f"unsupported lossless source subtype: {subtype}")
            start_frame = min(source.frames, max(0, round(start_seconds * source.samplerate)))
            end_frame = min(source.frames, max(start_frame, round(end_seconds * source.samplerate)))
            if end_frame <= start_frame:
                raise RuntimeError("requested clip has no source frames")
            source.seek(start_frame)
            with sf.SoundFile(
                str(temporary),
                "w",
                samplerate=source.samplerate,
                channels=source.channels,
                format="WAV",
                subtype=subtype,
            ) as destination:
                remaining = end_frame - start_frame
                while remaining:
                    block = source.read(
                        min(65_536, remaining),
                        dtype=dtype_by_subtype[subtype],
                        always_2d=True,
                    )
                    if len(block) == 0:
                        raise RuntimeError("source ended before requested clip boundary")
                    destination.write(block)
                    remaining -= len(block)
        os.replace(temporary, clip_path)

    def _detect_speech(
        self, clip_path: Path, *, start_seconds: float, end_seconds: float
    ) -> SpeechEvidence:
        detector = self.speech_detector
        try:
            if hasattr(detector, "detect"):
                raw = detector.detect(
                    clip_path, start_seconds=start_seconds, end_seconds=end_seconds
                )
            else:
                raw = detector(clip_path, start_seconds=start_seconds, end_seconds=end_seconds)
            return self._coerce_speech_evidence(raw)
        except Exception as exc:
            return SpeechEvidence(
                status="unknown",
                detector="speech_detector_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _coerce_speech_evidence(raw: Any) -> SpeechEvidence:
        if isinstance(raw, SpeechEvidence):
            return raw
        if isinstance(raw, str):
            raw = {"status": raw}
        if not isinstance(raw, dict):
            raise ValueError("speech detector must return SpeechEvidence, string, or object")
        status = raw.get("status")
        aliases = {
            "speech": "speech_detected",
            "no_speech": "no_speech_detected",
            "quiet": "no_speech_detected",
        }
        status = aliases.get(status, status)
        if status not in {
            "speech_detected",
            "no_speech_detected",
            "digital_silence",
            "unknown",
        }:
            raise ValueError("invalid speech detector status")
        intervals: list[tuple[float, float]] = []
        for item in raw.get("intervals", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("speech detector interval must be [start, end]")
            start = _finite_nonnegative(item[0], "speech interval start")
            end = _finite_nonnegative(item[1], "speech interval end")
            if end <= start:
                raise ValueError("speech interval must have positive duration")
            intervals.append((start, end))
        return SpeechEvidence(
            status=status,
            detector=str(raw.get("detector") or "custom_speech_detector"),
            intervals=tuple(intervals),
            detail=raw.get("detail"),
        )

    def _call_adapter_bounded(
        self,
        *,
        clip_path: Path,
        source_id: str,
        start_seconds: float,
        end_seconds: float,
        deadline: float,
    ) -> Any:
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise AdapterCallError("overall job deadline exceeded", timeout=True)
        call_timeout = min(self.request_timeout_seconds, remaining)
        adapter: Any = self.adapter or GeminiTranscriptionAdapter(
            self.model, self.request_timeout_seconds
        )
        context = self._multiprocessing_context(adapter)
        receiver, sender = context.Pipe(duplex=False)
        kwargs = {
            "audio_path": clip_path,
            "source_id": source_id,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "prompt": TRANSCRIPT_ONLY_PROMPT,
            "request_timeout_seconds": self.request_timeout_seconds,
        }
        worker = context.Process(
            target=_adapter_worker, args=(sender, adapter, kwargs), daemon=False
        )
        worker.start()
        sender.close()
        response: Optional[dict[str, Any]] = None
        end = self.clock() + call_timeout
        try:
            while self.clock() < end:
                wait = min(0.1, max(0.0, end - self.clock()))
                if receiver.poll(wait):
                    response = receiver.recv()
                    break
                if not worker.is_alive():
                    break
            if response is None and receiver.poll():
                response = receiver.recv()
            if response is None:
                if worker.is_alive():
                    self._terminate_worker(worker)
                    if self.clock() >= deadline:
                        raise AdapterCallError("overall job deadline exceeded", timeout=True)
                    raise AdapterCallError("request deadline exceeded", timeout=True)
                raise AdapterCallError(
                    f"adapter worker exited without a response (exit code {worker.exitcode})"
                )
            if not response.get("ok"):
                raise AdapterCallError(
                    f"adapter {response.get('error_type', 'error')}: {response.get('error', '')}"
                )
            return response.get("result")
        finally:
            receiver.close()
            if worker.is_alive():
                self._terminate_worker(worker)
            worker.join(timeout=0.2)

    def _multiprocessing_context(self, adapter: Any) -> multiprocessing.context.BaseContext:
        """Choose a clean interpreter for production Gemini requests.

        A watcher can have CoreAudio/Torch threads active.  Forking it into a
        live SDK request risks inheriting locks and device state, so the
        built-in Gemini adapter always uses spawn.  Fork is reserved for an
        explicitly injected local/test adapter on POSIX, where preserving a
        small in-process fake is useful and no production SDK state exists.
        """
        if isinstance(adapter, GeminiTranscriptionAdapter):
            return multiprocessing.get_context("spawn")
        if "fork" in multiprocessing.get_all_start_methods():
            return multiprocessing.get_context("fork")
        return multiprocessing.get_context("spawn")

    @staticmethod
    def _terminate_worker(worker: multiprocessing.Process) -> None:
        pid = worker.pid
        if not pid:
            return
        try:
            group_id = os.getpgid(pid)
            if group_id == pid:
                os.killpg(group_id, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        worker.join(timeout=1.0)
        if worker.is_alive():
            try:
                group_id = os.getpgid(pid)
                if group_id == pid:
                    os.killpg(group_id, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            worker.join(timeout=1.0)

    def _normalise_response(
        self,
        raw_response: Any,
        *,
        source_id: str,
        declared_start_seconds: float,
        timing_basis: str,
        timing_verified: bool,
        chunk: dict[str, Any],
        evidence: SpeechEvidence,
    ) -> dict[str, Any]:
        data = self._parse_raw_response(raw_response)
        requested_outcome = data.get("outcome")
        if requested_outcome not in {"transcribed", "no_speech", "uncertain", "failed"}:
            requested_outcome = "transcribed" if data.get("segments") else "uncertain"
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("response segments must be a list")
        valid_segments: list[dict[str, Any]] = []
        flags: list[str] = list(chunk.get("review_flags") or [])
        malformed = False
        for position, raw_segment in enumerate(raw_segments):
            parsed = self._normalise_segment(
                raw_segment,
                source_id=source_id,
                declared_start_seconds=declared_start_seconds,
                timing_basis=timing_basis,
                timing_verified=timing_verified,
                chunk=chunk,
                position=position,
            )
            if parsed is None:
                malformed = True
                continue
            valid_segments.append(parsed)
        valid_segments.sort(key=lambda segment: (segment["source_start_seconds"], segment["source_end_seconds"]))

        outcome = requested_outcome
        reason: Optional[str] = None
        if requested_outcome == "failed":
            raise ValueError("adapter returned failed outcome")
        if malformed:
            self._append_unique(flags, "malformed_segment_bounds")
            outcome = "uncertain"
            reason = "malformed_segment_bounds"
        if requested_outcome == "transcribed" and not valid_segments:
            self._append_unique(flags, "no_valid_segments")
            outcome = "uncertain"
            reason = reason or "no_valid_segments"
        if requested_outcome == "no_speech" and valid_segments:
            self._append_unique(flags, "model_no_speech_with_text")
            outcome = "uncertain"
            reason = reason or "model_no_speech_with_text"
        if requested_outcome == "no_speech" and not valid_segments:
            if evidence.status == "speech_detected":
                self._append_unique(flags, "unexplained_speech")
                self._append_unique(flags, "model_no_speech_conflicts_with_detector")
                outcome = "uncertain"
                reason = reason or "unexplained_speech"
            elif evidence.status == "digital_silence":
                outcome = "no_speech"
            elif evidence.status == "no_speech_detected":
                self._append_unique(flags, "vad_no_speech_not_conclusive")
                self._append_unique(flags, "model_no_speech_unverified")
                outcome = "uncertain"
                reason = reason or "model_no_speech_unverified"
            elif evidence.status == "unknown":
                self._append_unique(flags, "speech_evidence_unknown")
                self._append_unique(flags, "model_no_speech_unverified")
                outcome = "uncertain"
                reason = reason or "model_no_speech_unverified"
        elif valid_segments and evidence.status == "no_speech_detected":
            # Preserve text, but make the disagreement reviewable rather than
            # presenting model text as independently supported speech.
            self._append_unique(flags, "model_text_without_detected_speech")
            outcome = "uncertain"
            reason = reason or "model_text_without_detected_speech"
        elif valid_segments and evidence.status == "unknown":
            self._append_unique(flags, "speech_evidence_unknown")

        uncertain_ranges = self._normalise_uncertain_ranges(data, chunk)
        if uncertain_ranges:
            self._append_unique(flags, "model_uncertain_range")
            outcome = "uncertain"
            reason = reason or "model_uncertain_range"
        if requested_outcome == "uncertain":
            outcome = "uncertain"
            reason = reason or "model_uncertain"
        if chunk.get("known_capture_gaps"):
            self._append_unique(flags, "known_capture_gap")
        return {
            "outcome": outcome,
            "segments": valid_segments,
            "language": data.get("language") if isinstance(data.get("language"), str) else "unknown",
            "speech_evidence": evidence.as_dict(),
            "review_flags": flags,
            "reason": reason,
            "uncertain_ranges": uncertain_ranges,
        }

    @staticmethod
    def _parse_raw_response(raw_response: Any) -> dict[str, Any]:
        if isinstance(raw_response, dict) and "raw_text" in raw_response and len(raw_response) == 1:
            raw_response = raw_response["raw_text"]
        if isinstance(raw_response, str):
            text = raw_response.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            raw_response = json.loads(text.strip())
        if not isinstance(raw_response, dict):
            raise ValueError("adapter response must be a JSON object or JSON text")
        return raw_response

    def _normalise_segment(
        self,
        raw: Any,
        *,
        source_id: str,
        declared_start_seconds: float,
        timing_basis: str,
        timing_verified: bool,
        chunk: dict[str, Any],
        position: int,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        start_raw = raw.get("start_seconds", raw.get("start"))
        end_raw = raw.get("end_seconds", raw.get("end"))
        try:
            start = _finite_nonnegative(start_raw, "segment start_seconds")
            end = _finite_nonnegative(end_raw, "segment end_seconds")
        except ValueError:
            return None
        duration = float(chunk["duration_seconds"])
        if end <= start or start > duration or end > duration + 0.05:
            return None
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        speaker = self._anonymous_speaker_label(
            raw.get("speaker"), source_id=source_id, chunk_index=int(chunk["index"])
        )
        local_start = float(chunk["source_start_seconds"]) + start
        local_end = float(chunk["source_start_seconds"]) + min(end, duration)
        return {
            "segment_id": f"{source_id}-{int(chunk['index']):04d}-{position:04d}",
            "source_id": source_id,
            "chunk_index": int(chunk["index"]),
            "source_start_seconds": local_start,
            "source_end_seconds": local_end,
            "start_seconds": declared_start_seconds + local_start,
            "end_seconds": declared_start_seconds + local_end,
            "timing_basis": timing_basis,
            "timing_verified": timing_verified,
            "text": text.strip(),
            "speaker": speaker,
            "speaker_basis": "anonymous_source_chunk_local",
        }

    @staticmethod
    def _anonymous_speaker_label(raw: Any, *, source_id: str, chunk_index: int) -> str:
        label = str(raw or "speaker_unknown").strip().lower().replace(" ", "_")
        # Strip identity-like strings instead of preserving a model-invented
        # name.  Only anonymous speaker tokens are allowed through.
        # Numeric anonymous IDs are the only model labels we preserve.  A
        # string such as "Speaker Matthias" must not become a namespaced
        # leak; it is canonicalized to unknown instead.
        match = re.fullmatch(r"(?:speaker|anonymous_speaker)[_-]?(\d{1,4})", label)
        local = f"speaker_{match.group(1)}" if match else "speaker_unknown"
        return f"{source_id}:chunk{chunk_index:04d}:{local}"

    @staticmethod
    def _normalise_uncertain_ranges(
        data: dict[str, Any], chunk: dict[str, Any]
    ) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        duration = float(chunk["duration_seconds"])
        offset = float(chunk["source_start_seconds"])
        for raw in data.get("uncertain_ranges", []) or []:
            if not isinstance(raw, dict):
                continue
            try:
                start = _finite_nonnegative(raw.get("start_seconds"), "uncertain start")
                end = _finite_nonnegative(raw.get("end_seconds"), "uncertain end")
            except ValueError:
                continue
            if end <= start or start > duration or end > duration + 0.05:
                continue
            out.append({"source_start_seconds": offset + start, "source_end_seconds": offset + min(end, duration)})
        return out

    def _record_failed_attempt(
        self,
        manifest: dict[str, Any],
        manifest_path: Path,
        job_dir: Path,
        source: dict[str, Any],
        chunk: dict[str, Any],
        exc: Exception,
        *,
        retryable: bool,
        attempt_number: Optional[int] = None,
        attempt_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        attempt_number = attempt_number or len(chunk["attempts"]) + 1
        data = attempt_metadata or {
            "attempt": attempt_number,
            "started_at": _utc_now(),
        }
        data.update(
            {
                "completed_at": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        attempt_path = self._attempt_path(job_dir, source["source_id"], chunk, attempt_number)
        atomic_write_json(attempt_path, data)
        # Avoid duplicate attempt records when parsing happened after a raw
        # response had already been durably appended.
        if not any(a.get("attempt") == attempt_number for a in chunk["attempts"]):
            chunk["attempts"].append(
                {
                    "attempt": attempt_number,
                    "status": "failed",
                    "raw_attempt_path": str(attempt_path.relative_to(job_dir)),
                    "at": data["completed_at"],
                }
            )
        else:
            for item in chunk["attempts"]:
                if item.get("attempt") == attempt_number:
                    item["status"] = "failed"
                    item["raw_attempt_path"] = str(attempt_path.relative_to(job_dir))
        chunk["status"] = "failed"
        chunk["retryable"] = retryable
        chunk["error"] = str(exc)
        if isinstance(exc, AdapterCallError) and exc.timeout:
            self._add_flag(chunk, "request_timeout")
        atomic_write_json(manifest_path, manifest)

    @staticmethod
    def _attempt_path(
        job_dir: Path, source_id: str, chunk: dict[str, Any], attempt: int
    ) -> Path:
        return job_dir / "attempts" / source_id / chunk["chunk_id"] / f"attempt-{attempt:04d}.json"

    @staticmethod
    def _result_path(job_dir: Path, source_id: str, chunk: dict[str, Any]) -> Path:
        return job_dir / "segments" / source_id / f"{chunk['chunk_id']}.json"

    @staticmethod
    def _add_flag(chunk: dict[str, Any], flag: str) -> None:
        flags = chunk.setdefault("review_flags", [])
        if flag not in flags:
            flags.append(flag)

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    def _mark_deadline_chunk(self, chunk: dict[str, Any]) -> None:
        chunk["status"] = "failed"
        chunk["retryable"] = True
        chunk["error"] = "overall_job_deadline_exceeded"
        self._add_flag(chunk, "overall_job_deadline_exceeded")

    @staticmethod
    def _manifest_status(manifest: dict[str, Any]) -> str:
        statuses = [
            chunk.get("status")
            for source in manifest.get("sources", [])
            for chunk in source.get("chunks", [])
        ]
        if any(status in {"pending", "running"} for status in statuses):
            return "pending"
        if any(status == "failed" for status in statuses):
            return "retryable_failure"
        if any(status == "uncertain" for status in statuses):
            return "needs_review"
        if any(source.get("capture_gaps") for source in manifest.get("sources", [])):
            return "needs_review"
        if manifest.get("absent_source_capture_gaps"):
            return "needs_review"
        return "complete"

    def _build_payload(self, manifest: dict[str, Any], job_dir: Path) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        missing_ranges: list[dict[str, Any]] = []
        retryable_ranges: list[dict[str, Any]] = []
        review_flags: list[dict[str, Any]] = []
        source_provenance: list[dict[str, Any]] = []
        transcript_sections: list[str] = []
        languages: list[str] = []
        accounted = True
        structurally_complete = True

        for source in manifest["sources"]:
            source_id = source["source_id"]
            source_segments: list[dict[str, Any]] = []
            source_flags: list[str] = []
            if not source.get("timing_verified"):
                source_flags.append("timing_unverified")
            for gap in source.get("capture_gaps", []):
                missing_ranges.append(
                    self._range_payload(source, gap["start_seconds"], gap["end_seconds"], "known_capture_gap", False)
                )
                review_flags.append(
                    self._range_payload(source, gap["start_seconds"], gap["end_seconds"], "known_capture_gap", False)
                )
                structurally_complete = False
            for chunk in source["chunks"]:
                status = chunk["status"]
                if status in {"pending", "running"}:
                    accounted = False
                    structurally_complete = False
                flags = list(chunk.get("review_flags") or [])
                if flags:
                    for flag in flags:
                        review_flags.append(
                            self._range_payload(
                                source,
                                chunk["source_start_seconds"],
                                chunk["source_end_seconds"],
                                flag,
                                bool(chunk.get("retryable")),
                            )
                        )
                result = None
                result_path = chunk.get("result_path")
                if result_path:
                    try:
                        result = _read_json(job_dir / result_path)
                    except Exception:
                        status = "failed"
                        chunk["error"] = "durable segment result unreadable"
                        chunk["retryable"] = True
                if result:
                    source_segments.extend(result.get("segments") or [])
                    language = result.get("language")
                    if isinstance(language, str) and language.strip() and language.lower() != "unknown":
                        languages.append(language.strip())
                unresolved = status in {"failed", "pending", "running", "uncertain"}
                if unresolved:
                    reason = chunk.get("error") or (
                        "uncertain" if status == "uncertain" else status
                    )
                    unresolved_ranges = []
                    if status == "uncertain" and isinstance(result, dict):
                        unresolved_ranges = result.get("uncertain_ranges") or []
                    if not unresolved_ranges:
                        unresolved_ranges = [
                            {
                                "source_start_seconds": chunk["source_start_seconds"],
                                "source_end_seconds": chunk["source_end_seconds"],
                            }
                        ]
                    for unresolved_range in unresolved_ranges:
                        item = self._range_payload(
                            source,
                            unresolved_range["source_start_seconds"],
                            unresolved_range["source_end_seconds"],
                            reason,
                            bool(chunk.get("retryable")),
                        )
                        missing_ranges.append(item)
                        if item["retryable"]:
                            retryable_ranges.append(item)
                    structurally_complete = False
            source_segments.sort(key=lambda item: (item["source_start_seconds"], item["source_end_seconds"]))
            segments.extend(source_segments)
            timing_note = source["timing_basis"]
            verification = "verified" if source.get("timing_verified") else "unverified"
            lines = [f"## {source_id} ({verification} timing: {timing_note})"]
            for segment in source_segments:
                lines.append(
                    f"[{segment['source_start_seconds']:.2f}-{segment['source_end_seconds']:.2f}] "
                    f"{segment['speaker']}: {segment['text']}"
                )
            if len(lines) == 1:
                lines.append("[No durable text segments for this source]")
            transcript_sections.append("\n".join(lines))
            source_provenance.append(
                {
                    "source_id": source_id,
                    "input_sha256": source["input_sha256"],
                    "duration_seconds": source["duration_seconds"],
                    "declared_start_seconds": source["declared_start_seconds"],
                    "timing_basis": source["timing_basis"],
                    "timing_verified": source["timing_verified"],
                    "capture_gaps": source.get("capture_gaps", []),
                    "flags": source_flags,
                }
            )

        absent_by_source: dict[str, list[dict[str, Any]]] = {}
        for gap in manifest.get("absent_source_capture_gaps", []):
            item = self._absent_range_payload(gap)
            missing_ranges.append(item)
            review_flags.append(item)
            absent_by_source.setdefault(gap["source_id"], []).append(gap)
            structurally_complete = False
        for source_id, gaps in absent_by_source.items():
            source_provenance.append(
                {
                    "source_id": source_id,
                    "input_sha256": None,
                    "duration_seconds": None,
                    "declared_start_seconds": None,
                    "timing_basis": "source_absent",
                    "timing_verified": False,
                    "capture_gaps": gaps,
                    "flags": ["source_missing", "timing_unverified"],
                }
            )

        # Source grouping is intentional.  Even with numeric offsets, tracks
        # are never interleaved here: legacy offsets cannot establish a shared
        # clock and a user should not mistake this draft for a synchronized mix.
        segments.sort(key=lambda item: (item["source_id"], item["source_start_seconds"], item["source_end_seconds"]))
        status = self._manifest_status(manifest)
        language = "unknown"
        if languages:
            distinct_languages = list(dict.fromkeys(languages))
            language = distinct_languages[0] if len(distinct_languages) == 1 else "Mixed"
        return {
            "transcript": "\n\n".join(transcript_sections),
            "segments": segments,
            "language": language,
            "_meta": {
                "durable_job_status": status,
                "job_key": manifest["job_key"],
                "schema_version": SCHEMA_VERSION,
                "model": manifest["identity"]["model"],
                "prompt_sha256": manifest["identity"]["prompt_sha256"],
                "schema_sha256": manifest["identity"]["schema_sha256"],
                "request_contract_version": manifest["identity"][
                    "request_contract_version"
                ],
                "request_contract_sha256": manifest["identity"][
                    "request_contract_sha256"
                ],
                "session_ids": list(manifest.get("session_ids", [])),
                "audio_duration_seconds": max(
                    (float(source["duration_seconds"]) for source in manifest["sources"]),
                    default=0.0,
                ),
                "missing_ranges": missing_ranges,
                "retryable_ranges": retryable_ranges,
                "review_flags": review_flags,
                "source_provenance": source_provenance,
                "speaker_attribution": {
                    "status": "hold",
                    "basis": "anonymous_source_chunk_local",
                    "speaker_dependent_actions": "hold",
                    "reason": "No calendar, names, or cross-source voice identity was used.",
                },
                "structural_completeness": {
                    "all_regions_accounted": accounted,
                    "transcript_structurally_complete": structurally_complete,
                    "assessment_method": "durable per-region outcomes, not final transcript timestamp",
                    "accuracy_verified": False,
                    "accuracy_status": "not_measured",
                },
                "partial": bool(missing_ranges) or not structurally_complete,
            },
        }

    @staticmethod
    def _range_payload(
        source: dict[str, Any],
        source_start: float,
        source_end: float,
        reason: str,
        retryable: bool,
    ) -> dict[str, Any]:
        declared_start = float(source["declared_start_seconds"])
        return {
            "source_id": source["source_id"],
            "source_start_seconds": float(source_start),
            "source_end_seconds": float(source_end),
            # These fields are a declared source offset, not a synchronization
            # claim.  Consumers must consult timing_verified/timing_basis.
            "start_seconds": declared_start + float(source_start),
            "end_seconds": declared_start + float(source_end),
            "timing_basis": source["timing_basis"],
            "timing_verified": bool(source["timing_verified"]),
            "reason": reason,
            "retryable": retryable,
        }

    @staticmethod
    def _absent_range_payload(gap: dict[str, Any]) -> dict[str, Any]:
        """Represent missing capture evidence without fabricating a source."""
        return {
            "source_id": gap["source_id"],
            "source_start_seconds": float(gap["start_seconds"]),
            "source_end_seconds": float(gap["end_seconds"]),
            "start_seconds": float(gap["start_seconds"]),
            "end_seconds": float(gap["end_seconds"]),
            "timing_basis": "source_absent",
            "timing_verified": False,
            "reason": gap.get("reason") or "missing_source_capture_gap",
            "retryable": False,
        }


__all__ = [
    "AdapterCallError",
    "ConservativeSpeechDetector",
    "DEFAULT_CHUNK_SECONDS",
    "DEFAULT_JOB_TIMEOUT_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "GeminiTranscriptionAdapter",
    "JobDeadlineExceeded",
    "REQUEST_CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "SourceInput",
    "SourceTranscriptionPipeline",
    "SpeechEvidence",
    "TRANSCRIPTION_SCHEMA",
    "TRANSCRIPT_ONLY_PROMPT",
    "atomic_write_json",
]
