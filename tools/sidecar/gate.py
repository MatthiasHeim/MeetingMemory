"""Per-recording provider gate for the meeting sidecar.

This is intentionally independent of UI code so replay and CLI invocations get
the exact same fail-closed decision as the recorder menu.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .transcript import recordings_dir, safe_stem


class JudgeGateError(PermissionError):
    """A caller asked for Jev without the recording's explicit opt-in."""


@dataclass(frozen=True)
class RecordingSidecar:
    """Only the policy fields relevant to this first slice."""

    jev: bool = False
    # ``None`` means the meeting-attendance classification was absent or
    # malformed.  It must remain distinct from an explicit internal-only
    # ``false`` so the Jev route can fail closed.
    external_attendees: bool | None = None
    # A recorded acknowledgement is required when the owner explicitly opts
    # into Jev for an external or unresolved meeting.  It is deliberately a
    # separate boolean from the checkbox so the offline gate can distinguish
    # a normal internal choice from an informed external choice.
    jev_external_acknowledged: bool = False
    marks: tuple[float, ...] = ()
    # The first mic callback is later than the app's mark-zero boundary.  The
    # recorder persists that offset once it arrives, without doing I/O in its
    # audio callback.
    mic_first_sample_offset_seconds: float | None = None
    payload: dict[str, Any] | None = None


# Menu clicks and global-shortcut callbacks are normally delivered serially by
# AppKit, but the recorder also lets a worker finish while the menu remains
# usable.  Serialise read-modify-write mark updates in-process so two rapid
# shortcuts cannot overwrite one another's atomic replacement.
_SIDECAR_WRITE_LOCK = threading.RLock()
_JEV_AUDIT_WRITE_LOCK = threading.RLock()
_DEFAULT_AUDIT_ROOT = Path.home() / ".local" / "share" / "meeting-sidecar"
_UNSET = object()


def sidecar_path(stem: str, root: str | Path | None = None) -> Path:
    """Return the sidecar beside the corresponding WAV, never inside a path supplied by a stem."""
    return recordings_dir(root) / f"{safe_stem(stem)}.sidecar.json"


def _finite_nonnegative(value: Any) -> float | None:
    """Return a finite non-negative numeric sidecar field, or ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def load_recording_sidecar(stem: str, root: str | Path | None = None) -> RecordingSidecar:
    """Read a recording choice. Missing or malformed sidecars fail closed."""
    path = sidecar_path(stem, root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return RecordingSidecar()
    if not isinstance(raw, dict):
        return RecordingSidecar()
    marks: list[float] = []
    supplied_marks = raw.get("marks", [])
    if isinstance(supplied_marks, list):
        for mark in supplied_marks:
            value = _finite_nonnegative(mark)
            if value is not None:
                marks.append(value)
    raw_external_attendees = raw.get("external_attendees")
    return RecordingSidecar(
        jev=raw.get("jev") is True,
        external_attendees=(
            raw_external_attendees
            if type(raw_external_attendees) is bool
            else None
        ),
        jev_external_acknowledged=raw.get("jev_external_acknowledged") is True,
        marks=tuple(marks),
        mic_first_sample_offset_seconds=_finite_nonnegative(
            raw.get("mic_first_sample_offset_seconds")
        ),
        payload=raw,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _permitted_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep sidecars small and free of attendee/transcript content."""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when supplied")
    permitted: dict[str, Any] = {}
    if metadata:
        for key in ("calendar_title", "calendar_event_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                permitted[key] = value.strip()
        resolved = metadata.get("calendar_attendance_resolved")
        if type(resolved) is bool:
            permitted["calendar_attendance_resolved"] = resolved
    return permitted


def _validate_policy_values(
    *,
    jev: bool,
    external_attendees: bool | None,
    jev_external_acknowledged: bool,
) -> None:
    if type(jev) is not bool:
        raise ValueError("jev must be a boolean")
    if external_attendees is not None and type(external_attendees) is not bool:
        raise ValueError("external_attendees must be a boolean or null")
    if type(jev_external_acknowledged) is not bool:
        raise ValueError("jev_external_acknowledged must be a boolean")
    if jev_external_acknowledged and (not jev or external_attendees is False):
        raise ValueError(
            "jev_external_acknowledged requires jev: true and external/unresolved attendance"
        )


def _payload_for_recording(
    stem: str,
    *,
    jev: bool,
    external_attendees: bool | None,
    jev_external_acknowledged: bool,
    marks: list[float],
    metadata: Mapping[str, Any] | None = None,
    mic_first_sample_offset_seconds: float | None = None,
) -> dict[str, Any]:
    """Build the sole writable sidecar schema from policy-safe fields."""
    _validate_policy_values(
        jev=jev,
        external_attendees=external_attendees,
        jev_external_acknowledged=jev_external_acknowledged,
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "recording_stem": safe_stem(stem),
        "jev": jev,
        "external_attendees": external_attendees,
        "jev_external_acknowledged": jev_external_acknowledged,
        "marks": marks,
        "mark_clock": "seconds_since_recorder_start_monotonic",
    }
    if mic_first_sample_offset_seconds is not None:
        payload["mic_first_sample_offset_seconds"] = mic_first_sample_offset_seconds
    payload.update(_permitted_metadata(metadata))
    return payload


def initialise_recording_sidecar(
    stem: str,
    *,
    jev: bool,
    external_attendees: bool | None,
    jev_external_acknowledged: bool = False,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the menu decision before marks can be added.

    The format stays deliberately small and readable: ``marks`` are offsets in
    seconds from the recorder's monotonic start clock, not wall-clock dates.
    """
    path = sidecar_path(stem, root)
    payload = _payload_for_recording(
        stem,
        jev=jev,
        external_attendees=external_attendees,
        jev_external_acknowledged=jev_external_acknowledged,
        marks=[],
        metadata=metadata,
    )
    with _SIDECAR_WRITE_LOCK:
        _atomic_json(path, payload)
    return path


def amend_recording_sidecar(
    stem: str,
    *,
    root: str | Path | None = None,
    jev: bool | object = _UNSET,
    external_attendees: bool | None | object = _UNSET,
    jev_external_acknowledged: bool | object = _UNSET,
    metadata: Mapping[str, Any] | None = None,
    mic_first_sample_offset_seconds: float | None | object = _UNSET,
) -> Path:
    """Atomically amend local recorder facts without dropping prompt marks.

    Calendar resolution and first-sample timing arrive after recording starts.
    This helper intentionally preserves existing marks while replacing only
    the small policy/metadata fields supplied by the caller.
    """
    stem = safe_stem(stem)
    with _SIDECAR_WRITE_LOCK:
        existing = load_recording_sidecar(stem, root)
        raw = existing.payload or {}
        # A malformed or wrong-stem file is never a source of Jev authority.
        valid_existing = raw.get("recording_stem") == stem
        current_jev = existing.jev if valid_existing else False
        current_external = existing.external_attendees if valid_existing else None
        current_ack = existing.jev_external_acknowledged if valid_existing else False
        current_marks = list(existing.marks) if valid_existing else []
        current_mic_offset = (
            existing.mic_first_sample_offset_seconds if valid_existing else None
        )

        next_jev = current_jev if jev is _UNSET else jev
        next_external = (
            current_external if external_attendees is _UNSET else external_attendees
        )
        next_ack = (
            current_ack
            if jev_external_acknowledged is _UNSET
            else jev_external_acknowledged
        )
        if next_jev is False:
            next_ack = False
        if next_external is False:
            next_ack = False
        if mic_first_sample_offset_seconds is _UNSET:
            next_mic_offset = current_mic_offset
        else:
            next_mic_offset = _finite_nonnegative(mic_first_sample_offset_seconds)
            if mic_first_sample_offset_seconds is not None and next_mic_offset is None:
                raise ValueError("mic first-sample offset must be finite and non-negative")

        prior_metadata = _permitted_metadata(raw)
        prior_metadata.update(_permitted_metadata(metadata))
        payload = _payload_for_recording(
            stem,
            jev=next_jev,
            external_attendees=next_external,
            jev_external_acknowledged=next_ack,
            marks=current_marks,
            metadata=prior_metadata,
            mic_first_sample_offset_seconds=next_mic_offset,
        )
        path = sidecar_path(stem, root)
        _atomic_json(path, payload)
        return path


def append_mark(stem: str, offset_seconds: float, root: str | Path | None = None) -> float:
    """Atomically append a monotonic recording offset and return its normalised value."""
    try:
        mark = float(offset_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("mark offset must be a number") from exc
    if not math.isfinite(mark) or mark < 0:
        raise ValueError("mark offset must be finite and non-negative")
    path = sidecar_path(stem, root)
    with _SIDECAR_WRITE_LOCK:
        existing = load_recording_sidecar(stem, root)
        raw = existing.payload or {}
        valid_existing = raw.get("recording_stem") == safe_stem(stem)
        marks = list(existing.marks) if valid_existing else []
        # Millisecond precision is materially finer than a completed transcript's
        # timestamp resolution while keeping diffs/readability tidy.
        mark = round(mark, 3)
        marks.append(mark)
        payload = _payload_for_recording(
            stem,
            jev=existing.jev if valid_existing else False,
            external_attendees=(
                existing.external_attendees if valid_existing else None
            ),
            jev_external_acknowledged=(
                existing.jev_external_acknowledged if valid_existing else False
            ),
            marks=marks,
            metadata=_permitted_metadata(raw) if valid_existing else None,
            mic_first_sample_offset_seconds=(
                existing.mic_first_sample_offset_seconds if valid_existing else None
            ),
        )
        _atomic_json(path, payload)
    return mark


def _authoritative_sidecar_path(stem: str, root: str | Path | None) -> Path:
    """Return a regular, non-symlinked sidecar eligible to authorize Jev."""
    if root is None and os.environ.get("MEETING_SIDECAR_RECORDINGS_DIR"):
        raise JudgeGateError(
            "Jev refuses a recordings directory supplied through MEETING_SIDECAR_RECORDINGS_DIR"
        )
    root_path = recordings_dir(root)
    path = root_path / f"{safe_stem(stem)}.sidecar.json"
    try:
        resolved_root = root_path.resolve(strict=True)
        link_status = path.lstat()
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise JudgeGateError("Jev requires a readable local recording sidecar") from exc
    if stat.S_ISLNK(link_status.st_mode) or not stat.S_ISREG(link_status.st_mode):
        raise JudgeGateError("Jev requires a regular, non-symlinked recording sidecar")
    if resolved_path.parent != resolved_root:
        raise JudgeGateError("Jev sidecar is not inside the recordings root")
    return path


def _require_same_transcript_stem(stem: str, transcript_path: str | Path | None) -> None:
    if transcript_path is None:
        return
    path = Path(transcript_path)
    if path.suffix != ".json" or path.stem != safe_stem(stem):
        raise JudgeGateError("Jev sidecar stem does not match the finished transcript")


def judge_backend_for(
    stem: str,
    requested: str | None = None,
    *,
    recordings_root: str | Path | None = None,
    transcript_path: str | Path | None = None,
) -> str:
    """Choose the only permitted judge backend for a completed recording.

    Gemini is always the default. ``--judge jev`` is an offline override and
    succeeds only when this exact recording persisted an explicit ``jev: true``
    choice. A missing/corrupt JSON is therefore indistinguishable from opt-out.
    """
    normalised = (requested or "gemini").lower()
    if normalised not in {"gemini", "jev"}:
        raise ValueError("judge must be 'gemini' or 'jev'")
    if normalised == "gemini":
        return "gemini"
    stem = safe_stem(stem)
    _require_same_transcript_stem(stem, transcript_path)
    _authoritative_sidecar_path(stem, recordings_root)
    record = load_recording_sidecar(stem, recordings_root)
    if not record.payload or record.payload.get("recording_stem") != stem:
        raise JudgeGateError("Jev requires a sidecar bound to this recording stem")
    raw_external = record.payload.get("external_attendees", _UNSET)
    raw_acknowledgement = record.payload.get("jev_external_acknowledged", _UNSET)
    if (
        raw_external is _UNSET
        or (raw_external is not None and type(raw_external) is not bool)
        or type(raw_acknowledgement) is not bool
    ):
        raise JudgeGateError(
            "Jev requires explicit tri-state attendance and acknowledgement fields"
        )
    if not record.jev:
        raise JudgeGateError(
            "Jev is disabled for this recording; use Gemini or record an explicit jev: true choice"
        )
    if not (
        record.external_attendees is False
        or record.jev_external_acknowledged is True
    ):
        raise JudgeGateError(
            "Jev requires an internal recording or an explicit external-meeting acknowledgement"
        )
    return "jev"


def append_jev_audit_record(
    stem: str,
    record: RecordingSidecar,
    request_count: int,
    *,
    audit_root: str | Path | None = None,
) -> Path:
    """Append an audit-only Jev use record without transcript/request text.

    The record is written before helper invocation.  If it cannot be made
    durable, the caller must fail before an OpenRouter-capable helper sees any
    meeting material.
    """
    if type(request_count) is not int or request_count < 1:
        raise ValueError("Jev audit request count must be a positive integer")
    root = Path(audit_root or _DEFAULT_AUDIT_ROOT).expanduser()
    path = root / "jev-audit.jsonl"
    entry = {
        "schema_version": 1,
        "stem": safe_stem(stem),
        "external_attendees": record.external_attendees,
        "jev_external_acknowledged": record.jev_external_acknowledged,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_count": request_count,
    }
    encoded = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
    with _JEV_AUDIT_WRITE_LOCK:
        root.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return path
