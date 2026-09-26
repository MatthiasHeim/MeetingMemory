"""Per-recording provider gate for the meeting sidecar.

This is intentionally independent of UI code so replay and CLI invocations get
the exact same fail-closed decision as the recorder menu.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    marks: tuple[float, ...] = ()
    payload: dict[str, Any] | None = None


def sidecar_path(stem: str, root: str | Path | None = None) -> Path:
    """Return the sidecar beside the corresponding WAV, never inside a path supplied by a stem."""
    return recordings_dir(root) / f"{safe_stem(stem)}.sidecar.json"


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
            try:
                value = float(mark)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                marks.append(value)
    raw_external_attendees = raw.get("external_attendees")
    return RecordingSidecar(
        jev=raw.get("jev") is True,
        external_attendees=(
            raw_external_attendees
            if type(raw_external_attendees) is bool
            else None
        ),
        marks=tuple(marks),
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


def initialise_recording_sidecar(
    stem: str,
    *,
    jev: bool,
    external_attendees: bool,
    root: str | Path | None = None,
) -> Path:
    """Persist the menu decision before marks can be added.

    The format stays deliberately small and readable: ``marks`` are offsets in
    seconds from the recorder's monotonic start clock, not wall-clock dates.
    """
    if type(jev) is not bool or type(external_attendees) is not bool:
        raise ValueError("jev and external_attendees must be booleans")
    path = sidecar_path(stem, root)
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "jev": jev,
            "external_attendees": external_attendees,
            "marks": [],
            "mark_clock": "seconds_since_recorder_start_monotonic",
        },
    )
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
    existing = load_recording_sidecar(stem, root)
    payload = dict(existing.payload or {})
    marks = list(existing.marks)
    # Millisecond precision is materially finer than a completed transcript's
    # timestamp resolution while keeping diffs/readability tidy.
    mark = round(mark, 3)
    marks.append(mark)
    payload.update(
        {
            "schema_version": 1,
            "jev": existing.jev,
            "external_attendees": existing.external_attendees,
            "marks": marks,
            "mark_clock": "seconds_since_recorder_start_monotonic",
        }
    )
    _atomic_json(path, payload)
    return mark


def judge_backend_for(
    stem: str,
    requested: str | None = None,
    *,
    recordings_root: str | Path | None = None,
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
    record = load_recording_sidecar(stem, recordings_root)
    if not record.jev:
        raise JudgeGateError(
            "Jev is disabled for this recording; use Gemini or record an explicit jev: true choice"
        )
    if record.external_attendees is not False:
        # An explicit checkbox is necessary for Jev, never sufficient for
        # client material. Until the direct TypeSafe DPA condition in §14 is
        # met and represented by a verified policy field, an external attendee
        # or unresolved attendee status makes the recording fail closed to the
        # contract-covered backend.
        raise JudgeGateError(
            "Jev requires an explicitly internal-only recording; use the contract-covered Gemini backend"
        )
    return "jev"
