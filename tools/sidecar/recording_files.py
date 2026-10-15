"""Per-recording files the recorder writes next to the WAV.

``<stem>.live.json``         the live transcript as the window showed it, for
                             comparison with the final transcript.
``<stem>.diarization.json``  speaker names the owner confirmed in the window.
                             The watcher reads its ``roster`` before the final
                             Gemini call; ``authoritative`` drops calendar
                             invitees the owner removed.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .gate import _SIDECAR_WRITE_LOCK, _atomic_json
from .transcript import recordings_dir, safe_stem


def live_transcript_path(stem: str, root: str | Path | None = None) -> Path:
    return recordings_dir(root) / f"{safe_stem(stem)}.live.json"


def diarization_path(stem: str, root: str | Path | None = None) -> Path:
    return recordings_dir(root) / f"{safe_stem(stem)}.diarization.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_live_transcript(
    stem: str,
    lines: Sequence[Any],
    *,
    root: str | Path | None = None,
    model: str | None = None,
    recording_active: bool = True,
) -> Path:
    """Replace the live snapshot. Seconds are from the start of the recording."""
    path = live_transcript_path(stem, root)
    payload = {
        "schema_version": 1,
        "recording_stem": safe_stem(stem),
        "model": model,
        "recording_active": recording_active,
        "saved_at": _now(),
        "lines": [
            {
                "start": round(float(line.start), 2),
                "end": round(float(line.end), 2),
                "speaker": line.speaker,
                "text": line.text,
                "provisional": bool(getattr(line, "provisional", False)),
            }
            for line in lines
        ],
    }
    _atomic_json(path, payload)
    return path


def parse_speaker_names(text: str) -> list[str]:
    """Comma, semicolon or newline separated; case-insensitive duplicates dropped."""
    names: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\n]", text or ""):
        name = " ".join(part.split())
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def load_speaker_names(stem: str, root: str | Path | None = None) -> list[str]:
    try:
        data = json.loads(diarization_path(stem, root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    roster = data.get("roster") if isinstance(data, dict) else None
    names = []
    for item in roster or []:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str):
            names.append(name)
    return parse_speaker_names("\n".join(names))


def write_speaker_names(stem: str, names: Iterable[str], *, root: str | Path | None = None) -> Path:
    """Store confirmed names; other keys (e.g. ``num_speakers``) are preserved."""
    path = diarization_path(stem, root)
    with _SIDECAR_WRITE_LOCK:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        payload = existing if isinstance(existing, dict) else {}
        payload.update({
            "roster": [{"name": name, "role": "participant"} for name in parse_speaker_names("\n".join(names))],
            "authoritative": True,
            "source": "recorder",
            "updated_at": _now(),
        })
        _atomic_json(path, payload)
    return path


def calendar_speaker_names(resolution: Any) -> list[str]:
    """Invitee names from ``calendar_resolve.resolve``, without the owner."""
    if not isinstance(resolution, dict):
        return []
    names = [
        person.get("name", "")
        for person in resolution.get("participant_details") or []
        if isinstance(person, dict) and (person.get("role") or "").lower() != "self"
    ]
    return parse_speaker_names("\n".join(n for n in names if isinstance(n, str)))
