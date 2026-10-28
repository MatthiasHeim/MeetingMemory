"""Per-recording files the recorder writes next to the WAV.

``<stem>.live.json``         the live transcript and cards as the window showed
                             them. Written only by ``write_live_transcript``.
                             Read by ``live_store``: the watcher fills gaps in a
                             partial final transcript from ``lines``, and
                             Prompts… offers ``cards`` after the meeting.
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
    cards: Sequence[Any] = (),
) -> Path:
    """Replace the live snapshot. Seconds are from the start of the recording.

    ``cards`` was added after schema 1 shipped; readers treat a missing list as
    no cards, so the schema version is unchanged.
    """
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
        "cards": [_card_payload(card) for card in cards],
    }
    _atomic_json(path, payload)
    return path


def _card_payload(card: Any) -> dict[str, Any]:
    """Clean text plus the verbatim source lines, as ``live_store`` reads them."""
    return {
        "key": str(getattr(card, "key", "") or ""),
        "kind": str(getattr(card, "kind", "") or "request"),
        "audience": str(getattr(card, "audience", "") or "all"),
        "clean_text": str(getattr(card, "clean_text", "") or ""),
        "verbatim_lines": [
            {
                "start": round(float(getattr(line, "seconds", getattr(line, "start", 0.0))), 2),
                "speaker": str(getattr(line, "speaker", "")),
                "text": str(getattr(line, "text", "")),
            }
            for line in getattr(card, "lines", ()) or ()
        ],
        "start": round(float(getattr(card, "start_seconds", 0.0)), 2),
        "end": round(float(getattr(card, "end_seconds", 0.0)), 2),
    }


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
