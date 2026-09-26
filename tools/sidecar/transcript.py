"""Load the recorder's final-transcript JSON without changing it."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TRANSCRIPTS_DIR = Path.home() / "Documents" / "MeetingRecorder" / "Transcripts"
DEFAULT_RECORDINGS_DIR = Path.home() / "Documents" / "MeetingRecorder" / "Recordings"

_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRANSCRIPT_LINE = re.compile(
    r"^\s*\[(?P<timestamp>\d{1,3}:\d{2}(?::\d{2})?)\]\s*"
    r"(?P<speaker>[^:\n]+?)\s*:\s*(?P<text>.+?)\s*$"
)
logger = logging.getLogger(__name__)


def safe_stem(stem: str) -> str:
    """Reject a stem that could escape the configured recorder directories."""
    if not isinstance(stem, str) or not _SAFE_STEM.fullmatch(stem):
        raise ValueError("stem must contain only letters, digits, '.', '_' or '-'")
    return stem


def transcripts_dir(value: str | Path | None = None) -> Path:
    """Resolve the transcript root, with an opt-in environment override for tests."""
    if value is not None:
        return Path(value).expanduser()
    return Path(os.environ.get("MEETING_SIDECAR_TRANSCRIPTS_DIR", DEFAULT_TRANSCRIPTS_DIR)).expanduser()


def recordings_dir(value: str | Path | None = None) -> Path:
    """Resolve the recordings root, with an opt-in environment override for tests."""
    if value is not None:
        return Path(value).expanduser()
    return Path(os.environ.get("MEETING_SIDECAR_RECORDINGS_DIR", DEFAULT_RECORDINGS_DIR)).expanduser()


def timestamp_seconds(timestamp: str) -> float:
    """Convert ``MM:SS`` or ``HH:MM:SS`` transcript timestamps to seconds."""
    pieces = timestamp.split(":")
    if len(pieces) == 2:
        minutes, seconds = (int(piece) for piece in pieces)
        if not 0 <= seconds < 60:
            raise ValueError(f"invalid transcript timestamp: {timestamp!r}")
        return float(minutes * 60 + seconds)
    if len(pieces) == 3:
        hours, minutes, seconds = (int(piece) for piece in pieces)
        if not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError(f"invalid transcript timestamp: {timestamp!r}")
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"invalid transcript timestamp: {timestamp!r}")


def format_timestamp(seconds: float) -> str:
    """Format a non-negative transcript offset as ``MM:SS`` or ``HH:MM:SS``."""
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


@dataclass(frozen=True)
class TranscriptLine:
    """One verbatim final-transcript sentence/turn."""

    index: int
    seconds: float
    timestamp: str
    speaker: str
    text: str

    def display(self, *, timestamps: bool = True) -> str:
        prefix = f"[{self.timestamp}] " if timestamps else ""
        return f"{prefix}{self.speaker}: {self.text}"


@dataclass(frozen=True)
class Transcript:
    """A finished recorder transcript and only the fields the sidecar needs."""

    stem: str
    path: Path
    lines: tuple[TranscriptLine, ...]
    title: str
    payload: dict[str, Any]


def parse_transcript_lines(value: str) -> tuple[TranscriptLine, ...]:
    """Parse recorder transcript text, skipping blank/non-turn bookkeeping lines.

    Recorder transcripts are conventional ``[MM:SS] Speaker: text`` lines.
    A malformed line is not reinterpreted or sent to a judge; keeping it out is
    safer than inventing a timestamp or speaker.
    """
    parsed: list[TranscriptLine] = []
    for source_line in value.splitlines():
        match = _TRANSCRIPT_LINE.match(source_line)
        if not match:
            if source_line.strip():
                logger.debug("Skipping malformed transcript line: %r", source_line)
            continue
        timestamp = match.group("timestamp")
        try:
            seconds = timestamp_seconds(timestamp)
        except ValueError:
            logger.debug("Skipping transcript line with invalid timestamp: %r", source_line)
            continue
        text = match.group("text").strip()
        if not text:
            continue
        parsed.append(
            TranscriptLine(
                index=len(parsed),
                seconds=seconds,
                timestamp=timestamp,
                speaker=match.group("speaker").strip(),
                text=text,
            )
        )
    return tuple(parsed)


def _title_from_payload(payload: dict[str, Any], stem: str) -> str:
    for key in ("title", "meeting_title", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = payload.get("_meta")
    if isinstance(metadata, dict):
        for key in ("title", "meeting_title", "calendar_title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return stem


def load_transcript(stem: str, root: str | Path | None = None) -> Transcript:
    """Load ``Transcripts/<stem>.json`` and its final transcript turns."""
    stem = safe_stem(stem)
    path = transcripts_dir(root) / f"{stem}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"transcript not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"transcript JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"transcript JSON must be an object: {path}")
    raw = payload.get("transcript")
    if not isinstance(raw, str):
        raise ValueError(f"transcript JSON has no string 'transcript': {path}")
    lines = parse_transcript_lines(raw)
    if not lines:
        raise ValueError(f"transcript contains no timestamped speaker lines: {path}")
    return Transcript(stem, path, lines, _title_from_payload(payload, stem), payload)


def transcript_from_lines(
    stem: str,
    lines: Iterable[TranscriptLine],
    *,
    title: str | None = None,
) -> Transcript:
    """Create an in-memory transcript for tests and calibration adapters."""
    stem = safe_stem(stem)
    frozen_lines = tuple(lines)
    return Transcript(stem, Path(f"{stem}.json"), frozen_lines, title or stem, {})


def recent_transcripts(root: str | Path | None = None, limit: int = 10) -> list[Transcript]:
    """Return the newest valid final transcripts without ever modifying them."""
    directory = transcripts_dir(root)
    candidates = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    result: list[Transcript] = []
    for path in candidates:
        if len(result) >= limit:
            break
        try:
            result.append(load_transcript(path.stem, directory))
        except (OSError, ValueError):
            # Some housekeeping JSON files live next to real transcripts.
            continue
    return result
