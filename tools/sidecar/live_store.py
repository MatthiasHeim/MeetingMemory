"""Durable live-session file and partial-transcript fill.

``Recordings/<stem>.live.json`` is replaced atomically from the live worker.
The watcher later copies live lines into a partial final transcript for the
time ranges that transcript does not cover.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from .transcript import (
    TranscriptLine,
    _TRANSCRIPT_LINE,
    format_timestamp,
    recordings_dir,
    safe_stem,
    timestamp_seconds,
)

try:
    from transcript_validator import COVERAGE_MIN_PCT, validate_transcript
except ImportError:  # pragma: no cover - package import from the repo root
    from tools.transcript_validator import COVERAGE_MIN_PCT, validate_transcript


LIVE_SCHEMA_VERSION = 1
# A live line this close to an existing final-transcript timestamp is the same
# turn heard twice, not a gap that needs filling.
_BOUNDARY_SLACK_SECONDS = 0.4
# Consecutive final lines this close together already cover the span between
# them. A missing-range that overlaps that span is not a hole.
_COVERED_LINE_GAP_SECONDS = 30.0


def live_json_path(recordings_root: str | Path, stem: str) -> Path:
    """Path of the live-session file for one recording stem."""
    return recordings_dir(recordings_root) / f"{safe_stem(stem)}.live.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` with ``payload``. Readers never see a partial document.

    The live session file is mode 0600. ``os.open`` still applies the process
    umask, so the mode is set again after the bytes are written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def live_session_payload(stem: str, lines: Sequence[Any], cards: Sequence[Any]) -> dict[str, Any]:
    """Committed lines and prompt cards. Text is stored as the sidecar heard it."""
    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "recording_stem": safe_stem(stem),
        "lines": [_line_payload(line) for line in lines],
        "cards": [_card_payload(card) for card in cards],
    }


def load_live_payload(path: Path) -> dict[str, Any] | None:
    """Return a live-session object, or None when the file is missing or unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("lines"), list):
        payload = dict(payload)
        payload["lines"] = []
    if not isinstance(payload.get("cards"), list):
        payload = dict(payload)
        payload["cards"] = payload.get("cards") if isinstance(payload.get("cards"), list) else []
    return payload


def saved_live_prompt_results(stem: str, recordings_root: str | Path | None = None):
    """Prompt cards saved during the meeting, for the post-meeting Prompts menu."""
    from .prompts import PromptResult

    try:
        if recordings_root is None:
            path = recordings_dir(None) / f"{safe_stem(stem)}.live.json"
        else:
            path = live_json_path(recordings_root, stem)
    except ValueError:
        return []
    payload = load_live_payload(path)
    if payload is None:
        return []
    results = []
    for card in payload.get("cards") or []:
        result = _prompt_result_from_card(card, PromptResult)
        if result is not None:
            results.append(result)
    return results


def fill_transcript_from_live(
    transcript: str,
    live_lines: Sequence[Any],
    audio_duration: float,
    missing_time_ranges: Sequence[Sequence[float]] | None = None,
    channel_lag_seconds: float | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Insert live lines into uncovered ranges.

    A transcript that already reaches the coverage minimum, and has no explicit
    missing ranges, is returned unchanged. Ranges that already contain final
    lines are subtracted before any live line is inserted. Inserted lines are
    tagged ``[live]``. ``channel_lag_seconds`` is the alignment lag applied to
    the file Gemini transcribed: positive means the mic was late, so live
    timestamps move earlier by that many seconds.
    """
    ranges = _missing_ranges(transcript, audio_duration, missing_time_ranges)
    if not ranges:
        return transcript, None
    existing = _existing_starts(transcript)
    lag = _finite(channel_lag_seconds) or 0.0
    selected: list[dict[str, Any]] = []
    for item in live_lines or []:
        line = _coerce_live_line(item)
        if line is None:
            continue
        if lag:
            line["start"] -= lag
            line["end"] -= lag
            if line["end"] < 0:
                continue
            line["start"] = max(0.0, line["start"])
            if line["end"] < line["start"]:
                line["end"] = line["start"]
        if not _starts_in_ranges(line["start"], ranges):
            continue
        if _restates_existing(line, existing):
            continue
        if _already_tagged(transcript, line):
            continue
        selected.append(line)
    if not selected:
        return transcript, None
    selected.sort(key=lambda line: (line["start"], line["text"]))
    formatted = [
        (
            line["start"],
            f"[{format_timestamp(line['start'])}] {line['speaker']}: [live] {line['text']}",
        )
        for line in selected
    ]
    filled_ranges = []
    for start, end in ranges:
        if any(start - 1e-6 <= line["start"] < end for line in selected):
            filled_ranges.append({"start": round(start, 3), "end": round(end, 3)})
    if not filled_ranges:
        return transcript, None
    return _insert_formatted(transcript, formatted), {
        "ranges": filled_ranges,
        "line_count": len(selected),
    }


def apply_live_fill(
    result: Any,
    live_path: Path,
    audio_duration: float,
    channel_lag_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Mutate a Gemini result when a partial transcript has live lines for its gaps."""
    payload = load_live_payload(Path(live_path))
    if payload is None:
        return None
    new_text, meta = fill_transcript_from_live(
        getattr(result, "transcript", "") or "",
        payload.get("lines") or [],
        audio_duration,
        missing_time_ranges=getattr(result, "missing_time_ranges", None) or [],
        channel_lag_seconds=channel_lag_seconds,
    )
    if meta is None:
        return None
    result.transcript = new_text
    result.live_fill = meta
    return meta


def _line_payload(line: Any) -> dict[str, Any]:
    return {
        "start": float(line.start),
        "end": float(line.end),
        "speaker": str(line.speaker),
        "text": str(line.text),
    }


def _card_payload(card: Any) -> dict[str, Any]:
    verbatim = []
    for line in getattr(card, "lines", ()) or ():
        verbatim.append(
            {
                "start": float(getattr(line, "seconds", getattr(line, "start", 0.0))),
                "speaker": str(getattr(line, "speaker", "")),
                "text": str(getattr(line, "text", "")),
            }
        )
    return {
        "clean_text": str(getattr(card, "clean_text", "") or ""),
        "verbatim_lines": verbatim,
        "start": float(getattr(card, "start_seconds", 0.0)),
        "end": float(getattr(card, "end_seconds", 0.0)),
    }


def _prompt_result_from_card(card: Any, prompt_result_type: type):
    if not isinstance(card, dict):
        return None
    raw_lines = card.get("verbatim_lines") or []
    if isinstance(raw_lines, str):
        raw_lines = [raw_lines]
    lines: list[TranscriptLine] = []
    card_start = _finite(card.get("start"))
    for index, item in enumerate(raw_lines):
        if isinstance(item, str):
            text = " ".join(item.split())
            start = card_start if card_start is not None else 0.0
            speaker = "Live"
        elif isinstance(item, dict):
            text = " ".join(str(item.get("text") or "").split())
            start = _finite(item.get("start"))
            if start is None:
                start = card_start if card_start is not None else 0.0
            speaker = " ".join(str(item.get("speaker") or "Live").split()) or "Live"
        else:
            continue
        if not text:
            continue
        lines.append(
            TranscriptLine(
                index=index,
                seconds=start,
                timestamp=format_timestamp(start),
                speaker=speaker,
                text=text,
            )
        )
    if not lines:
        return None
    clean = str(card.get("clean_text") or "")
    return prompt_result_type(tuple(lines), "live", clean_text=clean)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _coerce_live_line(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    start = _finite(item.get("start"))
    if start is None or start < 0:
        return None
    end = _finite(item.get("end"))
    if end is None or end < start:
        end = start
    text = " ".join(str(item.get("text") or "").split())
    if not text:
        return None
    speaker = " ".join(str(item.get("speaker") or "Live").split()) or "Live"
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def _missing_ranges(
    transcript: str,
    audio_duration: float,
    missing_time_ranges: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for item in missing_time_ranges or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = _finite(item[0])
        end = _finite(item[1])
        if start is None or end is None or end <= start:
            continue
        ranges.append((max(0.0, start), end))
    duration = _finite(audio_duration)
    if duration is not None and duration > 0:
        report = validate_transcript(transcript or "", duration)
        if report.coverage_pct < COVERAGE_MIN_PCT:
            ranges.append((max(0.0, float(report.last_timestamp_sec)), duration))
    return _subtract_ranges(_merge_ranges(ranges), _covered_intervals(transcript))


def _covered_intervals(transcript: str) -> list[tuple[float, float]]:
    """Spans between nearby final lines. Those spans already contain speech."""
    starts = _existing_starts(transcript)
    covered: list[tuple[float, float]] = []
    for previous, nxt in zip(starts, starts[1:]):
        if nxt > previous and nxt - previous <= _COVERED_LINE_GAP_SECONDS:
            covered.append((previous, nxt))
    return covered


def _subtract_ranges(
    ranges: Sequence[tuple[float, float]],
    covered: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the parts of ``ranges`` that do not overlap ``covered``."""
    if not ranges:
        return []
    if not covered:
        return list(ranges)
    remaining: list[tuple[float, float]] = []
    for start, end in ranges:
        cursor = start
        for covered_start, covered_end in covered:
            if covered_end <= cursor:
                continue
            if covered_start >= end:
                break
            if covered_start > cursor:
                remaining.append((cursor, min(covered_start, end)))
            cursor = max(cursor, covered_end)
            if cursor >= end:
                break
        if cursor < end - 1e-9:
            remaining.append((cursor, end))
    return _merge_ranges(remaining)


def _merge_ranges(ranges: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(ranges, key=lambda item: (item[0], item[1]))
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _existing_starts(transcript: str) -> list[float]:
    starts: list[float] = []
    for raw in (transcript or "").splitlines():
        match = _TRANSCRIPT_LINE.match(raw.strip())
        if match is None:
            continue
        try:
            starts.append(timestamp_seconds(match.group("timestamp")))
        except ValueError:
            continue
    return starts


def _starts_in_ranges(start: float, ranges: Sequence[tuple[float, float]]) -> bool:
    return any(range_start - 1e-6 <= start < range_end for range_start, range_end in ranges)


def _restates_existing(line: dict[str, Any], existing: Sequence[float]) -> bool:
    """Skip a live line that only repeats a timestamp the transcript already has.

    A line that starts within the slack of an existing stamp but runs on into
    the gap is the missing speech, not a duplicate of the last covered line.
    """
    start = float(line["start"])
    end = float(line["end"])
    for other in existing:
        if abs(start - other) <= _BOUNDARY_SLACK_SECONDS and end <= other + _BOUNDARY_SLACK_SECONDS:
            return True
    return False


def _already_tagged(transcript: str, line: dict[str, Any]) -> bool:
    stamp = f"[{format_timestamp(line['start'])}]"
    for raw in transcript.splitlines():
        if stamp in raw and "[live]" in raw and line["text"] in raw:
            return True
    return False


def _insert_formatted(transcript: str, formatted: Sequence[tuple[float, str]]) -> str:
    pending = list(formatted)
    blocks: list[tuple[float | None, str]] = []
    for raw in transcript.splitlines():
        match = _TRANSCRIPT_LINE.match(raw.strip())
        seconds = None
        if match is not None:
            try:
                seconds = timestamp_seconds(match.group("timestamp"))
            except ValueError:
                seconds = None
        blocks.append((seconds, raw))
    output: list[str] = []
    index = 0
    for seconds, raw in blocks:
        if seconds is not None:
            while index < len(pending) and pending[index][0] < seconds:
                output.append(pending[index][1])
                index += 1
        output.append(raw)
    while index < len(pending):
        output.append(pending[index][1])
        index += 1
    text = "\n".join(output)
    if not transcript or transcript.endswith("\n"):
        text += "\n"
    return text
