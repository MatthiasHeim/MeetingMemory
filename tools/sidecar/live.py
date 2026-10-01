"""Live Hochdeutsch transcript, off the audio callback.

Every tick takes the last minute of microphone audio and, when present, the
tail of the system-audio WAV captured at that same moment. Gemini returns
speaker-labelled lines. Same-speaker overlaps replace or drop a re-hear;
other lines older than the frontier are dropped. A failure sets a status
string and does not touch recording.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .calibration_policy import OWNER_APPROVED_MODEL, OWNER_APPROVED_PROMPT_THRESHOLD
from .clean_prompt import clean_prompt_text
from .judges import (
    APPROVED_GEMINI_BASE_URL,
    Judge,
    JudgeError,
    _load_json_response,
    _verify_pinned_gemini_destination,
    pinned_gemini_client,
)
from .live_audio import (
    copy_chunk_references,
    read_wav_pcm_tail,
    stereo_tail_wav,
    tail_from_chunks,
)
from .prompts import PROMPT_CONSECUTIVE_GAP_SECONDS, _prompt_runs
from .questions import DICTATING_PROMPT_QUESTION
from .transcript import TranscriptLine, format_timestamp


LIVE_TICK_SECONDS = 20.0
LIVE_TAIL_SECONDS = 60.0
LIVE_SAMPLE_RATE = 16000
# A fraction past the tail is still "this clip". Anything further is either a
# meeting-clock timestamp or a hallucination, and must not be pinned to the edge.
_OFFSET_SLACK_SECONDS = 1.0
# Speech the first answer leaves uncovered is requested again. The floor is
# int16 RMS over 50 ms, above the room tone of a quiet tail and below the
# short remarks in the owner's test recordings.
_SPEECH_FRAME_SECONDS = 0.05
_SPEECH_RMS_FLOOR = 260.0
_SPEECH_MERGE_GAP_SECONDS = 0.35
_SPEECH_MIN_SECONDS = 0.45
_GAP_FILL_MIN_SECONDS = 0.5
# A full minute lets the model stop after the first dense stretch and skip
# short remarks later in the same tail. The second ask is a short slice.
_GAP_SLICE_SECONDS = 15.0


@dataclass(frozen=True)
class TailSnapshot:
    """Mic and system tails captured together, before any network call."""

    mic: np.ndarray
    mic_rate: int
    system: np.ndarray | None
    system_rate: int
    window_start: float
    window_end: float
    # A closed tail is the last one (stop, or the end of a replay). Open tails
    # may withhold one unfinished edge line, and only for a single tick.
    closed: bool = False
    # Stop archived or deleted the system WAV. The final tick must not call Gemini.
    system_gone: bool = False


@dataclass(frozen=True)
class LiveLine:
    """One committed Hochdeutsch line on the microphone clock."""

    start: float
    end: float
    speaker: str
    text: str

    def as_transcript_line(self, index: int) -> TranscriptLine:
        return TranscriptLine(
            index=index,
            seconds=self.start,
            timestamp=format_timestamp(self.start),
            speaker=self.speaker,
            text=self.text,
        )


@dataclass(frozen=True)
class LiveCard:
    """A prompt card: clean text for the clipboard, verbatim lines underneath."""

    key: str
    lines: tuple[TranscriptLine, ...]
    clean_text: str

    @property
    def verbatim_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def copy_text(self) -> str:
        return self.clean_text.strip() or self.verbatim_text

    @property
    def start_seconds(self) -> float:
        return self.lines[0].seconds if self.lines else 0.0

    @property
    def end_seconds(self) -> float:
        return self.lines[-1].seconds if self.lines else 0.0


# An incoming line must overlap a committed line of the same speaker by more
# than this share of the shorter interval before it can replace or drop it.
OVERLAP_REPLACE_RATIO = 0.5
# Lines that start earlier than this behind the committed frontier are old.
FRONTIER_SLACK_SECONDS = 0.5
# A failed clean-prompt call is retried on later ticks, then left verbatim.
CLEAN_PROMPT_ATTEMPTS = 3


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _looks_unfinished(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("...", "…", ",", ":", ";")):
        return True
    return stripped[-1] not in ".?!"


def _same_speaker(left: LiveLine, right: LiveLine) -> bool:
    return _norm(left.speaker) == _norm(right.speaker)


def _duration(line: LiveLine) -> float:
    return max(0.0, line.end - line.start)


def _overlap_seconds(left: LiveLine, right: LiveLine) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _overlap_ratio(left: LiveLine, right: LiveLine) -> float:
    """Fraction of the shorter interval covered by both lines."""
    short = min(_duration(left), _duration(right))
    overlap = _overlap_seconds(left, right)
    if short <= 1e-3:
        if _duration(left) <= 1e-3 and right.start - 1e-3 <= left.start <= right.end + 1e-3:
            return 1.0
        if _duration(right) <= 1e-3 and left.start - 1e-3 <= right.start <= left.end + 1e-3:
            return 1.0
        return 0.0
    return overlap / short


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.casefold().replace("…", " ").replace("...", " ").split():
        token = raw.strip(".,;:!?\"'“”«»()[]")
        if token:
            tokens.append(token)
    return tokens


def _cut_off_instruction(text: str) -> bool:
    """An unfinished line that names the assistant and the act of telling it."""
    if text.rstrip().endswith((".", "!", "?")):
        return False
    tokens = _tokens(text)
    if len(tokens) < 4:
        return False
    has_tool = any(token in {"claude", "agent"} for token in tokens)
    has_verb = any(token.startswith(("sag", "säg")) for token in tokens)
    return has_tool and has_verb


def _scores_keeping_cut_off_instructions(
    scores: Sequence[float], lines: Sequence[TranscriptLine], *, threshold: float
) -> list[float]:
    """Keep a cut-off instruction on the card when its continuation scores.

    A later window sometimes scores 'ich würde Claude sagen, er soll' under
    the threshold and then scores the rest of the same dictation above it.
    The opening would otherwise become its own missed line.
    """
    adjusted = [float(score) for score in scores]
    for index, line in enumerate(lines):
        if adjusted[index] >= threshold or not _cut_off_instruction(line.text):
            continue
        for later in range(index + 1, len(lines)):
            if lines[later].seconds - line.seconds > PROMPT_CONSECUTIVE_GAP_SECONDS:
                break
            if adjusted[later] >= threshold and lines[later].speaker == line.speaker:
                adjusted[index] = threshold
                break
    return adjusted


def _contains_committed(committed: str, incoming: str) -> bool:
    old = " ".join(_tokens(committed))
    new = " ".join(_tokens(incoming))
    return len(old) >= 12 and old in new


def _continuation_words(committed: str, incoming: str) -> list[str] | None:
    """Words ``incoming`` adds after the end of ``committed``.

    The match is a long suffix of the committed line and a prefix of the
    incoming line. A reworded overlap does not line up, so the caller drops
    it instead of appending a second copy.
    """
    old = _tokens(committed)
    new = _tokens(incoming)
    if len(old) < 3 or len(new) < 3:
        return None
    matched = 0
    for length in range(min(len(old), len(new)), 2, -1):
        if old[-length:] == new[:length]:
            matched = length
            break
    if matched < max(3, (len(new) + 1) // 2) or matched >= len(new):
        return None
    original = incoming.split()
    if len(original) == len(new):
        return original[matched:]
    return new[matched:]


def _covered(held: LiveLine, rows: Sequence[LiveLine]) -> bool:
    return any(_same_speaker(held, line) and _overlap_ratio(held, line) > OVERLAP_REPLACE_RATIO for line in rows)


def _held_is_due(held: LiveLine | None, line: LiveLine) -> bool:
    """True when ``line`` is the next tick's hearing of a line already held."""
    if held is None or not _same_speaker(held, line):
        return False
    if _overlap_ratio(held, line) > 0:
        return True
    return abs(line.start - held.start) <= 1.0


def _continues_committed(line: LiveLine, committed: Sequence[LiveLine]) -> bool:
    return any(_same_speaker(line, other) and _overlap_ratio(line, other) > OVERLAP_REPLACE_RATIO for other in committed)


def _select_lines_to_commit(
    lines: Sequence[LiveLine],
    window_end: float,
    *,
    closed: bool,
    held: LiveLine | None,
    committed: Sequence[LiveLine],
) -> tuple[list[LiveLine], LiveLine | None]:
    """Commit every line, holding an unfinished edge line for one tick only.

    The next tick commits that fragment even if it is still unpunctuated.
    Overlap replacement can then upgrade it. A finished sentence at the edge
    is committed immediately.
    """
    rows = [line for line in lines if line.text.strip()]
    if closed or not rows:
        extra = [held] if held is not None and not _covered(held, rows) else []
        return extra + rows, None
    last = max(rows, key=lambda line: (line.end, line.start))
    at_edge = window_end - last.end <= 2.0 and _looks_unfinished(last.text)
    if not at_edge or _held_is_due(held, last) or _continues_committed(last, committed):
        extra = [held] if held is not None and not _covered(held, rows) else []
        return extra + rows, None
    return [line for line in rows if line is not last], last


def _without_open_tail(
    lines: Sequence[LiveLine],
    window_end: float,
    *,
    closed: bool,
    held: LiveLine | None = None,
    committed: Sequence[LiveLine] = (),
) -> list[LiveLine]:
    """Lines from ``_select_lines_to_commit`` that should be committed now."""
    chosen, _held = _select_lines_to_commit(
        lines, window_end, closed=closed, held=held, committed=committed
    )
    return chosen


def _extends_fragment(committed: str, incoming: str) -> bool:
    """A short committed fragment is the opening of a longer nearby hearing."""
    if len(committed) > 40:
        return False
    old = " ".join(_tokens(committed))
    new = " ".join(_tokens(incoming))
    return bool(old) and new.startswith(old) and len(new) >= len(old) + 8


def _nearby_extension(lines: Sequence[LiveLine], stored: LiveLine) -> int | None:
    """Same-speaker fragment just before this line, when the clocks barely miss.

    Time overlap handles a re-hear of the same interval. A one-word fragment
    whose next hearing starts a fraction of a second later would otherwise
    stay as a second line.
    """
    for index in range(len(lines) - 1, -1, -1):
        other = lines[index]
        if not _same_speaker(other, stored):
            continue
        if abs(other.start - stored.start) > 15:
            continue
        gap = max(0.0, stored.start - other.end, other.start - stored.end)
        if gap > 2.0:
            continue
        old = " ".join(_tokens(other.text))
        new = " ".join(_tokens(stored.text))
        if old and (old == new or _extends_fragment(other.text, stored.text)):
            return index
    return None


def _drop_covered_lines(
    lines: list[LiveLine], primary: int, note: Callable[[int], None]
) -> tuple[list[LiveLine], int]:
    """Drop same-speaker lines the updated line now covers and is at least as long as."""
    updated = lines[primary]
    kept: list[LiveLine] = []
    shift = 0
    for index, other in enumerate(lines):
        covered = (
            index != primary
            and _same_speaker(other, updated)
            and _overlap_ratio(other, updated) > OVERLAP_REPLACE_RATIO
            and len(other.text) <= len(updated.text)
        )
        if covered:
            note(index)
            if index < primary:
                shift += 1
            continue
        kept.append(other)
    return kept, primary - shift


def commit_new_lines(
    committed: Sequence[LiveLine], incoming: Sequence[LiveLine]
) -> tuple[list[LiveLine], list[LiveLine], int | None]:
    """Keep new lines and collapse same-speaker overlaps.

    An incoming line that overlaps a committed line of the same speaker by
    more than half of the shorter interval replaces it when the new text is
    longer and still covers that line's start. Otherwise that re-hear is
    dropped. A tail that has slid forward keeps the committed prefix and
    appends only the aligned new suffix, so an unpunctuated monologue does
    not lose the words that left the window and does not gain a second copy.
    Anything else starting more than half a second before the frontier is
    old. ``revised_from`` is the earliest replaced index.
    """
    all_lines = list(committed)
    added: list[LiveLine] = []
    revised_from: int | None = None
    frontier = max((line.end for line in all_lines), default=-1.0)

    def _note_revision(index: int) -> None:
        nonlocal revised_from
        if revised_from is None or index < revised_from:
            revised_from = index

    for line in sorted(incoming, key=lambda item: (item.start, item.end)):
        text = line.text.strip()
        if not text:
            continue
        stored = LiveLine(line.start, max(line.end, line.start), line.speaker.strip() or "Ich", text)
        match_at: int | None = None
        match_ratio = 0.0
        for index, other in enumerate(all_lines):
            if not _same_speaker(other, stored):
                continue
            ratio = _overlap_ratio(other, stored)
            if ratio > OVERLAP_REPLACE_RATIO and (match_at is None or ratio >= match_ratio):
                match_ratio = ratio
                match_at = index
        if match_at is None:
            match_at = _nearby_extension(all_lines, stored)
        if match_at is not None:
            previous = all_lines[match_at]
            longer = len(stored.text) > len(previous.text)
            covers = (
                stored.start <= previous.start + 0.75
                or _contains_committed(previous.text, stored.text)
                or _extends_fragment(previous.text, stored.text)
            )
            updated = False
            if longer and covers:
                all_lines[match_at] = LiveLine(
                    min(previous.start, stored.start),
                    max(previous.end, stored.end),
                    previous.speaker,
                    stored.text,
                )
                updated = True
            else:
                extra = _continuation_words(previous.text, stored.text)
                if extra and stored.end > previous.end + FRONTIER_SLACK_SECONDS:
                    all_lines[match_at] = LiveLine(
                        previous.start,
                        max(previous.end, stored.end),
                        previous.speaker,
                        previous.text.rstrip() + " " + " ".join(extra),
                    )
                    updated = True
            if updated:
                _note_revision(match_at)
                all_lines, match_at = _drop_covered_lines(all_lines, match_at, _note_revision)
                frontier = max(line.end for line in all_lines)
            continue
        if all_lines and stored.start < frontier - FRONTIER_SLACK_SECONDS:
            continue
        all_lines.append(stored)
        added.append(stored)
        frontier = max(frontier, stored.end)
    return all_lines, added, revised_from


def snapshot_recording_tails(recorder: Any, tail_seconds: float = LIVE_TAIL_SECONDS) -> TailSnapshot | None:
    """Snapshot mic references and the system-file tail at one moment.

    Called only from the live worker. The audio callback is not involved.
    """
    chunks = copy_chunk_references(getattr(recorder, "audio_data", []) or [])
    rate = int(getattr(recorder, "sample_rate", 0) or 0)
    if rate <= 0:
        return None
    frames = getattr(recorder, "_mic_frames", None)
    total_samples = int(frames) if isinstance(frames, int) and frames > 0 else None
    mic, window_start, window_end = tail_from_chunks(
        chunks, rate, tail_seconds, total_samples=total_samples
    )
    system = None
    system_rate = rate
    system_gone = False
    path = getattr(recorder, "_sys_wav", None)
    if path is not None:
        wav_path = Path(path)
        try:
            if wav_path.is_file() and wav_path.stat().st_size > 44:
                system, system_rate = read_wav_pcm_tail(wav_path, tail_seconds)
            elif not wav_path.is_file():
                system_gone = True
        except (OSError, ValueError):
            system = None
    if mic.size == 0 and (system is None or getattr(system, "size", 0) == 0):
        return None
    return TailSnapshot(
        mic, rate, system, system_rate, window_start, window_end, system_gone=system_gone
    )


class GeminiLiveTranscriber:
    """Hochdeutsch transcript of one mic/system tail via pinned ``gemini-3.8-flash``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        types_module: Any | None = None,
        model: str = OWNER_APPROVED_MODEL,
        timeout_seconds: float = 22,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        if client is not None:
            _verify_pinned_gemini_destination(client)
            if _visible_base_url(client) is None:
                raise JudgeError("Gemini live client has no verifiable Google API destination")
            self.client = client
            self.types = types_module
            return
        self.client, self.types = pinned_gemini_client(api_key)

    def transcribe_tails(
        self, mic: np.ndarray, mic_rate: int, system: np.ndarray | None, system_rate: int
    ) -> list[dict[str, Any]]:
        """Return line dicts with offsets relative to the start of the tail.

        One retry, then the error propagates to the worker, which records a
        status line and leaves recording running.
        """
        last: Exception | None = None
        for _attempt in range(2):
            try:
                return self._once(mic, mic_rate, system, system_rate)
            except (JudgeError, OSError, ValueError) as exc:
                last = exc
        assert last is not None
        raise last

    def transcribe_gaps(
        self, mic: np.ndarray, mic_rate: int, system: np.ndarray | None, system_rate: int
    ) -> list[dict[str, Any]]:
        """Second ask for speech the meeting prompt left uncovered.

        A failure here keeps the first answer. It must not fail the tick.
        """
        has_system = system is not None and getattr(system, "size", 0) > 0
        try:
            return self._once(
                mic, mic_rate, system, system_rate, prompt=_gap_fill_prompt(has_system)
            )
        except (JudgeError, OSError, ValueError):
            return []

    def _once(
        self,
        mic: np.ndarray,
        mic_rate: int,
        system: np.ndarray | None,
        system_rate: int,
        prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        _verify_pinned_gemini_destination(self.client)
        audio = stereo_tail_wav(mic, mic_rate, system, system_rate, LIVE_SAMPLE_RATE)
        if len(audio) < 64:
            return []
        has_system = system is not None and getattr(system, "size", 0) > 0
        if prompt is None:
            prompt = _transcription_prompt(has_system=has_system)
        parts: list[Any] = [prompt]
        if self.types is None:
            parts.append({"mime_type": "audio/wav", "data": audio})
            config: Any = {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": _transcript_schema(),
                "http_options": {
                    "base_url": APPROVED_GEMINI_BASE_URL,
                    "api_version": "v1beta",
                    "timeout": int(self.timeout_seconds * 1000),
                },
            }
        else:
            parts.append(self.types.Part.from_bytes(data=audio, mime_type="audio/wav"))
            config = self.types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=4096,
                media_resolution=self.types.MediaResolution.MEDIA_RESOLUTION_LOW,
                thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=_transcript_schema(),
                http_options=self.types.HttpOptions(
                    base_url=APPROVED_GEMINI_BASE_URL,
                    api_version="v1beta",
                    timeout=int(self.timeout_seconds * 1000),
                ),
            )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=parts, config=config
            )
        except Exception as exc:
            raise JudgeError(f"live transcript request failed: {type(exc).__name__}") from exc
        return _lines_from_response(response)


def _visible_base_url(client: Any) -> str | None:
    from .judges import _visible_client_base_url

    return _visible_client_base_url(client)


def speech_intervals(samples: np.ndarray | None, sample_rate: int) -> list[tuple[float, float]]:
    """Loud regions of one tail, in seconds from the start of ``samples``.

    A silent partner channel does not hide speech: a frame counts when any
    channel is above the floor. Short gaps are merged, then blips are dropped.
    """
    if samples is None or sample_rate <= 0:
        return []
    array = np.asarray(samples)
    if array.size == 0:
        return []
    if array.ndim == 1:
        channels = [np.reshape(array, -1)]
    elif array.ndim == 2 and array.shape[1] >= 1:
        channels = [array[:, index] for index in range(array.shape[1])]
    else:
        return []
    frame = max(1, int(round(sample_rate * _SPEECH_FRAME_SECONDS)))
    usable = min(int(channel.shape[0]) for channel in channels)
    count = usable // frame
    if count <= 0:
        return []
    active = np.zeros(count, dtype=bool)
    for channel in channels:
        block = channel[: count * frame].astype(np.float32).reshape(count, frame)
        rms = np.sqrt(np.mean(block * block, axis=1))
        active |= rms >= _SPEECH_RMS_FLOOR
    runs: list[tuple[float, float]] = []
    index = 0
    while index < count:
        if not active[index]:
            index += 1
            continue
        end = index + 1
        while end < count and active[end]:
            end += 1
        runs.append((index * _SPEECH_FRAME_SECONDS, end * _SPEECH_FRAME_SECONDS))
        index = end
    merged: list[tuple[float, float]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= _SPEECH_MERGE_GAP_SECONDS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start >= _SPEECH_MIN_SECONDS]


def clip_speech_intervals(
    mic: np.ndarray,
    mic_rate: int,
    system: np.ndarray | None,
    system_rate: int,
) -> list[tuple[float, float]]:
    """Speech on the stereo tail, in seconds from the start of the clip.

    The WAV pads the shorter channel at the start so both ends match. Interval
    times use that same clock.
    """
    mic_intervals = speech_intervals(mic, mic_rate)
    system_intervals = speech_intervals(system, system_rate)
    mic_duration = (
        np.asarray(mic).shape[0] / float(mic_rate) if mic_rate and np.asarray(mic).size else 0.0
    )
    if system is None or system_rate <= 0 or np.asarray(system).size == 0:
        system_duration = 0.0
    else:
        system_duration = np.asarray(system).shape[0] / float(system_rate)
    clip = max(mic_duration, system_duration)
    shifted = _shift_intervals(mic_intervals, clip - mic_duration)
    shifted.extend(_shift_intervals(system_intervals, clip - system_duration))
    return _merge_intervals(shifted)


def uncovered_speech(
    intervals: Sequence[tuple[float, float]],
    lines: Sequence[Any],
    *,
    slack: float = 0.75,
) -> list[tuple[float, float]]:
    """Speech intervals that no line covers, in the same clock as ``lines``."""
    covered = [
        (float(line.start) - slack, float(line.end) + slack)
        for line in lines
    ]
    left: list[tuple[float, float]] = []
    for start, end in intervals:
        pieces = [(start, end)]
        for cover_start, cover_end in covered:
            nxt: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if piece_end <= cover_start or piece_start >= cover_end:
                    nxt.append((piece_start, piece_end))
                    continue
                if piece_start < cover_start:
                    nxt.append((piece_start, min(piece_end, cover_start)))
                if piece_end > cover_end:
                    nxt.append((max(piece_start, cover_end), piece_end))
            pieces = nxt
        left.extend((a, b) for a, b in pieces if b - a >= 0.3)
    return _merge_intervals(left)


def supplement_uncovered_lines(
    primary: Sequence[LiveLine],
    extra: Sequence[LiveLine],
    gaps: Sequence[tuple[float, float]],
    *,
    pad: float = 5.0,
) -> list[LiveLine]:
    """Keep gap-fill lines that land on, or just beside, speech the first answer missed.

    The second ask often times a short remark a few seconds off the energy
    burst. A line far from every gap stays out.
    """
    kept = list(primary)
    for line in extra:
        if any(
            _ranges_overlap(line.start, line.end, start - pad, end + pad)
            for start, end in gaps
        ):
            kept.append(line)
    kept.sort(key=lambda line: (line.start, line.end))
    return kept


def _shift_intervals(
    intervals: Sequence[tuple[float, float]], delta: float
) -> list[tuple[float, float]]:
    if delta <= 0:
        return [(start, end) for start, end in intervals]
    return [(start + delta, end + delta) for start, end in intervals]


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end - start <= 0:
            continue
        if merged and start - merged[-1][1] <= _SPEECH_MERGE_GAP_SECONDS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _ranges_overlap(start: float, end: float, other_start: float, other_end: float) -> bool:
    return min(end, other_end) - max(start, other_start) > 0.05


def _gap_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def gap_slices(
    gaps: Sequence[tuple[float, float]],
    duration: float,
    span: float = _GAP_SLICE_SECONDS,
) -> list[tuple[float, float]]:
    """Short windows centred on uncovered speech, in clip-relative seconds."""
    if duration <= 0 or not gaps:
        return []
    if duration <= span:
        return [(0.0, duration)]
    chosen: list[tuple[float, float]] = []
    for start, end in gaps:
        mid = (start + end) / 2.0
        left = min(max(0.0, mid - span / 2.0), duration - span)
        right = left + span
        if not chosen:
            chosen.append((left, right))
            continue
        union_left = min(chosen[-1][0], left)
        union_right = max(chosen[-1][1], right)
        if union_right - union_left <= span + 2.0:
            chosen[-1] = (union_left, min(duration, union_right))
        else:
            chosen.append((left, right))
    return chosen


def followup_slices(
    gaps: Sequence[tuple[float, float]],
    covered_end: float,
    duration: float,
    span: float = 10.0,
) -> list[tuple[float, float]]:
    """Slice the still-open tail, starting where the previous answer stopped.

    A window that holds two short remarks often returns only the first. The
    follow-up begins at that line. A gap near the end of the clip runs through
    the end, so the last words are in the audio.
    """
    chosen: list[tuple[float, float]] = []
    for start, end in gaps:
        if end <= covered_end + 0.25:
            continue
        if duration - end <= span:
            right = duration
            left = min(duration, max(0.0, covered_end, duration - span))
        else:
            left = min(max(0.0, covered_end), duration)
            right = min(duration, max(end + 0.75, left + span))
            if right - left > span:
                left = max(covered_end, right - span)
        if right - left < 0.5 or left >= right:
            continue
        if chosen and left <= chosen[-1][1]:
            chosen[-1] = (min(chosen[-1][0], left), max(chosen[-1][1], right))
        else:
            chosen.append((left, right))
    return chosen


def _slice_tail(samples: np.ndarray | None, sample_rate: int, start: float, end: float):
    if samples is None or sample_rate <= 0:
        return samples
    array = np.asarray(samples)
    if array.size == 0:
        return array
    first = min(array.shape[0], max(0, int(round(start * sample_rate))))
    last = min(array.shape[0], max(first, int(round(end * sample_rate))))
    return array[first:last]


def _offsets_from_slice(
    rows: Sequence[dict[str, Any]], slice_start: float, slice_duration: float
) -> list[dict[str, Any]]:
    """Shift clip-relative slice offsets onto the full tail.

    An offset past the slice is left alone so meeting-clock placement can
    rebase it. Adding the slice start on top of a meeting time would move it
    twice.
    """
    adjusted: list[dict[str, Any]] = []
    limit = slice_duration + _OFFSET_SLACK_SECONDS
    for row in rows:
        start = float(row["start_offset"])
        end = float(row["end_offset"])
        if start <= limit and end <= limit:
            row = dict(row)
            row["start_offset"] = start + slice_start
            row["end_offset"] = end + slice_start
        adjusted.append(row)
    return adjusted


def _transcription_prompt(has_system: bool) -> str:
    remote = (
        "The WAV is stereo and both channels end at the same moment. "
        "The left channel is the local microphone; label that speaker 'Ich'. "
        "The right channel is remote system audio; label those speakers 'Andere' unless a name is unmistakable."
        if has_system
        else "The WAV left channel is the local microphone. Label that speaker 'Ich'."
    )
    return (
        "Transcribe this meeting tail into Hochdeutsch, word for word. Do not summarise and do not shorten. "
        f"{remote} "
        "Normalise Swiss German into standard German. Keep English words and sentences in English. "
        "Keep names of AI tools as names. In particular, keep 'Claude' when it is spoken; do not rewrite it as an ordinary word. "
        "The local speaker sometimes dictates an instruction to Claude or to an agent in Swiss German "
        "(for example 'ich würde Claude sagen, er soll die Änderung umsetzen und nachfragen, was noch unklar ist'). "
        "Transcribe that instruction through to the last word you hear. "
        "Merge consecutive words from the same speaker into one line when the pause is under two seconds. "
        "A short remark by someone else does not end the local speaker's sentence: write the remark on its own line "
        "and continue the local speaker on the next line. "
        "Never write an ellipsis or three dots. Write the words that were spoken. Do not invent speech. Omit silence. "
        "start_offset and end_offset are seconds from the beginning of this audio, not from the start of the meeting. "
        "Do not add the elapsed meeting time. An offset past the end of this clip is wrong. "
        'Return JSON {"lines":[{"speaker":"...","text":"...","start_offset":0,"end_offset":0}]}.'
    )


def _gap_fill_prompt(has_system: bool) -> str:
    """Short second ask for speech the meeting prompt left out.

    The long prompt returns an empty list on a quiet tail that still contains
    short remarks. This ask is only used for those gaps. Offsets stay
    clip-relative so the same placement rules apply.
    """
    who = (
        "The left channel is the local microphone, speaker Ich. "
        "The right channel is remote audio, speaker Andere. "
        if has_system
        else "Label the speaker Ich. "
    )
    return (
        "Transcribe every spoken word in this audio into Hochdeutsch. "
        "Do not summarise. Include short remarks. "
        f"{who}"
        "start_offset and end_offset are seconds from the start of this clip. "
        "If you hear speech, you must include it. Omit only pure silence. "
        'Return JSON {"lines":[{"speaker":"...","text":"...","start_offset":0,"end_offset":0}]}.'
    )


def _transcript_schema() -> dict[str, Any]:
    line = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "text": {"type": "string"},
            "start_offset": {"type": "number"},
            "end_offset": {"type": "number"},
        },
        "required": ["speaker", "text", "start_offset", "end_offset"],
    }
    return {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": line}},
        "required": ["lines"],
    }


def _lines_from_response(response: Any) -> list[dict[str, Any]]:
    payload = _load_json_response(response)
    rows = payload.get("lines")
    if not isinstance(rows, list):
        raise JudgeError("live transcript JSON has no lines array")
    lines: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            start = float(row.get("start_offset"))
            end = float(row.get("end_offset"))
        except (TypeError, ValueError):
            continue
        if start != start or end != end:
            continue
        speaker = row.get("speaker")
        lines.append(
            {
                "speaker": speaker.strip() if isinstance(speaker, str) and speaker.strip() else "Ich",
                "text": text.strip(),
                "start_offset": max(0.0, start),
                "end_offset": max(0.0, end),
            }
        )
    return lines


def _absolute_lines(
    raw: Sequence[dict[str, Any]],
    window_start: float,
    window_end: float,
    trace: list[dict[str, Any]] | None = None,
) -> list[LiveLine]:
    """Place offsets on the microphone clock.

    Offsets within the tail are clip-relative. Offsets past the tail are
    treated as meeting time and kept only when they fall inside this window.
    Anything else is rejected. Clamping every outlier onto ``window_end``
    piled later lines on one instant and the deduper then dropped them.
    """
    span = max(0.0, float(window_end) - float(window_start))
    lines: list[LiveLine] = []
    for row in raw:
        start_offset = float(row["start_offset"])
        end_offset = float(row["end_offset"])
        decision, placed_start, placed_end = _place_offsets(
            start_offset, end_offset, float(window_start), float(window_end), span
        )
        if trace is not None:
            trace.append({
                "start_offset": start_offset,
                "end_offset": end_offset,
                "decision": decision,
                "placed_start": placed_start,
                "placed_end": placed_end,
            })
        if decision == "reject" or placed_start is None or placed_end is None:
            continue
        lines.append(
            LiveLine(
                placed_start,
                placed_end,
                str(row["speaker"]),
                str(row["text"]),
            )
        )
    return lines


def _place_offsets(
    start_offset: float,
    end_offset: float,
    window_start: float,
    window_end: float,
    span: float,
) -> tuple[str, float | None, float | None]:
    """Return ``(decision, absolute_start, absolute_end)``."""
    if (
        start_offset != start_offset
        or end_offset != end_offset
        or start_offset in (float("inf"), float("-inf"))
        or end_offset in (float("inf"), float("-inf"))
    ):
        return "reject", None, None
    slack = _OFFSET_SLACK_SECONDS
    meeting_clock = start_offset > span + slack or end_offset > span + slack
    if meeting_clock:
        if start_offset > window_end + slack or end_offset < window_start - slack:
            return "reject", None, None
        if end_offset < window_start - slack:
            return "reject", None, None
        placed_start = min(max(start_offset, window_start), window_end)
        placed_end = min(max(end_offset, placed_start), window_end)
        return "rebase", placed_start, placed_end
    if start_offset < -slack or end_offset < -slack:
        return "reject", None, None
    relative_start = min(max(0.0, start_offset), span)
    relative_end = min(max(relative_start, end_offset), span)
    return "relative", window_start + relative_start, window_start + relative_end


UpdateCallback = Callable[[tuple[LiveLine, ...], tuple[LiveCard, ...], str], None]
Cleaner = Callable[[str], str]


class LiveEngine:
    """Worker that ticks the live pipeline without joining the audio callback."""

    def __init__(
        self,
        *,
        snapshot: Callable[[float], TailSnapshot | None],
        transcriber: GeminiLiveTranscriber | None = None,
        transcriber_factory: Callable[[], GeminiLiveTranscriber] | None = None,
        judge: Judge | None = None,
        judge_factory: Callable[[], Judge | None] | None = None,
        threshold_factory: Callable[[], float] | None = None,
        clean_prompt: Cleaner | None = None,
        on_update: UpdateCallback | None = None,
        on_unavailable: Callable[[str], None] | None = None,
        on_commit: Callable[[Sequence[LiveLine], Sequence[LiveCard]], None] | None = None,
        on_offsets: Callable[[float, float, Sequence[dict[str, Any]]], None] | None = None,
        interval: float = LIVE_TICK_SECONDS,
        tail: float = LIVE_TAIL_SECONDS,
        prompt_threshold: float = OWNER_APPROVED_PROMPT_THRESHOLD,
    ):
        self._snapshot = snapshot
        self._transcriber = transcriber
        self._transcriber_factory = transcriber_factory
        self._judge = judge
        self._judge_factory = judge_factory
        self._threshold_factory = threshold_factory
        self._clean_prompt = clean_prompt
        self._on_update = on_update
        self._on_unavailable = on_unavailable
        self._on_commit = on_commit
        self._on_offsets = on_offsets
        self.interval = interval
        self.tail = tail
        self.prompt_threshold = prompt_threshold
        self.lines: list[LiveLine] = []
        self.cards: list[LiveCard] = []
        self.status = ""
        self.last_transcribe_seconds = 0.0
        self._scores: list[float] = []
        self._held: LiveLine | None = None
        self._clean_attempts: dict[str, int] = {}
        self._runtime_ready = False
        self._unavailable_notified = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="meeting-sidecar-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the worker to finish. Does not wait, so Stop stays responsive."""
        self._stop.set()

    def _loop(self) -> None:
        # Clients are built here, on the worker, before the first sleep. Start
        # has already returned on the main thread with only the panel open.
        try:
            self._ensure_runtime()
        except Exception as exc:
            self._fail(exc)
        while not self._stop.is_set():
            if self._stop.wait(self.interval):
                break
            self._safe_tick()
        # Committed lines are written at stop before the final transcription
        # tick, which may be skipped once the system WAV is archived.
        self._persist_tick(force=True)
        self._safe_tick(closed=True)

    def _ensure_runtime(self) -> None:
        """Build the Gemini clients on the worker, not on the menu thread."""
        if self._runtime_ready:
            return
        if self._transcriber is None:
            if self._transcriber_factory is None:
                raise RuntimeError("live transcriber is not configured")
            self._transcriber = self._transcriber_factory()
        if self._judge is None and self._judge_factory is not None:
            self._judge = self._judge_factory()
        if self._threshold_factory is not None:
            self.prompt_threshold = float(self._threshold_factory())
        self._runtime_ready = True

    def _safe_tick(self, *, closed: bool = False) -> None:
        try:
            snapshot = self._snapshot(self.tail)
            if snapshot is None:
                return
            if closed and not snapshot.closed:
                snapshot = replace(snapshot, closed=True)
            if closed and snapshot.system_gone:
                return
            self.process_snapshot(snapshot)
        except Exception as exc:
            self._fail(exc)
        finally:
            # At most once per tick, on this worker, never on the audio callback.
            self._persist_tick(force=closed)

    def _persist_tick(self, *, force: bool = False) -> None:
        if self._on_commit is None:
            return
        if not force and not self.lines and not self.cards:
            return
        try:
            self._on_commit(tuple(self.lines), tuple(self.cards))
        except Exception:
            self.status = (
                "Live-Sitzung konnte nicht gespeichert werden. Die Aufnahme läuft weiter."
            )

    def _fill_uncovered_speech(
        self,
        snapshot: TailSnapshot,
        incoming: list[LiveLine],
        trace: list[dict[str, Any]],
    ) -> list[LiveLine]:
        """Ask again when detected speech has no line, and keep only those gaps."""
        gap_fn = getattr(self._transcriber, "transcribe_gaps", None)
        if gap_fn is None:
            return incoming
        clip_gaps = uncovered_speech(
            [
                (snapshot.window_start + start, snapshot.window_start + end)
                for start, end in clip_speech_intervals(
                    snapshot.mic, snapshot.mic_rate, snapshot.system, snapshot.system_rate
                )
            ],
            incoming,
        )
        if _gap_duration(clip_gaps) < _GAP_FILL_MIN_SECONDS:
            return incoming
        absolute_speech = [
            (snapshot.window_start + start, snapshot.window_start + end)
            for start, end in clip_speech_intervals(
                snapshot.mic, snapshot.mic_rate, snapshot.system, snapshot.system_rate
            )
        ]
        return self._ask_gap_slices(snapshot, incoming, trace, gap_fn, absolute_speech, clip_gaps)

    def _ask_gap_slices(
        self,
        snapshot: TailSnapshot,
        incoming: list[LiveLine],
        trace: list[dict[str, Any]],
        gap_fn: Callable,
        absolute_speech: Sequence[tuple[float, float]],
        clip_gaps: Sequence[tuple[float, float]],
    ) -> list[LiveLine]:
        clip_duration = max(0.0, snapshot.window_end - snapshot.window_start)
        relative_gaps = [
            (start - snapshot.window_start, end - snapshot.window_start)
            for start, end in clip_gaps
        ]
        filled = supplement_uncovered_lines(
            incoming,
            self._transcribe_slices(
                snapshot, gap_fn, gap_slices(relative_gaps, clip_duration)[:3], trace
            ),
            clip_gaps,
        )
        follow_gaps = uncovered_speech(absolute_speech, filled)
        if _gap_duration(follow_gaps) < _GAP_FILL_MIN_SECONDS:
            return filled
        covered_end = max((line.end for line in filled), default=snapshot.window_start)
        follow = followup_slices(
            [
                (start - snapshot.window_start, end - snapshot.window_start)
                for start, end in follow_gaps
            ],
            covered_end - snapshot.window_start,
            clip_duration,
        )
        if not follow:
            return filled
        return supplement_uncovered_lines(
            filled,
            self._transcribe_slices(snapshot, gap_fn, follow[:2], trace),
            follow_gaps,
        )

    def _transcribe_slices(
        self,
        snapshot: TailSnapshot,
        gap_fn: Callable,
        slices: Sequence[tuple[float, float]],
        trace: list[dict[str, Any]],
    ) -> list[LiveLine]:
        extra_raw: list[dict[str, Any]] = []
        for slice_start, slice_end in slices:
            mic = _slice_tail(snapshot.mic, snapshot.mic_rate, slice_start, slice_end)
            system = _slice_tail(snapshot.system, snapshot.system_rate, slice_start, slice_end)
            rows: list[dict[str, Any]] = []
            for _attempt in range(2):
                try:
                    rows = gap_fn(mic, snapshot.mic_rate, system, snapshot.system_rate) or []
                except Exception:
                    rows = []
                if rows:
                    break
            extra_raw.extend(
                _offsets_from_slice(rows, slice_start, max(0.0, slice_end - slice_start))
            )
        return _absolute_lines(
            extra_raw, snapshot.window_start, snapshot.window_end, trace
        )

    def process_snapshot(self, snapshot: TailSnapshot) -> list[LiveLine]:
        """Transcribe one tail, commit new lines, and refresh prompt cards."""
        self._ensure_runtime()
        assert self._transcriber is not None
        started = time.perf_counter()
        raw = self._transcriber.transcribe_tails(
            snapshot.mic, snapshot.mic_rate, snapshot.system, snapshot.system_rate
        )
        trace: list[dict[str, Any]] = []
        incoming = _absolute_lines(
            raw, snapshot.window_start, snapshot.window_end, trace
        )
        incoming = self._fill_uncovered_speech(snapshot, incoming, trace)
        self.last_transcribe_seconds = time.perf_counter() - started
        if self._on_offsets is not None:
            try:
                self._on_offsets(snapshot.window_start, snapshot.window_end, trace)
            except Exception:
                pass
        incoming, self._held = _select_lines_to_commit(
            incoming,
            snapshot.window_end,
            closed=snapshot.closed,
            held=self._held,
            committed=self.lines,
        )
        self.lines, added, revised_from = commit_new_lines(self.lines, incoming)
        self.status = ""
        if added or revised_from is not None:
            try:
                if revised_from is not None:
                    self._scores = self._scores[:revised_from]
                    previous = revised_from
                else:
                    previous = len(self.lines) - len(added)
                self._extend_scores(previous)
                self._refresh_cards()
            except Exception as exc:
                self.status = (
                    f"Prompt-Erkennung unterbrochen ({type(exc).__name__}). "
                    "Die Aufnahme läuft weiter."
                )
        elif self._cards_need_clean_retry():
            try:
                self._refresh_cards()
            except Exception as exc:
                self.status = (
                    f"Sauberer Prompt fehlgeschlagen ({type(exc).__name__}). "
                    "Die Aufnahme läuft weiter."
                )
        self._publish()
        return added

    def _extend_scores(self, previous_count: int) -> None:
        if self._judge is None:
            self._scores = [0.0] * len(self.lines)
            return
        transcript = [line.as_transcript_line(index) for index, line in enumerate(self.lines)]
        context = 6
        start = max(0, previous_count - context)
        subset = transcript[start:]
        scores = self._judge.judge(
            subset, {"dictating_prompt": DICTATING_PROMPT_QUESTION}, context=context
        )["dictating_prompt"]
        if len(scores) != len(subset):
            raise JudgeError("prompt judge did not score every new line")
        # A later window can score the opening of a dictated prompt lower once
        # the rest of the sentence arrives. Keeping the higher score leaves
        # that opening in the run so the continuation stays on the same card.
        prior = self._scores[start:previous_count]
        merged: list[float] = []
        for index, score in enumerate(scores):
            value = float(score)
            if index < len(prior):
                value = max(prior[index], value)
            merged.append(value)
        self._scores = self._scores[:start] + merged

    def _refresh_cards(self) -> None:
        if self._judge is None or not self.lines or len(self._scores) != len(self.lines):
            return
        transcript = [line.as_transcript_line(index) for index, line in enumerate(self.lines)]
        scores = _scores_keeping_cut_off_instructions(
            self._scores, transcript, threshold=self.prompt_threshold
        )
        runs = _prompt_runs(scores, transcript, threshold=self.prompt_threshold)
        seen: set[str] = set()
        cards: list[LiveCard] = []
        for run in runs:
            if not run:
                continue
            key = f"{transcript[run[0]].seconds:.1f}"
            seen.add(key)
            verbatim_lines = tuple(transcript[index] for index in run)
            verbatim = "\n".join(line.text for line in verbatim_lines)
            previous = next((card for card in self.cards if card.key == key), None)
            cards.append(self._card_with_clean(key, verbatim_lines, verbatim, previous))
        # Keep a card that was already shown if a later rescore drops it for one tick.
        for card in self.cards:
            if card.key not in seen:
                cards.append(card)
        cards.sort(key=lambda card: card.start_seconds)
        self.cards = cards

    def _card_with_clean(
        self,
        key: str,
        verbatim_lines: tuple[TranscriptLine, ...],
        verbatim: str,
        previous: LiveCard | None,
    ) -> LiveCard:
        """Reuse a clean prompt only when one was actually produced.

        An empty result is a failed attempt. Later ticks retry until
        ``CLEAN_PROMPT_ATTEMPTS``, then the card stays on the verbatim text.
        """
        same = previous is not None and previous.verbatim_text == verbatim
        if same and previous is not None and previous.clean_text.strip():
            return previous
        attempts = self._clean_attempts.get(key, 0) if same else 0
        if not same:
            self._clean_attempts.pop(key, None)
        if same and previous is not None and attempts >= CLEAN_PROMPT_ATTEMPTS:
            return previous
        clean = ""
        if self._clean_prompt is not None:
            self._clean_attempts[key] = attempts + 1
            try:
                clean = (self._clean_prompt(verbatim) or "").strip()
            except Exception as exc:
                self.status = (
                    f"Sauberer Prompt fehlgeschlagen ({type(exc).__name__}). "
                    "Die Aufnahme läuft weiter."
                )
                clean = ""
        return LiveCard(key, verbatim_lines, clean)

    def _cards_need_clean_retry(self) -> bool:
        if self._clean_prompt is None or not self.cards:
            return False
        for card in self.cards:
            if card.clean_text.strip():
                continue
            if self._clean_attempts.get(card.key, 0) < CLEAN_PROMPT_ATTEMPTS:
                return True
        return False

    def _fail(self, exc: Exception) -> None:
        self.status = f"Live-Transkription unterbrochen ({type(exc).__name__}). Die Aufnahme läuft weiter."
        if not self._unavailable_notified and self._on_unavailable is not None:
            self._unavailable_notified = True
            try:
                self._on_unavailable(self.status)
            except Exception:
                pass
        self._publish()

    def _publish(self) -> None:
        if self._on_update is None:
            return
        self._on_update(tuple(self.lines), tuple(self.cards), self.status)


def replay_wav(
    path: Path,
    *,
    transcriber: GeminiLiveTranscriber,
    judge: Judge | None = None,
    clean_prompt: Cleaner | None = None,
    tick: float = LIVE_TICK_SECONDS,
    tail: float = LIVE_TAIL_SECONDS,
    prompt_threshold: float = OWNER_APPROVED_PROMPT_THRESHOLD,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Feed a finished WAV through the live pipeline on a simulated clock.

    Audio is revealed only up to the current tick. Lag is tick time plus the
    real transcribe duration, minus the line's audio time. No window is opened.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="int16", always_2d=True)
    duration = len(audio) / float(rate) if rate else 0.0
    begin = max(0.0, start_seconds)
    finish = duration if end_seconds is None else min(duration, end_seconds)
    engine = LiveEngine(
        snapshot=lambda _tail: None,
        transcriber=transcriber,
        judge=judge,
        clean_prompt=clean_prompt,
        interval=tick,
        tail=tail,
        prompt_threshold=prompt_threshold,
    )
    lags: list[float] = []
    printed_cards: dict[str, tuple[str, str]] = {}
    write = emit or print

    def _log_offsets(window_start: float, window_end: float, trace: Sequence[dict[str, Any]]) -> None:
        pieces = []
        for item in trace:
            piece = (
                f"{item['start_offset']:.3f}-{item['end_offset']:.3f}:{item['decision']}"
            )
            if item.get("placed_start") is not None:
                piece += f"->{item['placed_start']:.3f}-{item['placed_end']:.3f}"
            pieces.append(piece)
        write(f"OFFSETS {window_start:.3f}-{window_end:.3f} " + " ".join(pieces))

    engine._on_offsets = _log_offsets
    cursor = begin
    while cursor < finish - 1e-6:
        cursor = min(finish, cursor + tick)
        window_start = max(0.0, cursor - tail)
        start_index = int(window_start * rate)
        end_index = max(start_index, int(cursor * rate))
        frame = audio[start_index:end_index]
        mic = frame[:, 0] if frame.size else np.zeros(0, dtype=np.int16)
        system = frame[:, 1:] if frame.ndim == 2 and frame.shape[1] > 1 else None
        closed = cursor >= finish - 1e-6
        snapshot = TailSnapshot(
            mic, int(rate), system, int(rate), window_start, cursor, closed=closed
        )
        try:
            added = engine.process_snapshot(snapshot)
        except Exception as exc:
            write(f"STATUS {engine.status or type(exc).__name__}")
            continue
        # Lag is audio time to the transcript line. Prompt judging runs after
        # the line exists and does not count toward that lag.
        available = cursor + engine.last_transcribe_seconds
        for line in added:
            lag = available - line.start
            lags.append(lag)
            write(
                f"LINE {format_timestamp(line.start)} {line.speaker}: {line.text} lag={lag:.1f}s"
            )
        for card in engine.cards:
            signature = (card.verbatim_text, card.clean_text)
            if printed_cards.get(card.key) == signature:
                continue
            printed_cards[card.key] = signature
            write(
                f"CARD {format_timestamp(card.start_seconds)}-{format_timestamp(card.end_seconds)}"
            )
            write(f"CLEAN: {card.clean_text}")
            write(f"VERBATIM: {card.verbatim_text}")
        if engine.status:
            write(f"STATUS {engine.status}")
    median = _median(lags)
    write(
        f"median_lag_seconds={median:.2f}" if median is not None else "median_lag_seconds="
    )
    write(f"lines={len(engine.lines)} cards={len(engine.cards)}")
    return {
        "lines": tuple(engine.lines),
        "cards": tuple(engine.cards),
        "lags": tuple(lags),
        "median_lag_seconds": median,
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# Re-exported so a caller can see the merge gap the live cards use.
LIVE_PROMPT_GAP_SECONDS = PROMPT_CONSECUTIVE_GAP_SECONDS


def default_clean_prompt(api_key: str | None = None) -> Cleaner:
    def clean(verbatim: str) -> str:
        return clean_prompt_text(verbatim, api_key=api_key)

    return clean
