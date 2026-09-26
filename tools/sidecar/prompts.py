"""Prompt-card extraction from recorder marks and line-level judge decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .judges import (
    JUDGE_PROMPT_VERSION,
    JUDGE_PROTOCOL_FINGERPRINT,
    Judge,
    question_protocol_fingerprint,
)
from .questions import DICTATING_PROMPT_QUESTION, PROMPT_CONTENT_QUESTION
from .transcript import TranscriptLine


PROMPT_THRESHOLD = 0.45
PROMPT_CONTENT_THRESHOLD = 0.50
MARK_LOOKBACK_SECONDS = 5.0
MARK_MAX_SECONDS = 3 * 60.0
PROMPT_CONSECUTIVE_GAP_SECONDS = 12.0
DEFAULT_CALIBRATION_REPORT = Path.home() / ".local" / "share" / "meeting-sidecar" / "calibration" / "latest.json"


class PromptCalibrationError(RuntimeError):
    """The requested Gemini prompt selector has no passing threshold."""


def calibrated_prompt_threshold(model: str | None, report_path: str | Path | None = None) -> float:
    """Load the threshold selected by a passing Gemini calibration report.

    A failed, stale, or model-mismatched report is intentionally not a license
    to fall back to the historical ``0.45`` candidate. The runtime must use the
    exact question/model pairing whose threshold passed the acceptance gate.
    """
    if not model:
        raise PromptCalibrationError("Gemini prompt extraction needs a named calibrated model")
    path = Path(report_path or DEFAULT_CALIBRATION_REPORT).expanduser()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptCalibrationError("no readable Gemini calibration report is available") from exc
    if not isinstance(report, dict):
        raise PromptCalibrationError("the Gemini calibration report is not an object")
    chosen = report.get("chosen")
    if report.get("status") != "passed" or not isinstance(chosen, dict):
        raise PromptCalibrationError("Gemini prompt extraction is disabled until calibration criterion 2 passes")
    if report.get("judge_prompt_version") != JUDGE_PROMPT_VERSION:
        raise PromptCalibrationError("the Gemini calibration report was produced by a different judge prompt")
    if report.get("judge_protocol_sha256") != JUDGE_PROTOCOL_FINGERPRINT:
        raise PromptCalibrationError("the Gemini calibration report has a different batch protocol/schema")
    if report.get("dictating_prompt_protocol_sha256") != question_protocol_fingerprint(
        "dictating_prompt", DICTATING_PROMPT_QUESTION
    ):
        raise PromptCalibrationError("the Gemini calibration report has a different dictating-prompt definition")
    if chosen.get("model") != model:
        raise PromptCalibrationError(
            f"Gemini prompt extraction has no passing calibration for model {model!r}"
        )
    try:
        threshold = float(chosen["prompt_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromptCalibrationError("the selected Gemini calibration has no usable prompt threshold") from exc
    if not 0 <= threshold <= 1:
        raise PromptCalibrationError("the selected Gemini prompt threshold is outside 0..1")
    return threshold


@dataclass(frozen=True)
class PromptResult:
    """Verbatim consecutive prompt lines, either mark-backed or suggested."""

    lines: tuple[TranscriptLine, ...]
    source: str  # "mark" or "suggested"
    mark_seconds: float | None = None

    @property
    def text(self) -> str:
        # Prompt copying is verbatim content, not a transcript citation. The
        # card UI can still show its timestamp/source separately.
        return "\n".join(line.text for line in self.lines)

    @property
    def start_seconds(self) -> float:
        return self.lines[0].seconds


def mark_to_transcript_seconds(
    mark_offset_seconds: float,
    *,
    channel_lag_seconds: float = 0.0,
    mic_origin_delay_seconds: float = 0.0,
) -> float:
    """Map a mark to a transcript coordinate when the capture offsets are known.

    ``channel_align`` defines positive lag as a mic track that is late and
    advances it by that amount. The corrected mic coordinate is therefore
    ``mark - mic_origin_delay - lag``. The current recorder does not persist
    either offset alongside final transcript JSON, so normal operation uses the
    nominal zero-offset mapping and searches five seconds before the mark.
    """
    value = float(mark_offset_seconds) - float(mic_origin_delay_seconds) - float(channel_lag_seconds)
    return max(0.0, value)


def marked_window_indices(
    lines: Sequence[TranscriptLine],
    mark_seconds: float,
    *,
    lookback_seconds: float = MARK_LOOKBACK_SECONDS,
    max_seconds: float = MARK_MAX_SECONDS,
) -> tuple[int, ...]:
    """Return transcript positions in the bounded mark window.

    The public mark is a monotonic elapsed offset. It maps nominally to the
    final transcript's timeline; no hidden wall-clock conversion is attempted.
    """
    anchor = mark_to_transcript_seconds(mark_seconds)
    start = max(0.0, anchor - lookback_seconds)
    end = anchor + max_seconds
    return tuple(index for index, line in enumerate(lines) if start <= line.seconds <= end)


def _prompt_runs(
    probabilities: Sequence[float],
    lines: Sequence[TranscriptLine],
    *,
    threshold: float = PROMPT_THRESHOLD,
) -> list[tuple[int, int]]:
    """Merge consecutive prompt lines, allowing one low-confidence bridge."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    low_gap = 0
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            if start is None:
                start = index
            low_gap = 0
            continue
        if (
            start is not None
            and low_gap == 0
            and index > start
            and lines[index].seconds - lines[index - 1].seconds <= PROMPT_CONSECUTIVE_GAP_SECONDS
        ):
            # A short acknowledgement can appear between dictation sentences.
            low_gap = 1
            continue
        if start is not None:
            end = index - low_gap - 1
            runs.append((start, end))
            start = None
            low_gap = 0
    if start is not None:
        runs.append((start, len(probabilities) - low_gap - 1))
    return [(start, end) for start, end in runs if end >= start]


def _closest_run(
    runs: Sequence[tuple[int, int]],
    lines: Sequence[TranscriptLine],
    positions: set[int],
    mark_seconds: float,
) -> tuple[int, int] | None:
    candidate_runs = [
        run for run in runs if any(position in positions for position in range(run[0], run[1] + 1))
    ]
    if not candidate_runs:
        return None
    return min(
        candidate_runs,
        key=lambda run: min(abs(lines[position].seconds - mark_seconds) for position in range(run[0], run[1] + 1)),
    )


def _strip_leadin(
    lines: Sequence[TranscriptLine],
    positions: Sequence[int],
    judge: Judge,
    *,
    context: int,
) -> tuple[TranscriptLine, ...]:
    if not positions:
        return ()
    sentence_lines = _verbatim_sentence_lines(lines, positions)
    if not sentence_lines:
        return ()
    # A recorder turn can contain several sentences. Judge those verbatim
    # sentence substrings individually so a lead-in and the dictated prompt in
    # one turn are not forced to live or die together. The synthetic indexes
    # are unique for the judge protocol; timestamps/speakers stay attached to
    # their original turn and no text is generated or rewritten.
    scores = judge.judge(
        sentence_lines,
        {"prompt_content": PROMPT_CONTENT_QUESTION},
        context=context,
    )["prompt_content"]
    # This is lead-in stripping, not free-form sentence filtering. Once the
    # first prompt sentence is reached, retain every remaining verbatim
    # sentence—including a low-scored interior constraint—rather than silently
    # changing the meaning of a copied prompt.
    first_prompt = next(
        (index for index, score in enumerate(scores) if score >= PROMPT_CONTENT_THRESHOLD),
        None,
    )
    return () if first_prompt is None else tuple(sentence_lines[first_prompt:])


def _verbatim_sentence_lines(
    lines: Sequence[TranscriptLine], positions: Sequence[int]
) -> tuple[TranscriptLine, ...]:
    """Split candidate turns into exact sentence substrings for lead-in judging.

    The final fallback keeps an unpunctuated turn as a single sentence. This is
    deliberately a small deterministic boundary detector, not an LLM rewrite;
    punctuation and words are copied from the transcript unchanged.
    """
    result: list[TranscriptLine] = []
    for position in positions:
        source = lines[position]
        for text in _verbatim_sentences(source.text):
            result.append(
                TranscriptLine(
                    index=len(result),
                    seconds=source.seconds,
                    timestamp=source.timestamp,
                    speaker=source.speaker,
                    text=text,
                )
            )
    return tuple(result)


def _verbatim_sentences(text: str) -> tuple[str, ...]:
    """Return sentence substrings without changing their words or punctuation."""
    result: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".!?":
            continue
        after = index + 1
        # Decimal numbers and abbreviations without following whitespace are
        # not sentence boundaries. Include trailing closing punctuation in the
        # same copied substring before determining the next start.
        while after < len(text) and text[after] in "\\\"'”’)]":
            after += 1
        if after < len(text) and not text[after].isspace():
            continue
        sentence = text[start:after].strip()
        if sentence:
            result.append(sentence)
        start = after
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return tuple(result)


def prompts_from_marks(
    lines: Sequence[TranscriptLine],
    marks: Sequence[float],
    *,
    judge: Judge,
    context: int = 6,
    prompt_threshold: float = PROMPT_THRESHOLD,
) -> list[PromptResult]:
    """Return marked prompt cards first, then unmarked suggested prompt cards.

    A mark constrains candidate lines to ``mark - 5s`` through ``mark + 3min``.
    A line-level prompt judge decides where a prompt starts/ends; a separate
    sentence-level judge strips only non-prompt lead-ins. Consecutive lines are
    always returned together as one card.
    """
    if not lines:
        return []
    prompt_scores = judge.judge(
        lines,
        {"dictating_prompt": DICTATING_PROMPT_QUESTION},
        context=context,
    )["dictating_prompt"]
    if not 0 <= prompt_threshold <= 1:
        raise ValueError("prompt threshold must be in 0..1")
    runs = _prompt_runs(prompt_scores, lines, threshold=prompt_threshold)
    marked_positions: set[int] = set()
    results: list[PromptResult] = []
    for raw_mark in marks:
        try:
            mark = float(raw_mark)
        except (TypeError, ValueError):
            continue
        if mark < 0:
            continue
        positions = set(marked_window_indices(lines, mark))
        run = _closest_run(runs, lines, positions, mark)
        if run is None:
            continue
        run_positions = tuple(range(run[0], run[1] + 1))
        if any(position in marked_positions for position in run_positions):
            # Repeated hotkey presses around the same dictation must not make
            # duplicate cards for the same verbatim prompt run.
            continue
        # A mark suppresses only the exact run it selected in its bounded
        # [mark - 5s, mark + 3min] window. A different candidate merely near a
        # mark remains an honest suggested prompt.
        marked_positions.update(run_positions)
        kept = _strip_leadin(lines, run_positions, judge, context=context)
        if kept:
            results.append(PromptResult(kept, "mark", mark))

    for start, end in runs:
        positions = tuple(range(start, end + 1))
        if any(position in marked_positions for position in positions):
            continue
        kept = _strip_leadin(lines, positions, judge, context=context)
        if kept:
            results.append(PromptResult(kept, "suggested", None))
    return results
