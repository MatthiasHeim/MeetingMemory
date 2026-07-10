#!/usr/bin/env python3
"""transcript_validator — completeness/coverage gate for Gemini transcripts.

Sits between the Gemini result and the JSON write in
transcribe_watcher._process_with_gemini (see docs/RELIABILITY_PLAN_2026-07.md
Phase 1). A transcript must pass every check here before it is stored as a
clean success — the two failure modes it exists to catch both look like
valid, error-free output otherwise:

  - Silent truncation: pro can return well-formed JSON that stops far short
    of the audio's actual length (a 50.5min call once ended at [08:17], 16%
    coverage, no error, no disconnect).
  - Undeduped chunk overlap: the old chunked fallback concatenated a 30s
    overlap with no dedup and diarized each chunk independently, producing
    duplicated text with inverted speaker labels.

`validate_transcript` never raises on malformed input — a transcript that
fails to parse is itself a validation failure, not an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Coverage: max transcript timestamp must reach this fraction of the audio's
# actual duration. Below this, the transcript is treated as truncated.
COVERAGE_MIN_PCT = 90.0

# Upper coverage bound (F4, docs/SPEC-error-path-escalation-2026-07-10.md
# RC4): the drift-proof chunking continuity prefix shows Gemini the previous
# chunk's tail with chunk-local timestamps and says "continue" -- Gemini
# sometimes obeys too literally and keeps counting from there instead of
# restarting at [00:00], so _shift_timestamps then double-counts the chunk
# offset on top. That drift lands at ~130-200% coverage (observed 143.2%,
# source 427-adjacent incident 2026-07-09); 110% leaves headroom for normal
# rounding/dictation slop at the very end of a recording.
COVERAGE_MAX_PCT = 110.0

# A phrase of 1-4 words repeating more than this many times *consecutively*
# is a runaway-generation loop (observed: "s'heisst" x~400 in source 429).
REPETITION_MAX_CONSECUTIVE = 15
REPETITION_MAX_PHRASE_WORDS = 4

# An 8-word shingle appearing twice more than this many words apart indicates
# duplicated text from an undeduped chunk overlap, not natural repetition.
SHINGLE_SIZE = 8
SHINGLE_MIN_DISTANCE_WORDS = 30

# Marker spliced in where sanitize_repetition_loops() collapsed a runaway
# repeat (F6) -- makes the glitch visible in the stored transcript instead
# of silently vanishing.
GLITCH_MARKER = "[transcription glitch]"

_TIMESTAMP_RE = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2})\]")
_CHUNK_FAILED_RE = re.compile(r"\[CHUNK\s+\d+\s+FAILED\b", re.IGNORECASE)
_LEADING_TS_RE = re.compile(r"^\[(\d{1,2}(?::\d{2}){1,2})\]")


@dataclass
class ValidationResult:
    """Outcome of validating one transcript against its audio duration."""

    passed: bool
    coverage_pct: float
    last_timestamp_sec: float
    reasons: list[str] = field(default_factory=list)
    has_chunk_failure_marker: bool = False
    has_repetition_loop: bool = False
    has_duplicate_span: bool = False
    sanitized: bool = False
    sanitized_locations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable form for persistence into result `_meta`."""
        return {
            "passed": self.passed,
            "coverage_pct": self.coverage_pct,
            "last_timestamp_sec": self.last_timestamp_sec,
            "reasons": list(self.reasons),
            "has_chunk_failure_marker": self.has_chunk_failure_marker,
            "has_repetition_loop": self.has_repetition_loop,
            "has_duplicate_span": self.has_duplicate_span,
            "sanitized": self.sanitized,
            "sanitized_locations": list(self.sanitized_locations),
        }


def _ts_to_sec(ts: str) -> float:
    parts = [int(x) for x in ts.split(":")]
    return float(sum(p * 60 ** (len(parts) - 1 - i) for i, p in enumerate(parts)))


def _max_timestamp_sec(transcript: str) -> float:
    matches = _TIMESTAMP_RE.findall(transcript)
    if not matches:
        return -1.0
    return max(_ts_to_sec(m) for m in matches)


def _has_chunk_failure_marker(transcript: str) -> bool:
    return bool(_CHUNK_FAILED_RE.search(transcript))


def _consecutive_line_repetition(transcript: str) -> bool:
    """True if any non-empty line repeats >REPETITION_MAX_CONSECUTIVE times
    in a row (verbatim, after stripping surrounding whitespace)."""
    lines = [ln.strip() for ln in transcript.split("\n") if ln.strip()]
    run_line = None
    run_len = 0
    for ln in lines:
        if ln == run_line:
            run_len += 1
        else:
            run_line = ln
            run_len = 1
        if run_len > REPETITION_MAX_CONSECUTIVE:
            return True
    return False


def _consecutive_phrase_repetition(transcript: str) -> bool:
    """True if any 1-4 word phrase repeats >REPETITION_MAX_CONSECUTIVE times
    back-to-back anywhere in the token stream (catches loops inside a single
    turn's text, not just whole-line repeats)."""
    words = transcript.split()
    n_words = len(words)
    for n in range(1, REPETITION_MAX_PHRASE_WORDS + 1):
        i = 0
        while i + n <= n_words:
            j = i + n
            count = 1
            while j + n <= n_words and words[j:j + n] == words[i:i + n]:
                count += 1
                j += n
            if count > REPETITION_MAX_CONSECUTIVE:
                return True
            i = j if count > 1 else i + 1
    return False


def _duplicate_span(transcript: str) -> bool:
    """True if an 8-gram shingle recurs more than SHINGLE_MIN_DISTANCE_WORDS
    words apart — the signature of an undeduped chunk-overlap duplicate."""
    words = transcript.split()
    if len(words) < SHINGLE_SIZE + 1:
        return False
    first_seen: dict[tuple, int] = {}
    for i in range(len(words) - SHINGLE_SIZE + 1):
        shingle = tuple(words[i:i + SHINGLE_SIZE])
        prev = first_seen.get(shingle)
        if prev is not None and (i - prev) > SHINGLE_MIN_DISTANCE_WORDS:
            return True
        if prev is None:
            first_seen[shingle] = i
    return False


def validate_transcript(transcript: str, audio_duration_seconds: float) -> ValidationResult:
    """Validate a Gemini transcript against the recording's actual duration.

    Returns a ValidationResult; `passed` is False if any check fails. Never
    raises — malformed/empty input is itself represented as a failure.
    """
    reasons: list[str] = []

    last_ts = _max_timestamp_sec(transcript)
    if last_ts < 0:
        coverage_pct = 0.0
        reasons.append("No timestamps found in transcript — cannot verify coverage")
        last_timestamp_sec = 0.0
    else:
        last_timestamp_sec = last_ts
        if audio_duration_seconds and audio_duration_seconds > 0:
            coverage_pct = round(100.0 * last_ts / audio_duration_seconds, 4)
        else:
            coverage_pct = 0.0
            reasons.append("Invalid audio_duration_seconds — cannot compute coverage")
        if coverage_pct < COVERAGE_MIN_PCT:
            reasons.append(
                f"Coverage {coverage_pct:.1f}% is below the "
                f"{COVERAGE_MIN_PCT:.0f}% minimum (last timestamp "
                f"{last_timestamp_sec:.0f}s of {audio_duration_seconds:.0f}s)"
            )
        elif coverage_pct > COVERAGE_MAX_PCT:
            reasons.append(
                f"Coverage {coverage_pct:.1f}% exceeds the "
                f"{COVERAGE_MAX_PCT:.0f}% maximum (last timestamp "
                f"{last_timestamp_sec:.0f}s of {audio_duration_seconds:.0f}s) "
                f"— last timestamp exceeds audio duration — timestamp drift"
            )

    has_chunk_failure_marker = _has_chunk_failure_marker(transcript)
    if has_chunk_failure_marker:
        reasons.append("Transcript contains a [CHUNK N FAILED] marker — a chunk was silently dropped")

    has_repetition_loop = (
        _consecutive_line_repetition(transcript)
        or _consecutive_phrase_repetition(transcript)
    )
    if has_repetition_loop:
        reasons.append(
            f"Repetition loop detected — a line or short phrase repeats more "
            f"than {REPETITION_MAX_CONSECUTIVE}x consecutively"
        )

    has_duplicate_span = _duplicate_span(transcript)
    if has_duplicate_span:
        reasons.append(
            f"Duplicate span detected — an {SHINGLE_SIZE}-word sequence "
            f"recurs more than {SHINGLE_MIN_DISTANCE_WORDS} words apart "
            f"(likely an undeduped chunk overlap)"
        )

    passed = (
        COVERAGE_MIN_PCT <= coverage_pct <= COVERAGE_MAX_PCT
        and not has_chunk_failure_marker
        and not has_repetition_loop
        and not has_duplicate_span
    )

    return ValidationResult(
        passed=passed,
        coverage_pct=coverage_pct,
        last_timestamp_sec=last_timestamp_sec,
        reasons=reasons,
        has_chunk_failure_marker=has_chunk_failure_marker,
        has_repetition_loop=has_repetition_loop,
        has_duplicate_span=has_duplicate_span,
    )


def _collapse_phrase_repeats(words: list[str]) -> tuple[list[str], list[tuple[str, int]]]:
    """Collapse runs of a 1-4 word phrase repeating more than
    REPETITION_MAX_CONSECUTIVE times consecutively to one instance + a
    GLITCH_MARKER. Mirrors _consecutive_phrase_repetition's detection loop
    but rewrites the run instead of just flagging it."""
    out: list[str] = []
    hits: list[tuple[str, int]] = []
    n_words = len(words)
    i = 0
    while i < n_words:
        collapsed = False
        for n in range(1, REPETITION_MAX_PHRASE_WORDS + 1):
            if i + n > n_words:
                continue
            phrase = words[i:i + n]
            j = i + n
            count = 1
            while j + n <= n_words and words[j:j + n] == phrase:
                count += 1
                j += n
            if count > REPETITION_MAX_CONSECUTIVE:
                out.extend(phrase)
                out.append(GLITCH_MARKER)
                hits.append((" ".join(phrase), count))
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return out, hits


def sanitize_repetition_loops(transcript: str) -> tuple[str, list[dict]]:
    """Collapse runaway repetition loops down to one instance + GLITCH_MARKER.

    Handles two shapes, both observed in practice:
      - a whole line repeated verbatim (the 429 "s'heisst" case, when it
        lands as separate turns);
      - a 1-4 word phrase repeating within a single turn's line (source 463,
        2026-07-10: "de," x595 mid-sentence, ~5s of audio replaced by a
        runaway loop while the surrounding 23.9 minutes were fine).

    Scoped per-line rather than over the whole token stream: an observed
    runaway loop stays within one transcribed turn, so this is both simpler
    and safe -- it can never bridge a timestamp boundary into the next turn.

    Returns (sanitized_transcript, locations); locations is empty if nothing
    was collapsed (transcript returned unchanged, same string).
    """
    lines = transcript.split("\n")
    out_lines: list[str] = []
    locations: list[dict] = []

    # Pass 1: whole-line repeats.
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            i += 1
            continue
        j = i + 1
        count = 1
        while j < n_lines and lines[j].strip() == stripped:
            count += 1
            j += 1
        if count > REPETITION_MAX_CONSECUTIVE:
            out_lines.append(line)
            out_lines.append(GLITCH_MARKER)
            ts_match = _LEADING_TS_RE.match(stripped)
            locations.append({
                "kind": "line",
                "phrase": stripped[:80],
                "count": count,
                "timestamp": f"[{ts_match.group(1)}]" if ts_match else None,
            })
            i = j
        else:
            out_lines.extend(lines[i:j])
            i = j

    # Pass 2: in-line repeated 1-4 word phrases.
    final_lines: list[str] = []
    for line in out_lines:
        if line == GLITCH_MARKER:
            final_lines.append(line)
            continue
        words = line.split()
        if not words:
            final_lines.append(line)
            continue
        new_words, hits = _collapse_phrase_repeats(words)
        if hits:
            ts_match = _LEADING_TS_RE.match(line.strip())
            ts = f"[{ts_match.group(1)}]" if ts_match else None
            for phrase, count in hits:
                locations.append({
                    "kind": "phrase", "phrase": phrase, "count": count, "timestamp": ts,
                })
            final_lines.append(" ".join(new_words))
        else:
            final_lines.append(line)

    return "\n".join(final_lines), locations
