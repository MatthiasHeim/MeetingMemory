"""Tests for transcript_validator — the coverage/completeness gate that sits
between the Gemini result and the JSON write (see docs/RELIABILITY_PLAN_2026-07.md
Phase 1). A transcript must pass every check here or the pipeline must retry /
escalate instead of silently storing a truncated or corrupted result.

Written FIRST (red) before tools/transcript_validator.py exists (green).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from transcript_validator import validate_transcript  # noqa: E402


def _turns(specs: list[tuple[str, str]]) -> str:
    """Build a transcript from (timestamp, text) pairs, one turn per line."""
    return "\n".join(f"[{ts}] Matthias: {text}" for ts, text in specs)


# ── coverage check ──────────────────────────────────────────────────────


def test_coverage_passes_at_exactly_90_percent():
    transcript = _turns([("00:00", "hello"), ("54:00", "closing remarks")])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.coverage_pct == 90.0
    assert result.passed is True
    assert result.reasons == []


def test_coverage_fails_below_90_percent():
    """The empirical case: a 50.5min call whose JSON ends at [08:17] (16%)."""
    transcript = _turns([("00:00", "hello"), ("08:17", "still going")])
    result = validate_transcript(transcript, audio_duration_seconds=50.5 * 60)
    assert result.passed is False
    assert round(result.coverage_pct, 1) == 16.4
    assert any("coverage" in r.lower() for r in result.reasons)


def test_coverage_handles_hh_mm_ss_timestamps():
    """Recordings over an hour use [HH:MM:SS]."""
    transcript = _turns([("00:00:00", "hi"), ("01:05:30", "wrapping up")])
    result = validate_transcript(transcript, audio_duration_seconds=70 * 60)
    assert result.passed is True
    assert result.last_timestamp_sec == 1 * 3600 + 5 * 60 + 30


def test_no_timestamps_found_fails_with_zero_coverage():
    result = validate_transcript("Matthias: hello there, no timestamps here.",
                                  audio_duration_seconds=600)
    assert result.passed is False
    assert result.coverage_pct == 0.0
    assert any("no timestamps" in r.lower() for r in result.reasons)


# ── chunk-failure marker ────────────────────────────────────────────────


def test_chunk_failed_marker_fails_validation():
    transcript = (
        _turns([("00:00", "start"), ("14:59", "end of chunk 1")])
        + "\n\n[CHUNK 2 FAILED: Server disconnected without sending a response.]\n\n"
        + _turns([("30:00", "chunk 3 starts")])
    )
    result = validate_transcript(transcript, audio_duration_seconds=45 * 60)
    assert result.passed is False
    assert result.has_chunk_failure_marker is True
    assert any("chunk" in r.lower() and "fail" in r.lower() for r in result.reasons)


def test_no_chunk_marker_does_not_false_positive_on_word_failed():
    """The word 'failed' appearing in normal speech must not trip the marker
    check — only the literal [CHUNK N FAILED marker shape counts."""
    transcript = _turns([
        ("00:00", "the deployment failed yesterday, we should discuss"),
        ("54:00", "closing remarks"),
    ])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.has_chunk_failure_marker is False


# ── repetition loop ──────────────────────────────────────────────────────


def test_identical_line_repeated_over_15x_fails():
    lines = ["[10:00] Matthias: s'heisst."] * 20
    transcript = "\n".join(lines)
    result = validate_transcript(transcript, audio_duration_seconds=600)
    assert result.passed is False
    assert result.has_repetition_loop is True
    assert any("repetition" in r.lower() or "repeat" in r.lower()
               for r in result.reasons)


def test_short_phrase_repeated_over_15x_within_one_line_fails():
    """The 429 [52:07] bug: 's'heisst' repeated ~400x inside a single turn,
    not as separate lines."""
    loop = ("s'heisst " * 400).strip()
    transcript = _turns([("00:00", "normal start"), ("52:07", loop)])
    result = validate_transcript(transcript, audio_duration_seconds=3600)
    assert result.passed is False
    assert result.has_repetition_loop is True


def test_normal_backchannel_repetition_does_not_fail():
    """Conversational repeats ('ja, ja, genau') a handful of times are normal
    and must not trip the detector."""
    text = "ja ja genau, das stimmt, wir machen weiter mit dem naechsten Punkt"
    transcript = _turns([("00:00", text), ("54:00", "closing")])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.has_repetition_loop is False


def test_repetition_exactly_15_times_passes():
    """Boundary: >15 fails, so exactly 15 consecutive repeats must pass."""
    loop = ("echo " * 15).strip()
    transcript = _turns([("00:00", loop), ("54:00", "closing")])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.has_repetition_loop is False


# ── duplicate span (undeduped chunk overlap) ─────────────────────────────


def test_duplicate_span_far_apart_fails():
    """Simulates the 429 bug: a 30s overlap concatenated with no dedup,
    producing the same words far apart in word-index terms."""
    shared = "wir sollten definitiv auf die neue Plattform wechseln naechstes Jahr"
    filler = " ".join(f"wort{i}" for i in range(60))
    transcript = _turns([
        ("06:49", shared),
        ("07:30", filler),
        ("08:51", shared),
    ])
    result = validate_transcript(transcript, audio_duration_seconds=3600)
    assert result.passed is False
    assert result.has_duplicate_span is True
    assert any("duplicate" in r.lower() for r in result.reasons)


def test_close_repetition_of_short_phrase_not_flagged_as_duplicate_span():
    """An 8-gram that repeats within a small word window (e.g. natural
    immediate repetition) is a repetition-loop concern, not a duplicate-span
    one — spans within 30 words must not double-count as duplicate spans."""
    text = "wir muessen das nochmal pruefen wir muessen das nochmal pruefen genau"
    transcript = _turns([("00:00", text), ("54:00", "closing")])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.has_duplicate_span is False


def test_natural_short_shingle_repeat_within_30_words_not_flagged():
    shared = "vielen dank fuer eure zeit heute nachmittag"  # 8 words
    filler = " ".join(f"wort{i}" for i in range(10))
    transcript = _turns([
        ("00:00", shared + " " + filler + " " + shared),
        ("54:00", "closing"),
    ])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.has_duplicate_span is False


# ── clean transcript / combined result ───────────────────────────────────


def test_clean_full_coverage_transcript_passes():
    transcript = _turns([
        ("00:00", "Guten Morgen zusammen, lasst uns anfangen."),
        ("15:00", "Ich denke wir sollten das Budget nochmal anschauen."),
        ("30:00", "Einverstanden, machen wir das naechste Woche."),
        ("58:30", "Perfekt, dann sehen wir uns naechste Woche. Tschuess."),
    ])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    assert result.passed is True
    assert result.reasons == []
    assert result.has_chunk_failure_marker is False
    assert result.has_repetition_loop is False
    assert result.has_duplicate_span is False


def test_multiple_failures_all_reported():
    transcript = (
        _turns([("00:00", "start")])
        + "\n\n[CHUNK 1 FAILED: timeout]\n\n"
    )
    result = validate_transcript(transcript, audio_duration_seconds=3600)
    assert result.passed is False
    assert result.has_chunk_failure_marker is True
    # Coverage is also far below 90% here — both reasons should surface.
    assert len(result.reasons) >= 2


def test_to_dict_serializes_for_meta_persistence():
    transcript = _turns([("00:00", "hi"), ("54:00", "bye")])
    result = validate_transcript(transcript, audio_duration_seconds=60 * 60)
    d = result.to_dict()
    assert d["passed"] is True
    assert d["coverage_pct"] == 90.0
    assert d["reasons"] == []
