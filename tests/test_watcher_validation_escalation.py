"""Tests for the transcript-validation gate + escalation ladder that sits
between the Gemini result and the JSON write in _process_with_gemini
(docs/RELIABILITY_PLAN_2026-07.md Phase 1). A transcript must never be
stored as a clean success if it fails validation: retry a fresh single-call
on pro, then drift-proof chunked mode, then accept the best-available
result with a partial flag + Telegram alert naming what's missing.

Written FIRST (red) before the corresponding transcribe_watcher.py changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402
from gemini_processor import GeminiResult  # noqa: E402


class _StubLogger:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))

    def error(self, msg):
        self.lines.append(("error", msg))

    def debug(self, msg):
        self.lines.append(("debug", msg))


class _StubProcessor:
    """Stand-in for GeminiAudioProcessor exposing only what escalation needs."""

    CHUNK_DURATION_SEC = 15 * 60

    def __init__(self, single_shot_results=None, chunked_result=None):
        self._single_shot_results = list(single_shot_results or [])
        self._chunked_result = chunked_result
        self.single_shot_calls = 0
        self.chunked_calls = 0
        self.chunked_force_chunk_values = []

    def _process_single_shot(self, audio_path, custom_prompt, total_duration,
                              known_attendees=None, channel_segments=None,
                              diarization_segments=None):
        result = self._single_shot_results[self.single_shot_calls]
        self.single_shot_calls += 1
        return result

    def _process_chunked(self, audio_path, custom_prompt, total_duration,
                          known_attendees=None, channel_segments=None,
                          diarization_segments=None, force_chunk=False):
        self.chunked_calls += 1
        self.chunked_force_chunk_values.append(force_chunk)
        return self._chunked_result


def _watcher(gemini_processor):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    w.gemini_processor = gemini_processor
    w._telegram_alerts = []
    w._telegram_failures = []
    w._notify_telegram_partial = (
        lambda audio_file, validation: w._telegram_alerts.append((audio_file, validation))
    )
    w._notify_telegram_failure = (
        lambda audio_file, reason: w._telegram_failures.append((audio_file, reason))
    )
    return w


def _clean_result():
    return GeminiResult(transcript="[00:00] Matthias: hi\n[54:00] Matthias: bye", language="de")


# ── pure combining logic (missing_time_ranges must force a failure) ──────


def test_validate_gemini_result_passes_clean_transcript():
    result = _clean_result()
    validation = tw._validate_gemini_result(result, 60 * 60)
    assert validation.passed is True


def test_validate_gemini_result_fails_on_missing_time_ranges_despite_full_coverage():
    """A chunked result can reach the final timestamp (good raw coverage %)
    while still having a hole in the middle from a chunk that failed after
    retries -- transcript_validator's text-only coverage check can't see
    that, so GeminiResult.missing_time_ranges must force failure too."""
    result = GeminiResult(
        transcript="[00:00] Matthias: hi\n[59:00] Matthias: bye",
        language="de",
        missing_time_ranges=[(900.0, 1800.0)],
    )
    validation = tw._validate_gemini_result(result, 60 * 60)
    assert validation.passed is False
    assert any("missing time range" in r.lower() for r in validation.reasons)
    assert "15:00" in validation.reasons[-1]
    assert "30:00" in validation.reasons[-1]


# ── escalation ladder ──────────────────────────────────────────────────


def test_passing_validation_returns_immediately_no_escalation():
    proc = _StubProcessor()
    w = _watcher(proc)
    result = _clean_result()
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, result,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is result
    assert validation.passed is True
    assert partial is False
    assert proc.single_shot_calls == 0
    assert proc.chunked_calls == 0
    assert w._telegram_alerts == []


def test_failing_result_retries_single_shot_and_succeeds():
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")
    good = _clean_result()
    proc = _StubProcessor(single_shot_results=[good])
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, bad,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is good
    assert validation.passed is True
    assert partial is False
    assert proc.single_shot_calls == 1
    assert proc.chunked_calls == 0
    assert w._telegram_alerts == []


def test_failing_retry_falls_back_to_chunked_and_succeeds():
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")
    still_bad = GeminiResult(transcript="[00:02] Matthias: hi", language="de")
    good_chunked = _clean_result()
    proc = _StubProcessor(single_shot_results=[still_bad], chunked_result=good_chunked)
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, bad,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is good_chunked
    assert validation.passed is True
    assert partial is False
    assert proc.single_shot_calls == 1
    assert proc.chunked_calls == 1
    assert w._telegram_alerts == []


def test_all_escalation_exhausted_accepts_best_available_as_partial():
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")  # ~0.03% of 60min
    retry_bad = GeminiResult(transcript="[05:00] Matthias: better but still short", language="de")  # 8.3%
    chunked_bad = GeminiResult(transcript="[03:00] Matthias: worse than retry", language="de")  # 5%
    proc = _StubProcessor(single_shot_results=[retry_bad], chunked_result=chunked_bad)
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, bad,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert partial is True
    assert validation.passed is False
    # Highest-coverage candidate of the three wins: retry_bad (8.3%) beats
    # chunked_bad (5%) and the original (~0.03%).
    assert final_result is retry_bad
    assert len(w._telegram_alerts) == 1
    alerted_file, alerted_validation = w._telegram_alerts[0]
    assert alerted_file == Path("/tmp/rec.wav")
    assert alerted_validation is validation


def test_chunked_escalation_forces_real_chunking():
    """Empirical fix (source 427, 2026-07-08): without force_chunk, a
    sub-CHUNK_THRESHOLD_SEC recording's 'chunked mode' escalation step is a
    no-op disguise for a third single-shot call (_chunk_audio just returns
    the whole file as one chunk) -- the escalation ladder must force real
    chunking so it's actually a distinct strategy from the failed retries."""
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")
    still_bad = GeminiResult(transcript="[00:02] Matthias: hi", language="de")
    good_chunked = _clean_result()
    proc = _StubProcessor(single_shot_results=[still_bad], chunked_result=good_chunked)
    w = _watcher(proc)
    w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 30 * 60, bad,  # under 35min threshold
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert proc.chunked_force_chunk_values == [True]


def test_short_recording_skips_chunked_escalation():
    """A recording too short to meaningfully chunk shouldn't trigger a
    pointless chunked-mode attempt during escalation."""
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")
    still_bad = GeminiResult(transcript="[00:02] Matthias: hi", language="de")
    proc = _StubProcessor(single_shot_results=[still_bad])
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 10 * 60, bad,  # 10 min audio
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert proc.chunked_calls == 0
    assert partial is True
    assert len(w._telegram_alerts) == 1


def test_single_shot_retry_exception_falls_through_to_chunked():
    """A retry that raises (e.g. another disconnect) must not abort the
    ladder -- fall through to the next escalation step."""
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")
    good_chunked = _clean_result()

    class _RaisingThenChunkedProcessor(_StubProcessor):
        def _process_single_shot(self, *a, **k):
            self.single_shot_calls += 1
            raise RuntimeError("Server disconnected without sending a response.")

    proc = _RaisingThenChunkedProcessor(chunked_result=good_chunked)
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, bad,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is good_chunked
    assert partial is False
    assert proc.single_shot_calls == 1
    assert proc.chunked_calls == 1


def test_chunked_escalation_exception_still_returns_best_available():
    bad = GeminiResult(transcript="[00:01] Matthias: hi", language="de")
    retry_bad = GeminiResult(transcript="[05:00] Matthias: still short", language="de")

    class _RaisingChunkedProcessor(_StubProcessor):
        def _process_chunked(self, *a, **k):
            self.chunked_calls += 1
            raise RuntimeError("boom")

    proc = _RaisingChunkedProcessor(single_shot_results=[retry_bad])
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, bad,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert partial is True
    assert final_result is retry_bad
    assert len(w._telegram_alerts) == 1


# ── F1: error results must enter the escalation ladder (RC1) ─────────────


def test_error_result_enters_escalation_ladder_and_is_rescued():
    """RC1 regression (docs/SPEC-error-path-escalation-2026-07-10.md): a
    hard error result ("All chunks failed" after a disconnect storm) used to
    short-circuit before the ladder ever ran. _validate_gemini_result on an
    error-result already yields passed=False (empty transcript -> 0%
    coverage), so it must fall into the same ladder as any other
    validation failure and get a real rescue attempt."""
    error_result = GeminiResult(transcript="", language="unknown", error="All chunks failed")
    good = _clean_result()
    proc = _StubProcessor(single_shot_results=[good])
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, error_result,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is good
    assert validation.passed is True
    assert partial is False
    assert proc.single_shot_calls == 1
    assert w._telegram_failures == []


def test_junk_guard_no_json_when_every_step_returns_empty_transcript():
    """F1 junk guard: if every ladder step comes back with no usable
    transcript at all (0% coverage, empty), _validate_and_escalate must
    return result=None (the caller's contract: don't write a JSON, so F3's
    periodic rescan can retry later instead of the file being marked
    "processed" forever with a junk result) -- and fire exactly one
    (non-partial) failure alert, not the partial-transcript alert."""
    error_result = GeminiResult(transcript="", language="unknown", error="All chunks failed")
    still_empty_retry = GeminiResult(transcript="", language="unknown")
    still_empty_chunked = GeminiResult(transcript="", language="unknown", error="All chunks failed")
    proc = _StubProcessor(single_shot_results=[still_empty_retry], chunked_result=still_empty_chunked)
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 60 * 60, error_result,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is None
    assert partial is True
    assert validation.coverage_pct == 0.0
    assert len(w._telegram_failures) == 1
    assert w._telegram_alerts == []  # the partial-transcript alert must NOT fire


# ── F6: sanitize-and-revalidate, exercised through the watcher's gate ────


def _fixture_path() -> Path:
    return (
        Path.home() / "Documents" / "MeetingRecorder" / "Transcripts"
        / "2026-07-10_13-59-12.json"
    )


def test_validate_gemini_result_sanitizes_repetition_glitch_clean():
    """Mirrors the real incident (source 463): a runaway 'de,' loop covering
    ~5s of an otherwise-clean 24min recording must sanitize to a clean pass
    instead of entering the escalation ladder."""
    loop = " ".join(["de,"] * 595)
    transcript = (
        "[00:00] Matthias: hallo zusammen, los gehts heute\n"
        "[19:25] Philipp Baltensperger: Ich gseh im Moment nöd.\n"
        f"[19:28] Philipp Baltensperger: Aber drum würd ich jetzt mal, {loop} "
        "de-Stufe, dass mir mit de ablenkt.\n"
        "[19:33] Matthias: Okay, ja. Okay.\n"
        "[23:54] Matthias: Perfekt, dann sehen wir uns naechste Woche. Tschuess."
    )
    result = GeminiResult(transcript=transcript, language="de")
    validation = tw._validate_gemini_result(result, 24 * 60)

    assert validation.passed is True
    assert validation.sanitized is True
    assert len(validation.sanitized_locations) == 1
    assert validation.sanitized_locations[0]["count"] == 595
    # result.transcript was mutated in place to the sanitized text.
    assert "[transcription glitch]" in result.transcript
    assert result.transcript.count("de,") == 1


def test_validate_and_escalate_accepts_sanitized_result_without_escalating():
    """No ladder step should ever run when F6 sanitization alone is enough
    to pass -- that's the whole point (saves the expensive pro retries)."""
    loop = " ".join(["de,"] * 595)
    transcript = (
        "[00:00] Matthias: hallo zusammen, los gehts heute\n"
        f"[19:28] Matthias: jetzt mal, {loop} de-Stufe, weiter gehts.\n"
        "[23:54] Matthias: Perfekt, tschuess."
    )
    result = GeminiResult(transcript=transcript, language="de")
    proc = _StubProcessor()
    w = _watcher(proc)
    final_result, validation, partial = w._validate_and_escalate(
        Path("/tmp/rec.wav"), Path("/tmp/rec.mp3"), 24 * 60, result,
        known_attendees=None, channel_segments=None, diarization_segments=None,
    )
    assert final_result is result
    assert validation.passed is True
    assert validation.sanitized is True
    assert partial is False
    assert proc.single_shot_calls == 0
    assert proc.chunked_calls == 0


def test_validate_gemini_result_does_not_sanitize_pure_duplicate_span():
    """Sanitization must only trigger on has_repetition_loop -- a duplicate
    span with no runaway repeat must still fail and reach the ladder."""
    shared = "wir sollten definitiv auf die neue Plattform wechseln naechstes Jahr"
    filler = " ".join(f"wort{i}" for i in range(60))
    transcript = (
        f"[06:49] Matthias: {shared}\n"
        f"[07:30] Matthias: {filler}\n"
        f"[08:51] Matthias: {shared}\n"
        "[54:00] Matthias: closing"
    )
    result = GeminiResult(transcript=transcript, language="de")
    original_transcript = result.transcript
    validation = tw._validate_gemini_result(result, 60 * 60)

    assert validation.passed is False
    assert validation.has_duplicate_span is True
    assert validation.sanitized is False
    assert result.transcript == original_transcript  # untouched


@pytest.mark.skipif(not _fixture_path().exists(), reason="real incident fixture only available locally")
def test_real_fixture_source_463_sanitizes_clean():
    """Integration check against the actual stored incident JSON (source
    463, 2026-07-10) -- skipped automatically where the local recording
    archive isn't present (e.g. CI)."""
    data = json.loads(_fixture_path().read_text(encoding="utf-8"))
    audio_duration = data["_meta"]["audio_duration_seconds"]
    result = GeminiResult(transcript=data["transcript"], language=data.get("language", "de"))
    validation = tw._validate_gemini_result(result, audio_duration)
    assert validation.passed is True
    assert validation.sanitized is True
    assert "[transcription glitch]" in result.transcript
