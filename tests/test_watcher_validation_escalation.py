"""Tests for the transcript-validation gate + escalation ladder that sits
between the Gemini result and the JSON write in _process_with_gemini
(docs/RELIABILITY_PLAN_2026-07.md Phase 1). A transcript must never be
stored as a clean success if it fails validation: retry a fresh single-call
on pro, then drift-proof chunked mode, then accept the best-available
result with a partial flag + Telegram alert naming what's missing.

Written FIRST (red) before the corresponding transcribe_watcher.py changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    w._notify_telegram_partial = (
        lambda audio_file, validation: w._telegram_alerts.append((audio_file, validation))
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
