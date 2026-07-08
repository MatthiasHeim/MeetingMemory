"""Tests for drift-proof chunking (docs/RELIABILITY_PLAN_2026-07.md Phase 3):
zero-overlap chunk boundaries, continuity context passed between chunks, and
per-chunk validation + retry that replaces the old silent
"[CHUNK N FAILED]" concatenate-and-continue path.

Written FIRST (red) before the corresponding gemini_processor.py changes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gemini_processor import (  # noqa: E402
    GeminiAudioProcessor,
    GeminiResult,
    _build_continuity_prefix,
)


def _bare_processor() -> GeminiAudioProcessor:
    return GeminiAudioProcessor.__new__(GeminiAudioProcessor)


# ── zero overlap ──────────────────────────────────────────────────────────


def test_chunk_overlap_is_zero():
    """docs/RELIABILITY_PLAN_2026-07.md Phase 3: no overlap -> no duplicate
    text -> no conflicting-label duplicates (429's core defect)."""
    assert GeminiAudioProcessor.CHUNK_OVERLAP_SEC == 0


def test_chunk_audio_force_chunk_splits_even_under_threshold(monkeypatch, tmp_path):
    """Empirical finding (source 427 verification, 2026-07-08): the
    escalation ladder's 'drift-proof chunked mode' step is a no-op disguise
    for a single-shot retry when the recording is under CHUNK_THRESHOLD_SEC
    (the common case -- most meetings are under 35min) because _chunk_audio
    just returns the whole file as one "chunk". force_chunk=True makes
    escalation's chunked step genuinely split the audio for real diversity
    from the failing single-call path."""
    p = _bare_processor()
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    total_duration = 20 * 60  # under the 35min threshold
    monkeypatch.setattr(p, "_get_duration", lambda path: total_duration)

    class _FakeCompleted:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted())

    chunks = p._chunk_audio(f, force_chunk=True)
    assert len(chunks) > 1
    offsets = [off for _, off in chunks]
    assert offsets == [0.0, 900.0]


def test_chunk_audio_without_force_stays_single_chunk_under_threshold(monkeypatch, tmp_path):
    """Regression: default (force_chunk=False) behavior for sub-threshold
    audio is unchanged -- a single degenerate chunk covering the whole file."""
    p = _bare_processor()
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 20 * 60)
    chunks = p._chunk_audio(f)
    assert chunks == [(f, 0.0)]


def test_chunk_audio_produces_contiguous_non_overlapping_offsets(monkeypatch, tmp_path):
    p = _bare_processor()
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    total_duration = 40 * 60  # 40 min -> 3 contiguous chunks (15/15/10)
    monkeypatch.setattr(p, "_get_duration", lambda path: total_duration)

    class _FakeCompleted:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted())

    chunks = p._chunk_audio(f)
    offsets = [off for _, off in chunks]
    assert offsets == [0.0, 900.0, 1800.0]
    # Each chunk's start is exactly the previous chunk's end -- no overlap.
    for off in offsets[1:]:
        assert off % p.CHUNK_DURATION_SEC == 0


# ── continuity context prefix ─────────────────────────────────────────────


def test_continuity_prefix_empty_when_no_previous_context():
    assert _build_continuity_prefix(None) == ""
    assert _build_continuity_prefix("") == ""


def test_continuity_prefix_renders_instruction_and_lines():
    prefix = _build_continuity_prefix("[14:30] Matthias: line three")
    assert "CONTINUATION" in prefix
    assert "line three" in prefix
    assert "same speaker labels" in prefix.lower()


def test_process_single_shot_includes_continuity_context_in_prompt(monkeypatch, tmp_path):
    """The continuity block must actually reach the assembled prompt, not
    just exist as an unused helper function."""
    p = _bare_processor()
    p.model = "test-model"

    class _FakeUploadedFile:
        name = "files/fake"

        class state:
            name = "ACTIVE"

    class _FakeFiles:
        def upload(self, file, config):
            return _FakeUploadedFile()

        def delete(self, name):
            pass

    class _FakeClient:
        files = _FakeFiles()

    p.client = _FakeClient()

    f = tmp_path / "chunk.mp3"
    f.write_bytes(b"x")

    class _FakeResponse:
        usage_metadata = None

    captured = {}

    def fake_generate(prompt, audio_content=None, max_attempts=3):
        captured["prompt"] = prompt
        return '{"transcript": "[00:00] Matthias: hi", "language": "de"}', _FakeResponse()

    monkeypatch.setattr(p, "_generate_with_retry", fake_generate)

    p._process_single_shot(
        f, None, 900.0,
        continuity_context="[14:30] Matthias: line three",
    )

    assert "CONTINUATION" in captured["prompt"]
    assert "line three" in captured["prompt"]


def test_process_single_shot_no_continuity_block_when_none(monkeypatch, tmp_path):
    p = _bare_processor()
    p.model = "test-model"

    class _FakeUploadedFile:
        name = "files/fake"

        class state:
            name = "ACTIVE"

    class _FakeFiles:
        def upload(self, file, config):
            return _FakeUploadedFile()

        def delete(self, name):
            pass

    class _FakeClient:
        files = _FakeFiles()

    p.client = _FakeClient()
    f = tmp_path / "chunk.mp3"
    f.write_bytes(b"x")

    class _FakeResponse:
        usage_metadata = None

    captured = {}

    def fake_generate(prompt, audio_content=None, max_attempts=3):
        captured["prompt"] = prompt
        return '{"transcript": "[00:00] Matthias: hi", "language": "de"}', _FakeResponse()

    monkeypatch.setattr(p, "_generate_with_retry", fake_generate)

    p._process_single_shot(f, None, 900.0)

    assert "CONTINUATION" not in captured["prompt"]


# ── per-chunk validation + retry + missing-range tracking ────────────────


def test_chunk_loop_passes_previous_chunk_tail_as_continuity(monkeypatch, tmp_path):
    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    chunk0 = tmp_path / "chunk_00.mp3"
    chunk0.write_bytes(b"x")
    chunk1 = tmp_path / "chunk_01.mp3"
    chunk1.write_bytes(b"x")
    chunk_duration = 900.0
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(chunk0, 0.0), (chunk1, 900.0)])
    monkeypatch.setattr(p, "_get_duration", lambda path: chunk_duration)
    monkeypatch.setattr(
        p, "_generate_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")),
    )

    captured_continuity = []

    def fake_single_shot(audio_path, custom_prompt, total_dur, known_attendees=None,
                          channel_segments=None, diarization_segments=None,
                          continuity_context=None):
        captured_continuity.append(continuity_context)
        if audio_path == chunk0:
            return GeminiResult(
                transcript=(
                    "[00:00] Matthias: line one\n"
                    "[14:00] Speaker A: line two\n"
                    "[14:30] Matthias: line three"
                ),
                language="de",
            )
        return GeminiResult(transcript="[15:00] Matthias: chunk two", language="de")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    p._process_chunked(f, None, 1800.0)

    assert captured_continuity[0] is None
    assert captured_continuity[1] is not None
    assert "line three" in captured_continuity[1]
    assert "line one" in captured_continuity[1]


def test_chunk_retried_once_before_being_marked_missing(monkeypatch, tmp_path):
    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    chunk0 = tmp_path / "chunk_00.mp3"
    chunk0.write_bytes(b"x")
    chunk_duration = 900.0
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(chunk0, 0.0)])
    monkeypatch.setattr(p, "_get_duration", lambda path: chunk_duration)
    monkeypatch.setattr(
        p, "_generate_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")),
    )

    attempts = []

    def fake_single_shot(audio_path, custom_prompt, total_dur, known_attendees=None,
                          channel_segments=None, diarization_segments=None,
                          continuity_context=None):
        attempts.append(1)
        if len(attempts) == 1:
            return GeminiResult(transcript="[00:01] Matthias: hi", language="de")  # 0% coverage
        return GeminiResult(transcript="[15:00] Matthias: recovered on retry", language="de")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    result = p._process_chunked(f, None, chunk_duration)

    assert len(attempts) == 2
    assert result.missing_time_ranges == []
    assert "recovered on retry" in result.transcript


def test_chunk_failing_all_retries_recorded_as_missing_range_no_marker_text(monkeypatch, tmp_path):
    """REQUIRED fix: a chunk that fails validation after retries must be
    tracked as a missing time range, never papered over with a
    '[CHUNK N FAILED]' marker concatenated into the transcript."""
    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    chunk0 = tmp_path / "chunk_00.mp3"
    chunk0.write_bytes(b"x")
    chunk1 = tmp_path / "chunk_01.mp3"
    chunk1.write_bytes(b"x")
    chunk_duration = 900.0
    total_duration = 1800.0
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(chunk0, 0.0), (chunk1, 900.0)])
    monkeypatch.setattr(p, "_get_duration", lambda path: chunk_duration)
    monkeypatch.setattr(
        p, "_generate_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")),
    )

    def fake_single_shot(audio_path, custom_prompt, total_dur, known_attendees=None,
                          channel_segments=None, diarization_segments=None,
                          continuity_context=None):
        if audio_path == chunk0:
            # Always far too short -> fails validation on every attempt.
            return GeminiResult(transcript="[00:01] Matthias: hi", language="de")
        return GeminiResult(transcript="[15:00] Matthias: chunk two content here", language="de")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    result = p._process_chunked(f, None, total_duration)

    assert "[CHUNK" not in result.transcript
    assert result.missing_time_ranges == [(0.0, 900.0)]
    assert "chunk two content" in result.transcript
    assert result.error is None


def test_processing_continues_after_one_chunk_permanently_fails(monkeypatch, tmp_path):
    """A permanently-failed chunk must not abort the rest of the meeting."""
    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    chunk0 = tmp_path / "chunk_00.mp3"
    chunk0.write_bytes(b"x")
    chunk1 = tmp_path / "chunk_01.mp3"
    chunk1.write_bytes(b"x")
    chunk_duration = 900.0
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(chunk0, 0.0), (chunk1, 900.0)])
    monkeypatch.setattr(p, "_get_duration", lambda path: chunk_duration)
    monkeypatch.setattr(
        p, "_generate_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")),
    )

    def fake_single_shot(audio_path, custom_prompt, total_dur, known_attendees=None,
                          channel_segments=None, diarization_segments=None,
                          continuity_context=None):
        if audio_path == chunk0:
            return GeminiResult(transcript="[00:01] Matthias: hi", language="de")
        return GeminiResult(transcript="[15:00] Matthias: chunk two", language="de")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    result = p._process_chunked(f, None, 1800.0)

    assert len(result.raw_response["chunks"]) == 1  # only chunk1 made it in
    assert result.chunk_count == 2  # but the attempt count still reflects both


def test_missing_time_ranges_persisted_in_meta(monkeypatch, tmp_path):
    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    chunk0 = tmp_path / "chunk_00.mp3"
    chunk0.write_bytes(b"x")
    chunk1 = tmp_path / "chunk_01.mp3"
    chunk1.write_bytes(b"x")
    chunk_duration = 900.0
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(chunk0, 0.0), (chunk1, 900.0)])
    monkeypatch.setattr(p, "_get_duration", lambda path: chunk_duration)
    monkeypatch.setattr(
        p, "_generate_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")),
    )

    def fake_single_shot(audio_path, custom_prompt, total_dur, known_attendees=None,
                          channel_segments=None, diarization_segments=None,
                          continuity_context=None):
        if audio_path == chunk0:
            return GeminiResult(transcript="[00:01] Matthias: hi", language="de")
        return GeminiResult(transcript="[15:00] Matthias: chunk two", language="de")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    result = p._process_chunked(f, None, 1800.0)
    meta = result.parsed_response["_meta"]
    assert meta["missing_time_ranges"] == [[0.0, 900.0]]


def test_validation_report_and_partial_persist_in_meta():
    """Set by the watcher's _validate_and_escalate after the fact -- must
    round-trip into parsed_response()['_meta'] for the on-disk JSON."""
    result = GeminiResult(
        transcript="[00:00] Matthias: hi",
        language="de",
        validation_report={"passed": False, "coverage_pct": 42.0, "reasons": ["x"]},
        partial=True,
    )
    meta = result.parsed_response["_meta"]
    assert meta["partial"] is True
    assert meta["validation"] == {"passed": False, "coverage_pct": 42.0, "reasons": ["x"]}


def test_partial_defaults_false_and_no_validation_key_when_unset():
    result = GeminiResult(transcript="[00:00] Matthias: hi", language="de")
    meta = result.parsed_response["_meta"]
    assert meta["partial"] is False
    assert "validation" not in meta


def test_all_chunks_failing_still_returns_all_chunks_failed_error(monkeypatch, tmp_path):
    """Regression: with retries added, a deterministically-failing single
    chunk must still surface as 'All chunks failed', not loop forever or
    silently succeed."""
    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    chunk0 = tmp_path / "chunk_00.mp3"
    chunk0.write_bytes(b"x")
    chunk_duration = 900.0
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(chunk0, 0.0)])
    monkeypatch.setattr(p, "_get_duration", lambda path: chunk_duration)

    def fake_single_shot(*a, **k):
        return GeminiResult(transcript="[00:01] Matthias: hi", language="de")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    result = p._process_chunked(f, None, chunk_duration)
    assert result.error == "All chunks failed"
    assert result.missing_time_ranges == [(0.0, 900.0)]
