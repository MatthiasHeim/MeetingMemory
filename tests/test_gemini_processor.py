"""Tests for gemini_processor — covers calendar-attendee injection into the
audio prompt (Fix 1 for the speaker-mislabeling bug). The actual Gemini API
call is not exercised here; that's an integration concern."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gemini_processor import (  # noqa: E402
    AUDIO_ANALYSIS_PROMPT,
    REDUCE_PASS_PROMPT,
    GeminiAudioProcessor,
    _build_attendees_prefix,
    _build_diarization_map_prefix,
    process_audio_file,
)


# ── prompt prefix ─────────────────────────────────────────────────────


def test_prefix_empty_when_no_attendees():
    """Prompt must be byte-identical to legacy when no attendees provided."""
    assert _build_attendees_prefix(None) == ""
    assert _build_attendees_prefix([]) == ""


def test_prefix_empty_when_only_self_attendee():
    """Solo events / calendar matches with only Matthias must NOT inject a
    useless attendee block (the prompt would then say 'use these names'
    with only Matthias listed)."""
    only_self = [
        {"name": "Matthias Heim", "email": "matthias@lailix.com",
         "company": "Lailix", "role": "self"},
    ]
    assert _build_attendees_prefix(only_self) == ""


def test_prefix_renders_attendees_with_company():
    """Non-self attendees get rendered as 'Name (company)' or 'Name (role)'."""
    attendees = [
        {"name": "Matthias Heim", "role": "self", "company": "Lailix"},
        {"name": "Antonella Borromeo", "role": "participant",
         "company": "BlueCare"},
    ]
    prefix = _build_attendees_prefix(attendees)
    assert "KNOWN ATTENDEES" in prefix
    assert "Matthias Heim (host, Lailix)" in prefix
    assert "Antonella Borromeo (BlueCare)" in prefix
    assert "Speaker A" in prefix and "Speaker B" in prefix  # fallback guidance
    assert prefix.endswith("\n\n")


def test_diarization_prefix_empty_when_no_segments():
    assert _build_diarization_map_prefix(None) == ""
    assert _build_diarization_map_prefix([]) == ""


def test_diarization_prefix_renders_prior_wording():
    prefix = _build_diarization_map_prefix([
        {
            "start": 0.0,
            "end": 2.2,
            "label": "SPEAKER_00",
            "confidence": 0.82,
            "level": "high",
            "overlapped": False,
        },
        {
            "start": 2.2,
            "end": 3.0,
            "label": "SPEAKER_01",
            "confidence": 0.41,
            "level": "low",
            "overlapped": True,
        },
    ])
    assert "ACOUSTIC SPEAKER MAP (PRIOR)" in prefix
    assert "GROUND TRUTH" not in prefix
    assert "map wins" not in prefix
    assert "not as a final naming authority" in prefix
    assert "SPEAKER_00 confidence=high (0.82)" in prefix
    assert "SPEAKER_01 confidence=low (0.41) overlapped" in prefix
    assert "KNOWN ATTENDEES" in prefix


def test_prefix_falls_back_to_role_when_company_none():
    """`calendar_resolve._company_from_email` returns None for unknown domains.
    The prefix must still render those attendees — fall back to role."""
    attendees = [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "External Person", "role": "participant", "company": None},
    ]
    prefix = _build_attendees_prefix(attendees)
    assert "External Person (participant)" in prefix


def test_prefix_ignores_self_entries_by_name_too():
    """Even if calendar didn't tag the host with role='self', the prefix
    must dedupe by canonical SELF_NAME so the host appears once."""
    attendees = [
        {"name": "Matthias Heim", "role": "host"},  # no role='self'
        {"name": "Antonella Borromeo", "role": "participant"},
    ]
    prefix = _build_attendees_prefix(attendees)
    # Antonella present.
    assert "Antonella Borromeo" in prefix
    # Matthias appears in the host-anchor line; not duplicated as a participant.
    assert prefix.count("Matthias Heim") == 1


# ── process_audio signature ───────────────────────────────────────────


def test_process_audio_accepts_known_attendees_kwarg():
    """process_audio must accept known_attendees as a keyword argument so
    transcribe_watcher can pass calendar-resolved attendees through."""
    sig = inspect.signature(GeminiAudioProcessor.process_audio)
    assert "known_attendees" in sig.parameters
    # Must be optional (default None) — single-shot callers without calendar
    # context must keep working.
    assert sig.parameters["known_attendees"].default is None
    assert "diarization_segments" in sig.parameters
    assert sig.parameters["diarization_segments"].default is None


def test_process_audio_file_convenience_forwards_kwarg():
    """The module-level convenience wrapper must also accept known_attendees."""
    sig = inspect.signature(process_audio_file)
    assert "known_attendees" in sig.parameters
    assert sig.parameters["known_attendees"].default is None
    assert "diarization_segments" in sig.parameters
    assert sig.parameters["diarization_segments"].default is None


# ── prompt assembly with attendees ────────────────────────────────────


def test_prompt_unchanged_with_no_attendees(monkeypatch):
    """If known_attendees is None/empty, the prompt sent to Gemini must be
    byte-identical to the legacy AUDIO_ANALYSIS_PROMPT.

    We don't have a live Gemini key in tests, so we verify the prompt path
    by spying on the assembly: prefix + base must equal base when prefix is
    empty.
    """
    assert _build_attendees_prefix(None) + AUDIO_ANALYSIS_PROMPT == AUDIO_ANALYSIS_PROMPT
    assert _build_attendees_prefix([]) + AUDIO_ANALYSIS_PROMPT == AUDIO_ANALYSIS_PROMPT


def test_prompt_prepended_with_attendees():
    """With attendees, the prompt starts with the KNOWN ATTENDEES block."""
    attendees = [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Antonella Borromeo", "role": "participant",
         "company": "BlueCare"},
    ]
    composed = _build_attendees_prefix(attendees) + AUDIO_ANALYSIS_PROMPT
    assert composed.startswith("## KNOWN ATTENDEES")
    # Base prompt content still present.
    assert "TRANSCRIPTION (REQUIRED)" in composed
    # And the attendee appears verbatim.
    assert "Antonella Borromeo (BlueCare)" in composed


# ── named-person priming guard ────────────────────────────────────────


# Names that have biased Gemini in past incidents. Each broke a real meeting:
# - "Stefan" (Stefan Sieber at BlueCare) — the example JSON in
#   AUDIO_ANALYSIS_PROMPT used to hardcode Stefan + Swiss-German business
#   quotes, which primed Gemini to default to that name on any Swiss
#   business call (source_id 186, source_id 192). Fix: 346a7ad1.
# - Add more here as new bias incidents surface, with a short evidence note.
_FORBIDDEN_PRIMING_NAMES: tuple[str, ...] = (
    "Stefan",
    "Sieber",
)


def test_prompts_contain_no_real_person_names():
    """AUDIO_ANALYSIS_PROMPT and REDUCE_PASS_PROMPT must not name any real
    person other than Matthias (the host anchor).

    Few-shot examples that hardcode real names paired with realistic quotes
    biased Gemini's attribution — see commit 346a7ad1. This guard prevents
    that class of leak from regressing whenever someone iterates the prompt
    schema. Add to `_FORBIDDEN_PRIMING_NAMES` when new bias incidents surface.
    """
    for needle in _FORBIDDEN_PRIMING_NAMES:
        pat = re.compile(rf"\b{re.escape(needle)}\b")
        assert not pat.search(AUDIO_ANALYSIS_PROMPT), (
            f"AUDIO_ANALYSIS_PROMPT mentions {needle!r} — past-incident "
            f"bias source. Use 'Speaker B/C' placeholders instead."
        )
        assert not pat.search(REDUCE_PASS_PROMPT), (
            f"REDUCE_PASS_PROMPT mentions {needle!r} — past-incident "
            f"bias source. Use 'Speaker B/C' placeholders instead."
        )


# ── single-call vs chunked dispatch + fallback ────────────────────────
#
# The pipeline transcribes meetings ≤ CHUNK_THRESHOLD_SEC (60 min) in a single
# call to avoid cross-chunk speaker drift/swap, chunks longer ones, and falls
# back to chunked if a single call fails (so a long-audio disconnect never
# loses the whole meeting). These tests pin that dispatch without a live API.


def _bare_processor() -> GeminiAudioProcessor:
    """A processor instance without __init__ — no GEMINI_API_KEY / genai needed.
    Class-level CHUNK_* constants are still present."""
    return GeminiAudioProcessor.__new__(GeminiAudioProcessor)


def test_threshold_is_35_minutes():
    """Lowered from 60 min (docs/RELIABILITY_PLAN_2026-07.md Phase 0/3): pro
    silently truncates long single calls without erroring, and drift-proof
    chunking (zero overlap + continuity context) removes the reason to push
    single-call out to an hour."""
    assert GeminiAudioProcessor.CHUNK_THRESHOLD_SEC == 35 * 60


def test_audio_over_threshold_goes_straight_to_chunked(monkeypatch, tmp_path):
    p = _bare_processor()
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 45 * 60)  # 45 min
    seen: list[str] = []
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: seen.append("chunked") or "R")
    monkeypatch.setattr(p, "_process_single_shot", lambda *a, **k: seen.append("single") or "R")
    assert p.process_audio(f) == "R"
    assert seen == ["chunked"]


def test_audio_under_threshold_uses_single_shot(monkeypatch, tmp_path):
    p = _bare_processor()
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 20 * 60)  # 20 min
    seen: list[str] = []
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: seen.append("chunked") or "R")
    monkeypatch.setattr(p, "_process_single_shot", lambda *a, **k: seen.append("single") or "R")
    assert p.process_audio(f) == "R"
    assert seen == ["single"]


def test_single_shot_failure_falls_back_to_chunked(monkeypatch, tmp_path):
    """A 30-min recording (> CHUNK_DURATION_SEC, <= CHUNK_THRESHOLD_SEC)
    whose single call fails must fall back to chunked rather than lose the
    whole meeting."""
    p = _bare_processor()
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 30 * 60)
    seen: list[str] = []

    def boom(*a, **k):
        seen.append("single")
        raise RuntimeError("Server disconnected without sending a response.")

    monkeypatch.setattr(p, "_process_single_shot", boom)
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: seen.append("chunked") or "R")
    assert p.process_audio(f) == "R"
    assert seen == ["single", "chunked"]


def test_single_chunk_fallback_gets_full_channel_map(monkeypatch, tmp_path):
    """Regression (review W1): when a 15-60 min single-shot call fails and
    falls back to chunked, _chunk_audio returns ONE chunk covering the whole
    file. The channel map slice must then cover the full duration — not
    silently truncate at CHUNK_DURATION_SEC (15 min) while the prompt block
    claims ground truth."""
    from gemini_processor import GeminiResult

    p = _bare_processor()
    p.model = "test-model"  # _process_chunked reads it for the result
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    duration = 50 * 60  # 50 min: single chunk in _process_chunked
    monkeypatch.setattr(p, "_get_duration", lambda path: duration)
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(path, 0.0)])
    # Reduce-pass must not hit the API.
    monkeypatch.setattr(p, "_generate_with_retry",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")))

    captured: list = []

    def fake_single_shot(audio_path, custom_prompt, total_duration,
                         known_attendees=None, channel_segments=None,
                         diarization_segments=None, continuity_context=None):
        captured.append(channel_segments)
        # Full coverage (49:00 of 50:00 = 98%) so per-chunk validation passes
        # on the first attempt -- this test is about map slicing, not retry.
        return GeminiResult(transcript="[49:00] Matthias: hi", language="en")

    monkeypatch.setattr(p, "_process_single_shot", fake_single_shot)

    segments = [(0.0, 1200.0, 'host'), (1200.0, 2900.0, 'remote')]
    p._process_chunked(f, None, duration, channel_segments=segments)

    assert len(captured) == 1
    chunk_segments = captured[0]
    assert chunk_segments is not None
    # The remote span past 15:00 must survive the slice in full.
    assert max(t1 for _, t1, _ in chunk_segments) == 2900.0


def test_chunk_loop_does_not_recurse_into_process_audio(monkeypatch, tmp_path):
    """Regression (review S6): the chunk loop must call _process_single_shot
    directly. Routing through process_audio meant a single-chunk fallback
    (chunk == original 15-60 min file) re-entered the single-shot→chunked
    fallback on failure — unbounded mutual recursion with API calls."""
    from gemini_processor import GeminiResult

    p = _bare_processor()
    p.model = "test-model"
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    duration = 50 * 60
    monkeypatch.setattr(p, "_get_duration", lambda path: duration)
    monkeypatch.setattr(p, "_chunk_audio", lambda path, force_chunk=False: [(path, 0.0)])
    monkeypatch.setattr(p, "_generate_with_retry",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")))

    calls = {"single_shot": 0, "process_audio": 0}

    def failing_single_shot(*a, **k):
        calls["single_shot"] += 1
        raise RuntimeError("deterministic failure")

    orig_process_audio = p.process_audio

    def counting_process_audio(*a, **k):
        calls["process_audio"] += 1
        assert calls["process_audio"] < 5, "runaway recursion"
        return orig_process_audio(*a, **k)

    monkeypatch.setattr(p, "_process_single_shot", failing_single_shot)
    monkeypatch.setattr(p, "process_audio", counting_process_audio)

    result = p._process_chunked(f, None, duration)
    # Chunk failed on every retry attempt (CHUNK_RETRY_ATTEMPTS=2); no
    # re-entry into process_audio; the all-chunks-failed error result comes
    # back instead of a RecursionError.
    assert calls["single_shot"] == GeminiAudioProcessor.CHUNK_RETRY_ATTEMPTS
    assert calls["process_audio"] == 0
    assert result.error == "All chunks failed"


# ── malformed-JSON salvage + retry ────────────────────────────────────
#
# A Gemini stream can COMPLETE yet still be unusable: truncated mid-object
# ("Unterminated string") or wrapped in prose ("Extra data: line 1 col 2").
# Before this fix _parse_response just logged + returned an error result and
# the watcher dropped the recording silently. Two real losses prompted it:
# the 2026-06-22 10:17 recording (truncation, recovered on retry) and an
# 888 MB 2026-06-12 recording ("Extra data").


def test_strip_and_parse_json_salvages_trailing_prose():
    """'Extra data' — model appends commentary after the JSON object."""
    raw = '{"language": "en", "transcript": "[00:00] M: hi"}\n\nHope that helps!'
    assert GeminiAudioProcessor._strip_and_parse_json(raw) == {
        "language": "en", "transcript": "[00:00] M: hi"}


def test_strip_and_parse_json_salvages_leading_prose_and_fence():
    """Leading prose plus a ```json fence — extract the outermost object."""
    raw = 'Sure, here is the result:\n```json\n{"a": 1, "b": 2}\n```'
    assert GeminiAudioProcessor._strip_and_parse_json(raw) == {"a": 1, "b": 2}


def test_strip_and_parse_json_raises_on_truncation():
    """Genuine truncation has no closing brace — must RAISE so the caller
    retries the generation rather than salvaging a half object."""
    import json as _json

    import pytest
    with pytest.raises(_json.JSONDecodeError):
        GeminiAudioProcessor._strip_and_parse_json('{"transcript": "[00:00] M: hi')


def _scripted_processor(payloads, monkeypatch):
    """Bare processor whose streaming API yields each payload in turn."""
    import gemini_processor as gp

    p = _bare_processor()
    p.model = "test-model"
    p.temperature = 0.0
    p.max_output_tokens = 1024

    class _Types:
        @staticmethod
        def GenerateContentConfig(**kwargs):
            return kwargs

    p.types = _Types()

    class _Chunk:
        def __init__(self, text):
            self.text = text
            self.usage_metadata = None

    class _Models:
        def generate_content_stream(self, model, contents, config):
            return iter([_Chunk(payloads.pop(0))])

    class _Client:
        models = _Models()

    p.client = _Client()
    monkeypatch.setattr(gp.time, "sleep", lambda *_a, **_k: None)
    return p


def test_generate_with_retry_retries_on_unparseable_json(monkeypatch):
    """First attempt streams truncated JSON; the call must retry and return the
    clean second response — not pass the broken text downstream. This is the
    exact failure that silently dropped the 2026-06-22 10:17 recording."""
    bad = '{"transcript": "[00:00] M: hi'  # unterminated
    good = '{"transcript": "[00:00] M: hi", "language": "en"}'
    payloads = [bad, good]
    p = _scripted_processor(payloads, monkeypatch)

    text, _response = p._generate_with_retry(
        prompt="x", audio_content=None, max_attempts=3)

    assert text == good
    assert payloads == []  # both attempts consumed


def test_generate_with_retry_raises_after_persistent_bad_json(monkeypatch):
    """If every attempt is unparseable, raise (→ watcher fires the failure
    alert) rather than returning junk."""
    import pytest
    payloads = ['{"transcript": "oops'] * 3
    p = _scripted_processor(payloads, monkeypatch)

    with pytest.raises(RuntimeError):
        p._generate_with_retry(prompt="x", audio_content=None, max_attempts=3)


def test_generate_with_retry_skips_validation_when_disabled(monkeypatch):
    """validate_json=False returns raw text untouched (kept as an escape hatch
    for any non-JSON call)."""
    payloads = ['not json at all']
    p = _scripted_processor(payloads, monkeypatch)

    text, _response = p._generate_with_retry(
        prompt="x", audio_content=None, max_attempts=2, validate_json=False)

    assert text == 'not json at all'


def test_short_single_shot_failure_propagates(monkeypatch, tmp_path):
    """A sub-15-min recording can't meaningfully chunk; a single-shot failure
    must propagate, not silently no-op through the chunked path."""
    import pytest

    p = _bare_processor()
    f = tmp_path / "short.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 8 * 60)  # < CHUNK_DURATION_SEC

    def boom(*a, **k):
        raise RuntimeError("boom")

    chunked_called: list[int] = []
    monkeypatch.setattr(p, "_process_single_shot", boom)
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: chunked_called.append(1))
    with pytest.raises(RuntimeError):
        p.process_audio(f)
    assert chunked_called == []
