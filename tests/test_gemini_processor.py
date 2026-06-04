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


def test_process_audio_file_convenience_forwards_kwarg():
    """The module-level convenience wrapper must also accept known_attendees."""
    sig = inspect.signature(process_audio_file)
    assert "known_attendees" in sig.parameters
    assert sig.parameters["known_attendees"].default is None


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


def test_threshold_is_one_hour():
    """Regression: the single-call window is 60 min (raised from 15)."""
    assert GeminiAudioProcessor.CHUNK_THRESHOLD_SEC == 60 * 60


def test_audio_over_threshold_goes_straight_to_chunked(monkeypatch, tmp_path):
    p = _bare_processor()
    f = tmp_path / "long.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 90 * 60)  # 90 min
    seen: list[str] = []
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: seen.append("chunked") or "R")
    monkeypatch.setattr(p, "_process_single_shot", lambda *a, **k: seen.append("single") or "R")
    assert p.process_audio(f) == "R"
    assert seen == ["chunked"]


def test_audio_under_threshold_uses_single_shot(monkeypatch, tmp_path):
    p = _bare_processor()
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 45 * 60)  # 45 min
    seen: list[str] = []
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: seen.append("chunked") or "R")
    monkeypatch.setattr(p, "_process_single_shot", lambda *a, **k: seen.append("single") or "R")
    assert p.process_audio(f) == "R"
    assert seen == ["single"]


def test_single_shot_failure_falls_back_to_chunked(monkeypatch, tmp_path):
    """A 45-min recording (> CHUNK_DURATION_SEC) whose single call fails must
    fall back to chunked rather than lose the whole meeting."""
    p = _bare_processor()
    f = tmp_path / "mid.mp3"
    f.write_bytes(b"x")
    monkeypatch.setattr(p, "_get_duration", lambda path: 45 * 60)
    seen: list[str] = []

    def boom(*a, **k):
        seen.append("single")
        raise RuntimeError("Server disconnected without sending a response.")

    monkeypatch.setattr(p, "_process_single_shot", boom)
    monkeypatch.setattr(p, "_process_chunked", lambda *a, **k: seen.append("chunked") or "R")
    assert p.process_audio(f) == "R"
    assert seen == ["single", "chunked"]


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
