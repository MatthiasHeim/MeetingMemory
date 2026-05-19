"""Tests for gemini_processor — covers calendar-attendee injection into the
audio prompt (Fix 1 for the speaker-mislabeling bug). The actual Gemini API
call is not exercised here; that's an integration concern."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gemini_processor import (  # noqa: E402
    AUDIO_ANALYSIS_PROMPT,
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
