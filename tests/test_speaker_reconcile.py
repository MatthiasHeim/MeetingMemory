"""Tests for speaker_reconcile — canonicalising Gemini-guessed speaker names."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from speaker_reconcile import reconcile  # noqa: E402


def _cal(attendees: list[dict]) -> dict:
    return {"participant_details": attendees}


def _gem(transcript: str, participants: list[dict] | None = None,
         speaker_emotions: list[dict] | None = None,
         speaker_pacing: dict | None = None,
         interruptions: list[dict] | None = None,
         energy_levels: dict | None = None) -> dict:
    return {
        "transcript": transcript,
        "participants": participants or [],
        "speaker_emotions": speaker_emotions or [],
        "speaker_pacing": speaker_pacing or {},
        "interruptions": interruptions or [],
        "energy_levels": energy_levels or {},
    }


def test_singleton_rewrites_wrong_name_in_1to1_meeting():
    """Avosano case: 1:1 meeting, Gemini hallucinates a wrong name. The
    singleton rule must catch it and rewrite to the calendar attendee."""
    cal = _cal([
        {"name": "Matthias Heim", "email": "matthias@lailix.com",
         "company": "Lailix", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "email": "ladina.walicki@avosano.ch",
         "company": "Avosano", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Nadine Maricic: Hallo zusammen.\n"
            "Matthias: Schön, dass du da bist.\n"
            "Nadine Maricic: Danke, gerne."
        ),
        participants=[
            {"name": "Matthias", "role": "host", "speaking_pct": 40},
            {"name": "Nadine Maricic", "role": "participant", "speaking_pct": 60},
        ],
        speaker_emotions=[
            {"speaker": "Nadine Maricic", "arc": [{"time": "[01:00]", "tone": "warm"}]},
        ],
        speaker_pacing={"Nadine Maricic": {"wpm_avg": 110}},
        interruptions=[
            {"time": "[02:00]", "interrupter": "Matthias", "interruptee": "Nadine Maricic"},
        ],
        energy_levels={"Nadine Maricic": {"avg": "medium"}},
    )

    log = reconcile(g, cal)

    assert log["rewrote_speakers"] == 1
    assert any(d["rule"] == "singleton" and d["canonical_name"] == "Ladina Walicki-Kasper"
               for d in log["decisions"])
    # transcript label rewritten in both bracket-prefixed and bare positions
    assert "Nadine Maricic" not in g["transcript"]
    assert "Ladina Walicki-Kasper: Hallo zusammen." in g["transcript"]
    assert "Ladina Walicki-Kasper: Danke, gerne." in g["transcript"]
    # participants/emotions/pacing/interruptions/energy all rewritten
    assert any(p["name"] == "Ladina Walicki-Kasper" for p in g["participants"])
    assert g["speaker_emotions"][0]["speaker"] == "Ladina Walicki-Kasper"
    assert "Ladina Walicki-Kasper" in g["speaker_pacing"]
    assert g["interruptions"][0]["interruptee"] == "Ladina Walicki-Kasper"
    assert "Ladina Walicki-Kasper" in g["energy_levels"]


def test_confident_match_full_name():
    """First+last tokens both match a calendar attendee → 'confident' rule.
    Used when there are 2+ external attendees so singleton can't fire."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Müller", "role": "participant"},
        {"name": "Anna Weber", "role": "participant"},
    ])
    g = _gem(
        transcript="Stefan Müller: Hi.\nAnna Weber: Hallo.\nMatthias: Hi all.",
        participants=[{"name": "Stefan Müller"}, {"name": "Anna Weber"},
                       {"name": "Matthias"}],
    )
    log = reconcile(g, cal)
    rules = {d["gemini_name"]: d["rule"] for d in log["decisions"]}
    # Already canonical → 'confident' rule fires but no rewrite needed.
    assert rules["Stefan Müller"] == "confident"
    assert rules["Anna Weber"] == "confident"
    assert log["rewrote_speakers"] == 0  # names already match canonical


def test_fuzzy_first_name_only_when_unambiguous():
    """First-name match in a multi-attendee meeting (singleton can't fire) →
    fuzzy rewrite when exactly one external shares the first name."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Müller", "role": "participant"},
        {"name": "Anna Weber", "role": "participant"},
    ])
    g = _gem(
        transcript="Stefan: Hallo.\nAnna Weber: Hi.\nMatthias: Hi all.",
        participants=[{"name": "Stefan"}, {"name": "Anna Weber"},
                       {"name": "Matthias"}],
    )
    log = reconcile(g, cal)
    rules = {d["gemini_name"]: d["rule"] for d in log["decisions"]}
    assert rules["Stefan"] == "fuzzy"
    assert "Stefan Müller: Hallo." in g["transcript"]


def test_first_name_collision_keeps_gemini_guess():
    """If two external attendees share a first name, fuzzy must not fire."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Müller", "role": "participant"},
        {"name": "Stefan Weber", "role": "participant"},
    ])
    g = _gem(transcript="Stefan: Hallo.", participants=[{"name": "Stefan"}])
    log = reconcile(g, cal)
    assert log["rewrote_speakers"] == 0
    rules = {d["gemini_name"]: d["rule"] for d in log["decisions"]}
    assert rules["Stefan"] == "none"
    assert g["transcript"] == "Stefan: Hallo."


def test_no_calendar_match_skips_rewrite():
    """When calendar returned no external attendees (solo/no-match) skip."""
    cal = _cal([{"name": "Matthias Heim", "role": "self"}])
    g = _gem(transcript="Speaker 1: Hallo.", participants=[{"name": "Speaker 1"}])
    log = reconcile(g, cal)
    assert log["skipped_no_calendar"] is True
    assert g["transcript"] == "Speaker 1: Hallo."


def test_self_label_anchors_to_matthias():
    """'Matthias' (or 'host') always resolves to canonical SELF — never rewritten away."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(transcript="Matthias: Hi.")
    log = reconcile(g, cal)
    # Matthias should be flagged 'self' rule, but text label 'Matthias' is not a rewrite
    # candidate (the canonical full is 'Matthias Heim' — but rewriting to that would
    # break the convention; we expect 'Matthias' kept unless the canonical is identical).
    decisions = {d["gemini_name"]: d for d in log["decisions"]}
    assert decisions["Matthias"]["rule"] == "self"


def test_unmatched_external_speaker_kept():
    """Speaker not in calendar (e.g. ad-hoc joiner) is preserved, logged 'none'."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(
        transcript="[00:00] Some Other Person: Hi.\nLadina Walicki-Kasper: Hi.",
        participants=[{"name": "Some Other Person"}, {"name": "Ladina Walicki-Kasper"}],
    )
    log = reconcile(g, cal)
    decisions = {d["gemini_name"]: d for d in log["decisions"]}
    assert decisions["Some Other Person"]["rule"] == "none"
    assert "Some Other Person:" in g["transcript"]


def test_pattern_matches_bracket_prefixed_label():
    """`[00:00] Name:` form (timestamp before name) gets rewritten."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(
        transcript="[00:00] Nadine Maricic: Hallo.\n[02:15] Matthias: Hi.",
        participants=[{"name": "Nadine Maricic"}, {"name": "Matthias"}],
    )
    reconcile(g, cal)
    assert "[00:00] Ladina Walicki-Kasper: Hallo." in g["transcript"]
    assert "Nadine Maricic" not in g["transcript"]


def test_mid_sentence_fragments_filtered_from_candidates():
    """Real Gemini transcripts have inline `[MM:SS]` timestamps mid-paragraph.
    The transcript-scan must reject sentence fragments that happen to end at
    a colon, otherwise the decision log fills with junk like
    'Niederbipp und eine in Romont. [01' as a 'speaker name'."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Ladina: Wir haben Standorte in Niederbipp "
            "und eine in Romont. [01:23] Und es gibt einen Plan.\n"
            "Matthias: Aha. Frage? Sicher."
        ),
        participants=[{"name": "Ladina"}, {"name": "Matthias"}],
    )
    log = reconcile(g, cal)
    names = {d["gemini_name"] for d in log["decisions"]}
    # Real speaker names should appear:
    assert "Ladina" in names
    assert "Matthias" in names
    # Sentence fragments must NOT:
    for n in names:
        assert "[" not in n
        assert "." not in n
        assert "?" not in n
        assert not any(c.isdigit() for c in n)


def test_speaker_n_labels_ignored_as_candidates():
    """Generic 'Speaker 2' style labels do not produce a rename decision."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(transcript="Speaker 2: hi.\nMatthias: hi.")
    log = reconcile(g, cal)
    names = {d["gemini_name"] for d in log["decisions"]}
    assert "Speaker 2" not in names
