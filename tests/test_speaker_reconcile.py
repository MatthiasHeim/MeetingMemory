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


def test_speaker_n_labels_collapse_in_1to1_calendar():
    """Generic 'Speaker 2' style labels ARE candidates and, in a 1:1
    calendar meeting, collapse to the sole canonical attendee. (Before the
    singleton_collapse fix these labels were silently dropped — see the
    Antonella mislabel bug where 'Speaker 1' / 'Speaker 2' / 'Vivienne'
    persisted across chunks.)"""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(transcript="Speaker 2: hi.\nMatthias: hi.")
    log = reconcile(g, cal)
    names = {d["gemini_name"] for d in log["decisions"]}
    assert "Speaker 2" in names
    # Only one non-self gemini label → strict singleton fires.
    assert any(d["rule"] == "singleton" and d["canonical_name"] == "Ladina Walicki-Kasper"
               for d in log["decisions"])
    assert "Ladina Walicki-Kasper: hi." in g["transcript"]


# ── singleton_collapse (Fix 2) ────────────────────────────────────────


def test_singleton_collapse_rewrites_multiple_drifted_labels():
    """Antonella case: 1:1 calendar (one external attendee) and Gemini
    chunk-drift surfaces THREE different labels for the same physical
    speaker ('Speaker 1', 'Speaker 2', 'Vivienne'). Strict singleton can't
    fire (multiple non-self labels), but singleton_collapse must rewrite
    ALL three to the sole calendar attendee."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Antonella Borromeo", "company": "BlueCare",
         "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Speaker 1: Hallo Matthias.\n"
            "Matthias: Schön dich zu sehen.\n"
            "[14:30] Vivienne: Wir haben Standorte.\n"
            "Matthias: Aha.\n"
            "[31:12] Speaker 2: Und dann noch.\n"
        ),
        participants=[
            {"name": "Matthias", "role": "host", "speaking_pct": 40},
            {"name": "Speaker 1", "role": "participant", "speaking_pct": 60},
        ],
        speaker_emotions=[
            {"speaker": "Speaker 1", "arc": [{"time": "[00:30]", "tone": "warm"}]},
            {"speaker": "Vivienne", "arc": [{"time": "[15:00]", "tone": "focused"}]},
        ],
        speaker_pacing={
            "Vivienne": {"wpm_avg": 130, "hesitation_count": 5, "longest_pause_sec": 2.0},
            "Speaker 2": {"wpm_avg": 120, "hesitation_count": 3, "longest_pause_sec": 1.5},
        },
        interruptions=[
            {"time": "[14:31]", "interrupter": "Matthias", "interruptee": "Vivienne"},
        ],
        energy_levels={"Speaker 1": {"avg": "medium", "arc": []}},
    )

    log = reconcile(g, cal)

    # All three drifted labels collapsed to Antonella.
    collapse_decisions = [d for d in log["decisions"]
                          if d["rule"] == "singleton_collapse"]
    collapsed_names = {d["gemini_name"] for d in collapse_decisions}
    assert collapsed_names == {"Speaker 1", "Vivienne", "Speaker 2"}
    for d in collapse_decisions:
        assert d["canonical_name"] == "Antonella Borromeo"
    assert log["collapsed_labels"] == 3
    assert log["rewrote_speakers"] == 3

    # Transcript labels rewritten everywhere.
    assert "Speaker 1" not in g["transcript"]
    assert "Speaker 2" not in g["transcript"]
    assert "Vivienne" not in g["transcript"]
    assert "Antonella Borromeo:" in g["transcript"]
    # Matthias preserved.
    assert "Matthias: Schön dich zu sehen." in g["transcript"]

    # Structured fields rewritten too.
    assert any(p["name"] == "Antonella Borromeo" for p in g["participants"])
    assert all(p["name"] != "Speaker 1" for p in g["participants"])
    assert all(e["speaker"] == "Antonella Borromeo"
               for e in g["speaker_emotions"])
    assert "Antonella Borromeo" in g["speaker_pacing"]
    assert "Vivienne" not in g["speaker_pacing"]
    assert "Speaker 2" not in g["speaker_pacing"]
    assert g["interruptions"][0]["interruptee"] == "Antonella Borromeo"
    assert "Antonella Borromeo" in g["energy_levels"]


def test_singleton_collapse_skipped_for_multi_attendee_drift():
    """When calendar has 2+ external attendees, chunk drift must NOT
    collapse — we can't know which physical speaker each label maps to."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Müller", "role": "participant"},
        {"name": "Anna Weber", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Speaker 1: Hi.\n"
            "Matthias: Hallo.\n"
            "[10:00] Someone Else: Etwas.\n"
        ),
        participants=[
            {"name": "Matthias"},
            {"name": "Speaker 1"},
            {"name": "Someone Else"},
        ],
    )
    log = reconcile(g, cal)
    # No collapse: too many canonicals to disambiguate.
    assert log.get("collapsed_labels", 0) == 0
    assert log["rewrote_speakers"] == 0
    rules = {d["gemini_name"]: d["rule"] for d in log["decisions"]}
    assert rules["Speaker 1"] == "none"
    assert rules["Someone Else"] == "none"
    # Original labels preserved in transcript.
    assert "Speaker 1:" in g["transcript"]
    assert "Someone Else:" in g["transcript"]


def test_singleton_collapse_does_not_overwrite_confident_match():
    """Ad-hoc joiner case: calendar lists ONE external attendee, but Gemini
    correctly identifies the canonical (confident match) AND also surfaces
    a different label. The canonical match must not be overwritten, and
    the extra label must be preserved (could be an ad-hoc joiner). The
    collapse rule only fires when the canonical wasn't already matched."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Some Other Person: Hi.\n"
            "Ladina Walicki-Kasper: Hallo.\n"
            "Matthias: Hi all."
        ),
        participants=[
            {"name": "Some Other Person"},
            {"name": "Ladina Walicki-Kasper"},
            {"name": "Matthias"},
        ],
    )
    log = reconcile(g, cal)
    assert log.get("collapsed_labels", 0) == 0
    decisions = {d["gemini_name"]: d for d in log["decisions"]}
    assert decisions["Ladina Walicki-Kasper"]["rule"] == "confident"
    assert decisions["Some Other Person"]["rule"] == "none"
    assert "Some Other Person:" in g["transcript"]


def test_singleton_collapse_never_collapses_self_label():
    """Self labels are excluded from the collapse pass."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Antonella Borromeo", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "Matthias: Hi.\n"
            "Matthias Heim: Continuing.\n"
            "Speaker 1: Hallo.\n"
            "Vivienne: Sicher."
        ),
        participants=[
            {"name": "Matthias"}, {"name": "Speaker 1"}, {"name": "Vivienne"},
        ],
    )
    log = reconcile(g, cal)
    # Matthias / Matthias Heim resolve to 'self' and must NOT collapse.
    for d in log["decisions"]:
        if d["gemini_name"] in ("Matthias", "Matthias Heim"):
            assert d["rule"] == "self"
            assert d["canonical_name"] != "Antonella Borromeo"
    # Drift labels DO collapse.
    collapsed = {d["gemini_name"] for d in log["decisions"]
                 if d["rule"] == "singleton_collapse"}
    assert collapsed == {"Speaker 1", "Vivienne"}
    # Both Matthias variants survived as Matthias (not rewritten to Antonella).
    assert "Matthias:" in g["transcript"]
    assert "Antonella Borromeo:" in g["transcript"]
    # Self labels are intentionally NOT rewritten away from 'Matthias' —
    # they're anchored, not collapsed.


def test_singleton_collapse_merges_pacing_and_energy_collisions():
    """Regression for P1 review feedback: when multiple drifted labels
    collapse to one canonical, speaker_pacing/energy_levels/speaker_emotions
    must MERGE the colliding entries, not overwrite (last-wins). Before this
    fix, the dict comprehension dropped data silently — pacing for chunks
    1 and 2 of the same physical speaker would be lost.
    """
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Antonella Borromeo", "company": "BlueCare",
         "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Speaker 1: Erstes Chunk.\n"
            "[16:00] Vivienne: Zweites Chunk.\n"
            "[31:00] Speaker 2: Drittes Chunk.\n"
        ),
        participants=[
            {"name": "Matthias", "role": "host"},
            {"name": "Speaker 1", "role": "participant"},
        ],
        # Three drifted labels each carrying their own arc of emotions —
        # all three arcs must survive the collapse.
        speaker_emotions=[
            {"speaker": "Speaker 1",
             "arc": [{"time": "[00:30]", "tone": "warm"}]},
            {"speaker": "Vivienne",
             "arc": [{"time": "[16:30]", "tone": "focused"}]},
            {"speaker": "Speaker 2",
             "arc": [{"time": "[31:30]", "tone": "tired"}]},
        ],
        # Pacing across three chunks — each with different numbers. The
        # merged result must aggregate, not silently keep only the last.
        speaker_pacing={
            "Speaker 1": {"wpm_avg": 100, "hesitation_count": 5,
                          "longest_pause_sec": 2.0},
            "Vivienne":  {"wpm_avg": 140, "hesitation_count": 7,
                          "longest_pause_sec": 5.0},
            "Speaker 2": {"wpm_avg": 120, "hesitation_count": 3,
                          "longest_pause_sec": 3.0},
        },
        # Energy across three chunks — arcs must concat, avg must mode.
        energy_levels={
            "Speaker 1": {"avg": "medium",
                          "arc": [{"time": "[00:30]", "level": "medium"}]},
            "Vivienne":  {"avg": "high",
                          "arc": [{"time": "[16:30]", "level": "high"}]},
            "Speaker 2": {"avg": "medium",
                          "arc": [{"time": "[31:30]", "level": "low"}]},
        },
    )

    reconcile(g, cal)

    # speaker_emotions: one entry per canonical speaker, arcs concatenated.
    emotions_by_speaker = {e["speaker"]: e for e in g["speaker_emotions"]}
    assert "Antonella Borromeo" in emotions_by_speaker
    assert len(emotions_by_speaker["Antonella Borromeo"]["arc"]) == 3, (
        "All three arc events must survive collapse; got "
        f"{emotions_by_speaker['Antonella Borromeo']['arc']!r}"
    )

    # speaker_pacing: aggregated — mean wpm, sum hesitations, max pause.
    pacing = g["speaker_pacing"]["Antonella Borromeo"]
    assert pacing["wpm_avg"] == int((100 + 140 + 120) / 3), pacing
    assert pacing["hesitation_count"] == 5 + 7 + 3, pacing
    assert pacing["longest_pause_sec"] == 5.0, pacing

    # energy_levels: arcs concatenated, avg is the mode (medium ×2 > high ×1).
    energy = g["energy_levels"]["Antonella Borromeo"]
    assert len(energy["arc"]) == 3
    assert energy["avg"] == "medium"

    # No drifted labels survive.
    for stale in ("Speaker 1", "Vivienne", "Speaker 2"):
        assert stale not in g["speaker_pacing"]
        assert stale not in g["energy_levels"]
        assert not any(e["speaker"] == stale for e in g["speaker_emotions"])


# ── participants[] backfill (2026-08-10 binding-gap fix) ───────────────
#
# Gemini's own participants[] can come back empty even when the calendar
# resolved real attendees and the transcript carries real speaker labels
# (the Stefan-Sieber meeting: "Calendar attendees (3)" logged, a rename
# applied, but the saved JSON's participants[] stayed [] all the way to
# the DB). reconcile() must not leave a real meeting's participant list
# empty when it already has everything it needs to fill it in.


def test_backfills_empty_participants_from_transcript_labels():
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Sieber", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Stefan Sieber: Hoi.\n"
            "[00:05] Matthias: Hoi Stefan.\n"
            "[00:10] Stefan Sieber: Alles klar?\n"
        ),
        participants=[],  # Gemini returned no participants for this call
    )
    log = reconcile(g, cal)
    assert log["participants_backfilled"] == 2
    names_roles = {p["name"]: p["role"] for p in g["participants"]}
    assert names_roles == {"Matthias": "host", "Stefan Sieber": "participant"}


def test_backfill_uses_canonical_names_after_rename():
    """The seeded list must reflect the SAME rename the transcript itself
    got — not Gemini's raw (possibly wrong) guess."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Ladina Walicki-Kasper", "role": "participant"},
    ])
    g = _gem(
        transcript=(
            "[00:00] Nadine Maricic: Hallo.\n"
            "[00:05] Matthias: Hi.\n"
        ),
        participants=[],
    )
    log = reconcile(g, cal)
    assert log["participants_backfilled"] == 2
    names = {p["name"] for p in g["participants"]}
    assert names == {"Matthias", "Ladina Walicki-Kasper"}
    assert "Nadine Maricic" not in names


def test_backfill_skipped_when_participants_already_present():
    """SIDE B: an already-populated participants[] must never be
    clobbered by the backfill path — only a genuinely empty list qualifies."""
    cal = _cal([
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Sieber", "role": "participant"},
    ])
    existing = [{"name": "Stefan Sieber", "role": "participant", "speaking_pct": 60}]
    g = _gem(
        transcript="[00:00] Stefan Sieber: Hoi.\n[00:05] Matthias: Hoi.\n",
        participants=existing,
    )
    log = reconcile(g, cal)
    assert log["participants_backfilled"] == 0
    assert g["participants"] == existing


def test_backfill_skipped_when_no_transcript_labels_either():
    """No calendar attendee AND nothing to harvest from the transcript ->
    skipped_no_calendar fires first; nothing to backfill from anyway."""
    cal = _cal([{"name": "Matthias Heim", "role": "self"}])
    g = _gem(transcript="", participants=[])
    log = reconcile(g, cal)
    assert log["skipped_no_calendar"] is True
    assert g["participants"] == []
    assert log["participants_backfilled"] == 0
