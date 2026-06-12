"""Tests for apply_speaker_resolution — the label-rewrite logic.

DB and filesystem plumbing are exercised only via _rewrite_labels and
merge_participant_details (pure functions on the gemini dict); the CLI's
psycopg2 path needs a live DSN and is covered by the manual backfill run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from apply_speaker_resolution import (  # noqa: E402
    _rewrite_labels,
    merge_participant_details,
)


def _gem(transcript: str, participants=None, **kw) -> dict:
    d = {"transcript": transcript, "participants": participants or [],
         "speaker_emotions": [], "speaker_pacing": {},
         "interruptions": [], "energy_levels": {}}
    d.update(kw)
    return d


CP = {"name": "Margo Schorta", "email": None, "company": "BlueCare",
      "method": "topic_inference", "confidence": "medium"}


def test_pure_generic_labels_collapse_to_counterpart():
    """1:1 transcript, only generic labels → full reconcile path rewrites
    everything (transcript + structured fields)."""
    g = _gem(
        "[00:00] Speaker B: Hallo.\nMatthias: Hoi.\n"
        "[10:00] Speaker B: Genau.\n",
        participants=[{"name": "Matthias", "role": "host"},
                      {"name": "Speaker B", "role": "participant"}],
        speaker_pacing={"Speaker B": {"wpm_avg": 120}},
    )
    log = _rewrite_labels(g, CP)
    assert log["rewrote_speakers"] >= 1
    assert "Speaker B" not in g["transcript"]
    assert "Margo Schorta: Hallo." in g["transcript"]
    assert any(p["name"] == "Margo Schorta" for p in g["participants"])
    assert "Margo Schorta" in g["speaker_pacing"]


def test_drifted_labels_normalise_but_generic_sibling_survives():
    """Labels 'Speaker 1' + 'Margo': the first-name labels normalise to the
    full canonical name (fuzzy match). 'Speaker 1' is preserved — once the
    canonical is matched elsewhere in the transcript, reconcile's ad-hoc-
    joiner guard treats a separate generic label as a potential third
    voice rather than collapsing it. Conservative by design."""
    g = _gem(
        "Speaker 1: Hallo.\nMatthias: Hoi.\nMargo: Genau.\nMargo: Und so.\n",
        participants=[{"name": "Speaker 1"}],
    )
    log = _rewrite_labels(g, CP)
    assert log["rewrote_speakers"] >= 1
    assert "Margo Schorta: Genau." in g["transcript"]
    assert "Margo Schorta: Und so." in g["transcript"]
    assert "Speaker 1: Hallo." in g["transcript"]  # ad-hoc-joiner guard


def test_multiparty_real_names_never_swallowed():
    """The collapse trap: transcript has REAL other names (Gina, Vanessa)
    plus one generic label. Only the generic label may be rewritten —
    reconcile's singleton_collapse would swallow Gina and Vanessa too."""
    g = _gem(
        "[00:00] Gina: Hallo.\nMatthias: Hoi.\n"
        "[05:00] Speaker B: Ich bin neu da.\n"
        "[10:00] Vanessa: Genau.\n",
        participants=[{"name": "Gina"}, {"name": "Vanessa"},
                      {"name": "Speaker B"}],
    )
    log = _rewrite_labels(g, CP)
    assert log["rewrote_speakers"] == 1
    assert log["decisions"][0]["rule"] == "targeted_generic"
    assert "Gina: Hallo." in g["transcript"]
    assert "Vanessa: Genau." in g["transcript"]
    assert "Margo Schorta: Ich bin neu da." in g["transcript"]
    assert "Speaker B" not in g["transcript"]


def test_two_generic_labels_in_multiparty_abstain():
    """Two distinct generic labels alongside real names could be two
    different unknown people — no rewrite."""
    g = _gem(
        "Gina: Hallo.\nSpeaker B: Hi.\nSpeaker C: Hoi.\nMatthias: Ja.\n",
        participants=[],
    )
    log = _rewrite_labels(g, CP)
    assert log["rewrote_speakers"] == 0
    assert log.get("skipped_ambiguous_multiparty") is True
    assert "Speaker B: Hi." in g["transcript"]
    assert "Speaker C: Hoi." in g["transcript"]


def test_no_labels_no_change():
    g = _gem("Matthias: Selbstgespräch ohne Gegenseite.\n")
    before = g["transcript"]
    log = _rewrite_labels(g, CP)
    assert g["transcript"] == before
    assert log["rewrote_speakers"] == 0


# ── merge_participant_details ─────────────────────────────────────────


def test_merge_adds_counterpart_and_drops_placeholder():
    existing = [
        {"name": "Matthias Heim", "role": "self", "email": "matthias@lailix.com"},
        {"name": "Speaker B", "role": "counterpart", "confidence": "low"},
    ]
    out = merge_participant_details(existing, dict(CP, role_title="Qlik Dev"))
    names = [p["name"] for p in out]
    assert "Matthias Heim" in names
    assert "Margo Schorta" in names
    assert "Speaker B" not in names
    margo = next(p for p in out if p["name"] == "Margo Schorta")
    assert margo["resolution_method"] == "topic_inference"
    assert margo["title"] == "Qlik Dev"


def test_merge_enriches_existing_entry_instead_of_duplicating():
    existing = [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Margo", "role": "counterpart", "email": None},
    ]
    out = merge_participant_details(existing, dict(CP, email="m@bluecare.ch"))
    margos = [p for p in out if p["name"].startswith("Margo")]
    assert len(margos) == 1
    assert margos[0]["name"] == "Margo Schorta"   # normalised to full name
    assert margos[0]["email"] == "m@bluecare.ch"
    assert margos[0]["company"] == "BlueCare"
