"""Tests for speaker_hints — transcript-based counterpart inference.

All signals are validated against a synthetic directory; the Brain repo is
never touched. Patterns mirror the real failure cases from the 2026-06-12
A/B experiment (Natascha label drift, Gina third-person trap, Stefan-bias
label misattribution).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from speaker_hints import (  # noqa: E402
    detect_counterpart,
    load_client_directories,
)

DIR = [
    {"name": "Natascha Bekrić", "first": "natascha", "company": "BlueCare",
     "role": "Customer Service", "email": None},
    {"name": "Lorenz Fehr", "first": "lorenz", "company": "BlueCare",
     "role": "CEO", "email": "lorenz.fehr@bluecare.ch"},
    {"name": "Gina Keller", "first": "gina", "company": "Zur Rose",
     "role": "PMO", "email": None},
    {"name": "Florian Weber", "first": "florian", "company": "Zur Rose",
     "role": "Jira-Entwickler", "email": None},
    # Two Stefans at different clients — first-name ambiguity case.
    {"name": "Stefan Sieber", "first": "stefan", "company": "BlueCare",
     "role": "CTO", "email": "stefan.sieber@bluecare.ch"},
    {"name": "Stefan", "first": "stefan", "company": "SarahKitaApp",
     "role": "Kita-Leiter", "email": None},
]


def _gem(transcript: str) -> dict:
    return {"transcript": transcript, "participants": []}


# ── signal 1: transcript speaker label ────────────────────────────────


def test_label_drift_resolves_counterpart():
    """The Natascha pattern: Gemini starts with 'Speaker 1' and drifts to
    the real name as a label once it's audible. The label, validated
    against the directory, identifies the counterpart."""
    g = _gem(
        "Speaker 1: Es isch viel.\n"
        "Matthias: Was machsch du?\n"
        "Natascha: Ja, die sind im Jira.\n"
        "Matthias: Mhm.\n"
        "Natascha: Also das isch de Workflow.\n"
    )
    r = detect_counterpart(g, DIR)
    assert r is not None
    assert r["name"] == "Natascha Bekrić"
    assert r["method"] == "transcript_label"
    assert r["company"] == "BlueCare"


def test_unicode_directory_name_matches():
    """Bekrić has a non-ASCII letter — the directory loader and matcher
    must not drop it (regression: ASCII-umlaut-only name regex)."""
    g = _gem("Natascha: Hallo.\nMatthias: Hoi.\nNatascha: Guet.")
    r = detect_counterpart(g, DIR)
    assert r is not None and r["name"] == "Natascha Bekrić"


def test_ambiguous_first_name_label_abstains():
    """'Stefan' labels with TWO directory Stefans must abstain — a bare
    first-name label cannot disambiguate (and per the Stefan-bias
    incident, may itself be a Gemini hallucination)."""
    g = _gem(
        "Stefan: Mir sind da.\nMatthias: Ja.\nStefan: Genau.\n"
    )
    assert detect_counterpart(g, DIR) is None


def test_full_name_label_disambiguates_shared_first_name():
    g = _gem(
        "Stefan Sieber: Mir sind da.\nMatthias: Ja.\nStefan Sieber: Genau.\n"
    )
    r = detect_counterpart(g, DIR)
    assert r is not None and r["name"] == "Stefan Sieber"


def test_misattributed_label_rejected_by_self_reference():
    """A labeled speaker who talks about that very name in the third person
    (Swiss German article: 'de Stefan') cannot BE that person — the label
    is a misattribution and must not resolve. Uses a single-Stefan
    directory so the ambiguity guard can't mask the check."""
    single_stefan = [d for d in DIR if d["name"] != "Stefan"]
    g = _gem(
        "Stefan: Ich han mit em Stefan gredet und de Stefan het gseit, "
        "es gaht.\nMatthias: Aha.\nStefan: Genau, de Stefan macht das.\n"
    )
    assert detect_counterpart(g, single_stefan) is None


def test_one_off_label_ignored():
    """A label appearing once is usually a mid-paragraph capture."""
    g = _gem(
        "Speaker 1: Mir hand churz gluegt. Lorenz: het er gseit.\n"
        "Matthias: Ja.\nSpeaker 1: Genau.\n"
    )
    assert detect_counterpart(g, DIR) is None


# ── signal 2: direct address ─────────────────────────────────────────


def test_direct_address_by_host_resolves():
    g = _gem(
        "Matthias: Sali Gina, schön dich z'gseh.\n"
        "Speaker B: Hoi Matthias.\n"
        "Matthias: Gina, was meinsch du dezue?\n"
        "Speaker B: Ja, das passt.\n"
    )
    r = detect_counterpart(g, DIR)
    assert r is not None
    assert r["name"] == "Gina Keller"
    assert r["method"] == "direct_address"


def test_third_person_article_mentions_do_not_fire():
    """The Gina trap: the counterpart (Florian) talks ABOUT Gina constantly
    ('de Gina', 'd'Gina'). Article-marked mentions are third person and
    must never resolve Gina as the counterpart."""
    g = _gem(
        "Matthias: Wie gsehsch du das?\n"
        "Speaker B: Das het üs d'Gina gseit. Ich ha vo de Gina verstande, "
        "dass es guet isch. D'Gina luegt druf.\n"
        "Matthias: Okay, denn frage mer d'Gina.\n"
        "Speaker B: Genau, de Gina passt das.\n"
    )
    assert detect_counterpart(g, DIR) is None


def test_single_vocative_insufficient():
    """One direct-address hit can be a mis-hear; require repetition."""
    g = _gem(
        "Matthias: Sali Gina.\nSpeaker B: Hoi.\nMatthias: Was meinsch?\n"
        "Speaker B: Passt.\n"
    )
    assert detect_counterpart(g, DIR) is None


def test_two_addressed_people_abstains():
    """Host addressing two directory people → multi-party, ambiguous."""
    g = _gem(
        "Matthias: Sali Gina, hoi Florian.\n"
        "Speaker B: Hoi.\n"
        "Matthias: Gina, was meinsch? Und du, Florian?\n"
        "Speaker B: Ja.\nSpeaker C: Au guet.\n"
    )
    assert detect_counterpart(g, DIR) is None


# ── plumbing ──────────────────────────────────────────────────────────


def test_empty_inputs():
    assert detect_counterpart({}, DIR) is None
    assert detect_counterpart({"transcript": ""}, DIR) is None
    assert detect_counterpart(_gem("Matthias: Hallo."), []) is None


def test_load_client_directories_parses_tables(tmp_path):
    client = tmp_path / "Acme"
    client.mkdir()
    (client / "Acme_ClientContext.md").write_text(
        "# Profile\n\n"
        "## 📧 Contact Directory\n\n"
        "| Name | Role | Email |\n"
        "|---|---|---|\n"
        "| Anna Müller | CEO | anna.mueller@acme.ch |\n"
        "| **Bob Bekrić** | CTO (interim) | (TBD) |\n"
        "| not-a-name 123 | x | y |\n\n"
        "## Other Section\n\n"
        "| Ignored Person | Role | E |\n",
        encoding="utf-8",
    )
    people = load_client_directories(tmp_path)
    names = {p["name"] for p in people}
    assert "Anna Müller" in names
    assert "Bob Bekrić" in names          # bold markers + unicode survive
    assert "Ignored Person" not in names  # outside directory section
    anna = next(p for p in people if p["first"] == "anna")
    assert anna["email"] == "anna.mueller@acme.ch"
    assert anna["company"] == "Acme"


def test_load_client_directories_missing_dir(tmp_path):
    assert load_client_directories(tmp_path / "nope") == []
