"""Pure-function tests for calendar collision bind / unknown."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from apply_calendar_candidate import (  # noqa: E402
    apply_bind_to_transcript,
    apply_unknown_to_transcript,
    find_candidate,
    mark_bound,
    mark_unknown,
    primary_counterpart,
    publication_from_candidate,
)
from attribution_gate import gate  # noqa: E402


ROSTER = [
    {
        "event_id": "evt-tanja",
        "title": "Tanja / Matthias",
        "start": "2026-09-04T08:00:00Z",
        "company": "itesys",
        "attendees": [
            {"name": "Matthias Heim", "role": "self", "company": "Lailix"},
            {"name": "Tanja Example", "email": "tanja@example.com",
             "role": "participant", "company": "itesys"},
        ],
    },
    {
        "event_id": "evt-sarah",
        "title": "Sarah / Matthias",
        "start": "2026-09-04T08:05:00Z",
        "company": None,
        "attendees": [
            {"name": "Matthias Heim", "role": "self", "company": "Lailix"},
            {"name": "Sarah Stauffer", "email": "sarah@example.com",
             "role": "participant"},
        ],
    },
]


def test_find_candidate_refuses_ids_outside_roster():
    assert find_candidate(ROSTER, "evt-sarah")["title"] == "Sarah / Matthias"
    assert find_candidate(ROSTER, "evt-invented") is None
    assert find_candidate(ROSTER, None) is None


def test_bind_publishes_only_the_chosen_roster_event():
    sarah = find_candidate(ROSTER, "evt-sarah")
    published = publication_from_candidate(sarah)
    names = [p["name"] for p in published["participant_details"]]
    assert names == ["Matthias Heim", "Sarah Stauffer"]
    assert "Tanja" not in " ".join(names)
    assert published["calendar_event_id"] == "evt-sarah"
    search = mark_bound(
        {"match_count": 2, "ambiguous": True, "candidates": ROSTER},
        sarah, "high", "named intro",
    )
    assert search["identity_authoritative"] is True
    assert search["ambiguous"] is False
    assert search["resolution"]["event_id"] == "evt-sarah"


def test_unknown_stays_non_authoritative():
    search = mark_unknown(
        {"match_count": 2, "ambiguous": True, "candidates": ROSTER},
        "cannot tell",
    )
    assert search["identity_authoritative"] is False
    assert search["ambiguous"] is True
    assert search["resolution"]["decision"] == "unknown"


def test_bind_relabels_generic_speaker_within_roster_only():
    data = {
        "transcript": "[00:00] Speaker B: Hoi Matthias.\n[00:10] Matthias: Hoi.",
        "participants": [{"name": "Speaker B"}, {"name": "Matthias"}],
        "_meta": {"speaker_attribution": {
            "speaker_dependent_actions": "resolve_then_extract",
            "missing_stages": ["calendar_identity_review"],
            "calendar_search": {"match_count": 2, "ambiguous": True, "candidates": ROSTER},
        }},
    }
    recon = apply_bind_to_transcript(data, find_candidate(ROSTER, "evt-sarah"))
    assert recon["rewrote_speakers"] >= 1
    assert "Sarah Stauffer:" in data["transcript"]
    assert "Tanja" not in data["transcript"]
    assert primary_counterpart(find_candidate(ROSTER, "evt-tanja"))["name"] == "Tanja Example"
    attr = data["_meta"]["speaker_attribution"]
    assert attr["calendar_search"]["identity_authoritative"] is True
    assert "calendar_identity_review" not in attr["missing_stages"]
    assert gate(data)["speaker_dependent_actions"] == "require_turn_evidence"


def test_unknown_on_transcript_holds_named_extraction():
    data = {
        "transcript": "[00:00] Speaker B: Hello",
        "_meta": {"speaker_attribution": {
            "speaker_dependent_actions": "resolve_then_extract",
            "missing_stages": ["calendar_identity_review"],
            "calendar_search": {"match_count": 2, "candidates": ROSTER},
        }},
    }
    apply_unknown_to_transcript(data, "no roster name spoken")
    decision = gate(data)
    assert decision["speaker_dependent_actions"] == "hold"
    assert not decision["trigger_claude"]
    assert not decision["allow_named_commitments"]
