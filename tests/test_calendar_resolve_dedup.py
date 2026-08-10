"""Tests for calendar_resolve's attendee dedup — Fix 2 (P1.4) of
docs/../meeting-pipeline-investigation-2026-08-10.md.

2026-08-10: a Stefan-Sieber meeting's raw calendar attendee list was
"Matthias Heim, Matthias Heim, Stefan Sieber" (the host listed twice, e.g.
via an organizer/alias split). Left unhandled, the duplicate host entry
survived into participant_details and was then misread by
transcribe_watcher's namesake guard as a REMOTE participant sharing the
host's first name — disabling speaker verification entirely for that
meeting. `_dedupe_attendees` collapses duplicates before `resolve()` builds
participant_details; `resolve()` itself also treats an exact full-name
match to the host as the host, not a second attendee, even when the
duplicate's email differs from the raw-list dedup key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import calendar_resolve as cr  # noqa: E402


# ── _dedupe_attendees: pure-function unit tests ────────────────────────


def test_dedupes_identical_email_case_insensitive():
    attendees = [
        {"email": "Stefan.Sieber@BlueCare.ch", "displayName": "Stefan Sieber"},
        {"email": "stefan.sieber@bluecare.ch", "displayName": "Stefan Sieber"},
    ]
    out = cr._dedupe_attendees(attendees)
    assert len(out) == 1


def test_dedupes_by_display_name_when_email_missing():
    attendees = [
        {"displayName": "Matthias Heim"},
        {"displayName": "Matthias Heim"},
        {"displayName": "Stefan Sieber", "email": "stefan@bluecare.ch"},
    ]
    out = cr._dedupe_attendees(attendees)
    assert len(out) == 2


def test_keeps_distinct_people():
    """SIDE B: distinct attendees (different email AND different name) must
    survive dedup untouched — a guard that collapses everything is as
    broken as one that collapses nothing."""
    attendees = [
        {"email": "matthias@lailix.com", "displayName": "Matthias Heim", "self": True},
        {"email": "stefan.sieber@bluecare.ch", "displayName": "Stefan Sieber"},
        {"email": "anna.weber@bluecare.ch", "displayName": "Anna Weber"},
    ]
    out = cr._dedupe_attendees(attendees)
    assert len(out) == 3


def test_same_display_name_different_email_not_collapsed_by_dedupe_alone():
    """A same-name-different-email duplicate (e.g. host organizer alias)
    is NOT caught by _dedupe_attendees itself — both entries carry an
    email, so the email-based key treats them as distinct. This residual
    case is what resolve()'s exact-full-name-match-to-host check exists
    for (see test_resolve_collapses_duplicate_host_attendee_entries)."""
    attendees = [
        {"email": "matthias@lailix.com", "displayName": "Matthias Heim"},
        {"email": "matthias.alias@example.com", "displayName": "Matthias Heim"},
    ]
    out = cr._dedupe_attendees(attendees)
    assert len(out) == 2


# ── resolve(): end-to-end collapse of the host's own duplicate entry ──


def _event(attendees):
    return {
        "id": "evt1",
        "summary": "Stefan Sieber 1:1",
        "start": {"dateTime": "2026-08-10T13:59:30Z"},
        "attendees": attendees,
    }


def test_resolve_collapses_duplicate_host_attendee_entries(monkeypatch):
    """The exact reported bug: raw attendees "Matthias Heim, Matthias Heim,
    Stefan Sieber" must resolve to exactly 2 participant_details entries,
    with only ONE self entry and the duplicate logged as `self_duplicate`
    (not surfaced as a second "participant")."""
    event = _event([
        {"email": "matthias@lailix.com", "displayName": "Matthias Heim", "self": True},
        {"email": "matthias.alias@example.com", "displayName": "Matthias Heim"},
        {"email": "stefan.sieber@bluecare.ch", "displayName": "Stefan Sieber"},
    ])
    monkeypatch.setattr(cr, "_gws_calendar_events", lambda *a, **k: [event])
    monkeypatch.setattr(cr, "_load_client_names", lambda: [])

    result = cr.resolve("2026-08-10_15-59-30.json")

    pdetails = result["participant_details"]
    assert len(pdetails) == 2
    names = [p["name"] for p in pdetails]
    assert names == ["Matthias Heim", "Stefan Sieber"]
    roles = {p["name"]: p["role"] for p in pdetails}
    assert roles["Matthias Heim"] == "self"
    assert roles["Stefan Sieber"] == "participant"

    resolutions = result["participant_resolution_log"]["resolutions"]
    self_resolutions = [r for r in resolutions if r["method"] == "self"]
    assert len(self_resolutions) == 1
    assert any(r["method"] == "self_duplicate" for r in resolutions)


def test_resolve_keeps_genuine_namesake_as_separate_participant(monkeypatch):
    """SIDE B: a REMOTE attendee who genuinely shares the host's first name
    (different last name) is a real person and must stay a distinct
    'participant' entry — the exact-full-name check must not over-collapse."""
    event = _event([
        {"email": "matthias@lailix.com", "displayName": "Matthias Heim", "self": True},
        {"email": "matthias.mueller@example.com", "displayName": "Matthias Müller"},
    ])
    monkeypatch.setattr(cr, "_gws_calendar_events", lambda *a, **k: [event])
    monkeypatch.setattr(cr, "_load_client_names", lambda: [])

    result = cr.resolve("2026-08-10_15-59-30.json")

    pdetails = result["participant_details"]
    assert len(pdetails) == 2
    roles = {p["name"]: p["role"] for p in pdetails}
    assert roles["Matthias Heim"] == "self"
    assert roles["Matthias Müller"] == "participant"


def test_resolve_dedupes_identical_raw_attendee_entries(monkeypatch):
    """A plain double-invite of an external attendee (same email twice)
    must also collapse to one participant_details entry."""
    event = _event([
        {"email": "matthias@lailix.com", "displayName": "Matthias Heim", "self": True},
        {"email": "stefan.sieber@bluecare.ch", "displayName": "Stefan Sieber"},
        {"email": "Stefan.Sieber@bluecare.ch", "displayName": "Stefan Sieber"},
    ])
    monkeypatch.setattr(cr, "_gws_calendar_events", lambda *a, **k: [event])
    monkeypatch.setattr(cr, "_load_client_names", lambda: [])

    result = cr.resolve("2026-08-10_15-59-30.json")
    names = [p["name"] for p in result["participant_details"]]
    assert names == ["Matthias Heim", "Stefan Sieber"]
