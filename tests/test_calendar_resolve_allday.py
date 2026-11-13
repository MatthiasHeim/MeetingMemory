"""All-day / multi-day events must not count as meeting-bind candidates.

2026-11-12: an all-day event that started on 2026-11-06 fell inside the search
window and produced match_count 3, a false collision that put a normal
meeting on `resolve_then_extract`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import calendar_resolve as cr  # noqa: E402


def _timed(eid, start, attendees=None):
    return {"id": eid, "summary": f"Meeting {eid}", "start": {"dateTime": start},
            "attendees": attendees or [
                {"email": "matthias@lailix.com", "displayName": "Matthias Heim", "self": True},
                {"email": "anna@example.com", "displayName": "Anna Weber"}]}


def _all_day(eid, date):
    return {"id": eid, "summary": f"Trip {eid}", "start": {"date": date},
            "end": {"date": "2026-11-14"}}


def test_is_all_day_only_for_date_only_starts():
    assert cr._is_all_day(_all_day("a", "2026-11-06")) is True
    assert cr._is_all_day(_timed("t", "2026-11-12T15:00:00Z")) is False
    assert cr._is_all_day({"id": "no-start"}) is False


def test_all_day_events_do_not_create_a_false_collision(monkeypatch):
    events = [_all_day("a1", "2026-11-06"), _all_day("a2", "2026-11-12"),
              _timed("real", "2026-11-12T15:30:00Z")]
    monkeypatch.setattr(cr, "_gws_calendar_events", lambda *a, **k: events)
    monkeypatch.setattr(cr, "_load_client_names", lambda: [])
    out = cr.resolve("2026-11-12_16-30-49.json")
    search = out["participant_resolution_log"]["calendar_search"]
    assert search["match_count"] == 1
    assert search["ambiguous"] is False
    assert search["chosen_event_id"] == "real"
    assert [c["event_id"] if "event_id" in c else c.get("id") for c in search["candidates"]] == ["real"]
    assert out["calendar_event_id"] == "real"


def test_two_real_timed_events_still_collide(monkeypatch):
    """SIDE B: the filter must not hide a genuine collision."""
    events = [_timed("x", "2026-11-12T15:30:00Z"), _timed("y", "2026-11-12T15:45:00Z"),
              _all_day("a1", "2026-11-06")]
    monkeypatch.setattr(cr, "_gws_calendar_events", lambda *a, **k: events)
    monkeypatch.setattr(cr, "_load_client_names", lambda: [])
    search = cr.resolve("2026-11-12_16-30-49.json")["participant_resolution_log"]["calendar_search"]
    assert search["match_count"] == 2 and search["ambiguous"] is True


def test_only_all_day_events_means_no_match(monkeypatch):
    monkeypatch.setattr(cr, "_gws_calendar_events",
                        lambda *a, **k: [_all_day("a1", "2026-11-06")])
    monkeypatch.setattr(cr, "_load_client_names", lambda: [])
    out = cr.resolve("2026-11-12_16-30-49.json")
    assert out["participant_resolution_log"]["calendar_search"]["match_count"] == 0
    assert out["calendar_event_id"] is None
