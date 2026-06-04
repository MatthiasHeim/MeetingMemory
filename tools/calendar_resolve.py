#!/usr/bin/env python3
"""
calendar_resolve — Resolve meeting participants from Google Calendar.

Lifts the calendar-lookup logic that used to live in /meeting-actions Step
1b into pure Python. Called by the watcher's deterministic ingest phase so
that participant_details, calendar_event_id, and company are seeded BEFORE
the Claude session starts.

The forensic resolution log structure matches what /meeting-actions used to
write into `sources.participant_resolution_log`, so downstream consumers
(audit queries, future re-extraction) don't need to change.

Usage (library):
    from calendar_resolve import resolve
    result = resolve(transcript_path)
    # result = {
    #     "participant_details": [...],
    #     "participant_resolution_log": {...},
    #     "calendar_event_id": "...",
    #     "company": "...",
    # }

Usage (CLI):
    python calendar_resolve.py /path/to/transcript.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the Brain repo's gws account wrapper.
GWS_WRAPPER = os.path.expanduser("~/.claude/scripts/gws-account.sh")

# Path to the auto-generated client index. Used to map email domains and
# attendee names to known clients.
CLIENT_INDEX = os.path.expanduser(
    "~/Desktop/Repos/Brain/Areas/knowledge-mgmt/indexes/_INDEX_CLIENTS.md"
)

# How wide to search for matching calendar events around the recording start.
SEARCH_WINDOW_MIN = 15

SELF_EMAIL = "matthias@lailix.com"
SELF_NAME = "Matthias Heim"


# ── Filename → start-time parsing ─────────────────────────────────────

def _parse_start_time(transcript_path: Path) -> Optional[datetime]:
    """Filenames are `YYYY-MM-DD_HH-MM-SS.{json,html}` in local time."""
    stem = transcript_path.stem
    try:
        date_part, time_part = stem.split("_")
        y, mo, d = (int(x) for x in date_part.split("-"))
        h, mi, s = (int(x) for x in time_part.split("-"))
        local_dt = datetime(y, mo, d, h, mi, s)
        return local_dt.astimezone(timezone.utc)
    except (ValueError, IndexError):
        return None


# ── Client index ──────────────────────────────────────────────────────

def _load_client_names() -> list[str]:
    """Return list of known client folder names from _INDEX_CLIENTS.md."""
    if not os.path.exists(CLIENT_INDEX):
        logger.warning(f"Client index not found at {CLIENT_INDEX}")
        return []
    names: list[str] = []
    try:
        with open(CLIENT_INDEX, "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^\|\s*([A-Za-z][A-Za-z0-9_]+)\s*\|", line)
                if m:
                    names.append(m.group(1))
    except OSError as e:
        logger.warning(f"Failed to read client index: {e}")
    return names


def _company_from_email(email: str, client_names: list[str]) -> Optional[str]:
    """Best-effort: match the email's domain stem against known clients."""
    if not email or "@" not in email:
        return None
    domain = email.split("@", 1)[1].lower()
    # Strip common suffixes
    stem = domain.split(".")[0]
    if stem in {"gmail", "outlook", "hotmail", "yahoo", "icloud", "me", "lailix"}:
        return None
    for client in client_names:
        if client.lower() == stem or client.lower() in domain:
            return client
    return None


def _humanize_email_local(local: str) -> str:
    """Best-effort display name from an email local-part, used only when a
    calendar attendee has no displayName.

    External attendees (e.g. itesys) often arrive with just an email, so the
    raw local-part ("sascha.lioi") would otherwise leak verbatim into
    participant names, transcripts, insights, and the wiki. Convert the common
    firstname.lastname / first_last pattern into "Sascha Lioi". Leaves
    non-name-shaped locals (info, noreply, locals with digits, single tokens)
    unchanged so we never invent a name where the local-part isn't one.
    """
    parts = [p for p in re.split(r"[._\-]+", local) if p]
    if len(parts) >= 2 and all(p.isalpha() for p in parts):
        return " ".join(p.capitalize() for p in parts)
    return local


# ── Calendar query ────────────────────────────────────────────────────

def _gws_calendar_events(time_min: datetime, time_max: datetime) -> list[dict]:
    """Query Matthias's work calendar for events in [time_min, time_max]."""
    params = {
        "calendarId": "primary",
        "timeMin": time_min.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeMax": time_max.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "singleEvents": True,
        "orderBy": "startTime",
    }
    cmd = [GWS_WRAPPER, "work", "calendar", "events", "list",
           "--params", json.dumps(params)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("Calendar lookup timed out")
        return []
    if r.returncode != 0:
        logger.warning(f"gws calendar events list failed: {r.stderr[:200]}")
        return []
    try:
        body = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        logger.warning(f"Calendar response not JSON: {e}")
        return []
    return body.get("items", []) if isinstance(body, dict) else []


def _pick_best_event(events: list[dict], target: datetime) -> Optional[dict]:
    """Choose the event whose start is closest to `target`.

    Prefers events that have a video conference link (Teams/Zoom/Meet),
    then falls back to absolute time delta.
    """
    if not events:
        return None

    def event_start(ev: dict) -> Optional[datetime]:
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
        if not start:
            return None
        try:
            return datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            return None

    def has_conference(ev: dict) -> bool:
        if ev.get("hangoutLink") or ev.get("conferenceData"):
            return True
        loc = (ev.get("location") or "").lower()
        desc = (ev.get("description") or "").lower()
        for needle in ("teams.microsoft.com", "zoom.us", "meet.google.com"):
            if needle in loc or needle in desc:
                return True
        return False

    scored: list[tuple[float, int, dict]] = []
    for ev in events:
        start = event_start(ev)
        if not start:
            continue
        delta = abs((start - target).total_seconds())
        # Lower score wins; conference events get a discount.
        score = delta - (300 if has_conference(ev) else 0)
        scored.append((score, scored.__len__(), ev))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


# ── Resolution ────────────────────────────────────────────────────────

def resolve(transcript_path: str | Path) -> dict:
    """Resolve participants for a transcript via Matthias's calendar.

    Returns a dict with keys participant_details, participant_resolution_log,
    calendar_event_id, company. All keys are always present; values may be
    empty/None if no match was found.
    """
    p = Path(transcript_path)
    started = _parse_start_time(p)
    log: dict = {
        "calendar_search": {
            "window": None,
            "match_count": 0,
            "chosen_event_id": None,
            "chosen_event_title": None,
        },
        "resolutions": [],
    }
    out = {
        "participant_details": [],
        "participant_resolution_log": log,
        "calendar_event_id": None,
        "company": None,
    }

    if not started:
        logger.info(f"Could not parse start time from {p.name}")
        out["participant_details"] = [{
            "name": SELF_NAME, "email": SELF_EMAIL,
            "company": "Lailix", "role": "self",
        }]
        log["resolutions"] = [{
            "name": SELF_NAME, "method": "self",
            "confidence": "high",
            "evidence": "no calendar match (filename unparseable)",
        }]
        return out

    time_min = started - timedelta(minutes=SEARCH_WINDOW_MIN)
    time_max = started + timedelta(minutes=SEARCH_WINDOW_MIN)
    log["calendar_search"]["window"] = (
        f"{2*SEARCH_WINDOW_MIN}min around {started.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )

    events = _gws_calendar_events(time_min, time_max)
    log["calendar_search"]["match_count"] = len(events)

    chosen = _pick_best_event(events, started)
    if chosen is None:
        logger.info(f"No calendar match for {p.name}")
        out["participant_details"] = [{
            "name": SELF_NAME, "email": SELF_EMAIL,
            "company": "Lailix", "role": "self",
        }]
        log["resolutions"] = [{
            "name": SELF_NAME, "method": "self",
            "confidence": "high",
            "evidence": "Matthias's calendar (no event matched)",
        }]
        return out

    log["calendar_search"]["chosen_event_id"] = chosen.get("id")
    log["calendar_search"]["chosen_event_title"] = chosen.get("summary")
    out["calendar_event_id"] = chosen.get("id")

    client_names = _load_client_names()
    pdetails: list[dict] = []
    resolutions: list[dict] = []
    derived_company: Optional[str] = None

    attendees = chosen.get("attendees") or []
    if not attendees:
        # Solo event (no attendees listed); still record self.
        pdetails.append({
            "name": SELF_NAME, "email": SELF_EMAIL,
            "company": "Lailix", "role": "self",
        })
        resolutions.append({
            "name": SELF_NAME, "method": "self",
            "confidence": "high", "evidence": "calendar event with no attendees",
        })
    else:
        for att in attendees:
            email = (att.get("email") or "").lower()
            display = att.get("displayName")
            if not display and email:
                # No displayName (common for external attendees) → derive a
                # human name from the email local-part instead of leaking it.
                display = _humanize_email_local(email.split("@", 1)[0])
            if not display:
                continue
            is_self = email == SELF_EMAIL or att.get("self") is True
            if is_self:
                pdetails.append({
                    "name": SELF_NAME, "email": SELF_EMAIL,
                    "company": "Lailix", "role": "self",
                })
                resolutions.append({
                    "name": SELF_NAME, "method": "self",
                    "confidence": "high",
                    "evidence": "calendar attendee email match (self)",
                })
                continue
            company = _company_from_email(email, client_names)
            if company and not derived_company:
                derived_company = company
            pdetails.append({
                "name": display, "email": email,
                "company": company, "role": "participant",
            })
            resolutions.append({
                "name": display,
                "method": "calendar" if not company else "domain_lookup",
                "confidence": "high" if company else "medium",
                "evidence": (
                    f"calendar attendee + domain match against client index"
                    if company else "calendar attendee email"
                ),
            })

    out["participant_details"] = pdetails
    out["participant_resolution_log"] = {
        "calendar_search": log["calendar_search"],
        "resolutions": resolutions,
    }
    out["company"] = derived_company
    return out


# ── CLI ───────────────────────────────────────────────────────────────

def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve meeting participants from Google Calendar.")
    parser.add_argument("transcript", help="Path to transcript file (filename encodes timestamp).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = resolve(args.transcript)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
