"""Pure helpers used by the recorder's finished-transcript sidecar menu.

The menu-bar app imports this module, but it deliberately has no AppKit,
audio, or recorder dependency.  Keeping calendar classification, mark offsets,
and recent-meeting labels here makes the safety-sensitive pieces testable
without opening an audio stream or a macOS window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .gate import append_mark, load_recording_sidecar
from .transcript import Transcript


@dataclass(frozen=True)
class CalendarRecordingContext:
    """Conservative calendar facts available before a recording starts."""

    # True = confirmed external roster, False = confirmed internal-only roster,
    # None = no authoritative timed event/roster.  Unknown intentionally stays
    # distinct from external in the sidecar, while receiving the same warning.
    external_attendees: bool | None
    attendance_resolved: bool
    title: str | None = None
    event_id: str | None = None

    @property
    def jev_can_be_selected(self) -> bool:
        """The owner may explicitly choose Jev for every meeting type."""
        return True

    @property
    def requires_external_acknowledgement(self) -> bool:
        """External and unresolved meetings need the DPA warning/acknowledgement."""
        return self.external_attendees is not False


def calendar_context_from_resolution(result: Mapping[str, Any] | Any) -> CalendarRecordingContext:
    """Turn ``calendar_resolve.resolve`` output into a fail-closed UI context.

    A current event is considered internal only after the resolver reports one
    authoritative *timed* event, an explicit roster, and a verified host email.
    Empty rosters, synthetic host rows, ambiguous matches and all-day events
    are ``None``/unknown.  The owner can still select Jev after a clear DPA
    warning, but the offline gate then requires a separately persisted
    acknowledgement.
    """
    if not isinstance(result, Mapping):
        return CalendarRecordingContext(external_attendees=None, attendance_resolved=False)

    search = result.get("participant_resolution_log")
    if isinstance(search, Mapping):
        search = search.get("calendar_search")
    if not isinstance(search, Mapping):
        return CalendarRecordingContext(external_attendees=None, attendance_resolved=False)

    raw_title = search.get("chosen_event_title")
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
    raw_event_id = search.get("chosen_event_id")
    event_id = raw_event_id.strip() if isinstance(raw_event_id, str) and raw_event_id.strip() else None

    if search.get("identity_authoritative") is not True:
        return CalendarRecordingContext(
            external_attendees=None,
            attendance_resolved=False,
            title=title,
            event_id=event_id,
        )

    if search.get("matched_timed_event") is not True:
        return CalendarRecordingContext(
            external_attendees=None,
            attendance_resolved=False,
            title=title,
            event_id=event_id,
        )
    classification = search.get("attendance_classification")
    if classification == "external":
        return CalendarRecordingContext(
            external_attendees=True,
            attendance_resolved=True,
            title=title,
            event_id=event_id,
        )
    if classification != "internal" or search.get("attendee_roster_present") is not True:
        return CalendarRecordingContext(
            external_attendees=None,
            attendance_resolved=False,
            title=title,
            event_id=event_id,
        )

    # Defence in depth for callers that construct a resolver-shaped object:
    # never accept a role/display name as proof that a row is the owner.
    attendees = result.get("participant_details")
    if (
        search.get("self_email_verified") is not True
        or search.get("synthetic_self") is not False
        or not isinstance(attendees, list)
        or not attendees
        or any(
            not isinstance(attendee, Mapping)
            or str(attendee.get("role", "")).strip().casefold() != "self"
            or str(attendee.get("email", "")).strip().casefold() != "matthias@lailix.com"
            for attendee in attendees
        )
    ):
        return CalendarRecordingContext(
            external_attendees=None,
            attendance_resolved=False,
            title=title,
            event_id=event_id,
        )
    return CalendarRecordingContext(
        external_attendees=False,
        attendance_resolved=True,
        title=title,
        event_id=event_id,
    )


def sidecar_metadata_from_calendar(context: CalendarRecordingContext) -> dict[str, Any]:
    """Return small non-transcript metadata useful to the later picker."""
    metadata: dict[str, Any] = {
        "calendar_attendance_resolved": context.attendance_resolved,
    }
    if context.title:
        metadata["calendar_title"] = context.title
    if context.event_id:
        metadata["calendar_event_id"] = context.event_id
    return metadata


def append_prompt_mark(
    stem: str,
    *,
    recording_started_monotonic: float,
    recordings_root: str | Path | None = None,
    clock: Callable[[], float],
) -> float:
    """Append an elapsed monotonic mark without touching audio capture state."""
    offset = max(0.0, float(clock()) - float(recording_started_monotonic))
    return append_mark(stem, offset, root=recordings_root)


def recent_transcript_title(
    transcript: Transcript,
    *,
    recordings_root: str | Path | None = None,
) -> str:
    """Prefer transcript metadata, then recorder-side calendar metadata."""
    title = transcript.title.strip() if isinstance(transcript.title, str) else ""
    if title and title != transcript.stem:
        return title
    sidecar = load_recording_sidecar(transcript.stem, recordings_root)
    payload = sidecar.payload or {}
    for key in ("calendar_title", "meeting_title", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return title or transcript.stem
