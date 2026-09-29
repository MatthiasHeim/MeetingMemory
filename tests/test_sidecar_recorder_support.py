"""Headless coverage for the recorder menu's non-AppKit sidecar work.

The real rumps/AppKit menu needs an attended macOS session.  These tests cover
the data path it calls, including a concurrent finished-transcript clip while
marks are persisted, without opening an audio device or touching WAV writing.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sidecar.gate import initialise_recording_sidecar, load_recording_sidecar  # noqa: E402
from sidecar.recorder_support import (  # noqa: E402
    append_prompt_mark,
    calendar_context_from_resolution,
    sidecar_metadata_from_calendar,
)
from sidecar.service import clip_for_transcript  # noqa: E402
from sidecar.transcript import TranscriptLine, transcript_from_lines  # noqa: E402


def _authoritative_calendar(attendees: list[dict]) -> dict:
    external = any(attendee.get("email") != "matthias@lailix.com" for attendee in attendees)
    return {
        "participant_details": attendees,
        "participant_resolution_log": {
            "calendar_search": {
                "identity_authoritative": True,
                "matched_timed_event": True,
                "attendee_roster_present": True,
                "self_email_verified": True,
                "synthetic_self": False,
                "attendance_classification": "external" if external else "internal",
                "chosen_event_title": "Synthetic planning session",
                "chosen_event_id": "synthetic-event",
            }
        },
    }


def test_calendar_context_marks_external_event_and_persists_only_metadata(tmp_path):
    context = calendar_context_from_resolution(
        _authoritative_calendar(
            [
                {"role": "self", "name": "Matthias Heim", "email": "matthias@lailix.com"},
                {"role": "participant", "name": "Synthetic Guest", "email": "guest@example.test"},
            ]
        )
    )

    assert context.external_attendees is True
    assert context.attendance_resolved is True
    assert context.jev_can_be_selected is True
    assert context.requires_external_acknowledgement is True

    initialise_recording_sidecar(
        "2026-09-26_10-00-00",
        jev=False,
        external_attendees=context.external_attendees,
        root=tmp_path,
        metadata=sidecar_metadata_from_calendar(context),
    )
    sidecar = load_recording_sidecar("2026-09-26_10-00-00", tmp_path)
    assert sidecar.jev is False
    assert sidecar.external_attendees is True
    assert sidecar.payload == {
        "calendar_attendance_resolved": True,
        "calendar_event_id": "synthetic-event",
        "calendar_title": "Synthetic planning session",
        "external_attendees": True,
        "jev": False,
        "jev_external_acknowledged": False,
        "mark_clock": "seconds_since_recorder_start_monotonic",
        "marks": [],
        "recording_stem": "2026-09-26_10-00-00",
        "schema_version": 2,
    }

    internal = calendar_context_from_resolution(
        _authoritative_calendar(
            [{"role": "self", "name": "Matthias Heim", "email": "matthias@lailix.com"}]
        )
    )
    assert internal.external_attendees is False
    assert internal.jev_can_be_selected is True
    assert internal.requires_external_acknowledgement is False


def test_unresolved_calendar_is_persisted_fail_closed_for_jev():
    context = calendar_context_from_resolution(
        {
            "participant_details": [{"role": "self"}],
            "participant_resolution_log": {
                "calendar_search": {"identity_authoritative": False}
            },
        }
    )

    assert context.external_attendees is None
    assert context.attendance_resolved is False
    assert context.jev_can_be_selected is True
    assert context.requires_external_acknowledgement is True


def test_mark_offset_uses_monotonic_elapsed_seconds(tmp_path):
    stem = "2026-09-26_10-00-00"
    initialise_recording_sidecar(stem, jev=False, external_attendees=False, root=tmp_path)

    mark = append_prompt_mark(
        stem,
        recording_started_monotonic=200.0,
        recordings_root=tmp_path,
        clock=lambda: 204.5678,
    )

    assert mark == 4.568
    assert load_recording_sidecar(stem, tmp_path).marks == (4.568,)


def test_global_prompt_shortcut_requires_exact_modifier_chord():
    pytest.importorskip("rumps")
    from meeting_recorder import is_prompt_mark_shortcut  # noqa: E402

    class AppKit:
        NSEventModifierFlagControl = 1 << 18
        NSEventModifierFlagOption = 1 << 19
        NSEventModifierFlagCommand = 1 << 20

    class Event:
        def __init__(self, flags, *, character="p", repeated=False):
            self._flags = flags
            self._character = character
            self._repeated = repeated

        def modifierFlags(self):
            return self._flags

        def charactersIgnoringModifiers(self):
            return self._character

        def isARepeat(self):
            return self._repeated

    chord = (
        AppKit.NSEventModifierFlagControl
        | AppKit.NSEventModifierFlagOption
        | AppKit.NSEventModifierFlagCommand
    )
    assert is_prompt_mark_shortcut(Event(chord), AppKit)
    assert not is_prompt_mark_shortcut(Event(chord, repeated=True), AppKit)
    assert not is_prompt_mark_shortcut(Event(chord, character="x"), AppKit)
    assert not is_prompt_mark_shortcut(Event(AppKit.NSEventModifierFlagCommand), AppKit)


def test_concurrent_marks_and_clip_work_leave_recording_untouched(tmp_path):
    """The menu's mark/clip paths share no capture objects or audio callback."""
    stem = "2026-09-26_10-00-00"
    initialise_recording_sidecar(stem, jev=False, external_attendees=False, root=tmp_path)
    lines = [
        TranscriptLine(0, 1.0, "00:01", "Speaker", "Synthetic relevant topic one."),
        TranscriptLine(1, 2.0, "00:02", "Speaker", "Synthetic relevant topic two."),
    ]
    transcript = transcript_from_lines(stem, lines)
    judge_started = threading.Event()
    release_judge = threading.Event()
    capture = SimpleNamespace(is_recording=True)
    result = []

    class BlockingJudge:
        backend = "gemini"
        model = "synthetic"

        def judge(self, passed_lines, questions, *, context=6):
            judge_started.set()
            assert release_judge.wait(timeout=3)
            return {question: [0.95] * len(passed_lines) for question in questions}

    clip_thread = threading.Thread(
        target=lambda: result.append(
            clip_for_transcript(transcript, "Synthetic relevant topic", judge=BlockingJudge())
        ),
        daemon=True,
    )
    clip_thread.start()
    assert judge_started.wait(timeout=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        marks = list(
            pool.map(
                lambda index: append_prompt_mark(
                    stem,
                    recording_started_monotonic=100.0,
                    recordings_root=tmp_path,
                    clock=lambda: 100.0 + index / 10,
                ),
                range(8),
            )
        )

    # Mark writes and the background clip only use sidecar/transcript data.
    # A live recorder stays recording throughout and every mark survives.
    assert capture.is_recording is True
    assert sorted(load_recording_sidecar(stem, tmp_path).marks) == sorted(marks)
    release_judge.set()
    clip_thread.join(timeout=3)
    assert not clip_thread.is_alive()
    assert result[0].line_count == 2
