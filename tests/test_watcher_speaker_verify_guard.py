"""Tests for the watcher's speaker_verify namesake guard — Fix 2 (P1.4) of
docs/../meeting-pipeline-investigation-2026-08-10.md.

The guard used to treat ANY non-self calendar attendee whose first name is
"Matthias" as a remote namesake and skip channel-based verification
entirely. That included an exact duplicate of the host's own calendar
entry (2026-08-10: "Matthias Heim, Matthias Heim, Stefan Sieber" —
calendar_resolve now dedupes this before it reaches participant_details,
see test_calendar_resolve_dedup.py). The guard itself is hardened
defensively so it only fires on a genuinely different full name, in case a
duplicate reaches it via some other path (roster merge, counterpart
inference).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402


class _StubLogger:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))

    def error(self, msg):
        self.lines.append(("error", msg))

    def debug(self, msg):
        self.lines.append(("debug", msg))


class _Result:
    def __init__(self):
        self.parsed_response = {"transcript": "[00:00] Matthias: Hi.\n", "participants": []}
        self.transcript = self.parsed_response["transcript"]
        self.participants = []
        self.speaker_verification_log = None


def _watcher():
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _StubLogger()
    return w


def _run(monkeypatch, cal_match):
    monkeypatch.setattr(tw, "SPEAKER_VERIFY_AVAILABLE", True)
    calls = {"n": 0}
    monkeypatch.setattr(tw, "_verify_speakers",
                         lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), {})[1])
    w = _watcher()
    result = _Result()
    w._verify_speakers_inplace(result, object(), cal_match)
    return w, calls["n"]


# ── SIDE A: a genuine namesake still disables verification ─────────────


def test_genuine_namesake_skips_verification(monkeypatch):
    cal_match = {"participant_details": [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Matthias Müller", "role": "participant"},
    ]}
    w, n_calls = _run(monkeypatch, cal_match)
    assert n_calls == 0
    assert any("shares the host's first name" in msg for _, msg in w.logger.lines)


# ── SIDE B: an exact duplicate of the host must NOT be mistaken for one ─


def test_exact_duplicate_of_host_does_not_skip(monkeypatch):
    cal_match = {"participant_details": [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Matthias Heim", "role": "participant"},  # duplicate host entry
        {"name": "Stefan Sieber", "role": "participant"},
    ]}
    w, n_calls = _run(monkeypatch, cal_match)
    assert n_calls == 1
    assert not any("shares the host's first name" in msg for _, msg in w.logger.lines)


def test_self_role_entry_never_triggers_guard(monkeypatch):
    """Baseline: the host's normal self-role attendee entry alone never
    disables verification (unchanged behaviour)."""
    cal_match = {"participant_details": [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Stefan Sieber", "role": "participant"},
    ]}
    w, n_calls = _run(monkeypatch, cal_match)
    assert n_calls == 1
    assert not any("shares the host's first name" in msg for _, msg in w.logger.lines)


def test_no_matthias_attendee_proceeds_normally(monkeypatch):
    cal_match = {"participant_details": [
        {"name": "Matthias Heim", "role": "self"},
        {"name": "Anna Weber", "role": "participant"},
    ]}
    w, n_calls = _run(monkeypatch, cal_match)
    assert n_calls == 1
