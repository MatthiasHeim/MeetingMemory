"""Synthetic regressions derived from the PR #17 adversarial review.

These tests deliberately contain no customer transcript text, calendar records,
or provider credentials.  They cover the policy and mapping edges that a
normal happy-path suite is unlikely to exercise.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))

import sidecar.judges as judges  # noqa: E402
from calendar_resolve import _sidecar_attendance_facts  # noqa: E402
from sidecar.calibration import _scores_for_dataset, apply_owner_override  # noqa: E402
from sidecar.calibration_policy import (  # noqa: E402
    OWNER_APPROVED_MODEL,
    OWNER_APPROVED_PROMPT_THRESHOLD,
)
from sidecar.gate import (  # noqa: E402
    JudgeGateError,
    initialise_recording_sidecar,
    judge_backend_for,
    load_recording_sidecar,
)
from sidecar.judges import APPROVED_GEMINI_BASE_URL, GeminiJudge, JevJudge, JudgeError  # noqa: E402
from sidecar.prompts import prompts_from_marks  # noqa: E402
from sidecar.recorder_support import CalendarRecordingContext, calendar_context_from_resolution  # noqa: E402
from sidecar.selection import selected_indices  # noqa: E402
from sidecar.service import clip_for_transcript, create_judge  # noqa: E402
from sidecar.transcript import TranscriptLine, parse_transcript_lines, transcript_from_lines  # noqa: E402


STEM = "2026-09-26_10-00-00"


def _line(index: int, seconds: float, text: str) -> TranscriptLine:
    return TranscriptLine(index, seconds, f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}", "Host", text)


class _FakeGemini:
    backend = "gemini"
    model = "synthetic-gemini"

    def __init__(self, **_kwargs):
        pass

    def judge(self, lines, questions, *, context=6):
        return {question: [0.9] * len(lines) for question in questions}


class _FakeJev:
    backend = "jev"
    model = "synthetic-jev"
    constructed = 0

    def __init__(self, **_kwargs):
        type(self).constructed += 1

    def judge(self, lines, questions, *, context=6):
        return {question: [0.9] * len(lines) for question in questions}


def test_preference_jev_true_authorises_external_and_unknown_meetings(tmp_path):
    """The Preferences switch is the authority; acknowledgement is not required."""
    _FakeJev.constructed = 0
    initialise_recording_sidecar(
        STEM,
        jev=True,
        external_attendees=True,
        jev_external_acknowledged=False,
        root=tmp_path,
    )
    judge = create_judge(
        STEM,
        requested="jev",
        recordings_root=tmp_path,
        cache_root=tmp_path / "cache",
        gemini_factory=_FakeGemini,
        jev_factory=_FakeJev,
    )
    assert judge.backend == "jev"
    assert _FakeJev.constructed == 1

    initialise_recording_sidecar(
        STEM,
        jev=True,
        external_attendees=None,
        jev_external_acknowledged=False,
        root=tmp_path,
    )
    assert judge_backend_for(STEM, "jev", recordings_root=tmp_path) == "jev"

    # A malformed/omitted attendance field is not interchangeable with the
    # explicitly stored unknown (`null`) state, even if an attacker adds ack.
    (tmp_path / f"{STEM}.sidecar.json").write_text(
        json.dumps(
            {
                "recording_stem": STEM,
                "jev": True,
                "external_attendees": "false",
                "jev_external_acknowledged": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(JudgeGateError, match="tri-state"):
        judge_backend_for(STEM, "jev", recordings_root=tmp_path)


@pytest.mark.parametrize(
    "event, expected",
    [
        ({"start": {"dateTime": "2026-09-26T10:00:00Z"}, "attendees": []}, "unknown"),
        ({"start": {"date": "2026-09-26"}, "attendees": [{"email": "matthias@lailix.com"}]}, "unknown"),
        ({"start": {"dateTime": "2026-09-26T10:00:00Z"}, "attendees": [{"email": "matthias@lailix.com"}]}, "internal"),
        (
            {
                "start": {"dateTime": "2026-09-26T10:00:00Z"},
                "attendees": [{"email": "not-matthias@example.test", "displayName": "Matthias Heim"}],
            },
            "unknown",
        ),
    ],
)
def test_calendar_facts_never_accept_display_name_or_synthetic_self(event, expected):
    facts = _sidecar_attendance_facts(event)
    assert facts["attendance_classification"] == expected
    if expected == "internal":
        assert facts["self_email_verified"] is True
        assert facts["synthetic_self"] is False
    else:
        assert facts["synthetic_self"] is True


def test_context_preserves_unknown_and_keeps_external_dialog_switchable():
    context = calendar_context_from_resolution(
        {
            "participant_details": [{"role": "self", "email": "matthias@lailix.com"}],
            "participant_resolution_log": {
                "calendar_search": {
                    "identity_authoritative": True,
                    "matched_timed_event": True,
                    "attendee_roster_present": False,
                    "attendance_classification": "unknown",
                }
            },
        }
    )
    assert context.external_attendees is None
    assert context.attendance_resolved is False
    assert context.jev_can_be_selected is True
    assert context.requires_external_acknowledgement is True


def test_jev_sidecar_authority_rejects_symlinks_environment_and_wrong_transcript_stem(tmp_path, monkeypatch):
    initialise_recording_sidecar(
        STEM,
        jev=True,
        external_attendees=False,
        root=tmp_path,
    )
    assert judge_backend_for(
        STEM,
        "jev",
        recordings_root=tmp_path,
        transcript_path=tmp_path / f"{STEM}.json",
    ) == "jev"
    with pytest.raises(JudgeGateError):
        judge_backend_for(
            STEM,
            "jev",
            recordings_root=tmp_path,
            transcript_path=tmp_path / "other.json",
        )

    original = tmp_path / f"{STEM}.sidecar.json"
    target = tmp_path / "elsewhere.json"
    target.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(JudgeGateError, match="non-symlinked"):
        judge_backend_for(STEM, "jev", recordings_root=tmp_path)

    forged = tmp_path / "forged"
    forged.mkdir()
    initialise_recording_sidecar(STEM, jev=True, external_attendees=False, root=forged)
    monkeypatch.setenv("MEETING_SIDECAR_RECORDINGS_DIR", str(forged))
    with pytest.raises(JudgeGateError, match="MEETING_SIDECAR_RECORDINGS_DIR"):
        judge_backend_for(STEM, "jev")


def test_jev_audit_is_text_free_and_helper_contract_stays_internal(tmp_path, monkeypatch):
    initialise_recording_sidecar(
        STEM,
        jev=True,
        external_attendees=True,
        jev_external_acknowledged=True,
        root=tmp_path,
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"q": {"probability": 0.8}}),
            stderr="",
        )

    monkeypatch.setattr(judges.subprocess, "run", fake_run)
    judge = JevJudge(
        stem=STEM,
        recordings_root=tmp_path,
        helper_path=Path(__file__),
        audit_root=tmp_path / "audit",
    )
    lines = (_line(0, 1, "synthetic alpha"), _line(1, 2, "synthetic beta"))
    assert judge.judge(lines, {"q": "synthetic criterion"})["q"] == [0.8, 0.8]

    assert all(command[command.index("--classification") + 1] == "internal" for command in commands)
    assert all("--external-processing-approved" in command for command in commands)
    entry = json.loads((tmp_path / "audit" / "jev-audit.jsonl").read_text(encoding="utf-8"))
    assert entry == {
        "schema_version": 1,
        "stem": STEM,
        "external_attendees": True,
        "jev_external_acknowledged": True,
        "timestamp": entry["timestamp"],
        "request_count": 2,
    }
    audit_text = (tmp_path / "audit" / "jev-audit.jsonl").read_text(encoding="utf-8")
    assert "synthetic alpha" not in audit_text and "synthetic criterion" not in audit_text


def test_gemini_endpoint_is_pinned_for_clip_prompts_and_calibration(monkeypatch):
    """Even an inherited OpenRouter override cannot redirect judge text."""
    from google import genai

    constructed = []

    class Models:
        def __init__(self, owner):
            self.owner = owner

        def generate_content(self, *, model, contents, config):
            self.owner.requests.append(
                (
                    self.owner._api_client._http_options.base_url,
                    config.http_options.base_url,
                )
            )
            indexes = [int(value) for value in re.findall(r'"current_line":\{"line_index":(\d+)', contents)]
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {"line_index": index, "probabilities": {"relevant": 0.9, "dictating_prompt": 0.95, "prompt_content": 0.95}}
                            for index in indexes
                        ]
                    }
                )
            )

    class Client:
        def __init__(self, *, api_key, vertexai, http_options):
            assert api_key == "synthetic-key" and vertexai is False
            self._api_client = SimpleNamespace(_http_options=http_options)
            self.requests: list[str] = []
            self.models = Models(self)
            constructed.append(self)

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://openrouter.example/api/v1")
    judge = GeminiJudge(api_key="synthetic-key", model=OWNER_APPROVED_MODEL)
    assert judge.effective_base_url == APPROVED_GEMINI_BASE_URL

    lines = (_line(0, 10, "synthetic instruction"), _line(1, 12, "synthetic continuation"))
    transcript = transcript_from_lines(STEM, lines)
    clip_for_transcript(transcript, "synthetic", judge=judge)
    prompts_from_marks(
        lines,
        [10],
        judge=judge,
        channel_lag_seconds=0.0,
        mic_origin_delay_seconds=0.0,
    )
    _scores_for_dataset(
        judge,
        "A",
        {
            "topic": "synthetic",
            "lines": [{"ts": "00:01", "spk": "Host"}],
            "versions": {"en": {"0": "synthetic calibration line"}},
        },
        "en",
    )
    assert constructed and constructed[0].requests
    assert {
        destination
        for request_destinations in constructed[0].requests
        for destination in request_destinations
    } == {APPROVED_GEMINI_BASE_URL}


def test_owner_override_cannot_waive_failed_english_or_prompt_measurements(tmp_path):
    report = apply_owner_override(
        {
            "schema_version": 2,
            "models": [
                {
                    "model": OWNER_APPROVED_MODEL,
                    "prompt_threshold": OWNER_APPROVED_PROMPT_THRESHOLD,
                    "pass_relevance_a": False,
                    "pass_prompt_c": False,
                }
            ],
            "metrics": [],
        }
    )
    assert report["status"] == "failed_unwaived_acceptance_criteria"
    assert report["chosen"] is None
    assert "owner_override" not in report


class _PromptJudge:
    backend = "synthetic"
    model = "synthetic"

    def judge(self, lines, questions, *, context=6):
        question = next(iter(questions))
        if question == "dictating_prompt":
            return {question: [0.95, 0.05, 0.95]}
        return {question: [0.95] * len(lines)}


def test_prompt_mapping_uses_persisted_offsets_or_labels_the_card_uncertain():
    lines = (
        _line(0, 72.54, "synthetic first prompt sentence"),
        _line(1, 75.0, "synthetic acknowledgement"),
        _line(2, 78.0, "synthetic final prompt sentence"),
    )
    exact = prompts_from_marks(
        lines,
        [90.0],
        judge=_PromptJudge(),
        channel_lag_seconds=15.46,
        mic_origin_delay_seconds=2.0,
    )
    assert exact[0].mapping_uncertain is False
    assert exact[0].association_label is None
    # The low-scored acknowledgement bridges detection but must not enter the
    # copied card.
    assert exact[0].text == "synthetic first prompt sentence\nsynthetic final prompt sentence"

    uncertain = prompts_from_marks(
        lines,
        [90.0],
        judge=_PromptJudge(),
        channel_lag_seconds=None,
        mic_origin_delay_seconds=2.0,
    )
    assert uncertain[0].mapping_uncertain is True
    assert uncertain[0].association_label == "Zuordnung unsicher"


def test_watcher_alignment_metadata_is_strictly_additive():
    """The allowed watcher change must leave every non-alignment byte intact."""
    from transcribe_watcher import _transcript_payload_with_channel_alignment

    payload = {
        "transcript": "[00:01] Host: synthetic line.",
        "language": "de",
        "_meta": {"model": "synthetic", "chunked": False},
    }
    before = json.dumps(payload, ensure_ascii=False, indent=2)
    untouched = _transcript_payload_with_channel_alignment(payload, None)
    assert untouched is payload
    assert json.dumps(untouched, ensure_ascii=False, indent=2) == before

    amended = _transcript_payload_with_channel_alignment(payload, 15.46)
    assert amended["transcript"] == payload["transcript"]
    assert amended["language"] == payload["language"]
    assert amended["_meta"] == {
        "model": "synthetic",
        "chunked": False,
        "channel_alignment": {"lag_seconds": 15.46},
    }
    assert "channel_alignment" not in payload["_meta"]


def test_filler_is_not_a_clip_seed_and_malformed_turns_are_debug_logged(caplog):
    lines = (
        _line(0, 1, "Ja."),
        _line(1, 2, "synthetic substantive sentence."),
        _line(2, 3, "unrelated sentence."),
    )
    assert selected_indices([0.99, 0.99, 0.0], lines) == ()

    caplog.set_level("DEBUG")
    parsed = parse_transcript_lines("[00:01] Host: synthetic valid.\nsynthetic malformed")
    assert len(parsed) == 1
    assert "Skipping malformed transcript line" in caplog.text


def _load_recorder_with_fake_platform(monkeypatch):
    """Import the menu module headlessly so capture ordering stays testable."""
    fake_rumps = types.ModuleType("rumps")

    class App:
        pass

    fake_rumps.App = App
    fake_rumps.notification = lambda **_kwargs: None
    fake_rumps.alert = lambda **_kwargs: 1
    fake_rumps.MenuItem = object
    fake_sounddevice = types.ModuleType("sounddevice")
    monkeypatch.setitem(sys.modules, "rumps", fake_rumps)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    sys.modules.pop("meeting_recorder", None)
    return importlib.import_module("meeting_recorder")


def test_recorder_starts_capture_before_calendar_or_sidecar_and_keeps_recording_on_write_failure(tmp_path, monkeypatch):
    recorder_module = _load_recorder_with_fake_platform(monkeypatch)
    events: list[str] = []
    worker_targets = []

    class Recorder:
        is_recording = False

        def start(self, _output):
            events.append("capture-start")
            self.is_recording = True
            return True

    app = recorder_module.MeetingRecorderApp.__new__(recorder_module.MeetingRecorderApp)
    app.recordings_dir = tmp_path
    app.recorder = Recorder()
    app._prompt_mark_item = None
    app._sidecar_write_failed_stems = set()
    app._notify_from_worker = lambda **_kwargs: events.append("sidecar-notice")
    app._start_sidecar_worker = lambda **kwargs: worker_targets.append(kwargs["target"])
    app._calendar_context_for_start = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("calendar lookup ran on the capture path")
    )
    monkeypatch.setattr(
        recorder_module,
        "initialise_recording_sidecar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic sidecar failure")),
    )

    app._start_recording(SimpleNamespace(title="Start Recording"))

    assert events[:2] == ["capture-start", "sidecar-notice"]
    assert app.recorder.is_recording is True
    assert len(worker_targets) == 2
    assert app._sidecar_write_failed_stems


def test_calendar_resolution_worker_updates_metadata_without_a_dialog(tmp_path, monkeypatch):
    recorder_module = _load_recorder_with_fake_platform(monkeypatch)
    app = recorder_module.MeetingRecorderApp.__new__(recorder_module.MeetingRecorderApp)
    app.recordings_dir = tmp_path
    app.recorder = SimpleNamespace(is_recording=True)
    app._recording_stem = STEM
    app._sidecar_write_failed_stems = set()
    app._notify_from_worker = lambda **_kwargs: None
    initialise_recording_sidecar(STEM, jev=True, external_attendees=None, root=tmp_path)
    observed = {}
    dialogs = []
    worker_thread = []

    def resolve(_output, *, timeout_seconds):
        observed["timeout"] = timeout_seconds
        worker_thread.append(threading.get_ident())
        return CalendarRecordingContext(True, True, title="Synthetic", event_id="evt")

    app._calendar_context_for_start = resolve
    app._dispatch_ui = lambda callback: dialogs.append(callback)
    recorder_module.rumps.alert = lambda **_kwargs: dialogs.append("alert")
    app._resolve_calendar_after_capture(STEM, tmp_path / f"{STEM}.wav")
    assert observed["timeout"] == 3.0
    assert dialogs == []
    assert worker_thread == [threading.get_ident()]
    sidecar = load_recording_sidecar(STEM, tmp_path)
    assert sidecar.jev is True
    assert sidecar.external_attendees is True
    assert sidecar.jev_external_acknowledged is False
    assert judge_backend_for(STEM, "jev", recordings_root=tmp_path) == "jev"


def test_recorder_uses_configured_dotenv_key_name_and_persists_first_mic_offset(tmp_path, monkeypatch):
    recorder_module = _load_recorder_with_fake_platform(monkeypatch)
    app = recorder_module.MeetingRecorderApp.__new__(recorder_module.MeetingRecorderApp)
    app.recordings_dir = tmp_path
    app.config = {"gemini": {"api_key_env": "SYNTHETIC_SIDECAR_GEMINI_KEY"}}
    monkeypatch.setenv("SYNTHETIC_SIDECAR_GEMINI_KEY", "synthetic-secret")
    assert app._gemini_api_key() == "synthetic-secret"

    initialise_recording_sidecar(STEM, jev=False, external_attendees=None, root=tmp_path)
    app.recorder = SimpleNamespace(wait_for_mic_first_sample=lambda **_kwargs: 105.4321)
    app._sidecar_write_failed_stems = set()
    app._notify_from_worker = lambda **_kwargs: None
    app._persist_mic_first_sample_offset(STEM, 100.0)
    assert load_recording_sidecar(STEM, tmp_path).mic_first_sample_offset_seconds == pytest.approx(5.4321)
