"""Partial transcripts gain live lines only for the time they do not cover."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))

import transcribe_watcher as tw  # noqa: E402
from gemini_processor import GeminiResult  # noqa: E402
from sidecar.live_store import fill_transcript_from_live  # noqa: E402


PARTIAL = "[00:04] Host: synthetic start.\n[01:18] Host: synthetic middle.\n"
COMPLETE = "[00:00] Host: synthetic start.\n[02:50] Host: synthetic end.\n"
LIVE_LINES = [
    {"start": 40.0, "end": 45.0, "speaker": "Host", "text": "synthetic covered line"},
    {"start": 80.0, "end": 88.0, "speaker": "Host", "text": "synthetic gap line"},
    {"start": 700.0, "end": 710.0, "speaker": "Host", "text": "synthetic hole line"},
]


def test_complete_transcript_is_not_modified_by_live_lines():
    filled, meta = fill_transcript_from_live(COMPLETE, LIVE_LINES, 175.0)
    assert filled == COMPLETE
    assert meta is None


def test_coverage_gap_inserts_only_the_missing_live_lines_and_is_idempotent():
    filled, meta = fill_transcript_from_live(PARTIAL, LIVE_LINES, 175.0)
    assert "[01:20] Host: [live] synthetic gap line" in filled
    assert "synthetic covered line" not in filled
    assert "synthetic hole line" not in filled
    assert filled.startswith(PARTIAL.rstrip("\n")) or "[00:04]" in filled
    assert "[01:18] Host: synthetic middle." in filled
    assert meta == {"ranges": [{"start": 78.0, "end": 175.0}], "line_count": 1}
    again, second = fill_transcript_from_live(filled, LIVE_LINES, 175.0)
    assert again == filled
    assert second is None


def test_line_that_starts_on_the_last_stamp_but_runs_into_the_gap_is_inserted():
    lines = [
        {"start": 78.2, "end": 78.35, "speaker": "Host", "text": "synthetic restatement"},
        {"start": 78.32, "end": 100.0, "speaker": "Host", "text": "synthetic continuation"},
    ]
    filled, meta = fill_transcript_from_live(PARTIAL, lines, 175.0)
    assert "synthetic restatement" not in filled
    assert "[01:18] Host: [live] synthetic continuation" in filled
    assert meta["line_count"] == 1


def test_explicit_missing_range_is_filled_when_coverage_already_reaches_the_end():
    transcript = "[00:00] Host: synthetic start.\n[30:00] Host: synthetic end.\n"
    filled, meta = fill_transcript_from_live(
        transcript, LIVE_LINES, 1800.0, missing_time_ranges=[(600.0, 1200.0)]
    )
    assert "[11:40] Host: [live] synthetic hole line" in filled
    assert "synthetic gap line" not in filled
    assert meta["line_count"] == 1
    assert meta["ranges"] == [{"start": 600.0, "end": 1200.0}]


class _StubLogger:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


class _StubProcessor:
    CHUNK_DURATION_SEC = 15 * 60

    def __init__(self, single_shot_results=None, chunked_result=None):
        self._single_shot_results = list(single_shot_results or [])
        self._chunked_result = chunked_result
        self.single_shot_calls = 0
        self.chunked_calls = 0

    def _process_single_shot(self, *args, **kwargs):
        result = self._single_shot_results[self.single_shot_calls]
        self.single_shot_calls += 1
        return result

    def _process_chunked(self, *args, **kwargs):
        self.chunked_calls += 1
        return self._chunked_result


def _watcher(processor, recordings: Path):
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = _StubLogger()
    watcher.gemini_processor = processor
    watcher.recordings_dir = recordings
    watcher._telegram_alerts = []
    watcher._telegram_failures = []
    watcher._notify_telegram_partial = (
        lambda audio_file, validation: watcher._telegram_alerts.append((audio_file, validation))
    )
    watcher._notify_telegram_failure = (
        lambda audio_file, reason: watcher._telegram_failures.append((audio_file, reason))
    )
    return watcher


def _write_live(recordings: Path, stem: str) -> None:
    (recordings / f"{stem}.live.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recording_stem": stem,
                "lines": LIVE_LINES,
                "cards": [],
            }
        ),
        encoding="utf-8",
    )


def test_passing_validation_leaves_a_complete_transcript_untouched(tmp_path):
    _write_live(tmp_path, "rec")
    original = COMPLETE
    result = GeminiResult(transcript=original, language="de")
    processor = _StubProcessor()
    watcher = _watcher(processor, tmp_path)
    final, validation, partial = watcher._validate_and_escalate(
        tmp_path / "rec.wav",
        tmp_path / "rec.mp3",
        175.0,
        result,
        known_attendees=None,
        channel_segments=None,
        diarization_segments=None,
    )
    assert partial is False
    assert validation.passed is True
    assert final.transcript == original
    assert final.live_fill is None
    assert "live_fill" not in final.parsed_response["_meta"]
    assert processor.single_shot_calls == 0
    assert watcher._telegram_alerts == []


def test_partial_transcript_is_filled_after_escalation_and_the_alert_says_so(tmp_path, monkeypatch):
    _write_live(tmp_path, "rec")
    original = GeminiResult(transcript=PARTIAL, language="de")
    retry = GeminiResult(transcript="[00:02] Host: synthetic shorter.\n", language="de")
    processor = _StubProcessor(single_shot_results=[retry])
    watcher = _watcher(processor, tmp_path)
    final, validation, partial = watcher._validate_and_escalate(
        tmp_path / "rec.wav",
        tmp_path / "rec.mp3",
        175.0,
        original,
        known_attendees=None,
        channel_segments=None,
        diarization_segments=None,
    )
    assert processor.single_shot_calls == 1
    assert processor.chunked_calls == 0
    assert partial is True
    assert validation.passed is False
    assert final is original
    assert "[01:20] Host: [live] synthetic gap line" in final.transcript
    assert "synthetic covered line" not in final.transcript
    assert final.live_fill["line_count"] == 1
    final.partial = partial
    final.validation_report = validation.to_dict()
    assert final.parsed_response["_meta"]["live_fill"]["ranges"] == [{"start": 78.0, "end": 175.0}]
    assert final.parsed_response["_meta"]["partial"] is True
    assert len(watcher._telegram_alerts) == 1
    message = tw._partial_alert_message(
        Path("rec.wav"), validation, watcher._last_live_fill
    )
    assert "Stored as partial — needs review/reprocessing." in message
    assert "Gaps were filled from the live transcript." in message
    plain = tw._partial_alert_message(Path("rec.wav"), validation, None)
    assert "Gaps were filled from the live transcript." not in plain

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tw.subprocess, "run", fake_run)
    notifier = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    notifier.logger = _StubLogger()
    notifier._last_live_fill = watcher._last_live_fill
    notifier._telegram_notify_script = lambda: "/tmp/telegram_notify.py"
    notifier._notify_telegram_partial(Path("rec.wav"), validation)
    assert "Gaps were filled from the live transcript." in seen["cmd"][-1]
