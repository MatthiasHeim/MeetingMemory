"""Live lines and prompt cards survive a closed window.

``<stem>.live.json`` has one writer (``recording_files.write_live_transcript``,
called by the live session) and one format. These tests cover what the
persist-fallback branch needs from it: cards beside the lines, an owner-only
atomic file, Prompts… after the meeting, and gap fill from a file written
before cards were added.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))

from sidecar.live import LiveCard, LiveLine  # noqa: E402
from sidecar.live_session import RecordingLiveSession  # noqa: E402
from sidecar.live_store import apply_live_fill, saved_live_prompt_results  # noqa: E402
from sidecar.live_window import LivePanel  # noqa: E402
from sidecar.prompts import PromptResult, prompt_choice_label  # noqa: E402
from sidecar.recording_files import write_live_transcript  # noqa: E402
from sidecar.service import prompts_for_stem  # noqa: E402
from sidecar.transcript import TranscriptLine  # noqa: E402

STEM = "2026-09-28_synthetic"


class _Transcriber:
    def transcribe_tails(self, mic, mic_rate, _system, _system_rate):
        return [
            {
                "speaker": "Ich",
                "text": "Synthetic prompt for the test.",
                "start_offset": 0.2,
                "end_offset": 0.8,
            }
        ]


class _CardJudge:
    """Contextual-card judge: every line is a new request."""

    backend = "synthetic"
    model = "synthetic"

    def choose(self, _state, _questions):
        return {"kind": "request", "relation": "new"}


class _Assistant:
    def __init__(self, *_args, **_kwargs):
        pass

    def clean(self, _card):
        return "Clean synthetic prompt."

    def suggest(self, *_args, **_kwargs):
        return []


def _card(text: str, seconds: float, *, kind: str = "request", clean: str = "") -> LiveCard:
    line = TranscriptLine(0, seconds, "00:00", "Ich", text)
    return LiveCard(f"key-{text}", (line,), clean, kind)


def test_session_writes_lines_and_cards_only_when_they_change(tmp_path, monkeypatch):
    import sidecar.live_session as session_mod

    writes: list[int] = []
    real_write = session_mod.write_live_transcript

    def counting_write(*args, **kwargs):
        writes.append(len(kwargs.get("cards") or ()))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(session_mod, "write_live_transcript", counting_write)
    session = RecordingLiveSession(SimpleNamespace(), api_key=None, copy_text=lambda _: None,
                                   open_window=False, recordings_root=tmp_path, stem=STEM)
    lines = (LiveLine(0, 3, "Ich", "Hallo."),)
    session._on_update(lines, (), "")
    session._on_update(lines, (), "status only")
    card = _card("Hallo.", 0.0, clean="Bitte Hallo sagen.")
    session._on_update(lines, (card,), "")
    assert writes == [0, 1]
    data = json.loads((tmp_path / f"{STEM}.live.json").read_text())
    assert data["cards"] == [{
        "key": "key-Hallo.", "kind": "request", "audience": "all",
        "clean_text": "Bitte Hallo sagen.",
        "verbatim_lines": [{"start": 0.0, "speaker": "Ich", "text": "Hallo."}],
        "start": 0.0, "end": 0.0,
    }]
    # Stop rewrites the file without losing the cards last shown.
    session.engine.lines = list(lines)
    session.stop()
    data = json.loads((tmp_path / f"{STEM}.live.json").read_text())
    assert data["recording_active"] is False
    assert [c["clean_text"] for c in data["cards"]] == ["Bitte Hallo sagen."]


def test_live_json_is_owner_readable_only(tmp_path):
    import os

    previous = os.umask(0)
    try:
        path = write_live_transcript("stem", [], root=tmp_path)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600


def test_atomic_replace_is_always_a_complete_document(tmp_path):
    path = tmp_path / "stem.live.json"
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        number = 0
        while not stop.is_set():
            write_live_transcript(
                "stem",
                [SimpleNamespace(start=0, end=1, speaker="Ich", text=str(number))],
                root=tmp_path,
            )
            number += 1

    def reader() -> None:
        while not stop.is_set():
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if payload.get("schema_version") != 1 or len(payload.get("lines") or []) != 1:
                errors.append("partial")

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
    assert errors == []
    assert list(path.parent.glob(".stem.live.json.*.tmp")) == []


def test_closed_window_keeps_live_cards_for_prompts(tmp_path, monkeypatch):
    recordings = tmp_path / "recordings"
    transcripts = tmp_path / "transcripts"
    recordings.mkdir()
    transcripts.mkdir()
    (transcripts / f"{STEM}.json").write_text(
        json.dumps({"transcript": "[00:01] Ich: synthetic finished line.\n"}),
        encoding="utf-8",
    )
    import sidecar.live_session as session_mod

    threads: list[str] = []
    real_write = session_mod.write_live_transcript

    def recording_write(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(session_mod, "write_live_transcript", recording_write)
    monkeypatch.setattr(session_mod, "CardAssistant", _Assistant)
    panel = LivePanel(on_copy_clip=lambda _topic: None, on_copy_card=lambda _text: None)
    panel.close()
    recorder = SimpleNamespace(
        audio_data=[np.ones((16000, 1), dtype=np.int16)],
        sample_rate=16000,
        _sys_wav=None,
        _mic_frames=16000,
    )
    session = RecordingLiveSession(
        recorder,
        api_key="test-key",
        copy_text=lambda _text: None,
        open_window=False,
        transcriber=_Transcriber(),
        judge=_CardJudge(),
        prompt_threshold=0.9,
        recordings_root=recordings,
        stem=STEM,
    )
    session.panel = panel
    session.engine.interval = 0.05
    session.start()
    path = recordings / f"{STEM}.live.json"
    deadline = time.perf_counter() + 3
    payload = None
    while time.perf_counter() < deadline:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("lines") and any(c.get("clean_text") for c in payload.get("cards") or []):
                break
        time.sleep(0.02)
    session.stop()
    assert session.engine._thread is not None
    session.engine._thread.join(timeout=2)
    assert payload is not None
    assert payload["lines"][0]["speaker"] == "Ich"
    assert payload["lines"][0]["text"] == "Synthetic prompt for the test."
    assert payload["lines"][0]["end"] >= payload["lines"][0]["start"]
    card = payload["cards"][0]
    assert card["clean_text"] == "Clean synthetic prompt."
    assert card["verbatim_lines"][0]["text"] == "Synthetic prompt for the test."
    assert card["end"] >= card["start"]
    assert panel._lines == ()
    assert panel._closed is True
    # The worker kept writing while the window was closed. Stop adds one write
    # on the caller's thread (main's existing final save).
    assert "meeting-sidecar-live" in threads
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["recording_active"] is False
    assert final["cards"]

    monkeypatch.setattr(
        "sidecar.service.create_judge",
        lambda *args, **kwargs: SimpleNamespace(
            backend="gemini", model="gemini-3.8-flash",
            judge=lambda lines, questions, **_k: {name: [0.0] * len(lines) for name in questions},
        ),
    )
    monkeypatch.setattr("sidecar.service.calibrated_prompt_threshold", lambda _model: 0.9)
    prompts = prompts_for_stem(
        STEM,
        transcripts_root=transcripts,
        recordings_root=recordings,
        gemini_api_key="test-key",
    )
    live = [prompt for prompt in prompts if prompt.source == "live"]
    assert len(live) == 1
    assert live[0].copy_text == "Clean synthetic prompt."
    assert live[0].text == "Synthetic prompt for the test."
    assert prompt_choice_label(live[0]) == "Live bei 00:00"


def test_prompts_offer_spoken_cards_but_not_ai_suggestions(tmp_path):
    write_live_transcript(
        STEM,
        [LiveLine(0, 3, "Ich", "Bitte den Bericht senden.")],
        root=tmp_path,
        cards=[
            _card("Bitte den Bericht senden.", 0.0, clean="Bericht senden."),
            _card("Bitte den Bericht senden.", 0.0, kind="suggestion_question", clean="Wann?"),
        ],
    )
    results = saved_live_prompt_results(STEM, tmp_path)
    assert [r.copy_text for r in results] == ["Bericht senden."]


def test_file_written_before_cards_existed_still_fills_gaps(tmp_path):
    """Recordings from main's writer before the merge have no ``cards`` key."""
    path = tmp_path / f"{STEM}.live.json"
    path.write_text(json.dumps({
        "schema_version": 1, "recording_stem": STEM, "model": "gemini-3.8-flash",
        "recording_active": False, "saved_at": "2026-10-20T10:00:00+00:00",
        "lines": [
            {"start": 5.0, "end": 5.2, "speaker": "Ich", "text": "covered", "provisional": False},
            {"start": 80.0, "end": 84.0, "speaker": "Andere", "text": "synthetic gap line",
             "provisional": True},
        ],
    }), encoding="utf-8")
    assert saved_live_prompt_results(STEM, tmp_path) == []
    result = SimpleNamespace(transcript="[00:05] Ich: covered\n", missing_time_ranges=[])
    meta = apply_live_fill(result, path, 175.0)
    assert meta is not None and meta["line_count"] == 1
    assert "[01:20] Andere: [live] synthetic gap line" in result.transcript
    assert "Ich: [live] covered" not in result.transcript


def test_prompt_choice_label_keeps_mark_and_suggestion_wording():
    line = TranscriptLine(0, 12, "00:12", "Ich", "synthetic")
    marked = PromptResult((line,), "mark", mark_seconds=12, mapping_uncertain=True)
    suggested = PromptResult((line,), "suggested")
    assert prompt_choice_label(marked) == "Markierung bei 00:12 — Zuordnung unsicher"
    assert prompt_choice_label(suggested) == "Vorschlag bei 00:12"
