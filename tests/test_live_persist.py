"""Live lines and prompt cards survive a closed window."""

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

from sidecar.live import LiveEngine, TailSnapshot  # noqa: E402
from sidecar.live_session import RecordingLiveSession  # noqa: E402
from sidecar.live_store import atomic_write_json  # noqa: E402
from sidecar.live_window import LivePanel  # noqa: E402
from sidecar.prompts import PromptResult, prompt_choice_label  # noqa: E402
from sidecar.service import prompts_for_stem  # noqa: E402
from sidecar.transcript import TranscriptLine  # noqa: E402


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


class _Judge:
    backend = "gemini"
    model = "gemini-3.8-flash"

    def judge(self, lines, questions, *, context=6):
        return {name: [0.95] * len(lines) for name in questions}


def _snapshot(_tail: float) -> TailSnapshot:
    return TailSnapshot(np.zeros(16000, dtype=np.int16), 16000, None, 16000, 0.0, 1.0)


def test_each_tick_persists_once_and_a_direct_snapshot_does_not():
    commits: list[str] = []
    engine = LiveEngine(
        snapshot=_snapshot,
        transcriber=_Transcriber(),
        judge=_Judge(),
        clean_prompt=lambda _text: "Clean synthetic prompt.",
        on_commit=lambda _lines, _cards: commits.append(threading.current_thread().name),
        prompt_threshold=0.9,
        interval=30,
    )
    engine.process_snapshot(_snapshot(60))
    assert commits == []
    engine._safe_tick()
    engine._safe_tick()
    assert commits == ["MainThread", "MainThread"]
    assert len(engine.lines) == 1
    assert engine.lines[0].text == "Synthetic prompt for the test."
    assert engine.cards[0].clean_text == "Clean synthetic prompt."


def test_live_json_is_owner_readable_only(tmp_path):
    import os

    path = tmp_path / "stem.live.json"
    previous = os.umask(0)
    try:
        atomic_write_json(
            path,
            {"schema_version": 1, "recording_stem": "stem", "lines": [], "cards": []},
        )
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
            atomic_write_json(
                path,
                {
                    "schema_version": 1,
                    "recording_stem": "stem",
                    "n": number,
                    "lines": [],
                    "cards": [],
                },
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
            if payload.get("schema_version") != 1 or "n" not in payload:
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
    stem = "2026-09-28_synthetic"
    recordings = tmp_path / "recordings"
    transcripts = tmp_path / "transcripts"
    recordings.mkdir()
    transcripts.mkdir()
    (transcripts / f"{stem}.json").write_text(
        json.dumps({"transcript": "[00:01] Ich: synthetic finished line.\n"}),
        encoding="utf-8",
    )
    threads: list[str] = []
    import sidecar.live_session as session_mod

    real_write = session_mod.atomic_write_json

    def recording_write(path, payload):
        threads.append(threading.current_thread().name)
        real_write(path, payload)

    monkeypatch.setattr(session_mod, "atomic_write_json", recording_write)
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
        judge=_Judge(),
        prompt_threshold=0.9,
        recordings_dir=recordings,
        stem=stem,
        clean_prompt=lambda _text: "Clean synthetic prompt.",
    )
    session.panel = panel
    session.engine.interval = 0.05
    session.start()
    path = recordings / f"{stem}.live.json"
    deadline = time.perf_counter() + 2
    payload = None
    while time.perf_counter() < deadline:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("lines") and payload.get("cards"):
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
    assert threads
    assert all(name == "meeting-sidecar-live" for name in threads)
    assert threading.main_thread().name not in threads

    monkeypatch.setattr(
        "sidecar.service.create_judge",
        lambda *args, **kwargs: _Judge(),
    )
    monkeypatch.setattr("sidecar.service.calibrated_prompt_threshold", lambda _model: 0.9)
    prompts = prompts_for_stem(
        stem,
        transcripts_root=transcripts,
        recordings_root=recordings,
        gemini_api_key="test-key",
    )
    live = [prompt for prompt in prompts if prompt.source == "live"]
    assert len(live) == 1
    assert live[0].copy_text == "Clean synthetic prompt."
    assert live[0].text == "Synthetic prompt for the test."
    assert prompt_choice_label(live[0]) == "Live bei 00:00"


def test_prompt_choice_label_keeps_mark_and_suggestion_wording():
    line = TranscriptLine(0, 12, "00:12", "Ich", "synthetic")
    marked = PromptResult((line,), "mark", mark_seconds=12, mapping_uncertain=True)
    suggested = PromptResult((line,), "suggested")
    assert prompt_choice_label(marked) == "Markierung bei 00:12 — Zuordnung unsicher"
    assert prompt_choice_label(suggested) == "Vorschlag bei 00:12"
