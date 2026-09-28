"""Slice 2: no start dialog, an unchanged audio callback, and an isolated live worker.

Synthetic audio only. The owner's 2026-09-28 recording is exercised by the
replay command, not by this file.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))

from sidecar.live import (  # noqa: E402
    GeminiLiveTranscriber,
    LiveEngine,
    LiveLine,
    TailSnapshot,
    commit_new_lines,
    replay_wav,
)
from sidecar.live_audio import read_wav_pcm_tail, tail_from_chunks  # noqa: E402
from sidecar.live_window import (  # noqa: E402
    NONACTIVATING_PANEL_MASK,
    LivePanel,
    panel_style_mask,
)
from sidecar.prompts import PromptResult  # noqa: E402
from sidecar.questions import DICTATING_PROMPT_QUESTION  # noqa: E402
from sidecar.recorder_support import jev_preference_enabled, write_jev_preference  # noqa: E402
from sidecar.transcript import TranscriptLine  # noqa: E402


def test_prompt_question_counts_indirect_swiss_german_instructions():
    instructions = str(DICTATING_PROMPT_QUESTION["instructions"])
    assert "ich würd em Claude säge, er söll" in instructions
    assert "mir müessted em Agent säge" in instructions
    assert "English dictation inside dialect" in instructions


def test_short_interruptions_stay_inside_one_prompt_run():
    from sidecar.prompts import _prompt_runs

    lines = [
        TranscriptLine(0, 440.0, "07:20", "Ich", "Ich würde jetzt Claude sagen, er soll"),
        TranscriptLine(1, 454.0, "07:34", "Ich", "er soll"),
        TranscriptLine(2, 457.0, "07:37", "Andere", "Papi"),
        TranscriptLine(3, 458.0, "07:38", "Ich", "den Plan Schritt für Schritt implementieren und die Fragen fragen"),
    ]
    runs = _prompt_runs([0.95, 0.2, 0.1, 0.95], lines, threshold=0.90)
    assert runs == [(0, 3)]
    substantial = [
        TranscriptLine(0, 10.0, "00:10", "Ich", "Ich würde Claude sagen, er soll den ersten Entwurf schreiben."),
        TranscriptLine(1, 16.0, "00:16", "Ich", "Wir reden jetzt über das Wetter und den Zugfahrplan nach Hause."),
        TranscriptLine(2, 22.0, "00:22", "Ich", "Danach essen wir zu Mittag und holen die Kinder von der Schule ab."),
        TranscriptLine(3, 30.0, "00:30", "Ich", "Und dann soll der Agent die Tests ergänzen und den Diff erklären."),
    ]
    split = _prompt_runs([0.95, 0.1, 0.1, 0.95], substantial, threshold=0.90)
    assert split == [(0,), (3,)]


def test_copy_text_prefers_the_clean_prompt():
    line = TranscriptLine(0, 1.0, "00:01", "Ich", "verbatim instruction")
    verbatim = PromptResult((line,), "suggested", clean_text="")
    assert verbatim.copy_text == "verbatim instruction"
    cleaned = PromptResult((line,), "suggested", clean_text=" Implement the plan step by step. ")
    assert cleaned.copy_text == "Implement the plan step by step."
    assert cleaned.text == "verbatim instruction"


def test_jev_preference_defaults_off_and_roundtrips(tmp_path):
    config = tmp_path / "config.yaml"
    assert jev_preference_enabled(config) is False
    assert jev_preference_enabled(None) is False
    write_jev_preference(config, False)
    assert jev_preference_enabled(config) is False
    write_jev_preference(config, True)
    assert jev_preference_enabled(config) is True
    with pytest.raises(ValueError):
        write_jev_preference(config, "true")  # type: ignore[arg-type]


def test_commit_keeps_only_lines_newer_than_the_frontier():
    committed = [LiveLine(10.0, 12.0, "Ich", "Der erste Satz steht schon.")]
    incoming = [
        LiveLine(10.2, 12.1, "Ich", "Der erste Satz steht schon."),
        LiveLine(18.0, 20.0, "Andere", "Ein neuer Satz kommt dazu."),
    ]
    all_lines, added, revised = commit_new_lines(committed, incoming)
    assert revised is None
    assert [line.text for line in added] == ["Ein neuer Satz kommt dazu."]
    assert [line.text for line in all_lines] == [
        "Der erste Satz steht schon.",
        "Ein neuer Satz kommt dazu.",
    ]


def test_commit_replaces_a_cut_off_line_with_the_complete_hearing():
    committed = [LiveLine(20.0, 24.0, "Ich", "Ich würde Claude sagen, er soll...")]
    incoming = [
        LiveLine(
            20.1,
            32.0,
            "Ich",
            "Ich würde Claude sagen, er soll den Entwurf umsetzen und nachfragen, was unklar ist.",
        )
    ]
    all_lines, added, revised = commit_new_lines(committed, incoming)
    assert added == []
    assert revised == 0
    assert "nachfragen" in all_lines[0].text
    assert all_lines[0].start == 20.0
    assert all_lines[0].end == 32.0


def test_open_tail_withholds_only_an_unfinished_line_at_the_window_edge():
    from sidecar.live import _without_open_tail

    running = LiveLine(18.0, 19.6, "Ich", "und dann soll er")
    done = LiveLine(8.0, 12.0, "Ich", "Der Satz davor ist fertig.")
    finished_at_edge = LiveLine(16.0, 19.8, "Ich", "Das ist ein ganzer Satz.")
    assert _without_open_tail([done, running], 20.0, closed=False) == [done]
    assert _without_open_tail([done, running], 20.0, closed=True) == [done, running]
    assert _without_open_tail([done, finished_at_edge], 20.0, closed=False) == [done, finished_at_edge]


def test_mic_tail_copies_references_without_a_callback_lock():
    rate = 16000
    chunks = [np.full((rate, 1), index, dtype=np.int16) for index in range(3)]
    tail, start, end = tail_from_chunks(chunks, rate, tail_seconds=1.5)
    assert start == pytest.approx(1.5)
    assert end == pytest.approx(3.0)
    assert tail.shape == (int(1.5 * rate),)
    assert int(tail[0]) == 1
    assert int(tail[-1]) == 2


def test_system_tail_reads_a_growing_wav_with_an_unfinalised_data_size(tmp_path):
    path = tmp_path / "growing.wav"
    rate = 8000
    frames = np.arange(rate * 2, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    raw = bytearray(path.read_bytes())
    data_at = raw.find(b"data")
    raw[data_at + 4 : data_at + 8] = (0xFFFFFFFF).to_bytes(4, "little")
    path.write_bytes(raw)

    tail, read_rate = read_wav_pcm_tail(path, tail_seconds=0.5)
    assert read_rate == rate
    assert tail.shape[0] == rate // 2
    assert int(tail[-1, 0]) == int(frames[-1])


def test_audio_callback_source_is_unchanged_and_a_slow_worker_cannot_delay_it():
    import meeting_recorder

    source = inspect.getsource(meeting_recorder.AudioRecorder._audio_callback)
    assert "self.audio_data.append(indata.copy())" in source
    assert "Lock" not in source
    assert "live" not in source.lower()
    assert "gemini" not in source.lower()
    assert "open(" not in source
    start_source = inspect.getsource(meeting_recorder.MeetingRecorderApp._start_recording)
    assert "NSAlert" not in start_source
    assert "runModal" not in start_source
    assert "_ask_jev_choice" not in start_source
    assert not hasattr(meeting_recorder.MeetingRecorderApp, "_ask_jev_choice")

    recorder = meeting_recorder.AudioRecorder.__new__(meeting_recorder.AudioRecorder)
    recorder.audio_data = []
    recorder.recording = True
    recorder.sample_rate = 48000
    recorder._mic_frames = 0
    recorder._previous_adc_end = None
    recorder._next_timing_sample = 10**9
    recorder._mic_first_sample_monotonic = None
    recorder._mic_first_sample_event = threading.Event()
    recorder._capture_meta = {"discontinuities": [], "mic_timing_samples": []}

    entered = threading.Event()
    release = threading.Event()

    class BlockingTranscriber:
        def transcribe_tails(self, *_args, **_kwargs):
            entered.set()
            release.wait(timeout=2)
            raise RuntimeError("synthetic live crash")

    mic = np.zeros(1600, dtype=np.int16)
    engine = LiveEngine(
        snapshot=lambda _tail: TailSnapshot(mic, 16000, None, 16000, 0.0, 0.1),
        transcriber=BlockingTranscriber(),
        interval=0.01,
    )
    engine.start()
    assert entered.wait(timeout=1)
    info = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.0)
    block = np.zeros((160, 1), dtype=np.int16)
    started = time.perf_counter()
    for _ in range(40):
        recorder._audio_callback(block, 160, info, None)
    assert time.perf_counter() - started < 0.25
    assert len(recorder.audio_data) == 40
    assert recorder.recording is True
    stopped = time.perf_counter()
    engine.stop()
    assert time.perf_counter() - stopped < 0.05
    release.set()
    deadline = time.perf_counter() + 2
    while "unterbrochen" not in engine.status and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert "unterbrochen" in engine.status
    assert recorder.recording is True


def test_replay_is_headless_and_reports_simulated_lag(tmp_path, capsys):
    path = tmp_path / "synthetic.wav"
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * rate * 2)

    class FakeTranscriber:
        def transcribe_tails(self, mic, mic_rate, _system, _system_rate):
            duration = len(mic) / float(mic_rate)
            return [
                {
                    "speaker": "Ich",
                    "text": f"synthetic {duration:.1f}",
                    "start_offset": max(0.0, duration - 0.4),
                    "end_offset": max(0.1, duration - 0.1),
                }
            ]

    result = replay_wav(
        path,
        transcriber=FakeTranscriber(),
        tick=1.0,
        tail=2.0,
        prompt_threshold=0.9,
    )
    captured = capsys.readouterr().out
    assert "LINE " in captured
    assert "median_lag_seconds=" in captured
    assert result["median_lag_seconds"] is not None
    assert result["median_lag_seconds"] <= 30
    assert result["cards"] == ()


def test_panel_show_does_not_run_modal_or_become_key():
    source = inspect.getsource(LivePanel.show)
    assert "orderFrontRegardless" in source
    assert "runModal" not in source
    assert "makeKeyAndOrderFront" not in source
    import AppKit

    mask = panel_style_mask(AppKit)
    assert mask & NONACTIVATING_PANEL_MASK
    panel = LivePanel(on_copy_clip=lambda _topic: None, on_copy_card=lambda _text: None)
    try:
        panel.show()
        assert panel.panel is not None
        assert int(panel.panel.styleMask()) & NONACTIVATING_PANEL_MASK
        assert panel.panel.isKeyWindow() is False
        assert panel.panel.level() >= 3
    finally:
        if panel.panel is not None:
            panel.panel.orderOut_(None)
            panel.panel.close()


def test_live_transcriber_sends_audio_to_the_pinned_endpoint():
    seen = {}

    class Models:
        def generate_content(self, *, model, contents, config):
            seen["model"] = model
            seen["contents"] = contents
            seen["base"] = config["http_options"]["base_url"]
            return SimpleNamespace(
                text='{"lines":[{"speaker":"Ich","text":"Hallo","start_offset":1,"end_offset":2}]}'
            )

    client = SimpleNamespace(
        models=Models(),
        _api_client=SimpleNamespace(
            _http_options=SimpleNamespace(base_url="https://generativelanguage.googleapis.com/")
        ),
    )
    transcriber = GeminiLiveTranscriber(client=client, types_module=None, model="gemini-3.8-flash")
    lines = transcriber.transcribe_tails(np.zeros(16000, dtype=np.int16), 16000, None, 16000)
    assert lines[0]["text"] == "Hallo"
    assert seen["model"] == "gemini-3.8-flash"
    assert seen["base"] == "https://generativelanguage.googleapis.com/"
    assert any(isinstance(part, dict) and part.get("mime_type") == "audio/wav" for part in seen["contents"])


def test_start_records_preference_and_opens_no_dialog(tmp_path, monkeypatch):
    import importlib
    import types

    fake_rumps = types.ModuleType("rumps")
    dialogs: list[object] = []

    class App:
        pass

    class Window:
        def __init__(self, **kwargs):
            dialogs.append(("window", kwargs))

        def run(self):
            dialogs.append("run")
            return SimpleNamespace(clicked=False, text="")

    fake_rumps.App = App
    fake_rumps.notification = lambda **_kwargs: None
    fake_rumps.alert = lambda **kwargs: dialogs.append(("alert", kwargs))
    fake_rumps.Window = Window
    fake_rumps.MenuItem = object
    fake_sounddevice = types.ModuleType("sounddevice")
    monkeypatch.setitem(sys.modules, "rumps", fake_rumps)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    sys.modules.pop("meeting_recorder", None)
    recorder_module = importlib.import_module("meeting_recorder")

    write_jev_preference(tmp_path / "config.yaml", True)
    app = recorder_module.MeetingRecorderApp.__new__(recorder_module.MeetingRecorderApp)
    app.recordings_dir = tmp_path
    app.config_path = tmp_path / "config.yaml"
    app._prompt_mark_item = None
    app._sidecar_write_failed_stems = set()
    app._notify_from_worker = lambda **_kwargs: None
    app._sidecar_threads = set()

    class Recorder:
        is_recording = False
        audio_data = []
        sample_rate = 16000

        def start(self, _output):
            self.is_recording = True
            return True

        def stop(self):
            self.is_recording = False
            return None

    app.recorder = Recorder()
    calendar_threads: list[int] = []

    def resolve(*_args, **_kwargs):
        calendar_threads.append(threading.get_ident())
        from sidecar.recorder_support import CalendarRecordingContext

        return CalendarRecordingContext(None, False)

    app._calendar_context_for_start = resolve
    live_calls = []

    class Session:
        def start(self):
            live_calls.append("start")

        def stop(self):
            live_calls.append("stop")

    app._live_session_factory = lambda _app: Session()
    started = time.perf_counter()
    app._start_recording(SimpleNamespace(title="Start Recording"))
    assert time.perf_counter() - started < 0.5
    assert threading.get_ident() not in calendar_threads
    assert dialogs == []
    assert live_calls == ["start"]
    assert app.recorder.is_recording is True
    from sidecar.gate import load_recording_sidecar

    sidecar_files = list(tmp_path.glob("*.sidecar.json"))
    assert len(sidecar_files) == 1
    stem = sidecar_files[0].name.removesuffix(".sidecar.json")
    sidecar = load_recording_sidecar(stem, tmp_path)
    assert sidecar.jev is True
    assert sidecar.external_attendees is None
    assert sidecar.jev_external_acknowledged is False
    stopped = time.perf_counter()
    app._stop_recording(SimpleNamespace(title="Stop Recording"))
    assert time.perf_counter() - stopped < 0.5
    assert live_calls == ["start", "stop"]
    assert app.recorder.is_recording is False
