"""Model migration off gemini-2.5-* (Google auto-routes them from 2026-12-01),
thinking control and true-cost accounting, fail-fast on errors no retry can
fix, and the watcher's speech gate (speech_trim) in the real flow.

See docs/model-eval-2026-11.md for why the defaults are what they are.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gemini_processor as gp  # noqa: E402
import transcribe_watcher as tw  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ── no path may stay on a deprecated model ─────────────────────────────

def test_defaults_are_not_deprecated():
    assert gp.DEFAULT_MODEL not in gp.DEPRECATED_MODELS
    assert gp.GeminiResult(transcript="", language="x").model == gp.DEFAULT_MODEL
    assert {"gemini-2.5-pro", "gemini-2.5-flash"} <= gp.DEPRECATED_MODELS


def test_no_runtime_code_names_a_deprecated_model():
    """The only mentions allowed are the deny-list itself and history notes."""
    offenders = []
    for path in (REPO / "tools").rglob("*.py"):
        if path.name in ("model_eval.py",):          # eval may run references
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            for m in ("gemini-2.5-pro", "gemini-2.5-flash"):
                if f'"{m}"' in code or f"'{m}'" in code:
                    if "DEPRECATED_MODELS" in code or path.name == "gemini_processor.py" and n < 45:
                        continue
                    offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert offenders == []


def test_processor_refuses_deprecated_model():
    with pytest.raises(ValueError, match="deprecated"):
        gp.GeminiAudioProcessor(api_key="k", model="gemini-2.5-pro")


def test_processor_eval_escape_hatch():
    p = gp.GeminiAudioProcessor(api_key="k", model="gemini-2.5-pro", allow_deprecated=True)
    assert p.model == "gemini-2.5-pro"


def _watcher():
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = SimpleNamespace(lines=[])
    for lvl in ("info", "warning", "error", "debug"):
        setattr(w.logger, lvl, lambda msg, _l=lvl: w.logger.lines.append((_l, msg)))
    return w


def test_watcher_replaces_deprecated_config_model_and_says_so():
    w = _watcher()
    model, thinking = w._configured_model({"model_long": "gemini-2.5-pro"},
                                          "model_long", "thinking_level_long")
    assert (model, thinking) == (gp.DEFAULT_MODEL, gp.DEFAULT_THINKING_LEVEL)
    assert any(l == "error" and "deprecated" in m for l, m in w.logger.lines)


def test_watcher_routes_model_and_thinking_from_config():
    w = _watcher()
    cfg = {"model": "gemini-3.6-flash", "thinking_level": "low",
           "model_long": "gemini-4-hydrogen", "thinking_level_long": "medium"}
    assert w._configured_model(cfg, "model", "thinking_level") == ("gemini-3.6-flash", "low")
    assert w._configured_model(cfg, "model_long", "thinking_level_long") == (
        "gemini-4-hydrogen", "medium")


def test_unset_model_gets_default_model_and_its_thinking_level():
    w = _watcher()
    assert w._configured_model({}, "model", "thinking_level") == (
        gp.DEFAULT_MODEL, gp.DEFAULT_THINKING_LEVEL)
    # A non-default model without an explicit level keeps the model's own default.
    assert w._configured_model({"model": "gemini-4-hydrogen"}, "model",
                               "thinking_level") == ("gemini-4-hydrogen", None)


def test_live_config_file_has_no_deprecated_model():
    cfg = Path.home() / "Documents/MeetingRecorder/config.yaml"
    if not cfg.exists():
        pytest.skip("no local MeetingRecorder config")
    import yaml
    g = (yaml.safe_load(cfg.read_text()) or {}).get("gemini", {})
    assert g.get("model") not in gp.DEPRECATED_MODELS
    assert g.get("model_long") not in gp.DEPRECATED_MODELS


# ── thinking + usage accounting ────────────────────────────────────────

class _Types:
    @staticmethod
    def GenerateContentConfig(**kw):
        return kw

    @staticmethod
    def ThinkingConfig(**kw):
        return kw


def _scripted(payloads, monkeypatch, thinking="low", errors=None):
    p = gp.GeminiAudioProcessor.__new__(gp.GeminiAudioProcessor)
    p.model, p.temperature, p.max_output_tokens = "gemini-3.8-flash", 0.1, 1024
    p.thinking_level = thinking
    p.usage_log = []
    p.types = _Types()
    seen = []
    errors = list(errors or [])

    class _Chunk:
        def __init__(self, text):
            self.text = text
            self.usage_metadata = SimpleNamespace(
                prompt_token_count=1000, candidates_token_count=200, thoughts_token_count=50)

    class _Models:
        def generate_content_stream(self, model, contents, config):
            seen.append(config)
            if errors:
                raise errors.pop(0)
            return iter([_Chunk(payloads.pop(0))])

    p.client = SimpleNamespace(models=_Models())
    monkeypatch.setattr(gp.time, "sleep", lambda *_a, **_k: None)
    return p, seen


def test_thinking_level_is_sent(monkeypatch):
    p, seen = _scripted(['{"transcript": ""}'], monkeypatch, thinking="low")
    p._generate_with_retry("x", audio_content=None)
    assert seen[0]["thinking_config"] == {"thinking_level": "low"}


def test_no_thinking_level_leaves_model_default(monkeypatch):
    p, seen = _scripted(['{"transcript": ""}'], monkeypatch, thinking=None)
    p._generate_with_retry("x", audio_content=None)
    assert "thinking_config" not in seen[0]


def test_usage_log_counts_billed_retries_and_thinking(monkeypatch):
    # The truncated first reply is billed too; the old token count missed it.
    p, _ = _scripted(['{"transcript": "cut', '{"transcript": ""}'], monkeypatch)
    p._generate_with_retry("x", audio_content=object())
    assert [u["attempt"] for u in p.usage_log] == [1, 2]
    assert all(u["thinking_tokens"] == 50 and u["audio"] for u in p.usage_log)
    total = tw._summarize_usage(p.usage_log)
    assert total == {"calls": 2, "input_tokens": 2000, "output_tokens": 400,
                     "thinking_tokens": 100}


def test_parse_response_reads_thinking_tokens():
    p = gp.GeminiAudioProcessor.__new__(gp.GeminiAudioProcessor)
    p.model, p.thinking_level = "gemini-3.8-flash", "low"
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7))
    r = p._parse_response(resp, 1.0, '{"transcript": "[00:00] A: hi"}')
    assert (r.thinking_tokens, r.thinking_level) == (7, "low")
    assert r.parsed_response["_meta"]["thinking_tokens"] == 7


# ── errors no retry can fix ────────────────────────────────────────────

class _ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{code} {msg}")
        self.code = code


@pytest.mark.parametrize("exc", [
    _ApiError(400, "INVALID_ARGUMENT. {'message': 'API key not valid. Please pass a valid API key.'}"),
    _ApiError(403, "PERMISSION_DENIED"),
    _ApiError(429, "RESOURCE_EXHAUSTED. Your project is temporarily rate-limited to prevent "
                   "exceeding its monthly spending cap."),
    _ApiError(404, "models/gemini-9-flash is not found for API version v1beta"),
])
def test_fatal_errors_are_not_retried(monkeypatch, exc):
    p, seen = _scripted(['{"transcript": ""}'], monkeypatch, errors=[exc])
    with pytest.raises(gp.GeminiFatalError):
        p._generate_with_retry("x", audio_content=None, max_attempts=3)
    assert len(seen) == 1


@pytest.mark.parametrize("exc", [
    _ApiError(429, "RESOURCE_EXHAUSTED. Quota exceeded for requests per minute."),
    _ApiError(503, "UNAVAILABLE"),
    RuntimeError("Server disconnected without sending a response."),
    _ApiError(400, "INVALID_ARGUMENT. Request contains an invalid argument."),
])
def test_transient_errors_are_still_retried(monkeypatch, exc):
    p, seen = _scripted(['{"transcript": ""}'], monkeypatch, errors=[exc])
    text, _ = p._generate_with_retry("x", audio_content=None, max_attempts=3)
    assert len(seen) == 2 and text


def test_fatal_error_skips_chunk_retries_and_single_shot_fallback(monkeypatch):
    p = gp.GeminiAudioProcessor.__new__(gp.GeminiAudioProcessor)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise gp.GeminiFatalError("API key not valid")

    p._get_duration = lambda _p: 1500.0
    p._process_single_shot = boom
    p._process_chunked = lambda *a, **k: pytest.fail("must not fall back to chunking")
    audio = Path(__file__)
    with pytest.raises(gp.GeminiFatalError):
        p.process_audio(audio)
    assert calls == [1]
    with pytest.raises(gp.GeminiFatalError):
        p._process_chunk_with_retries(audio, None, 60)
    assert calls == [1, 1]


def test_escalation_ladder_does_not_swallow_fatal_errors(monkeypatch):
    w = _watcher()
    w._fill_partial_from_live = lambda *a: None
    w._notify_telegram_failure = lambda *a: pytest.fail("fatal must propagate, not alert here")
    proc = SimpleNamespace(CHUNK_DURATION_SEC=900)

    def boom(*a, **k):
        raise gp.GeminiFatalError("PERMISSION_DENIED")

    proc._process_single_shot = boom
    proc._process_chunked = boom
    bad = gp.GeminiResult(transcript="", language="x")
    with pytest.raises(gp.GeminiFatalError):
        w._validate_and_escalate(Path("a.wav"), Path("a.mp3"), 1800, bad,
                                 known_attendees=None, channel_segments=None,
                                 diarization_segments=None, processor=proc)


def test_chunk_dirs_are_unique_per_call(tmp_path, monkeypatch):
    p = gp.GeminiAudioProcessor.__new__(gp.GeminiAudioProcessor)
    p._get_duration = lambda _p: 1000.0
    made = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (
        made.append(Path(cmd[-1]).parent), SimpleNamespace(returncode=0, stderr=""))[1])
    p._chunk_audio(tmp_path / "x.mp3", force_chunk=True)
    p._chunk_audio(tmp_path / "x.mp3", force_chunk=True)
    assert len({d for d in made}) == 2


# ── the speech gate inside _process_with_gemini ────────────────────────

SR = 16000


def _wav(path: Path, mono: np.ndarray, channels: int = 3) -> Path:
    data = np.stack([mono] + [np.zeros_like(mono)] * (channels - 1), axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _talk(sec, rng):
    t = np.arange(int(sec * SR)) / SR
    return 0.3 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0) \
        + 0.0003 * rng.standard_normal(len(t))


def _room(sec, rng):
    return 0.0003 * rng.standard_normal(int(sec * SR))


class _StubProcessor:
    CHUNK_DURATION_SEC = 900

    def __init__(self):
        self.model, self.thinking_level, self.usage_log = "gemini-3.8-flash", "low", []
        self.calls = []

    def process_audio(self, path, custom_prompt=None, **kw):
        self.calls.append(Path(path))
        dur = float(subprocess.run(
            ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout)
        self.calls.append(dur)
        raise RuntimeError("stop after the Gemini call")  # end the flow here


def _flow_watcher(tmp_path, monkeypatch, wav: Path):
    w = _watcher()
    w.config = {"diarization": {"enabled": False}}
    w.transcripts_dir = tmp_path
    w.gemini_processor = _StubProcessor()
    w._gemini_model_short = w._gemini_model_long = "gemini-3.8-flash"
    w._gemini_thinking_short = w._gemini_thinking_long = "low"
    w._gemini_long_min = 30
    failures = []
    w._notify_telegram_failure = lambda f, reason: failures.append(reason)
    w._resolve_calendar = lambda p: None
    w._merge_configured_diarization_roster = lambda f, a: a
    w._classify_source_topology_safe = lambda f: None
    w._align_channels_safe = lambda f, t: f
    w._discard_aligned_copy = lambda *a: None
    w._compute_channel_vad_safe = lambda f: None

    def convert(f, vad, ok):
        mp3 = tmp_path / f"{f.stem}.mp3"
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-y", "-i", str(f),
                        "-ac", "1", str(mp3)], check=True)
        return mp3

    w._convert_for_gemini_routed = convert
    monkeypatch.setattr(tw, "save_trial_stage", lambda *a, **k: None)
    monkeypatch.setattr(tw, "trial_input_digest", lambda *a, **k: None)
    monkeypatch.setattr(tw, "digital_silence", lambda f: False)
    return w, failures


def test_flow_skips_near_empty_recording_without_any_gemini_call(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    wav = _wav(tmp_path / "2026-11-30_10-00-00.wav", _room(600, rng))
    w, failures = _flow_watcher(tmp_path, monkeypatch, wav)
    w._process_with_gemini(wav)
    assert w.gemini_processor.calls == []
    data = json.loads((tmp_path / f"{wav.stem}.json").read_text())
    assert data["_meta"]["speech_gate"] == "vad_no_speech"
    assert data["_meta"]["speech_trim"]["skip"] is True
    assert data["_meta"]["speaker_attribution"]["status"] == "no_speech"
    assert failures == []


def test_flow_sends_only_the_speech_to_gemini(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    audio = np.concatenate([_talk(120, rng), _room(400, rng)])   # forgot to stop
    wav = _wav(tmp_path / "2026-11-30_11-00-00.wav", audio)
    w, _ = _flow_watcher(tmp_path, monkeypatch, wav)
    w._process_with_gemini(wav)
    sent_path, sent_dur = w.gemini_processor.calls
    assert sent_path.name.endswith(".speech.mp3")
    assert sent_dur < 140                      # ~120 s talk + 10 s pad, not 520 s
    assert not sent_path.exists()              # derived file cleaned up
    assert (tmp_path / f"{wav.stem}.mp3").exists()  # full MP3 kept as evidence


def test_flow_sends_a_normal_meeting_untouched(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    wav = _wav(tmp_path / "2026-11-30_12-00-00.wav", _talk(300, rng))
    w, _ = _flow_watcher(tmp_path, monkeypatch, wav)
    w._process_with_gemini(wav)
    sent_path, sent_dur = w.gemini_processor.calls
    assert sent_path == tmp_path / f"{wav.stem}.mp3"
    assert sent_dur == pytest.approx(300, abs=1)


def test_flow_honours_speech_trim_disabled(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    wav = _wav(tmp_path / "2026-11-30_13-00-00.wav", _room(400, rng))
    w, _ = _flow_watcher(tmp_path, monkeypatch, wav)
    w.config["speech_trim"] = {"enabled": False}
    w._process_with_gemini(wav)
    assert w.gemini_processor.calls[0] == tmp_path / f"{wav.stem}.mp3"
