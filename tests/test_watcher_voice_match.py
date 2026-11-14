"""Watcher wiring for the voice-matching stage (embedding worker stubbed)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcribe_watcher as tw  # noqa: E402
import voice_match as vm  # noqa: E402
from gemini_processor import GeminiResult  # noqa: E402
from voice_library import BASIS_HUMAN, BASIS_SELF, VoiceLibrary  # noqa: E402
from voice_synth import base_vectors, noisy  # noqa: E402

B = base_vectors(["Person A", "Person B", "host"])
STEM = "2026-11-01_10-00-00"
TEXT_FILLER = "Das ist ein ausreichend langer Satz fuer die Messung hier."


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, level):
        return lambda msg: self.lines.append((level, msg))


class _Vad:
    """Two-track VAD stub (only segments + duration, like the system-track tests)."""
    def __init__(self, segments, duration=240.0):
        self.segments, self.duration_sec = segments, duration

    def separation_report(self):
        return {"admissible": False}


def _meeting(labels=("Matthias", "Speaker 1")):
    lines, segs = [], []
    for i in range(12):
        t = i * 20
        label = labels[i % 2]
        if i % 2:
            segs.append((t, t + 19, "remote"))
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {label}: {TEXT_FILLER}")
    return "\n".join(lines), segs


def _watcher(tmp_path, voice_cfg=None):
    w = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.logger = _Log()
    cfg = {"library_dir": str(tmp_path / "lib")}
    cfg.update(voice_cfg or {})
    w.config = {"voice_matching": cfg}
    return w


def _library(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem="2026-09-01_10-00-00", segments=9, basis=BASIS_HUMAN)
    lib.add_sample("Matthias Heim", B["host"], stem="2026-09-01_10-00-00", segments=9, basis=BASIS_SELF, role="self")
    return lib


def _stub(monkeypatch, remote_voice="Person A"):
    def fake(source, plans, **kw):
        out = []
        for i, p in enumerate(plans):
            base = B["host"] if p.role == "host" else B[remote_voice]
            out.append(noisy(base, 1, seed=50 + i)[0])
        return out, {"ok": True, "segments": len(plans), "device": "stub"}
    monkeypatch.setattr(vm, "embed_segments", fake)


def _result(text, verified=True):
    r = GeminiResult(text, "de")
    r.participants = [{"name": "Matthias"}, {"name": "Speaker 1"}]
    r.system_track_attribution = ({"ran": True, "ok": True, "verified_coverage": 0.9,
                                   "uncertain_regions": []} if verified else None)
    return r


def _run(w, result, vad, tmp_path, dur=240.0):
    w._voice_matching_inplace(result, tmp_path / f"{STEM}.wav", tmp_path / f"{STEM}.aligned.wav", vad, dur)


def test_stage_is_wired_after_system_track_and_before_coherence():
    src = inspect.getsource(tw.TranscribeWatcher._process_with_gemini)
    a = src.index("_system_track_attribution_inplace(")
    b = src.index("_voice_matching_inplace(")
    c = src.index("_coherence_repair_inplace(")
    assert a < b < c


def test_default_config_logs_a_match_and_changes_no_text(tmp_path, monkeypatch):
    _stub(monkeypatch)
    _library(tmp_path)
    text, segs = _meeting()
    result = _result(text)
    # No voice_matching keys at all except the temp library dir: defaults apply.
    w = _watcher(tmp_path)
    _run(w, result, _Vad(segs), tmp_path)
    assert result.transcript == text
    assert [p["name"] for p in result.participants] == ["Matthias", "Speaker 1"]
    log = result.parsed_response["_meta"]["voice_match"]
    assert log["ran"] and log["ok"] and log["auto_apply"] is False
    assert log["labels"]["Speaker 1"]["verdict"] == "clear"
    assert log["labels"]["Speaker 1"]["best"]["name"] == "Person A"
    assert log["renames"] == []
    json.dumps(result.parsed_response)                    # lands in the transcript JSON as plain data


def test_absent_voice_matching_key_means_enabled_report_only(tmp_path, monkeypatch):
    _stub(monkeypatch)
    text, segs = _meeting()
    result = _result(text)
    w = _watcher(tmp_path)
    w.config = {}                                         # no key at all
    monkeypatch.setattr(vm, "default_library_dir", lambda cfg=None: tmp_path / "lib")
    _run(w, result, _Vad(segs), tmp_path)
    assert result.voice_match["ran"] is True and result.transcript == text


def test_auto_apply_renames_only_generic_labels_on_a_clear_conflict_free_match(tmp_path, monkeypatch):
    _stub(monkeypatch)
    _library(tmp_path)
    text, segs = _meeting()
    result = _result(text)
    _run(_watcher(tmp_path, {"auto_apply": True}), result, _Vad(segs), tmp_path)
    assert "] Person A:" in result.transcript and "] Speaker 1:" not in result.transcript
    assert "] Matthias:" in result.transcript                       # the host is never renamed
    assert [p["name"] for p in result.participants] == ["Matthias", "Person A"]
    assert result.voice_match["renames"][0]["to"] == "Person A"
    assert result.voice_match["lines_renamed"] == 6


def test_auto_apply_never_renames_a_real_name(tmp_path, monkeypatch):
    _stub(monkeypatch)
    _library(tmp_path)
    text, segs = _meeting(labels=("Matthias", "Philipp"))
    result = _result(text)
    result.participants = [{"name": "Matthias"}, {"name": "Philipp"}]
    _run(_watcher(tmp_path, {"auto_apply": True}), result, _Vad(segs), tmp_path)
    assert result.transcript == text
    # ...but the disagreement is on record, for a human.
    assert any(c["type"] == "name_disagrees_with_voice" for c in result.voice_match["conflicts"])


def test_auto_apply_does_nothing_when_a_conflict_exists(tmp_path, monkeypatch):
    _stub(monkeypatch)
    _library(tmp_path)
    lines, segs = [], []
    for i in range(12):                                   # two generic labels, one voice
        t = i * 20
        label = ("Matthias", "Speaker 1", "Speaker 2")[i % 3]
        if i % 3:
            segs.append((t, t + 19, "remote"))
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {label}: {TEXT_FILLER}")
    text = "\n".join(lines)
    result = _result(text)
    _run(_watcher(tmp_path, {"auto_apply": True}), result, _Vad(segs), tmp_path)
    assert result.transcript == text
    assert result.voice_match["conflicts"][0]["type"] == "same_person_two_labels"


def test_worker_failure_or_timeout_leaves_text_untouched(tmp_path, monkeypatch):
    _library(tmp_path)
    monkeypatch.setattr(vm, "embed_segments",
                        lambda *a, **k: (None, {"ok": False, "error": "TimeoutError: voice embedding exceeded 600s"}))
    text, segs = _meeting()
    result = _result(text)
    _run(_watcher(tmp_path, {"auto_apply": True}), result, _Vad(segs), tmp_path)
    assert result.transcript == text
    assert result.voice_match["ok"] is False and "TimeoutError" in result.voice_match["reason"]


def test_stage_swallows_unexpected_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "process", lambda *a, **k: 1 / 0)
    text, segs = _meeting()
    result = _result(text)
    _run(_watcher(tmp_path), result, _Vad(segs), tmp_path)
    assert result.transcript == text and result.voice_match is None


def test_single_source_recording_is_skipped_with_a_reason(tmp_path, monkeypatch):
    _stub(monkeypatch)
    text, _ = _meeting()
    result = _result(text)
    _run(_watcher(tmp_path, {"auto_apply": True}), result, None, tmp_path)      # channel_vad is None
    assert result.transcript == text
    assert result.voice_match["ran"] is False
    assert result.voice_match["reason"] == "single_source_or_no_system_track"


def test_disabled_in_config_writes_nothing(tmp_path, monkeypatch):
    _stub(monkeypatch)
    text, segs = _meeting()
    result = _result(text)
    _run(_watcher(tmp_path, {"enabled": False}), result, _Vad(segs), tmp_path)
    assert result.voice_match is None and result.transcript == text
    assert not (tmp_path / "lib").exists()


def test_self_enrolment_only_after_a_verified_host_label(tmp_path, monkeypatch):
    _stub(monkeypatch)
    _library(tmp_path)
    text, segs = _meeting()
    for verified, expected in ((False, 1), (True, 2)):
        result = _result(text, verified=verified)
        _run(_watcher(tmp_path), result, _Vad(segs), tmp_path)
        lib = VoiceLibrary(tmp_path / "lib")
        n = sum(len(p["samples"]) for p in lib.people() if p["role"] == "self")
        assert n == expected, (verified, n)


def test_voice_log_is_copied_by_neon_insert_next_to_system_track(tmp_path, monkeypatch):
    import neon_insert
    src = inspect.getsource(neon_insert)
    assert '"system_track_attribution", "voice_match"' in src


def test_no_vector_reaches_the_persisted_payload(tmp_path, monkeypatch):
    _stub(monkeypatch)
    _library(tmp_path)
    text, segs = _meeting()
    result = _result(text)
    _run(_watcher(tmp_path), result, _Vad(segs), tmp_path)
    payload = json.dumps(result.parsed_response)
    for lib_vec in (B["Person A"], B["host"]):
        for v in lib_vec[:6]:
            assert f"{abs(float(v)):.5f}" not in payload
    assert isinstance(np.asarray([0.0]), np.ndarray)      # numpy imported for the check above
