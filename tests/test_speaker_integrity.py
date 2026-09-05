import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from capture_provenance import archive_capture, merge_filter
from channel_vad import channel_separation_report
from gemini_processor import GeminiResult, _build_channel_map_prefix
from speaker_integrity import coherence_complete, digital_silence, finalize_attribution, save_trial_stage
from speaker_integrity import trial_input_digest
from speaker_verify import verify


class Probe:
    def __init__(self, segments):
        self.segments = segments
        self.duration_sec = max(x[1] for x in segments)
    def separation_report(self):
        return channel_separation_report(self.segments)
    def shares(self, s, e):
        amounts = {"host_only": 0, "remote_only": 0, "both": 0}
        for a, b, label in self.segments:
            amounts[{"host": "host_only", "remote": "remote_only", "both": "both"}[label]] += max(0, min(b,e)-max(a,s))
        return {**amounts, "speech": sum(amounts.values())}


def test_phone_after_startup_sounds_cannot_rewrite_remote_to_host():
    vad = Probe([(0, 2, "remote"), (2, 1720, "host")])
    d = {"transcript": "[00:05] Robert: I moved to the phone.\n[01:00] Matthias: Okay.",
         "participants": [{"name": "Robert"}, {"name": "Matthias"}]}
    before = copy.deepcopy(d)
    assert verify(d, vad)["flips"] == []
    assert d == before
    assert _build_channel_map_prefix(vad.segments) == ""


def test_reference_loss_midway_does_not_disable_a_proven_remote_interval():
    vad = Probe([(0, 90, "remote"), (90, 600, "host")])
    d = {"transcript": "[00:00] Matthias: Remote voice.\n[01:00] Robert: Remote.\n[02:00] Robert: Now on phone.",
         "participants": [{"name": "Robert"}, {"name": "Matthias"}]}
    log = verify(d, vad)
    assert len(log["flips"]) == 1
    assert "[00:00] Robert:" in d["transcript"]
    assert "[02:00] Robert:" in d["transcript"]
    assert log["skipped"]["reference_unknown"] >= 1


def test_failed_probe_cannot_flip():
    vad = Probe([(0, 100, "host")])
    vad.separation_report = lambda: None
    d = {"transcript": "[00:00] Speaker B: Voice."}
    assert verify(d, vad)["flips"] == []


def test_partial_coherence_success_is_not_completion():
    assert not coherence_complete({"ok": True, "failed_windows": [{"lines": "1-30"}]})
    assert not coherence_complete({"ok": True, "refused_runaway": True})
    assert coherence_complete({"ok": True, "ran": True, "failed_windows": []})


def test_relabel_invalidates_speaker_statistics_and_is_not_accuracy():
    result = GeminiResult("[00:00] Robert: Hi.", "de", participants=[
        {"name": "Robert", "speaking_pct": 99, "total_seconds": 500}],
        speaker_pacing={"Matthias": {"wpm_avg": 250}})
    report = finalize_attribution(result, {"transcript": "[00:00] Matthias: Hi."})
    assert "speaking_pct" not in result.participants[0]
    assert result.speaker_pacing == {}
    assert report["speaker_dependent_actions"] == "hold"
    assert result.parsed_response["_meta"]["speaker_attribution"]["accuracy_measured"] is False


def test_stage_archive_keeps_both_revisions_and_queue(tmp_path):
    cfg = {"attribution_trial": {"enabled": True, "state_dir": str(tmp_path)}}
    for label in ("A", "B"):
        save_trial_stage(cfg, Path("meeting.wav"), "candidate", {"transcript": label,
             "_meta": {"speaker_attribution": {"missing_stages": ["semantic_audit"]}}})
    assert len(list((tmp_path / "recordings/meeting").glob("candidate-*.json"))) == 2
    assert json.loads((tmp_path / "recordings/meeting/verification_queue.json").read_text())["state"] == "pending"


def test_exact_silence_blocks_but_quiet_audio_does_not(tmp_path):
    path = tmp_path / "x.wav"
    sf.write(path, np.zeros((8000, 3)), 8000, subtype="PCM_16")
    assert digital_silence(path)
    signal = np.zeros((8000, 3)); signal[200, 0] = 1/32768
    sf.write(path, signal, 8000, subtype="PCM_16")
    assert not digital_silence(path)


def test_candidate_certifies_original_audio_and_rejects_midrun_replacement(tmp_path):
    cfg = {"attribution_trial": {"enabled": True, "state_dir": str(tmp_path)}}
    path = tmp_path / "meeting.wav"; path.write_bytes(b"original audio")
    digest = trial_input_digest(cfg, path)
    save_trial_stage(cfg, path, "candidate", {"transcript": "original"}, audio_sha256=digest)
    revision = json.loads(next((tmp_path / "recordings/meeting").glob("candidate-*.json")).read_text())
    assert revision["audio_input_verified"] and revision["audio_sha256"] == digest
    path.write_bytes(b"replacement audio")
    save_trial_stage(cfg, path, "candidate", {"transcript": "replacement"}, audio_sha256=digest)
    revisions = [json.loads(p.read_text()) for p in (tmp_path / "recordings/meeting").glob("candidate-*.json")]
    assert not next(r for r in revisions if r["payload"]["transcript"] == "replacement")["audio_input_verified"]


def test_merge_keeps_longer_tail_and_physical_channel_order(tmp_path):
    sr = 8000
    sf.write(tmp_path / "mic.wav", np.full(sr, 0.1), sr, subtype="PCM_16")
    sf.write(tmp_path / "sys.wav", np.tile([0.2, 0.3], (2*sr, 1)), sr, subtype="PCM_16")
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "mic.wav"),
                    "-i", str(tmp_path / "sys.wav"), "-filter_complex", merge_filter(2), "-map", "[a]",
                    str(tmp_path / "merged.wav")], check=True)
    data, actual_sr = sf.read(tmp_path / "merged.wav")
    assert actual_sr == sr and data.shape == (2*sr, 3)
    assert np.allclose(data[0], [0.1, 0.2, 0.3], atol=1/32768)
    assert np.allclose(data[-1], [0, 0.2, 0.3], atol=1/32768)
    archive = archive_capture(tmp_path / "archive", "test", [tmp_path / "mic.wav", tmp_path / "sys.wav"], {})
    assert len(json.loads((archive / "capture.json").read_text())["retained_files"]) == 2
