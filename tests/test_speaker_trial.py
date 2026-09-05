import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from speaker_trial import RULES, evaluate, metrics, verify_baseline, verify_audio, sha256, collect


def pair():
    return {"stem": "meeting", "baseline_sha256": "b", "candidate_sha256": "c"}


def label(**values):
    return {"meeting": "meeting", "start": 0, "end": 120,
            "audio_evidence": "human checked original audio", "truth_source": "human_confirmed",
            "baseline_sha256": "b", "candidate_sha256": "c",
            "baseline_speaker_wrong": False, "candidate_speaker_wrong": False, **values}


def test_new_confirmed_speaker_regression_requires_revert():
    report = evaluate([pair()], [label(candidate_speaker_wrong=True)], RULES)
    assert report["decision"] == "revert"


def test_improved_small_sample_is_honestly_inconclusive():
    assert evaluate([pair()], [label(baseline_speaker_wrong=True)], RULES)["decision"] == "keep_inconclusive"


def test_model_confidence_and_stale_labels_cannot_be_counted_as_accuracy():
    report = evaluate([pair()], [label(truth_source="model_confidence"), label(candidate_sha256="old")], RULES)
    assert report["reviewed_seconds"] == 0
    assert report["accuracy_claim_permitted"] is False


def test_new_critical_owner_mistake_triggers_revert_even_in_small_sample():
    assert evaluate([pair()], [label(candidate_critical_owner_error=True)], RULES)["decision"] == "revert"


def test_duplicate_annotations_do_not_manufacture_sample_size():
    assert evaluate([pair()], [label()]*20, RULES)["reviewed_seconds"] == 120


def test_model_last_timestamp_is_not_full_coverage():
    report = metrics({"transcript": "[00:00] A: Hi.\n[59:59] B: Bye.", "_meta": {
        "audio_duration_seconds": 3600, "missing_time_ranges": [[900,2700]], "partial": False}})
    assert report["partial"] and report["missing_chunk_seconds"] == 1800


def test_no_speech_hallucination_is_a_hard_failure():
    row = {**pair(), "candidate": {"no_speech": True, "word_count": 2}}
    assert evaluate([row], [], RULES)["decision"] == "revert"


def test_word_loss_and_overlap_loss_are_independent_of_correct_speaker():
    assert evaluate([pair()], [label(reference_words=100, baseline_word_errors=0,
                                     candidate_word_errors=3)], RULES)["decision"] == "revert"
    assert evaluate([pair()], [label(overlapping_host_seconds=100,
               baseline_host_recalled_seconds=100, candidate_host_recalled_seconds=94)], RULES)["decision"] == "revert"


def test_dirty_or_mismatched_baseline_and_changed_config_are_rejected(tmp_path):
    import subprocess
    import pytest
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q"); git("config", "user.email", "test@example.test"); git("config", "user.name", "Test")
    source = tmp_path / "code.py"; source.write_text("original")
    config = tmp_path.parent / (tmp_path.name + ".yaml"); config.write_text("original config")
    git("add", "code.py"); git("commit", "-qm", "fixture")
    manifest = {"baseline_repo": str(tmp_path), "baseline_commit": git("rev-parse", "HEAD"),
                "config_backup": str(config), "config_sha256": sha256(config)}
    assert verify_baseline(manifest) == manifest["baseline_commit"]
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="baseline is dirty"):
        verify_baseline(manifest)
    git("restore", "code.py")
    with pytest.raises(RuntimeError, match="different commit"):
        verify_baseline({**manifest, "baseline_commit": "incorrect"})
    config.write_text("changed config")
    with pytest.raises(RuntimeError, match="configuration changed"):
        verify_baseline(manifest)


def test_replay_rejects_missing_or_changed_audio_digest(tmp_path):
    import pytest
    path = tmp_path / "audio.wav"; path.write_bytes(b"original")
    expected = sha256(path)
    assert verify_audio(path, expected) == expected
    path.write_bytes(b"changed")
    for digest in (None, expected):
        with pytest.raises(RuntimeError, match="audio changed"):
            verify_audio(path, digest)


def test_collection_rejects_baseline_from_different_audio(tmp_path):
    from speaker_integrity import atomic_json, save_trial_stage, trial_input_digest
    stem = "2026-09-06_10-00-00"
    recordings = tmp_path / "audio"; recordings.mkdir()
    transcripts = tmp_path / "transcripts"; transcripts.mkdir()
    audio = recordings / (stem + ".wav"); audio.write_bytes(b"candidate audio")
    data = {"transcript": "[00:00] A: Hello", "_meta": {"audio_duration_seconds": 600}}
    atomic_json(transcripts / (stem + ".json"), data)
    cfg = {"attribution_trial": {"enabled": True, "state_dir": str(tmp_path)}}
    save_trial_stage(cfg, audio, "candidate", data, audio_sha256=trial_input_digest(cfg, audio))
    atomic_json(tmp_path / "manifest.json", {"start": "2026-09-05T00:00:00+08:00", "end": "2026-09-12T09:00:00+08:00",
        "timezone": "Asia/Hong_Kong", "recordings_dir": str(recordings), "transcripts_dir": str(transcripts),
        "baseline_commit": "baseline", "config_sha256": "config"})
    shadow = tmp_path / "shadows" / stem / "baseline" / (stem + ".json")
    atomic_json(shadow, data)
    certificate = {"baseline_commit": "baseline", "config_sha256": "config", "transcript_sha256": sha256(shadow),
                   "audio_sha256": "different audio"}
    atomic_json(shadow.parent / "run.json", certificate)
    assert collect(tmp_path)["paired_baselines"] == 0
    certificate["audio_sha256"] = sha256(audio)
    atomic_json(shadow.parent / "run.json", certificate)
    assert collect(tmp_path)["paired_baselines"] == 1
