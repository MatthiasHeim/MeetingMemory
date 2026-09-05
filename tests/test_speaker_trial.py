import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from speaker_trial import RULES, evaluate, metrics


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
