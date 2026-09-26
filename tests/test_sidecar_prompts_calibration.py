from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sidecar.calibration import (  # noqa: E402
    _scores_for_dataset,
    apply_owner_override,
    choose_prompt_threshold,
    precision_recall,
)
from sidecar.calibration_policy import (  # noqa: E402
    OWNER_APPROVED_MODEL,
    OWNER_APPROVED_PROMPT_THRESHOLD,
    OWNER_OVERRIDE_STATUS,
    valid_owner_override,
)
from sidecar.judges import (  # noqa: E402
    JUDGE_PROMPT_VERSION,
    JUDGE_PROTOCOL_FINGERPRINT,
    question_protocol_fingerprint,
)
from sidecar.questions import DICTATING_PROMPT_QUESTION  # noqa: E402
from sidecar.prompts import (  # noqa: E402
    PromptCalibrationError,
    calibrated_prompt_threshold,
    mark_to_transcript_seconds,
    marked_window_indices,
    prompts_from_marks,
)
from sidecar.transcript import TranscriptLine  # noqa: E402


def _line(index: int, seconds: int, text: str) -> TranscriptLine:
    return TranscriptLine(index, float(seconds), f"{seconds // 60:02d}:{seconds % 60:02d}", "Matthias", text)


class _PromptJudge:
    backend = "fake"
    model = "fake"

    def judge(self, lines, questions, *, context=6):
        question = next(iter(questions))
        if question == "dictating_prompt":
            return {question: [0.02, 0.80, 0.92, 0.01, 0.89, 0.93]}
        if question == "prompt_content":
            # The lead-in is rejected while both marked prompt content and the
            # later unmarked pair are preserved as whole verbatim sentences.
            scores = {
                "I will record this now": 0.1,
                "Create a concise implementation plan": 0.95,
                "List the risks": 0.96,
                "and give mitigations": 0.95,
            }
            return {question: [scores.get(line.text, 0.0) for line in lines]}
        raise AssertionError(question)


def test_mark_mapping_accounts_for_known_channel_alignment_and_bounds_window():
    assert mark_to_transcript_seconds(90, channel_lag_seconds=15.46, mic_origin_delay_seconds=2) == pytest.approx(72.54)
    lines = [_line(0, 68, "before"), _line(1, 72, "at mark"), _line(2, 252, "edge"), _line(3, 254, "after")]
    assert marked_window_indices(lines, 73) == (0, 1, 2)


def test_marked_prompt_strips_only_leadin_and_unmarked_run_is_suggested():
    lines = [
        _line(0, 5, "ordinary discussion"),
        _line(1, 10, "I will record this now"),
        _line(2, 12, "Create a concise implementation plan"),
        _line(3, 40, "ordinary discussion again"),
        _line(4, 80, "List the risks"),
        _line(5, 82, "and give mitigations"),
    ]

    results = prompts_from_marks(lines, [14.0], judge=_PromptJudge())

    assert [(result.source, result.mark_seconds, result.text) for result in results] == [
        ("mark", 14.0, "Create a concise implementation plan"),
        ("suggested", None, "List the risks\nand give mitigations"),
    ]


def test_leadin_is_stripped_per_sentence_when_a_turn_contains_the_prompt():
    class _SentenceJudge:
        backend = "fake"
        model = "fake"

        def judge(self, lines, questions, *, context=6):
            question = next(iter(questions))
            if question == "dictating_prompt":
                return {question: [0.9]}
            if question == "prompt_content":
                assert [line.text for line in lines] == [
                    "I will record this now.",
                    "Create a concise implementation plan.",
                ]
                return {question: [0.1, 0.95]}
            raise AssertionError(question)

    lines = [_line(0, 10, "I will record this now. Create a concise implementation plan.")]
    results = prompts_from_marks(lines, [10.0], judge=_SentenceJudge())

    assert [(result.source, result.text) for result in results] == [
        ("mark", "Create a concise implementation plan."),
    ]


def test_leadin_filter_keeps_a_low_scored_interior_prompt_sentence_verbatim():
    class _InteriorJudge:
        backend = "fake"
        model = "fake"

        def judge(self, lines, questions, *, context=6):
            question = next(iter(questions))
            if question == "dictating_prompt":
                return {question: [0.95]}
            if question == "prompt_content":
                assert [line.text for line in lines] == [
                    "Write a deployment plan.",
                    "It must preserve all data.",
                    "Return the risks.",
                ]
                return {question: [0.95, 0.10, 0.95]}
            raise AssertionError(question)

    lines = [
        _line(
            0,
            10,
            "Write a deployment plan. It must preserve all data. Return the risks.",
        )
    ]
    results = prompts_from_marks(lines, [10.0], judge=_InteriorJudge())

    assert results[0].text == "Write a deployment plan.\nIt must preserve all data.\nReturn the risks."


def test_mark_suppresses_only_its_selected_run_not_a_nearby_suggested_prompt():
    class _MarkJudge:
        backend = "fake"
        model = "fake"

        def judge(self, lines, questions, *, context=6):
            question = next(iter(questions))
            if question == "dictating_prompt":
                return {question: [0.95, 0.01, 0.01, 0.95]}
            if question == "prompt_content":
                return {question: [0.95] * len(lines)}
            raise AssertionError(question)

    lines = [
        _line(0, 92, "Suggested prompt before the mark."),
        _line(1, 93, "Ordinary meeting discussion."),
        _line(2, 94, "More ordinary discussion."),
        _line(3, 100, "Prompt selected by the mark."),
    ]
    results = prompts_from_marks(lines, [100.0], judge=_MarkJudge())

    assert [(result.source, result.text) for result in results] == [
        ("mark", "Prompt selected by the mark."),
        ("suggested", "Suggested prompt before the mark."),
    ]


def test_positive_prompt_lines_far_apart_produce_separate_suggestions():
    class _GapJudge:
        backend = "fake"
        model = "fake"

        def judge(self, lines, questions, *, context=6):
            question = next(iter(questions))
            return {question: [0.95] * len(lines)}

    lines = [
        _line(0, 10, "First dictated prompt."),
        _line(1, 500, "Second dictated prompt."),
    ]
    results = prompts_from_marks(lines, [], judge=_GapJudge())

    assert [result.text for result in results] == ["First dictated prompt.", "Second dictated prompt."]


def test_marked_card_intersects_the_run_with_its_bounded_window():
    class _BoundedJudge:
        backend = "fake"
        model = "fake"

        def judge(self, lines, questions, *, context=6):
            question = next(iter(questions))
            return {question: [0.95] * len(lines)}

    lines = [_line(index, seconds, f"Prompt sentence at {seconds}.") for index, seconds in enumerate(range(0, 200, 5))]
    results = prompts_from_marks(
        lines,
        [10.0],
        judge=_BoundedJudge(),
        channel_lag_seconds=0.0,
        mic_origin_delay_seconds=0.0,
    )

    # The marked card must not leak the detected run's text outside the
    # documented [mark - 5s, mark + 3min] window.  Its global run is consumed
    # by the mark, so this test deliberately checks the bounded card itself
    # rather than creating duplicate residual suggestion fragments.
    assert len(results) == 1
    marked = results[0]
    assert marked.source == "mark"
    assert [line.seconds for line in marked.lines][0] == 5
    assert [line.seconds for line in marked.lines][-1] == 190


def test_prompt_threshold_prefers_a_passing_recall_false_positive_envelope():
    labels = [True, True, True, True, True, True, False, False, False]
    probabilities = [0.91, 0.82, 0.72, 0.61, 0.51, 0.46, 0.45, 0.20, 0.10]

    threshold = choose_prompt_threshold(labels, probabilities)
    _, recall, _, false_positives, _ = precision_recall(labels, probabilities, threshold)

    assert recall >= 5 / 6
    assert false_positives <= 2


def test_calibration_uses_the_same_single_question_layouts_as_runtime():
    class _CaptureJudge:
        backend = "fake"
        model = "fake"

        def __init__(self):
            self.calls = []

        def judge(self, lines, questions, *, context=6):
            self.calls.append(tuple(questions))
            return {question: [0.5] * len(lines) for question in questions}

    dataset = {
        "topic": "Synthetic topic",
        "lines": [{"ts": "00:01", "spk": "Speaker"}],
        "versions": {"en": {"0": "Synthetic transcript line"}},
    }
    judge = _CaptureJudge()
    scores = _scores_for_dataset(judge, "A", dataset, "en")

    assert judge.calls == [("relevant",), ("dictating_prompt",)]
    assert set(scores) == {"relevant", "dictating_prompt"}


def test_runtime_uses_only_a_passing_threshold_for_the_exact_gemini_model(tmp_path):
    report = tmp_path / "latest.json"
    report.write_text('{"status":"failed_acceptance_criterion_2","chosen":null}', encoding="utf-8")
    with pytest.raises(PromptCalibrationError):
        calibrated_prompt_threshold("gemini-3.8-flash", report)

    report.write_text(
        '{"status":"passed","judge_prompt_version":"'
        + JUDGE_PROMPT_VERSION
        + '","judge_protocol_sha256":"'
        + JUDGE_PROTOCOL_FINGERPRINT
        + '","dictating_prompt_protocol_sha256":"'
        + question_protocol_fingerprint("dictating_prompt", DICTATING_PROMPT_QUESTION)
        + '","chosen":{"model":"gemini-3.8-flash","prompt_threshold":0.62}}',
        encoding="utf-8",
    )
    assert calibrated_prompt_threshold("gemini-3.8-flash", report) == pytest.approx(0.62)
    with pytest.raises(PromptCalibrationError):
        calibrated_prompt_threshold("gemini-3.1-flash-lite", report)


def test_owner_override_enables_only_the_recorded_single_judge_threshold(tmp_path):
    report = apply_owner_override(
        {
            "schema_version": 2,
            "models": [
                {
                    "model": OWNER_APPROVED_MODEL,
                    "prompt_threshold": OWNER_APPROVED_PROMPT_THRESHOLD,
                    "pass_relevance_a": True,
                    "pass_prompt_c": True,
                }
            ],
            "metrics": [],
        }
    )

    assert report["status"] == OWNER_OVERRIDE_STATUS
    assert valid_owner_override(report)
    assert report["owner_override"]["date"] == "2026-09-26"
    assert report["owner_override"]["swiss_german_relevance_auc_range"] == [0.788, 0.884]
    assert report["chosen"]["prompt_threshold"] == OWNER_APPROVED_PROMPT_THRESHOLD

    destination = tmp_path / "latest.json"
    destination.write_text(__import__("json").dumps(report), encoding="utf-8")
    assert calibrated_prompt_threshold(OWNER_APPROVED_MODEL, destination) == pytest.approx(0.90)
    with pytest.raises(PromptCalibrationError):
        calibrated_prompt_threshold("gemini-3.1-flash-lite", destination)
