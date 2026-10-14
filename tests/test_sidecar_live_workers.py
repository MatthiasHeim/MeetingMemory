"""Concurrency regressions with synthetic text and event-controlled providers."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sidecar.live import LiveEngine, TailSnapshot


@pytest.fixture(autouse=True)
def no_wall_clock_retry_delay(monkeypatch):
    monkeypatch.setattr("sidecar.live.PROMPT_RETRY_SECONDS", 0.0)


def row(text="Please ask the agent to prepare a report.", start=1.0, end=5.0):
    return {"speaker": "Ich", "text": text, "start_offset": start, "end_offset": end}


def snapshot():
    return TailSnapshot(np.zeros(16, dtype=np.int16), 16000, None, 16000, 0, 300, closed=True)


class Transcriber:
    def __init__(self, rows=None):
        self.rows = [row()] if rows is None else rows

    def transcribe_tails(self, *_args):
        return list(self.rows)


class PositiveJudge:
    backend = "synthetic"

    def judge(self, lines, questions, *, context=6):
        return {name: [0.99] * len(lines) for name in questions}


def drain(engine):
    """Bounded join of optional workers, including work queued by the judge."""
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline:
        with engine._lock:
            workers = [worker for worker in (engine._prompt_thread, engine._clean_thread) if worker is not None]
        if not workers:
            return
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.perf_counter()))
    pytest.fail("synthetic workers did not finish")


def test_blocked_judge_does_not_delay_publication_next_tick_or_stop():
    entered, release, three_ticks = threading.Event(), threading.Event(), threading.Event()
    updates = []

    class TickTranscriber:
        calls = 0

        def transcribe_tails(self, *_args):
            self.calls += 1
            if self.calls >= 3:
                three_ticks.set()
            return [row(f"Synthetic instruction {i}.", i * 10 + 1, i * 10 + 5) for i in range(self.calls)]

    class BlockingJudge(PositiveJudge):
        def judge(self, lines, questions, *, context=6):
            entered.set()
            assert release.wait(3)
            return super().judge(lines, questions, context=context)

    engine = LiveEngine(snapshot=lambda _: snapshot(), transcriber=TickTranscriber(),
                        judge=BlockingJudge(), interval=0.01,
                        on_update=lambda lines, cards, status: updates.append((lines, cards, status)))
    try:
        engine.start()
        assert entered.wait(1)
        assert updates and updates[0][0] and not updates[0][1]
        assert three_ticks.wait(1), "transcription must continue while classification is blocked"
        start = time.monotonic()
        engine.stop()
        assert time.monotonic() - start < 0.1
        engine._thread.join(timeout=1)
        assert not engine._thread.is_alive()
        assert len(engine.lines) >= 3
    finally:
        engine.stop()
        release.set()
        engine._thread.join(timeout=2)
        drain(engine)
    assert len(engine._scores) == len(engine.lines)


def test_blocked_judge_initialization_does_not_delay_first_transcript():
    entered, release, published = threading.Event(), threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(3)
        return PositiveJudge()

    engine = LiveEngine(snapshot=lambda _: snapshot(), transcriber=Transcriber(), judge_factory=factory,
                        interval=0.01, on_update=lambda lines, *_: published.set() if lines else None)
    try:
        engine.start()
        assert entered.wait(1)
        assert published.wait(1)
        assert not engine.cards
    finally:
        engine.stop()
        release.set()
        engine._thread.join(timeout=2)
        drain(engine)
    assert engine.cards


def test_failed_factory_recovers_on_an_unchanged_transcript_without_fallback():
    calls = []

    def factory():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("private diagnostic details")
        return PositiveJudge()

    engine = LiveEngine(snapshot=lambda _: None, transcriber=Transcriber(), judge_factory=factory)
    engine.process_snapshot(snapshot())
    drain(engine)
    assert engine.lines and not engine.cards and engine._scores == []
    assert "Prompt-Erkennung unterbrochen" in engine.status
    engine.process_snapshot(snapshot())
    drain(engine)
    assert len(calls) == 2 and len(engine.cards) == 1
    assert engine.status == ""


@pytest.mark.parametrize("add_lines", [False, True])
def test_failed_batch_retries_every_unscored_line_even_after_silence(add_lines):
    observed = []

    class FailingOnce(PositiveJudge):
        def judge(self, lines, questions, *, context=6):
            observed.append([line.index for line in lines])
            if len(observed) == 1:
                raise TimeoutError("synthetic")
            return super().judge(lines, questions, context=context)

    transcriber = Transcriber([row(f"Instruction {i}.", i * 10, i * 10 + 3) for i in range(10)])
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=FailingOnce())
    engine.process_snapshot(snapshot())
    drain(engine)
    assert engine._scores == []
    if add_lines:
        transcriber.rows += [row("Another instruction.", 110, 113)]
    engine.process_snapshot(snapshot())
    drain(engine)
    assert observed[1] == list(range(8)), "retry the original bounded range, including its first line"
    assert len(engine._scores) == len(engine.lines)
    assert engine.cards[0].lines[0].index == 0


def test_pending_transcripts_are_coalesced_without_losing_lines_or_order():
    entered, release = threading.Event(), threading.Event()
    observed, update_lengths = [], []

    class BlockingOnce(PositiveJudge):
        def judge(self, lines, questions, *, context=6):
            observed.append([line.index for line in lines])
            if len(observed) == 1:
                entered.set()
                assert release.wait(3)
            return super().judge(lines, questions, context=context)

    transcriber = Transcriber()
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=BlockingOnce(),
                        on_update=lambda lines, *_: update_lengths.append(len(lines)))
    try:
        engine.process_snapshot(snapshot())
        assert entered.wait(1)
        for i in range(1, 10):
            transcriber.rows.append(row(f"Continuation {i}.", i * 10 + 1, i * 10 + 5))
            engine.process_snapshot(snapshot())
        assert len(engine.lines) == 10 and engine._scores == []
    finally:
        release.set()
        drain(engine)
    assert observed == [[0], list(range(9)), list(range(3, 10))]
    assert len(engine._scores) == 10
    assert update_lengths == sorted(update_lengths)


def test_inflight_score_for_replaced_text_is_discarded_before_card_publication():
    entered, release = threading.Event(), threading.Event()
    observed, shown = [], []

    class Judge:
        def judge(self, lines, questions, *, context=6):
            observed.append([line.text for line in lines])
            if len(observed) == 1:
                entered.set()
                assert release.wait(3)
                return {name: [0.99] * len(lines) for name in questions}
            return {name: [0.01] * len(lines) for name in questions}

    transcriber = Transcriber([row("The agent should", 1, 4)])
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=Judge(),
                        on_update=lambda _, cards, __: shown.extend(cards))
    try:
        engine.process_snapshot(snapshot())
        assert entered.wait(1)
        transcriber.rows = [row("The agent should be mentioned in ordinary discussion.", 1, 8)]
        engine.process_snapshot(snapshot())
    finally:
        release.set()
        drain(engine)
    assert len(observed) == 2 and engine._scores == [0.01]
    assert shown == [] and engine.cards == []


def test_revision_invalidates_affected_scores_but_keeps_earlier_context():
    observed = []

    class Judge:
        def judge(self, lines, questions, *, context=6):
            observed.append([line.index for line in lines])
            scores = [0.99 if line.text == "The agent should" else 0.01 for line in lines]
            return {name: scores for name in questions}

    rows = [row(f"Ordinary discussion {i}.", i * 10, i * 10 + 4) for i in range(9)]
    rows += [row("The agent should", 90, 94)]
    transcriber = Transcriber(rows)
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=Judge())
    engine.process_snapshot(snapshot())
    drain(engine)
    assert len(engine.cards) == 1
    transcriber.rows[-1] = row("The agent should be mentioned in ordinary discussion.", 90, 98)
    engine.process_snapshot(snapshot())
    drain(engine)
    assert observed[-1] == list(range(3, 10)), "preserve the existing six-line context policy"
    assert engine._scores == [0.01] * 10 and engine.cards == []


def test_blocked_cleaner_does_not_hide_verbatim_cards_or_block_new_detection():
    entered, release = threading.Event(), threading.Event()
    two_cards = threading.Event()

    def cleaner(text):
        entered.set()
        assert release.wait(3)
        return "Clean: " + text

    transcriber = Transcriber()
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=PositiveJudge(),
                        clean_prompt=cleaner,
                        on_update=lambda _, cards, __: two_cards.set() if len(cards) == 2 else None)
    try:
        engine.process_snapshot(snapshot())
        assert entered.wait(1)
        assert engine.cards[0].copy_text == transcriber.rows[0]["text"]
        transcriber.rows.append(row("Please ask the agent to make a separate plan.", 150, 155))
        engine.process_snapshot(snapshot())
        assert two_cards.wait(1), "another detected card must not wait for the first rewrite"
        assert len(engine._scores) == 2
    finally:
        release.set()
        drain(engine)


def test_old_clean_result_cannot_replace_revised_card():
    entered, release = threading.Event(), threading.Event()
    shown, cleaned = [], []

    def cleaner(text):
        cleaned.append(text)
        if len(cleaned) == 1:
            entered.set()
            assert release.wait(3)
            return "OUTDATED REWRITE"
        return "CURRENT REWRITE"

    transcriber = Transcriber([row("Ask the agent to", 1, 4)])
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=PositiveJudge(),
                        clean_prompt=cleaner,
                        on_update=lambda _, cards, __: shown.extend(card.clean_text for card in cards))
    try:
        engine.process_snapshot(snapshot())
        assert entered.wait(1)
        transcriber.rows = [row("Ask the agent to prepare a complete plan.", 1, 9)]
        engine.process_snapshot(snapshot())
        # The scoring worker must finish without waiting for the cleaner.
        worker = engine._prompt_thread
        if worker is not None:
            worker.join(timeout=1)
            assert not worker.is_alive()
    finally:
        release.set()
        drain(engine)
    assert "OUTDATED REWRITE" not in shown
    assert engine.cards[0].clean_text == "CURRENT REWRITE"


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [-0.1], [1.1], []])
def test_invalid_scores_remain_unscored_and_visible_as_failure(scores):
    class Judge:
        def judge(self, _lines, questions, *, context=6):
            return {name: scores for name in questions}

    engine = LiveEngine(snapshot=lambda _: None, transcriber=Transcriber(), judge=Judge())
    engine.process_snapshot(snapshot())
    drain(engine)
    assert engine.lines and not engine.cards and engine._scores == []
    assert "Prompt-Erkennung unterbrochen" in engine.status


def test_diagnostics_distinguish_scored_negatives_from_failure_without_content():
    diagnostics = []

    class Judge:
        backend = "decider"

        def judge(self, lines, questions, *, context=6):
            return {name: [0.919] * len(lines) for name in questions}

    text = "PRIVATE SYNTHETIC INSTRUCTION, never log me."
    engine = LiveEngine(snapshot=lambda _: None, transcriber=Transcriber([row(text)]), judge=Judge(),
                        prompt_threshold=0.92, on_diagnostics=diagnostics.append)
    engine.process_snapshot(snapshot())
    drain(engine)
    prompts = next(event for event in diagnostics if event["stage"] == "prompts")
    assert prompts["scored_count"] == 1 and prompts["above_threshold_count"] == 0
    assert prompts["card_count"] == 0 and prompts["max_score"] == 0.919
    assert prompts["backend"] == "decider" and prompts["threshold"] == 0.92
    assert text not in json.dumps(diagnostics)
    assert engine.status == "", "a completed negative decision is distinct from an error"


def test_errors_do_not_log_provider_response_or_clear_on_transcription_success():
    diagnostics = []

    class Judge:
        def judge(self, *_args, **_kwargs):
            raise TimeoutError("PRIVATE PROVIDER RESPONSE")

    engine = LiveEngine(snapshot=lambda _: None, transcriber=Transcriber(), judge=Judge(),
                        on_diagnostics=diagnostics.append)
    for _ in range(2):
        engine.process_snapshot(snapshot())
        drain(engine)
        assert "Prompt-Erkennung unterbrochen" in engine.status
    assert diagnostics[-1]["error_type"] == "TimeoutError"
    assert "PRIVATE" not in json.dumps(diagnostics)


def test_permanently_failing_line_cannot_stall_later_prompts_or_grow_retries():
    observed, diagnostics = [], []

    class PoisonedJudge(PositiveJudge):
        def judge(self, lines, questions, *, context=6):
            observed.append([line.index for line in lines])
            if any(line.index == 9 for line in lines):
                raise RuntimeError("synthetic persistent line failure")
            return super().judge(lines, questions, context=context)

    transcriber = Transcriber([row(f"Instruction {i}.", i * 8, i * 8 + 2) for i in range(30)])
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=PoisonedJudge(),
                        on_diagnostics=diagnostics.append)
    for _ in range(3):
        engine.process_snapshot(snapshot())
        drain(engine)
    assert len(engine._scores) == 30
    assert engine._scores[9] is None, "failure must never masquerade as a scored negative"
    assert engine._scores[-1] == 0.99
    assert any(card.lines[-1].index == 29 for card in engine.cards)
    assert all(9 not in [line.index for line in card.lines] for card in engine.cards)
    assert max(map(len, observed)) <= 14, "at most eight new lines plus six context lines"
    poison_requests = [indices for indices in observed if 9 in indices]
    assert len(poison_requests) == 3 and poison_requests[0] == poison_requests[1] == poison_requests[2]
    assert "nicht ausgewertet" in engine.status
    failed = [event for event in diagnostics if event.get("error_type")]
    assert failed[-1]["skipped_count"] == 8
    assert failed[-1]["consecutive_failures"] == 3
    assert failed[-1]["pending_count"] == 14
    # No endless retry in silence once failed ranges have been reported.
    count = len(observed)
    engine.process_snapshot(snapshot())
    drain(engine)
    assert len(observed) == count


def test_failure_backoff_prevents_immediate_retries(monkeypatch):
    monkeypatch.setattr("sidecar.live.PROMPT_RETRY_SECONDS", 20.0)
    clock = [100.0]
    monkeypatch.setattr("sidecar.live.time.monotonic", lambda: clock[0])
    calls = []

    class FailingJudge:
        def judge(self, *_args, **_kwargs):
            calls.append(True)
            raise TimeoutError("synthetic")

    engine = LiveEngine(snapshot=lambda _: None, transcriber=Transcriber(), judge=FailingJudge())
    engine.process_snapshot(snapshot())
    drain(engine)
    for _ in range(4):
        engine.process_snapshot(snapshot())
        drain(engine)
    assert len(calls) == 1
    clock[0] += 20
    engine.process_snapshot(snapshot())
    drain(engine)
    assert len(calls) == 2
    clock[0] += 39
    engine.process_snapshot(snapshot())
    drain(engine)
    assert len(calls) == 2
    clock[0] += 1
    engine.process_snapshot(snapshot())
    drain(engine)
    assert len(calls) == 3 and engine._scores == [None]


def test_unknown_score_is_a_hard_boundary_for_prompt_cards():
    from sidecar.live import LiveLine

    engine = LiveEngine(snapshot=lambda _: None, judge=PositiveJudge())
    engine.lines = [LiveLine(i * 2, i * 2 + 1, "Ich", text) for i, text in enumerate(
        ["First task.", "I would tell the agent to", "Second task."])]
    engine._scores = [0.99, None, 0.99]
    engine._refresh_cards()
    assert [[line.index for line in card.lines] for card in engine.cards] == [[0], [2]]


def test_new_score_diagnostics_are_not_hidden_by_an_earlier_positive():
    diagnostics = []

    class Judge:
        def judge(self, lines, questions, *, context=6):
            return {name: [0.99 if line.index == 0 else 0.90 for line in lines] for name in questions}

    transcriber = Transcriber()
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=Judge(),
                        prompt_threshold=0.92, on_diagnostics=diagnostics.append)
    engine.process_snapshot(snapshot())
    drain(engine)
    transcriber.rows.append(row("Another potential request.", 150, 155))
    engine.process_snapshot(snapshot())
    drain(engine)
    event = [event for event in diagnostics if event["stage"] == "prompts"][-1]
    assert event["max_score"] == 0.99
    assert event["new_max_score"] == 0.90
    assert event["new_above_threshold_count"] == 0
    assert event["new_line_from"] == 1 and event["new_line_to"] == 2
    assert event["requested_count"] == 2 and event["pending_count"] == 0


def test_persistent_context_failure_eventually_expires_without_blocking_the_rest():
    observed = []

    class Judge(PositiveJudge):
        def judge(self, lines, questions, *, context=6):
            observed.append([line.index for line in lines])
            if any(line.index == 15 for line in lines):
                raise RuntimeError("synthetic failure also when the line is context")
            return super().judge(lines, questions, context=context)

    transcriber = Transcriber([row(f"Instruction {i}.", i * 8, i * 8 + 2) for i in range(30)])
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=Judge())
    for _ in range(6):
        engine.process_snapshot(snapshot())
        drain(engine)
    assert len(engine._scores) == 30 and engine._scores[15] is None
    assert engine._scores[-1] == 0.99
    assert any(card.lines[-1].index == 29 for card in engine.cards)
    assert max(map(len, observed)) <= 14


def test_unrelated_index_shift_preserves_existing_card_and_clean_text():
    entered, release = threading.Event(), threading.Event()
    clean_calls = []

    class Judge:
        calls = 0

        def judge(self, lines, questions, *, context=6):
            self.calls += 1
            if self.calls == 2:
                entered.set()
                assert release.wait(3)
            return {name: [0.99 if "report" in line.text else 0.01 for line in lines] for name in questions}

    def cleaner(text):
        clean_calls.append(text)
        return "Clean report request."

    transcriber = Transcriber([row("Before.", 1, 3), row("A fragment.", 4, 6),
                               row("Please prepare a report.", 100, 104)])
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=Judge(), clean_prompt=cleaner)
    engine.process_snapshot(snapshot())
    drain(engine)
    assert engine.cards[0].lines[0].index == 2
    transcriber.rows = [row("Before and a fragment from earlier.", 0, 7)]
    try:
        engine.process_snapshot(snapshot())
        assert entered.wait(1)
        assert len(engine.lines) == 2
        assert len(engine.cards) == 1 and engine.cards[0].lines[0].index == 1
        assert engine.cards[0].clean_text == "Clean report request."
    finally:
        release.set()
        drain(engine)
    assert len(clean_calls) == 1
