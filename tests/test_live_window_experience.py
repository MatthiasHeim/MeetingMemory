"""Live window: cards survive rewrites (A), settled text is stable (B), no client questions when alone (E)."""
import time

import numpy as np

from tools.sidecar.card_semantics import split_sources
from tools.sidecar.live import (
    LiveCard, LiveEngine, LiveLine, TailSnapshot, commit_new_lines, rebind_card,
)
from tools.sidecar.transcript import TranscriptLine, format_timestamp


def _card_on(lines, kind="request", key="k"):
    spans = tuple(split_sources(lines))
    rows = tuple(TranscriptLine(s.line_index, s.seconds, format_timestamp(s.seconds), s.speaker, s.text) for s in spans)
    return LiveCard(key, rows, "clean text", kind, spans)


# ---- A: cards follow rewritten lines -------------------------------------

def test_card_follows_lines_merged_by_a_later_pass():
    old = [LiveLine(50, 52, "Ich", "We need to improve this."), LiveLine(52.75, 58.9, "Ich", "We need to improve the overall interface.")]
    card = _card_on(old[1:], key="52.8")
    merged = [LiveLine(50, 59, "Ich", "We need to improve this. We need to improve the whole interface as well.")]
    moved = rebind_card(card, merged)
    assert moved is not None
    assert moved.key == "52.8" and moved.clean_text == "clean text"
    assert all(span.id in {s.id for s in split_sources(merged)} for span in moved.spans)


def test_card_dropped_only_when_its_speech_is_gone():
    old = [LiveLine(50, 58, "Ich", "Please fix the scroll bug.")]
    card = _card_on(old)
    assert rebind_card(card, [LiveLine(200, 205, "Ich", "Something else entirely.")]) is None
    assert rebind_card(card, [LiveLine(50, 58, "Andere", "Other speaker only.")]) is None
    assert rebind_card(card, [LiveLine(49, 60, "Ich", "Please fix the scroll bug now.")]) is not None


def test_snapshot_rewrite_keeps_request_and_suggestion_cards():
    class T:
        rows = []
        def transcribe_tails(self, *a):
            return list(self.rows)
    t = T()
    t.rows = [{"speaker": "Ich", "text": "Okay.", "start_offset": 0, "end_offset": 1},
              {"speaker": "Ich", "text": "We need to improve the interface.", "start_offset": 2, "end_offset": 6}]
    engine = LiveEngine(snapshot=lambda _: None, transcriber=t)
    snap = TailSnapshot(np.zeros(16), 16000, None, 16000, 0, 8)
    engine.process_snapshot(snap)
    request = _card_on(engine.lines[1:], key="r")
    suggestion = _card_on(engine.lines, kind="suggestion_question", key="s")
    engine.cards = [request]
    engine._suggestions["client"] = [suggestion]
    # A later pass hears the same speech as one longer line starting earlier.
    t.rows = [{"speaker": "Ich", "text": "Okay. We need to improve the interface and the layout.", "start_offset": 0, "end_offset": 7}]
    engine.process_snapshot(snap)
    assert len(engine.lines) == 1
    assert [c.key for c in engine.cards] == ["r"]
    assert [c.key for c in engine._suggestions["client"]] == ["s"]
    ids = {s.id for s in split_sources(engine.lines)}
    assert all(span.id in ids for span in engine._suggestions["client"][0].spans)


# ---- B: settled lines are not replaced by a worse merge ------------------

FOUR = [
    LiveLine(7.0, 10.0, "Ich", "Okay, so I have not really used this much."),
    LiveLine(11.0, 17.0, "Ich", "My problems are that, you know, the cards vanish."),
    LiveLine(18.0, 23.0, "Ich", "Actually this is cool, I like it."),
    LiveLine(24.0, 28.0, "Ich", "Let me think about it for a moment."),
]


def test_shorter_merge_of_settled_lines_is_rejected():
    garbled = LiveLine(6.9, 28.2, "Ich", "as much, um my problems are that, I see this is cool, think about it.")
    lines, _added, revised = commit_new_lines(FOUR, [garbled])
    assert lines == FOUR and revised is None


def test_lossless_merge_that_keeps_all_words_is_accepted():
    merged = LiveLine(6.9, 28.2, "Ich", " ".join(line.text for line in FOUR) + " Really.")
    lines, _added, revised = commit_new_lines(FOUR, [merged])
    assert len(lines) == 1 and lines[0].text == merged.text and revised is not None


def test_single_line_refinement_still_allowed():
    prev = [LiveLine(0, 3, "Ich", "Das ist ein")]
    lines, _a, revised = commit_new_lines(prev, [LiveLine(0, 4, "Ich", "Das ist ein Test mit mehr Worten.")])
    assert lines[0].text == "Das ist ein Test mit mehr Worten." and revised == 0


# ---- E: nobody but the host has spoken -----------------------------------

def _suggest_engine(lines, calls):
    engine = LiveEngine(snapshot=lambda _: None, judge=object(), contextual_cards=True,
                        suggest_cards=lambda *a: calls.append(a) or [])
    engine.lines = lines
    engine._contextual_ready_key = tuple(s.id for s in split_sources(lines))
    return engine


def _wait(engine):
    deadline = time.monotonic() + 2
    while engine._suggest_thread is not None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_no_suggestions_when_only_host_spoke():
    calls = []
    engine = _suggest_engine([LiveLine(0, 4, "Ich", "One thing."), LiveLine(5, 9, "Ich", "Another thing."),
                              LiveLine(10, 14, "Ich", "A third thing.")], calls)
    engine._request_suggestions()
    _wait(engine)
    assert calls == []


def test_suggestions_run_once_someone_else_spoke():
    calls = []
    engine = _suggest_engine([LiveLine(0, 4, "Ich", "One thing."), LiveLine(5, 9, "Andere", "Another thing."),
                              LiveLine(10, 14, "Ich", "A third thing.")], calls)
    engine._request_suggestions()
    _wait(engine)
    assert len(calls) == 1
