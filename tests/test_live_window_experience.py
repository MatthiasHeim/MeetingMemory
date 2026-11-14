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


# ---- C: scroll follow ------------------------------------------------------

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from sidecar.panel_layout import (  # noqa: E402
    MEETING_CONTROLS, INTERNAL_CONTROLS, is_at_bottom, meeting_layout, scroll_offset_after_update, visible_controls,
)
from sidecar.recorder_support import (  # noqa: E402
    jev_preference_enabled, preference_path, read_window_mode, write_jev_preference, write_window_mode,
)


def test_follow_only_at_the_bottom():
    assert is_at_bottom(offset=900, viewport=100, content=1000)
    assert is_at_bottom(offset=0, viewport=100, content=80)  # nothing to scroll
    assert not is_at_bottom(offset=300, viewport=100, content=1000)


def test_update_keeps_reader_position_unless_following():
    assert scroll_offset_after_update(True, 900, 100, 1300) == 1200      # follows new content
    assert scroll_offset_after_update(False, 300, 100, 1300) == 300      # reader stays put
    assert scroll_offset_after_update(False, 5000, 100, 1300) == 1200    # clamped if content shrank


# ---- D: two modes ----------------------------------------------------------

def test_meeting_mode_is_quiet_and_internal_has_everything():
    assert visible_controls("meeting") == MEETING_CONTROLS
    assert visible_controls("internal") == INTERNAL_CONTROLS
    assert visible_controls("garbage") == MEETING_CONTROLS  # fails to the shareable set
    assert MEETING_CONTROLS < INTERNAL_CONTROLS
    for name in ("speakers", "topic", "clip", "audience", "fullscreen", "help", "status", "consent", "copy_transcript"):
        assert name not in MEETING_CONTROLS and name in INTERNAL_CONTROLS


@pytest.mark.parametrize("size", [(480, 760), (1000, 700), (700, 640), (1440, 900)])
def test_meeting_layout_uses_the_whole_window_without_overlap(size):
    width, height = size
    rects = meeting_layout(width, height)
    for name in ("transcript", "cards"):
        x, y, w, h = rects[name]
        assert x >= 0 and y >= 0 and x + w <= width + 1e-6 and y + h <= height + 1e-6 and w > 100 and h > 40
    tx, ty, tw, th = rects["transcript"]
    cx, cy, cw, ch = rects["cards"]
    separate = tx + tw <= cx or cx + cw <= tx or ty + th <= cy or cy + ch <= ty
    assert separate
    assert (tw * th + cw * ch) / (width * height) > 0.7  # little wasted chrome


def test_window_mode_preference_roundtrip_and_legacy_compat(tmp_path):
    cfg = tmp_path / "config.yaml"
    prefs = preference_path(cfg)
    prefs.write_text(json.dumps({"use_decider": True}))          # file as it exists today
    assert read_window_mode(prefs) == "meeting"                  # default, old files stay valid
    write_window_mode(prefs, "internal")
    assert json.loads(prefs.read_text()) == {"use_decider": True, "window_mode": "internal"}
    assert read_window_mode(prefs) == "internal" and jev_preference_enabled(cfg)
    write_jev_preference(cfg, False)                             # the other switch keeps the mode
    assert read_window_mode(prefs) == "internal"
    prefs.write_text("not json")
    assert read_window_mode(prefs) == "meeting"
    assert read_window_mode(None) == "meeting"
    with pytest.raises(ValueError):
        write_window_mode(prefs, "presentation")


def _panel(tmp_path, monkeypatch):
    pytest.importorskip("AppKit")
    from sidecar import live_window
    monkeypatch.setattr(live_window, "_dispatch_main", lambda callback: callback())
    return live_window.LivePanel(on_copy_clip=lambda _t: None, on_copy_card=lambda _t: None,
                                 preferences_path=tmp_path / "sidecar-preferences.json")


def _many_lines(count=120):
    return tuple(LiveLine(i * 5.0, i * 5.0 + 4, "Ich", f"Sentence number {i} is spoken here for the test.") for i in range(count))


def test_panel_defaults_to_meeting_mode_and_hides_internal_controls(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        assert panel.window_mode == "meeting"
        panel.show()
        assert panel._record_button.isHidden() is False
        for view in (panel.topic_field, panel.speakers_field, panel._clip_button, panel._audience_control,
                     panel._fullscreen_button, panel._help_button, panel._copy_transcript_button):
            assert view.isHidden() is True
        panel.set_window_mode("internal")
        assert panel.topic_field.isHidden() is False and panel._audience_control.isHidden() is False
        assert read_window_mode(tmp_path / "sidecar-preferences.json") == "internal"
    finally:
        panel.close()


def test_meeting_mode_forces_client_audience(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.show()
        panel.set_window_mode("internal")
        panel.set_audience("internal")
        panel.set_window_mode("meeting")
        assert panel.audience == "client"
    finally:
        panel.close()


def test_scrolled_up_view_is_not_pulled_back_and_jump_resumes(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.show()
        lines = _many_lines()
        panel.apply_update(lines[:100], (), "")
        scroll = panel._transcript_scroll
        clip = scroll.contentView()
        assert panel._follow["transcript"] is True
        at_end = clip.bounds().origin.y
        assert at_end > 500                                   # followed the newest line
        clip.scrollToPoint_((0, 400))                         # the reader scrolls up
        scroll.reflectScrolledClipView_(clip)
        panel._sync_follow()
        assert panel._follow["transcript"] is False
        assert panel._jump_buttons["transcript"].isHidden() is False
        panel.apply_update(lines, (), "")                     # more speech arrives
        assert clip.bounds().origin.y == 400
        panel.jump_to_latest("transcript")
        assert panel._follow["transcript"] is True
        assert clip.bounds().origin.y > at_end
        assert panel._jump_buttons["transcript"].isHidden() is True
    finally:
        panel.close()
