"""Live window: three views (Meeting / Ambient / Intern), calm reveal rule, Claude jump request."""
import itertools
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from sidecar import jump  # noqa: E402
from sidecar.live import LiveCard, LiveLine  # noqa: E402
from sidecar.panel_layout import (  # noqa: E402
    AMBIENT_CONTROLS, INTERNAL_CONTROLS, MEETING_CONTROLS, PANEL_HELP, VIEWS, card_columns, cards_for_view,
    context_lines, context_text, latest_line_text, normalize_view, panel_mode, split_columns, view_layout,
    visible_controls,
)
from sidecar.recorder_support import read_window_mode, write_window_mode  # noqa: E402
from sidecar.reveal import RevealState  # noqa: E402
from sidecar.transcript import TranscriptLine  # noqa: E402


def card(key, kind="request", audience="all", text=None, start=0.0):
    return LiveCard(key, (TranscriptLine(0, start, "00:00", "Ich", "said"),), text or f"text {key}", kind, (), audience)


ALL_CARDS = (
    card("r", "request"), card("d", "decision"), card("a", "action", "client"),
    card("sq", "suggestion_question", "client"), card("si", "suggestion_idea", "all"),
    card("priv", "suggestion_idea", "internal"), card("privact", "action", "internal"),
)


# ---- view values and persistence -------------------------------------------------

def test_unknown_view_values_fall_back_to_meeting():
    assert VIEWS == ("meeting", "ambient", "internal")
    for value in ("meeting", "ambient", "internal"):
        assert normalize_view(value) == value
    for value in ("presentation", "compact", "", None, 3, "Internal"):
        assert normalize_view(value) == "meeting"
    assert visible_controls("garbage") == MEETING_CONTROLS


def test_view_persistence_roundtrip_and_old_values(tmp_path):
    prefs = tmp_path / "prefs.json"
    for view in VIEWS:
        write_window_mode(prefs, view)
        assert read_window_mode(prefs) == view
    prefs.write_text(json.dumps({"window_mode": "internal", "use_decider": True}))
    assert read_window_mode(prefs) == "internal"          # the old persisted value keeps working
    prefs.write_text(json.dumps({"window_mode": "presentation"}))
    assert read_window_mode(prefs) == "meeting"
    with pytest.raises(ValueError):
        write_window_mode(prefs, "presentation")


# ---- visible controls ---------------------------------------------------------------

def test_visible_controls_per_view():
    for forbidden in ("audience", "speakers", "topic", "clip", "help", "copy_transcript", "transcript", "ctx_claude"):
        assert forbidden not in MEETING_CONTROLS
    assert {"window_mode", "state", "record", "cards", "recent", "marker"} <= MEETING_CONTROLS
    assert {"ctx_label", "ctx_range", "ctx_copy", "ctx_claude", "cards"} <= AMBIENT_CONTROLS
    for forbidden in ("audience", "speakers", "topic", "clip", "help", "copy_transcript", "recent"):
        assert forbidden not in AMBIENT_CONTROLS
    assert {"transcript", "copy_transcript", "audience", "help", "speakers", "topic", "clip", "cards"} <= INTERNAL_CONTROLS
    assert "ctx_claude" not in INTERNAL_CONTROLS and "recent" not in INTERNAL_CONTROLS
    for view in VIEWS:
        assert {"window_mode", "record", "state"} <= visible_controls(view)
    # full screen adds only what the roomier layout draws
    assert "transcript" in visible_controls("ambient", fullscreen=True) and "transcript" not in visible_controls("ambient")
    assert "rail" in visible_controls("internal", fullscreen=True)
    assert visible_controls("meeting", fullscreen=True) == MEETING_CONTROLS


# ---- which points each view shows ----------------------------------------------------

def keys(cards):
    return [c.key for c in cards]


def test_meeting_shows_only_client_safe_request_decision_action():
    assert keys(cards_for_view(ALL_CARDS, "meeting")) == ["r", "d", "a"]
    assert keys(cards_for_view(ALL_CARDS, "meeting", show_private=True)) == ["r", "d", "a"]


def test_ambient_shows_non_suggestion_client_points_never_private():
    assert keys(cards_for_view(ALL_CARDS, "ambient")) == ["r", "d", "a"]
    assert keys(cards_for_view(ALL_CARDS, "ambient", show_private=True)) == ["r", "d", "a"]


def test_internal_shows_everything_private_only_with_the_tick():
    assert keys(cards_for_view(ALL_CARDS, "internal")) == ["r", "d", "a", "sq", "si"]
    assert keys(cards_for_view(ALL_CARDS, "internal", show_private=True)) == keys(ALL_CARDS)
    assert keys(cards_for_view(ALL_CARDS, "nonsense")) == ["r", "d", "a"]   # unknown view fails to meeting


def test_meeting_columns_by_mode():
    assert [t for t, _ in card_columns("meeting", "narrow")] == [None]
    assert len(card_columns("meeting", "wide")) == 2
    full = card_columns("meeting", "fullscreen")
    assert [t for t, _ in full] == ["Entscheidungen", "Nächste Schritte", "Anfragen"]
    cols = split_columns(cards_for_view(ALL_CARDS, "meeting"), full)
    assert [keys(c) for c in cols] == [["d"], ["a"], ["r"]]
    assert len(card_columns("ambient", "fullscreen")) == 1 and len(card_columns("internal", "wide")) == 1


# ---- layout ------------------------------------------------------------------------

SIZES = [(480, 760, False), (1000, 700, False), (1440, 900, True)]


def inside(rect, width, height):
    x, y, w, h = rect
    return x >= -1e-6 and y >= -1e-6 and x + w <= width + 1e-6 and y + h <= height + 1e-6 and w > 0 and h > 0


def overlap(a, b):
    return not (a[0] + a[2] <= b[0] + 1e-6 or b[0] + b[2] <= a[0] + 1e-6
                or a[1] + a[3] <= b[1] + 1e-6 or b[1] + b[3] <= a[1] + 1e-6)


def test_panel_mode_has_a_distinct_fullscreen_value():
    assert panel_mode(480, 760) == "narrow"
    assert panel_mode(1000, 700) == "wide"
    assert panel_mode(480, 760, fullscreen=True) == "fullscreen"
    assert panel_mode(1440, 900, fullscreen=True) == "fullscreen"
    assert "presentation" not in {panel_mode(w, h, fullscreen=f) for w, h, f in SIZES}


@pytest.mark.parametrize("view,size", list(itertools.product(VIEWS, SIZES)))
def test_layout_rects_fit_and_content_areas_do_not_overlap(view, size):
    width, height, fullscreen = size
    rects = view_layout(view, width, height, fullscreen=fullscreen)
    allowed = visible_controls(view, fullscreen=fullscreen)
    for name, rect in rects.items():
        assert inside(rect, width, height), (name, rect)
    for name in allowed - {"jump"}:
        if name in {"state", "help", "status"}:
            continue  # optional on narrow windows
        assert name in rects, name
    areas = [n for n in ("cards", "transcript", "recent", "rail", "ctx_range", "ctx_copy", "ctx_claude", "speakers",
                         "topic", "clip", "window_mode", "record") if n in rects]
    if view == "internal" and fullscreen:
        areas = [n for n in areas if n not in {"speakers", "topic", "clip"}]  # those sit on the rail
    for a, b in itertools.combinations(areas, 2):
        assert not overlap(rects[a], rects[b]), (a, b)
    assert rects["cards"][3] > 60 and rects["cards"][2] > 100
    # nothing the view hides gets a rect that could be mistaken for content
    assert set(rects) - {"help_body", "marker"} <= allowed | {"marker"}


def test_narrow_is_one_column_and_wide_is_columns():
    narrow = view_layout("internal", 480, 760)
    assert narrow["transcript"][0] == narrow["cards"][0] and narrow["transcript"][1] > narrow["cards"][1]
    wide = view_layout("internal", 1000, 700)
    assert wide["transcript"][0] < wide["cards"][0]
    assert abs(wide["transcript"][1] - wide["cards"][1]) < 60  # side by side, not stacked


def test_fullscreen_layouts_are_roomier_and_view_specific():
    window = view_layout("meeting", 1440, 900)
    full = view_layout("meeting", 1440, 900, fullscreen=True)
    assert full["cards"][0] > window["cards"][0]                    # wider margins
    amb = view_layout("ambient", 1440, 900, fullscreen=True)
    assert amb["cards"][0] < amb["transcript"][0] and amb["cards"][2] > amb["transcript"][2]   # list left, transcript right
    internal = view_layout("internal", 1440, 900, fullscreen=True)
    assert internal["transcript"][0] < internal["cards"][0] < internal["rail"][0]
    assert internal["rail"][0] > internal["transcript"][0] + internal["transcript"][2]
    for name in ("speakers", "topic", "clip"):
        assert inside(internal[name], *internal["rail"][2:]) or internal[name][0] >= internal["rail"][0]
    assert "transcript" not in view_layout("meeting", 1440, 900, fullscreen=True)


def test_narrow_window_has_room_for_switch_state_and_record():
    for view in VIEWS:
        r = view_layout(view, 480, 760)
        assert r["window_mode"][0] + r["window_mode"][2] < r["record"][0]
        assert "state" in r


# ---- context and the quiet line -------------------------------------------------------

LINES = tuple(LiveLine(i * 100.0, i * 100.0 + 5, "Ich" if i % 2 else "Andere", f"line {i}") for i in range(10))


def test_context_ranges():
    assert context_lines(LINES, "all") == LINES
    last = context_lines(LINES, "last10")
    assert [l.text for l in last] == [f"line {i}" for i in range(3, 10)]   # latest end 905 -> since 305
    assert context_lines((), "last10") == ()
    text = context_text(LINES, "last10")
    assert text.startswith("[") and "line 3" in text and "line 2" not in text
    assert latest_line_text(LINES) == "line 9"
    assert latest_line_text((LiveLine(0, 1, "Ich", "a  b\nc"),)) == "a b c"
    assert latest_line_text(()) == ""


# ---- calm rule bookkeeping ---------------------------------------------------------------

def test_first_points_show_immediately_then_new_ones_wait():
    s = RevealState()
    s.observe(["a", "b"])
    assert s.shown == ["a", "b"] and s.pending == [] and s.marker_text is None and s.divider_key is None
    s.observe(["a", "b", "c"])
    assert s.shown == ["a", "b"] and s.pending == ["c"] and s.marker_text == "1 neuer Punkt ↓"
    s.observe(["a", "b", "c", "d", "e"])
    assert s.shown == ["a", "b"] and s.marker_text == "3 neue Punkte ↓"


def test_reveal_moves_pending_in_and_sets_divider():
    s = RevealState()
    s.observe(["a", "b"])
    s.observe(["a", "b", "c", "d"])
    assert s.reveal("11:02") == 2
    assert s.shown == ["a", "b", "c", "d"] and s.pending == [] and s.marker_text is None
    assert s.divider_key == "c" and s.divider_label == "Neu seit 11:02"
    assert s.divider_before(["a", "b", "c", "d"]) == "c"
    assert s.divider_before(["a", "d"]) == "d"          # per column: first point of the batch in it
    assert s.divider_before(["c", "d"]) is None         # a divider at the top of a column says nothing
    assert s.divider_before(["a", "b"]) is None
    assert s.reveal("11:03") == 0


def test_revealed_points_stay_revealed_when_rewritten_and_nothing_reorders():
    s = RevealState()
    s.observe(["a", "b"])
    s.observe(["b", "a", "z"])                          # engine reorders and adds one
    assert s.shown == ["a", "b"] and s.pending == ["z"]
    s.observe(["a", "b", "z"])                          # same keys, rewritten text: no change
    assert s.shown == ["a", "b"]
    s.reveal("10:00")
    s.observe(["z", "b", "a"])
    assert s.shown == ["a", "b", "z"]


def test_dropped_points_are_pruned_and_divider_follows():
    s = RevealState()
    s.observe(["a"])
    s.observe(["a", "b", "c"])
    s.reveal("09:00")
    s.observe(["a", "c"])                               # b is gone from the engine
    assert s.shown == ["a", "c"] and s.divider_key == "c"
    s.observe(["a"])
    assert s.divider_key is None and s.divider_label == ""


def test_new_session_resets_everything():
    s = RevealState()
    s.observe(["a"]); s.observe(["a", "b"]); s.reveal("09:00"); s.observe(["a", "b", "c"])
    s.reset()
    assert (s.shown, s.pending, s.divider_key, s.batch) == ([], [], None, [])
    s.observe(["x", "y"])
    assert s.shown == ["x", "y"] and s.pending == []


# ---- jump request ----------------------------------------------------------------------------

def token_file(tmp_path, text="secret-token\n"):
    path = tmp_path / "jump-server-token"
    path.write_text(text)
    return path


def test_jump_payload_and_prompt(tmp_path):
    calls = []
    result = jump.start_claude("[00:01] Ich\nHallo", range_label="Letzte 10 Min", token_path=token_file(tmp_path),
                               post=lambda url, payload, timeout: calls.append((url, payload, timeout)) or {"ok": True})
    assert result.ok and result.message
    (url, payload, timeout), = calls
    assert url == "http://localhost:7842/jump" and timeout > 0
    assert set(payload) == {"token", "prompt", "default", "topic", "repo"}
    assert payload["token"] == "secret-token" and payload["repo"] == "Brain" and payload["topic"]
    assert "[00:01] Ich\nHallo" in payload["prompt"] and "Letzte 10 Min" in payload["prompt"]
    assert "What do you want me to do?" in payload["prompt"] and "keine Tools" in payload["prompt"]
    assert payload["default"] == payload["prompt"]
    assert "secret-token" not in payload["prompt"]


def test_jump_missing_token_never_posts_or_raises(tmp_path):
    def boom(*_a):
        raise AssertionError("must not post without a token")
    for path in (tmp_path / "nope", token_file(tmp_path, "  \n")):
        result = jump.start_claude("text", range_label="x", token_path=path, post=boom)
        assert not result.ok and "Token" in result.message


def test_jump_server_down_or_error_reply_is_a_message(tmp_path):
    def down(*_a):
        raise ConnectionRefusedError("refused")
    result = jump.start_claude("text", range_label="x", token_path=token_file(tmp_path), post=down)
    assert not result.ok and "nicht erreichbar" in result.message
    result = jump.start_claude("text", range_label="x", token_path=token_file(tmp_path),
                               post=lambda *_a: {"ok": False, "error": "spinoff timed out"})
    assert not result.ok and "spinoff timed out" in result.message
    result = jump.start_claude("text", range_label="x", token_path=token_file(tmp_path), post=lambda *_a: "garbage")
    assert not result.ok
    assert not jump.start_claude("  ", range_label="x", token_path=token_file(tmp_path), post=lambda *_a: {"ok": True}).ok


def test_jump_default_http_call_is_not_used_when_injected(tmp_path, monkeypatch):
    monkeypatch.setattr(jump.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert jump.start_claude("t", range_label="x", token_path=token_file(tmp_path), post=lambda *_a: {"ok": True}).ok
    # and without injection a failing network is still just a message
    monkeypatch.setattr(jump.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert not jump.start_claude("t", range_label="x", token_path=token_file(tmp_path)).ok


# ---- help text -----------------------------------------------------------------------------------

def test_help_describes_three_views_and_not_the_removed_switches():
    for word in ("Meeting", "Ambient", "Intern", "Kontext für Claude", "Private Hinweise zeigen"):
        assert word in PANEL_HELP
    for gone in ("Kompakt", "Präsentation", "Kundenansicht", "Vollbild-Knopf", "Vollbild)"):
        assert gone not in PANEL_HELP
    assert "ue" not in {w for w in PANEL_HELP.split() if w in {"fuer", "ueber", "Loesung"}}


# ---- the panel itself (AppKit hosts only) -------------------------------------------------------------

def _panel(tmp_path, monkeypatch, **kwargs):
    pytest.importorskip("AppKit")
    from sidecar import live_window
    monkeypatch.setattr(live_window, "_dispatch_main", lambda callback: callback())
    copied = []
    panel = live_window.LivePanel(on_copy_clip=lambda _t: None, on_copy_card=copied.append,
                                  preferences_path=tmp_path / "prefs.json", **kwargs)
    panel.copied = copied
    return panel


def test_panel_reads_persisted_view_and_falls_back(tmp_path, monkeypatch):
    (tmp_path / "prefs.json").write_text(json.dumps({"window_mode": "internal"}))
    assert _panel(tmp_path, monkeypatch).window_mode == "internal"
    (tmp_path / "prefs.json").write_text(json.dumps({"window_mode": "presentation"}))
    assert _panel(tmp_path, monkeypatch).window_mode == "meeting"
    panel = _panel(tmp_path, monkeypatch)
    panel.set_window_mode("ambient")
    assert read_window_mode(tmp_path / "prefs.json") == "ambient"
    with pytest.raises(ValueError):
        panel.set_window_mode("compact")


def test_panel_controls_per_view(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.show()
        shown = lambda *views: all(v.isHidden() is False for v in views)
        hidden = lambda *views: all(v.isHidden() is True for v in views)
        assert shown(panel._window_control, panel._record_button, panel.cards_scroll, panel._recent_label)
        assert hidden(panel._ctx_claude, panel._transcript_scroll, panel._help_button, panel._audience_check)
        panel.set_window_mode("ambient")
        assert shown(panel._ctx_claude, panel._ctx_copy, panel._ctx_range, panel.cards_scroll)
        assert hidden(panel._recent_label, panel._transcript_scroll, panel._help_button, panel.topic_field)
        panel.set_window_mode("internal")
        assert shown(panel._transcript_scroll, panel._help_button, panel._audience_check, panel.topic_field,
                     panel.speakers_field, panel._copy_transcript_button)
        assert hidden(panel._ctx_claude, panel._recent_label)
        assert panel.view_mode in {"narrow", "wide"}
    finally:
        panel.close()


def _lines(n):
    return tuple(LiveLine(i * 10.0, i * 10.0 + 5, "Ich", f"Satz {i}") for i in range(n))


def test_panel_holds_new_points_back_until_the_marker_is_clicked(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.show()
        panel.apply_update(_lines(2), [card("a", start=1), card("b", "decision", start=2)], "")
        assert keys(panel.rendered_cards) == ["a", "b"] and panel._marker_button.isHidden() is True
        rewritten = card("a", text="rewritten", start=1)
        panel.apply_update(_lines(3), [rewritten, card("b", "decision", start=2), card("c", "action", start=3)], "")
        assert keys(panel.rendered_cards) == ["a", "b"] and panel.pending_count == 1
        assert panel._marker_button.isHidden() is False
        assert panel.rendered_cards[0].clean_text == "rewritten"            # same key, newest text
        panel.apply_update(_lines(4), [card("b", "decision", start=2), card("a", start=1), card("c", "action", start=3),
                                       card("d", start=4)], "")
        assert keys(panel.rendered_cards) == ["a", "b"] and panel.pending_count == 2   # order did not change
        assert panel.reveal_pending() == 2
        assert keys(panel.rendered_cards) == ["a", "b", "c", "d"] and panel._marker_button.isHidden() is True
        assert panel._tracker().divider_label.startswith("Neu seit ")
        panel.reset_session()                                               # a new recording starts clean
        assert panel.rendered_cards == () and panel.pending_count == 0
    finally:
        panel.close()


def test_panel_private_cards_only_in_intern_with_the_tick(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.show()
        panel.apply_update(_lines(1), ALL_CARDS, "")
        assert keys(panel.rendered_cards) == ["r", "d", "a"]
        panel.set_window_mode("internal")
        assert "priv" not in keys(panel.displayed_cards)
        panel.set_audience("internal")
        assert "priv" in keys(panel.displayed_cards) and panel._audience_check.state() == 1
        panel.set_window_mode("ambient")
        assert panel.audience == "client" and all(c.audience != "internal" for c in panel.displayed_cards)
    finally:
        panel.close()


def test_panel_context_copy_and_claude_start_use_injected_post(tmp_path, monkeypatch):
    posted = []
    done = threading.Event()
    def post(url, payload, timeout):
        posted.append(payload)
        done.set()
        return {"ok": True}
    panel = _panel(tmp_path, monkeypatch, jump_post=post, jump_token_path=token_file(tmp_path))
    try:
        panel.show()
        panel.set_window_mode("ambient")
        panel.copy_context()
        assert panel.status_field.stringValue() == "Noch kein Transkript zum Kopieren."
        panel.apply_update(_lines(3), [], "")
        panel.set_context_range("all")
        panel.copy_context()
        deadline = time.monotonic() + 2
        while not panel.copied and time.monotonic() < deadline:
            time.sleep(0.01)
        assert panel.copied and "Satz 2" in panel.copied[0] and "Satz 0" in panel.copied[0]
        panel.start_claude()
        assert done.wait(2)
        assert posted[0]["repo"] == "Brain" and "Satz 1" in posted[0]["prompt"]
    finally:
        panel.close()


def test_panel_claude_start_failure_shows_a_message_and_never_raises(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch, jump_post=lambda *_a: (_ for _ in ()).throw(OSError("down")),
                   jump_token_path=token_file(tmp_path))
    try:
        panel.show()
        panel.set_window_mode("ambient")
        panel.apply_update(_lines(2), [], "")
        panel.start_claude()
        deadline = time.monotonic() + 2
        while "nicht erreichbar" not in str(panel.status_field.stringValue()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "nicht erreichbar" in str(panel.status_field.stringValue())
    finally:
        panel.close()
