"""One native recorder window with three views: Meeting, Ambient, Intern.

Closing hides the window. View changes never touch capture or live workers.
Only explicit callbacks may start/stop recording or copy text.
"""
from __future__ import annotations

import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from .live import LiveCard, LiveLine
from . import jump
from .panel_layout import (COMPACT_SIZE, CONTEXT_RANGES, METRICS, MINIMUM_SIZE, PANEL_HELP, VIEW_LABELS, VIEWS,
                           card_columns, cards_for_view, context_range_label, context_text, is_at_bottom,
                           latest_line_text, panel_mode, scroll_offset_after_update, source_ranges, split_columns,
                           transcript_content, transcript_copy_text, view_layout, visible_controls)
from .recorder_support import PREFERENCE_FILENAME, read_window_mode, write_window_mode
from .reveal import RevealState
from .transcript import format_timestamp

NONACTIVATING_PANEL_MASK = 1 << 7
FLOATING_WINDOW_LEVEL = 3
DEFAULT_PREFERENCES_PATH = Path.home() / "Documents" / "MeetingRecorder" / PREFERENCE_FILENAME

try:
    import AppKit as A
    import Quartz  # Register CoreGraphics metadata used by CALayer colours.
    import objc
    from Foundation import NSObject, NSMakeRect, NSMakeSize, NSMakeRange, NSString, NSNotificationCenter
except ImportError:  # Geometry/help remain importable on non-macOS test hosts.
    A = None
    NSObject = object


def _constant(module: object, modern: str, legacy: str, fallback: int) -> int:
    return int(getattr(module, modern, getattr(module, legacy, fallback)))


def panel_style_mask(appkit: object) -> int:
    return (
        _constant(appkit, "NSWindowStyleMaskTitled", "NSTitledWindowMask", 1)
        | _constant(appkit, "NSWindowStyleMaskClosable", "NSClosableWindowMask", 2)
        | _constant(appkit, "NSWindowStyleMaskMiniaturizable", "NSMiniaturizableWindowMask", 4)
        | _constant(appkit, "NSWindowStyleMaskResizable", "NSResizableWindowMask", 8)
        | _constant(appkit, "NSWindowStyleMaskNonactivatingPanel", "NSNonactivatingPanelMask", NONACTIVATING_PANEL_MASK)
    )


def panel_collection_behavior(appkit: object) -> int:
    return (
        _constant(appkit, "NSWindowCollectionBehaviorCanJoinAllSpaces", "", 1)
        | _constant(appkit, "NSWindowCollectionBehaviorStationary", "", 1 << 4)
        | _constant(appkit, "NSWindowCollectionBehaviorFullScreenAuxiliary", "", 1 << 8)
    )


def copy_to_clipboard(text: str) -> None:
    result = subprocess.run(["pbcopy"], input=text, text=True, capture_output=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "pbcopy failed")


def _dispatch_main(callback: Callable[[], None]) -> None:
    from PyObjCTools import AppHelper
    if threading.current_thread() is threading.main_thread():
        callback()
    else:
        AppHelper.callAfter(callback)


def _color(hex_value: str, alpha: float = 1.0):
    rgb = [int(hex_value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return A.NSColor.colorWithSRGBRed_green_blue_alpha_(*rgb, alpha)


def _font(size: float, *, display=False, utility=False, bold=False):
    if display:
        return A.NSFont.fontWithName_size_("AvenirNext-DemiBold", size) or A.NSFont.boldSystemFontOfSize_(size)
    if utility:
        return A.NSFont.monospacedSystemFontOfSize_weight_(size, A.NSFontWeightRegular)
    return A.NSFont.systemFontOfSize_weight_(size, A.NSFontWeightSemibold if bold else A.NSFontWeightRegular)


def _text_height(text: str, width: float, font) -> float:
    # Measure with the same NSTextFieldCell that actually renders these labels.
    # NSString bounding boxes undercount multiline field padding/newlines.
    cell = A.NSTextFieldCell.alloc().initTextCell_(text or " ")
    cell.setFont_(font)
    cell.setWraps_(True)
    cell.setScrollable_(False)
    cell.setLineBreakMode_(A.NSLineBreakByWordWrapping)
    measured = cell.cellSizeForBounds_(NSMakeRect(0, 0, max(1, width - 4), 100000))
    return max(22, math.ceil(measured.height) + 10)



class _TopView(A.NSView if A is not None else object):
    def isFlipped(self):
        return True


class _FollowScroll(A.NSScrollView if A is not None else object):
    """A scroll view that tells its panel when the reader moved it."""
    on_user_scroll = None

    def scrollWheel_(self, event):
        objc.super(_FollowScroll, self).scrollWheel_(event)
        if self.on_user_scroll is not None:
            self.on_user_scroll()


class _PanelTarget(NSObject):
    panel = None

    def changeWindowMode_(self, sender):
        self.panel.set_window_mode(VIEWS[max(0, min(len(VIEWS) - 1, int(sender.selectedSegment())))])

    def jumpLatest_(self, _sender):
        self.panel.jump_to_latest("transcript")

    def revealPending_(self, _sender):
        self.panel.reveal_pending()

    def liveScrolled_(self, _notification):
        self.panel._sync_follow()

    def copyClip_(self, _sender):
        self.panel.copy_clip()

    def saveSpeakers_(self, _sender):
        self.panel.save_speakers()

    def copyTranscript_(self, _sender):
        self.panel.copy_transcript()

    def copyCard_(self, sender):
        self.panel.copy_card(int(sender.tag()))

    def showSource_(self, sender):
        self.panel.toggle_source(int(sender.tag()))

    def togglePrivate_(self, sender):
        self.panel.set_audience("internal" if int(sender.state()) else "client")

    def changeContextRange_(self, sender):
        self.panel.set_context_range(CONTEXT_RANGES[max(0, int(sender.indexOfSelectedItem()))][0])

    def copyContext_(self, _sender):
        self.panel.copy_context()

    def startClaude_(self, _sender):
        self.panel.start_claude()

    def toggleRecording_(self, _sender):
        self.panel.toggle_recording()

    def toggleHelp_(self, _sender):
        self.panel.show_help(not self.panel._help_open)

    def windowDidResize_(self, _notification):
        self.panel._layout()

    def windowShouldZoom_toFrame_(self, _window, _frame):
        # The green button opens native full screen instead of a plain zoom.
        self.panel.toggle_fullscreen()
        return False

    def windowDidEnterFullScreen_(self, _notification):
        self.panel._fullscreen = True
        self.panel._layout()

    def windowDidExitFullScreen_(self, _notification):
        self.panel._fullscreen = False
        self.panel.panel.setCollectionBehavior_(panel_collection_behavior(A))
        self.panel.panel.setLevel_(FLOATING_WINDOW_LEVEL)
        self.panel._layout()

    def windowDidFailToEnterFullScreen_(self, _notification):
        self.windowDidExitFullScreen_(_notification)

    def windowShouldClose_(self, _sender):
        self.panel.hide()
        return False


KIND_LABELS = {"request": "ANFRAGE", "decision": "ENTSCHEIDUNG", "action": "NÄCHSTER SCHRITT",
               "suggestion_question": "KI-FRAGE", "suggestion_idea": "KI-IDEE"}
KIND_COLORS = {"request": "3A5D7C", "decision": "7A5A12", "action": "2F6B4F"}
# View-specific names of the quiet text above the list.
CARDS_LABELS = {"ambient": "THEMEN BISHER", "internal": "ALLE KARTEN"}
RECORDING_LABELS = {"idle": "Bereit", "recording": "● Läuft", "stopped": "Beendet", "preview": "Vorschau"}


class LivePanel:
    def __init__(self, *, on_copy_clip: Callable[[str], None], on_copy_card: Callable[[str], None],
                 on_toggle_recording: Callable[[], None] | None = None,
                 preferences_path: Path | None = None,
                 jump_post: Callable[[str, dict, float], dict] | None = None,
                 jump_token_path: Path | None = None):
        self._preferences_path = Path(preferences_path) if preferences_path else DEFAULT_PREFERENCES_PATH
        self.window_mode = read_window_mode(self._preferences_path)
        # Only the transcript follows the end; points enter the list when the reader asks.
        self._follow = {"transcript": True}
        self._jump_post = jump_post
        self._jump_token_path = jump_token_path
        self._context_range = CONTEXT_RANGES[0][0]
        self._notice: tuple[str, float] | None = None
        self._reveals: dict[tuple[str, bool], RevealState] = {}
        self._on_copy_clip = on_copy_clip
        self._on_copy_card = on_copy_card
        self._on_toggle_recording = on_toggle_recording
        self._on_speakers = None
        self._speakers_text = ""
        self._cards: tuple[LiveCard, ...] = ()
        self.audience = "client"
        self._on_audience = None
        self._highlighted_key = None
        self._session_generation = 0
        self._on_diagnostics = None
        self._lines: tuple[LiveLine, ...] = ()
        self._status = ""
        self._recording_state = "idle"
        self._built = False
        self._closed = False
        self._fullscreen = False
        self._help_open = False
        self._expanded_keys: set[str] = set()
        self._view_mode = "narrow"
        self._laying_out = False
        self.panel = None
        self.transcript_text = None
        self.cards_scroll = None
        self.cards_document = None
        self.status_field = None
        self.topic_field = None
        self.speakers_field = None
        self._speakers_unconfirmed = False
        self._target = None
        self._card_rows = []
        self._rendered: tuple[LiveCard, ...] = ()
        self._divider_y = None

    @property
    def view_mode(self):
        """'narrow', 'wide' or 'fullscreen' (see panel_layout.panel_mode)."""
        return self._view_mode

    # ---- calm rule: which points are on screen -------------------------------

    def _tracker(self, view: str | None = None, private: bool | None = None) -> RevealState:
        view = view or self.window_mode
        private = (self.audience == "internal" and view == "internal") if private is None else private
        return self._reveals.setdefault((view, bool(private) and view == "internal"), RevealState())

    def _observe_all(self) -> None:
        """Every view keeps its own bookkeeping, so switching view never moves or hides points."""
        for view, private in (("meeting", False), ("ambient", False), ("internal", False), ("internal", True)):
            self._tracker(view, private).observe(c.key for c in cards_for_view(self._cards, view, private))

    @property
    def displayed_cards(self):
        """Every point this view may show, including those still held back."""
        return cards_for_view(self._cards, self.window_mode, self.audience == "internal")

    @property
    def rendered_cards(self) -> tuple[LiveCard, ...]:
        """The points actually on screen, in the order they entered."""
        by_key = {card.key: card for card in self.displayed_cards}
        return tuple(by_key[key] for key in self._tracker().shown if key in by_key)

    @property
    def pending_count(self) -> int:
        return self._tracker().pending_count

    def reveal_pending(self) -> int:
        entered = self._tracker().reveal(time.strftime("%H:%M"))
        if entered and self._built:
            self._render_cards()
            self._render_state()
            self._scroll_to_divider()
        return entered

    # ---- window lifecycle ----------------------------------------------------

    def close(self) -> None:
        """Dispose a retired session's window; title-bar close uses hide instead."""
        self._closed = True
        panel, self.panel = self.panel, None
        if panel is not None:
            def dispose():
                panel.setDelegate_(None)
                panel.orderOut_(None)
                panel.close()
            _dispatch_main(dispose)

    def hide(self) -> None:
        if self.panel is not None and not self._closed:
            _dispatch_main(lambda: self.panel.orderOut_(None) if self.panel is not None else None)

    def order_front(self) -> None:
        if self.panel is None or self._closed:
            return
        def reveal():
            if self.panel.isMiniaturized():
                self.panel.deminiaturize_(None)
            self.panel.orderFrontRegardless()
        _dispatch_main(reveal)

    def show(self) -> None:
        if self._closed:
            return
        if self._built:
            self.order_front()
            return
        if A is None:
            raise RuntimeError("The recorder window requires macOS AppKit")
        A.NSApplication.sharedApplication()
        width, height = COMPACT_SIZE
        panel = A.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), panel_style_mask(A), A.NSBackingStoreBuffered, False)
        panel.setTitle_("MeetingRecorder")
        panel.setFloatingPanel_(True)
        panel.setLevel_(FLOATING_WINDOW_LEVEL)
        panel.setHidesOnDeactivate_(False)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setWorksWhenModal_(False)
        panel.setCollectionBehavior_(panel_collection_behavior(A))
        panel.setReleasedWhenClosed_(False)
        panel.setContentMinSize_(NSMakeSize(*MINIMUM_SIZE))
        panel.setAppearance_(A.NSAppearance.appearanceNamed_(A.NSAppearanceNameAqua))
        panel.setBackgroundColor_(_color("F7F8F4"))
        self.panel = panel
        screen = A.NSScreen.mainScreen()
        if screen is not None:
            visible = screen.visibleFrame()
            panel.setFrame_display_(NSMakeRect(
                visible.origin.x + visible.size.width - width - 20,
                visible.origin.y + max(16, visible.size.height - height - 20), width, height), False)
        self._target = _PanelTarget.alloc().init()
        self._target.panel = self
        panel.setDelegate_(self._target)
        content = panel.contentView()
        self._surface = A.NSView.alloc().initWithFrame_(content.bounds())
        self._surface.setWantsLayer_(True)
        self._surface.layer().setBackgroundColor_(_color("F9FAF6", 0.97).CGColor())
        content.addSubview_(self._surface)

        self._rail = A.NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
        self._rail.setWantsLayer_(True)
        self._rail.layer().setBackgroundColor_(_color("EEF0EB").CGColor())
        self._rail.layer().setCornerRadius_(14)
        self._state_label = self._label("", 11, utility=True)
        self._window_control = A.NSSegmentedControl.alloc().initWithFrame_(NSMakeRect(0, 0, 216, 28))
        self._window_control.setSegmentCount_(len(VIEWS))
        for i, view in enumerate(VIEWS):
            self._window_control.setLabel_forSegment_(VIEW_LABELS[view], i)
        self._window_control.setTarget_(self._target)
        self._window_control.setAction_("changeWindowMode:")
        self._window_control.setToolTip_("Meeting: ruhig, zum Teilen. Ambient: läuft nebenher, Kontext für Claude. Intern: alles, mit Werkzeugen.")
        self._help_button = self._button("?", "toggleHelp:", "Die Recorder-Funktionen kurz erklärt.")
        self._transcript_label = self._label("TRANSKRIPT", 11, utility=True)
        self._copy_transcript_button = self._button("Transkript kopieren", "copyTranscript:", "Kopiert das ganze bisherige Transkript, auch nach dem Stoppen.")
        self._copy_transcript_button.setFont_(_font(11))
        self._cards_label = self._label("ALLE KARTEN", 11, utility=True)
        self._audience_check = A.NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 22))
        self._audience_check.setButtonType_(_constant(A, "NSButtonTypeSwitch", "NSSwitchButton", 3))
        self._audience_check.setTitle_("Private Hinweise zeigen")
        self._audience_check.setFont_(_font(12))
        self._audience_check.setTarget_(self._target)
        self._audience_check.setAction_("togglePrivate:")
        self._audience_check.setToolTip_("Zeigt zusätzlich private Beratungshinweise. Meeting und Ambient zeigen sie nie.")
        self._transcript_scroll, self.transcript_text = self._scroll_text()
        self.cards_scroll = self._scroll()
        self._jump_buttons = {}
        button = self._button("↓ Zum Neuesten", "jumpLatest:", "Springt zum neuesten Eintrag und folgt wieder automatisch.")
        button.setFont_(_font(11))
        button.setHidden_(True)
        self._jump_buttons["transcript"] = button
        self._transcript_scroll.on_user_scroll = self._sync_follow
        self._transcript_scroll.setPostsBoundsChangedNotifications_(False)
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self._target, "liveScrolled:", "NSScrollViewDidLiveScrollNotification", self._transcript_scroll)
        self.cards_document = _TopView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
        self.cards_scroll.setDocumentView_(self.cards_document)
        self._help_scroll, self._help_text = self._scroll_text()
        self._help_text.setString_(PANEL_HELP)
        self._help_text.setFont_(_font(14))
        self.status_field = self._label("", 11, wrap=True)
        self.status_field.setTextColor_(_color("59666A"))
        self.speakers_field = A.NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 26))
        self.speakers_field.setPlaceholderString_("Teilnehmende")
        self.speakers_field.setFont_(_font(13))
        self.speakers_field.setStringValue_(self._speakers_text)
        self.speakers_field.cell().setSendsActionOnEndEditing_(True)
        self.speakers_field.setTarget_(self._target)
        self.speakers_field.setAction_("saveSpeakers:")
        self.speakers_field.setToolTip_("Wer spricht in diesem Meeting? Enter speichert. Das finale Transkript ordnet die Stimmen diesen Namen zu.")
        self._speakers_label = self._label("TEILNEHMENDE", 11, utility=True)
        self._topic_label = self._label("THEMENAUSZUG", 11, utility=True)
        self.topic_field = A.NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 26))
        self.topic_field.setPlaceholderString_("Thema für Auszug")
        self.topic_field.setFont_(_font(13))
        self.topic_field.setToolTip_("Kopiert passende Stellen aus dem Transkript dieser Aufnahme.")
        self._clip_button = self._button("Auszug kopieren", "copyClip:", "Sucht passende Textstellen zu deinem Thema und kopiert sie.")
        self._record_button = self._button("Start", "toggleRecording:", "Start oder Stop ist immer eine ausdrückliche Aktion. Vorher alle Beteiligten informieren.")
        self._marker_button = A.NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 190, 32))
        self._marker_button.setBordered_(False)
        self._marker_button.setWantsLayer_(True)
        self._marker_button.layer().setBackgroundColor_(_color("22323B").CGColor())
        self._marker_button.layer().setCornerRadius_(16)
        self._marker_button.setTarget_(self._target)
        self._marker_button.setAction_("revealPending:")
        self._marker_button.setToolTip_("Holt die neuen Punkte in die Liste. Bis dahin bleibt alles, wie es ist.")
        self._recent_label = self._label("", 12.5)
        self._recent_label.setTextColor_(_color("8A9496"))
        self._recent_label.cell().setLineBreakMode_(A.NSLineBreakByTruncatingTail)
        self._ctx_label = self._label("KONTEXT FÜR CLAUDE", 11, utility=True)
        self._ctx_range = A.NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(0, 0, 140, 28), False)
        self._ctx_range.addItemsWithTitles_([title for _key, title in CONTEXT_RANGES])
        self._ctx_range.setFont_(_font(12))
        self._ctx_range.setTarget_(self._target)
        self._ctx_range.setAction_("changeContextRange:")
        self._ctx_range.setToolTip_("Welcher Teil des Transkripts als Kontext gilt.")
        self._ctx_copy = self._button("Kopieren", "copyContext:", "Kopiert den gewählten Transkript-Ausschnitt in die Zwischenablage.")
        self._ctx_claude = self._button("Claude starten", "startClaude:", "Öffnet eine Claude-Sitzung mit dem Ausschnitt als Hintergrund. Sie fragt zuerst, was du willst.")
        for view in (self._rail, self._state_label, self._window_control, self._help_button, self._audience_check,
                     self._transcript_label, self._copy_transcript_button, self._cards_label, self._transcript_scroll,
                     self.cards_scroll, self._help_scroll, self.status_field, self._speakers_label, self.speakers_field,
                     self._topic_label, self.topic_field, self._clip_button, self._record_button, self._recent_label,
                     self._ctx_label, self._ctx_range, self._ctx_copy, self._ctx_claude, self._marker_button,
                     *self._jump_buttons.values()):
            self._surface.addSubview_(view)
        # Which views belong to which named control (see panel_layout.visible_controls).
        self._control_views = {
            "window_mode": (self._window_control,), "state": (self._state_label,), "record": (self._record_button,),
            "help": (self._help_button,), "audience": (self._audience_check,), "cards": (self.cards_scroll,),
            "cards_label": (self._cards_label,), "transcript": (self._transcript_scroll,),
            "transcript_label": (self._transcript_label,), "copy_transcript": (self._copy_transcript_button,),
            "speakers": (self.speakers_field,), "speakers_label": (self._speakers_label,),
            "topic": (self.topic_field,), "topic_label": (self._topic_label,), "clip": (self._clip_button,),
            "status": (self.status_field,), "recent": (self._recent_label,), "rail": (self._rail,),
            "ctx_label": (self._ctx_label,), "ctx_range": (self._ctx_range,), "ctx_copy": (self._ctx_copy,),
            "ctx_claude": (self._ctx_claude,), "marker": (self._marker_button,)}
        self._built = True
        self._observe_all()
        self._layout()
        self._render_transcript()
        self._render_state()
        # Opening a panel must not steal keyboard focus from the user's work.
        panel.orderFrontRegardless()

    # ---- layout --------------------------------------------------------------

    def _layout(self) -> None:
        if not self._built or self._closed or self._laying_out:
            return
        self._laying_out = True
        try:
            bounds = self.panel.contentView().bounds()
            width, height = bounds.size.width, bounds.size.height
            view = self.window_mode
            if view != "internal":
                self._help_open = False
            self._view_mode = panel_mode(width, height, fullscreen=self._fullscreen)
            metrics = METRICS[self._view_mode]
            self._surface.setFrame_(NSMakeRect(0, 0, width, height))
            rects = view_layout(view, width, height, fullscreen=self._fullscreen)
            allowed = visible_controls(view, fullscreen=self._fullscreen)
            help_cover = self._help_open and view == "internal"
            covered = {"transcript", "transcript_label", "copy_transcript", "cards", "cards_label", "audience", "marker"}
            for name, views in self._control_views.items():
                visible = name in allowed and name in rects and not (help_cover and name in covered)
                for control in views:
                    control.setHidden_(not visible)
                    if name in rects:
                        control.setFrame_(NSMakeRect(*rects[name]))
            self._help_scroll.setHidden_(not help_cover)
            if "help_body" in rects:
                self._help_scroll.setFrame_(NSMakeRect(*rects["help_body"]))
            self._window_control.setSelectedSegment_(VIEWS.index(view))
            self._audience_check.setState_(1 if self.audience == "internal" else 0)
            self._state_label.setAlignment_(A.NSTextAlignmentRight)
            self.transcript_text.setFont_(_font(metrics["transcript"]))
            self.transcript_text.setTextColor_(_color("22323B" if self._lines else "6D777A"))
            self._fit_text_view(self._transcript_scroll, self.transcript_text)
            self._fit_text_view(self._help_scroll, self._help_text)
            self._place_jump_buttons()
            self._render_cards()
            self._render_state()
        finally:
            self._laying_out = False

    def _place_jump_buttons(self) -> None:
        frame = self._transcript_scroll.frame()
        self._jump_buttons["transcript"].setFrame_(
            NSMakeRect(frame.origin.x + frame.size.width - 142, frame.origin.y + 6, 134, 26))
        self._update_jump_buttons()

    def _update_jump_buttons(self) -> None:
        shown = "transcript" in visible_controls(self.window_mode, fullscreen=self._fullscreen) and "jump" in visible_controls(self.window_mode)
        for which, button in self._jump_buttons.items():
            button.setHidden_(self._follow[which] or not shown or (self._help_open and self.window_mode == "internal")
                              or self._transcript_scroll.isHidden())

    def _scroll_geometry(self, which: str):
        scroll = self._transcript_scroll if which == "transcript" else self.cards_scroll
        clip = scroll.contentView()
        document = scroll.documentView()
        return scroll, clip, float(clip.bounds().origin.y), float(clip.bounds().size.height), float(document.frame().size.height)

    def _sync_follow(self) -> None:
        """Called only for reader-made scrolls: following resumes at the end, stops anywhere else."""
        if not self._built:
            return
        _scroll, _clip, offset, viewport, content = self._scroll_geometry("transcript")
        self._follow["transcript"] = is_at_bottom(offset, viewport, content)
        self._update_jump_buttons()

    def _apply_scroll(self, which: str, old_offset: float) -> None:
        scroll, clip, _offset, viewport, content = self._scroll_geometry(which)
        target = scroll_offset_after_update(self._follow.get(which, False), old_offset, viewport, content)
        clip.scrollToPoint_((0, target))
        scroll.reflectScrolledClipView_(clip)
        self._update_jump_buttons()

    def _scroll_to_divider(self) -> None:
        """After the reader opened the held-back points, show where the new ones start."""
        y = getattr(self, "_divider_y", None)
        if y is None:
            return
        scroll, clip, _offset, viewport, content = self._scroll_geometry("cards")
        clip.scrollToPoint_((0, max(0.0, min(y - 8, max(0.0, content - viewport)))))
        scroll.reflectScrolledClipView_(clip)

    def jump_to_latest(self, which: str) -> None:
        if which not in self._follow or not self._built:
            return
        self._follow[which] = True
        self._apply_scroll(which, 0.0)

    def set_window_mode(self, mode: str, *, persist: bool = True) -> None:
        if mode not in VIEWS:
            raise ValueError("unknown window mode")
        self.window_mode = mode
        if mode != "internal":
            self._help_open = False
            if self.audience != "client":
                self.set_audience("client")  # private hints only ever show in Intern
        if persist:
            try:
                write_window_mode(self._preferences_path, mode)
            except OSError:
                pass
        if self._built:
            self._layout()

    def toggle_fullscreen(self) -> None:
        """Native full screen; reached through the green window button."""
        if self.panel is None or self._closed:
            return
        if not self._fullscreen:
            self.panel.setCollectionBehavior_(A.NSWindowCollectionBehaviorFullScreenPrimary)
            self.panel.setLevel_(A.NSNormalWindowLevel)
            # Focus changes only on this explicit user action, never at Start.
            self.panel.makeKeyAndOrderFront_(None)
        self.panel.toggleFullScreen_(None)

    def show_help(self, visible: bool = True) -> None:
        self._help_open = visible and self.window_mode == "internal"
        self.show()
        self._layout()
        self.order_front()

    def set_recording_state(self, state: str) -> None:
        if state not in {"idle", "recording", "stopped", "preview"}:
            raise ValueError("unknown recording state")
        def update():
            self._recording_state = state
            if self._built:
                self._render_state()
                if not self._lines:
                    self._render_transcript()
        _dispatch_main(update)

    def reset_session(self) -> None:
        """Only a newly started recording resets content; view controls never do."""
        self._session_generation += 1
        self.audience = "client"
        self._on_audience = None
        self._reveals.clear()
        self._notice = None
        if self._built:
            self._audience_check.setState_(0)
        self._expanded_keys.clear()
        self._highlighted_key = None
        self._on_speakers = None
        self._speakers_unconfirmed = False
        self.set_speakers("")
        self.apply_update((), (), "")

    def set_diagnostics_callback(self, callback) -> None:
        self._on_diagnostics = callback

    def set_audience_callback(self, callback) -> None:
        self._on_audience = callback

    def set_audience(self, audience: str) -> None:
        if audience not in {"client", "internal"}:
            raise ValueError("unknown audience")
        self.audience = audience
        self._highlighted_key = None
        self._expanded_keys.intersection_update(card.key for card in self.rendered_cards)
        if self._built:
            self._audience_check.setState_(1 if audience == "internal" else 0)
            self._render_transcript()
            self._render_cards()
        if self._on_audience is not None:
            self._on_audience(audience)

    def toggle_recording(self) -> None:
        if self._on_toggle_recording is not None:
            self._on_toggle_recording()

    # ---- actions ---------------------------------------------------------------

    def _set_notice(self, text: str, seconds: float = 10.0) -> None:
        def update():
            self._notice = (text, time.monotonic() + seconds)
            if self._built:
                self._render_state()
        _dispatch_main(update)

    def copy_clip(self) -> None:
        topic = str(self.topic_field.stringValue() or "") if self.topic_field is not None else ""
        threading.Thread(target=self._on_copy_clip, args=(topic,), name="meeting-sidecar-live-clip", daemon=True).start()

    def set_context_range(self, range_key: str) -> None:
        if range_key in dict(CONTEXT_RANGES):
            self._context_range = range_key

    def copy_context(self) -> None:
        """Copy the chosen transcript range with the existing clipboard path."""
        text = context_text(self._lines, self._context_range)
        if not text:
            self._set_notice("Noch kein Transkript zum Kopieren.")
            return
        threading.Thread(target=self._on_copy_card, args=(text,), name="meeting-sidecar-live-context", daemon=True).start()
        self._set_notice(f"Kontext kopiert ({context_range_label(self._context_range)}).")

    def start_claude(self) -> None:
        """Open a Claude session with the range as background. Off the main thread, never raises."""
        text = context_text(self._lines, self._context_range)
        label = context_range_label(self._context_range)
        if not text:
            self._set_notice("Noch kein Transkript für Claude.")
            return
        self._set_notice("Claude wird gestartet …", 30.0)

        def work():
            try:
                result = jump.start_claude(text, range_label=label, token_path=self._jump_token_path, post=self._jump_post)
                message = result.message
            except Exception:  # start_claude already catches; this keeps the thread silent no matter what
                message = "Claude nicht gestartet."
            self._set_notice(message)
        threading.Thread(target=work, name="meeting-sidecar-live-claude", daemon=True).start()

    def set_speakers_callback(self, callback) -> None:
        self._on_speakers = callback

    def set_speakers(self, text: str, *, proposed: bool = False) -> None:
        """Show names without saving them. A proposal (from the calendar) is saved on Enter."""
        def update():
            self._speakers_text = text
            self._speakers_unconfirmed = proposed
            if self._built:
                self.speakers_field.setStringValue_(text)
        _dispatch_main(update)

    def save_speakers(self) -> None:
        text = str(self.speakers_field.stringValue() or "") if self.speakers_field is not None else ""
        if self._on_speakers is None or (text == self._speakers_text and not self._speakers_unconfirmed):
            return
        try:
            self._status = self._on_speakers(text) or self._status
            self._speakers_text, self._speakers_unconfirmed = text, False
        except Exception as exc:
            self._status = f"Teilnehmende nicht gespeichert ({type(exc).__name__})."
        if self._built:
            self._render_state()

    def copy_transcript(self) -> None:
        """Copy every line this window holds; works during and after a recording."""
        text = transcript_copy_text(self._lines)
        if not text:
            return
        threading.Thread(target=self._on_copy_card, args=(text,), name="meeting-sidecar-live-transcript", daemon=True).start()
        self._set_notice(f"Transkript kopiert ({len(self._lines)} Abschnitte).")

    def copy_card(self, index: int) -> None:
        rendered = self.rendered_cards
        if 0 <= index < len(rendered):
            text = rendered[index].copy_text
            threading.Thread(target=self._on_copy_card, args=(text,), name="meeting-sidecar-live-card", daemon=True).start()

    def toggle_source(self, index: int) -> None:
        rendered = self.rendered_cards
        if not 0 <= index < len(rendered):
            return
        key = rendered[index].key
        if key in self._expanded_keys:
            self._expanded_keys.remove(key)
        else:
            self._expanded_keys.add(key)
        self._highlighted_key = key if key in self._expanded_keys else None
        self._render_transcript()
        self._render_cards()
        selected = rendered[index]
        ranges = source_ranges(self._lines, selected.spans) if key in self._expanded_keys else []
        if ranges:
            self._follow["transcript"] = False
            self.transcript_text.scrollRangeToVisible_(NSMakeRange(*ranges[0]))
            self._update_jump_buttons()

    def apply_update(self, lines: Sequence[LiveLine], cards: Sequence[LiveCard], status: str) -> None:
        snapshot_lines, snapshot_cards = tuple(lines), tuple(cards)
        generation = self._session_generation
        queued_at = time.monotonic()
        def render():
            if self._closed or generation != self._session_generation:
                return
            self._lines, self._cards, self._status = snapshot_lines, snapshot_cards, status
            self._observe_all()
            self._expanded_keys.intersection_update(card.key for card in self.rendered_cards)
            if self._built:
                self._render_transcript()
                self._render_cards()
                self._render_state()
                if self._on_diagnostics is not None:
                    self._on_diagnostics({"stage": "ui_publish", "elapsed_seconds": time.monotonic() - queued_at,
                                          "line_count": len(snapshot_lines), "card_count": len(self.displayed_cards)})
        _dispatch_main(render)

    # ---- rendering ---------------------------------------------------------------

    def _render_state(self) -> None:
        if not self._built:
            return
        self._state_label.setStringValue_(RECORDING_LABELS[self._recording_state])
        self._state_label.setTextColor_(_color("A23F32" if self._recording_state == "recording" else "59666A"))
        self._record_button.setTitle_("Stopp" if self._recording_state == "recording" else "Start")
        self._record_button.setEnabled_(self._on_toggle_recording is not None and self._recording_state != "preview")
        self._copy_transcript_button.setEnabled_(bool(self._lines))
        has_committed = any(not line.provisional for line in self._lines)
        self._clip_button.setEnabled_(has_committed)
        self.topic_field.setEnabled_(has_committed)
        self._ctx_copy.setEnabled_(bool(self._lines))
        self._ctx_claude.setEnabled_(bool(self._lines))
        self._ctx_range.selectItemAtIndex_([key for key, _t in CONTEXT_RANGES].index(self._context_range))
        self._cards_label.setStringValue_(self._cards_label_text())
        self._transcript_label.setStringValue_("TRANSKRIPT · IM HINTERGRUND" if self.window_mode == "ambient" else "TRANSKRIPT")
        self._render_recent()
        if self.window_mode == "ambient":
            hint = "Beispieldaten · Mikrofon ist aus." if self._recording_state == "preview" else "Läuft nebenher."
        elif self._recording_state == "preview":
            hint = "Beispieldaten · Mikrofon ist aus."
        else:
            hint = "Vorher alle Beteiligten informieren. Schliessen versteckt nur dieses Fenster."
        notice = self._notice[0] if self._notice and time.monotonic() < self._notice[1] else None
        self.status_field.setStringValue_(notice or self._status or hint)
        self._update_marker()

    def _cards_label_text(self) -> str:
        base = CARDS_LABELS.get(self.window_mode, "")
        return f"{base} · {len(self.rendered_cards)}" if base else ""

    def _update_marker(self) -> None:
        text = self._tracker().marker_text
        allowed = "marker" in visible_controls(self.window_mode, fullscreen=self._fullscreen)
        self._marker_button.setHidden_(text is None or not allowed or (self._help_open and self.window_mode == "internal"))
        if text:
            title = A.NSAttributedString.alloc().initWithString_attributes_(
                text, {A.NSForegroundColorAttributeName: _color("FFFFFF"), A.NSFontAttributeName: _font(12.5, bold=True)})
            self._marker_button.setAttributedTitle_(title)

    def _render_transcript(self) -> None:
        old_offset = float(self._transcript_scroll.contentView().bounds().origin.y)
        value, _offsets = transcript_content(self._lines)
        if not value:
            value = ("Sobald du sprichst, erscheint hier das Transkript." if self._recording_state == "recording"
                     else "Starte die Aufnahme, wenn alle Beteiligten einverstanden sind.")
        self.transcript_text.setString_(value)
        quiet = self.window_mode == "ambient"
        self.transcript_text.setTextColor_(_color(("8A9496" if quiet else "22323B") if self._lines else "6D777A"))
        self._fit_text_view(self._transcript_scroll, self.transcript_text)
        selected = next((card for card in self.rendered_cards if card.key == self._highlighted_key), None)
        ranges = source_ranges(self._lines, selected.spans) if selected else []
        storage = self.transcript_text.textStorage()
        for start, length in ranges:
            storage.addAttribute_value_range_(A.NSBackgroundColorAttributeName, _color("F5E6AE"), NSMakeRange(start, length))
        # The highlight is shown once, when the reader opens a card's source
        # (toggle_source); a later update never pulls the view back to it.
        self._apply_scroll("transcript", old_offset)
        self._render_recent()

    def _render_recent(self) -> None:
        recent = latest_line_text(self._lines)
        self._recent_label.setStringValue_(f"Zuletzt   {recent}" if recent else "Zuletzt   noch nichts gehört")

    def _render_cards(self) -> None:
        if self.cards_document is None:
            return
        view = self.window_mode
        mode = self._view_mode
        metrics = METRICS[mode]
        old_origin = self.cards_scroll.contentView().bounds().origin
        for subview in list(self.cards_document.subviews()):
            subview.removeFromSuperview()
        self._card_rows = []
        self._divider_y = None
        width = float(self.cards_scroll.contentView().bounds().size.width)
        viewport_h = self.cards_scroll.contentView().bounds().size.height
        rendered = self.rendered_cards
        self._rendered = rendered
        tracker = self._tracker()
        columns = card_columns(view, mode)
        grouped = split_columns(rendered, columns)
        col_gap = metrics["gap"]
        col_w = (width - col_gap * (len(columns) - 1)) / len(columns)
        fullscreen = mode == "fullscreen"
        total = 0.0
        if not rendered:
            texts = {"meeting": "Noch nichts.\nEntscheidungen, nächste Schritte und Anfragen erscheinen hier.",
                     "ambient": "Noch nichts erfasst.\nHier erscheint eine Zeile pro Punkt.",
                     "internal": "Noch keine Karte.\nAnfragen, Entscheidungen und nächste Schritte erscheinen hier. KI-Ideen sind separat beschriftet."}
            text = self._label(texts[view], 16 if fullscreen else 14, wrap=True)
            text.setTextColor_(_color("6D777A"))
            text.setFrame_(NSMakeRect(0, 8, max(30, width), _text_height(str(text.stringValue()), width, text.font())))
            self.cards_document.addSubview_(text)
        index_of = {card.key: i for i, card in enumerate(rendered)}
        for ci, ((title, _kinds), group) in enumerate(zip(columns, grouped)):
            x = ci * (col_w + col_gap)
            y = 0.0
            if title and rendered:
                head = self._label(title, 34 if fullscreen else 15, display=fullscreen, utility=False)
                if not fullscreen:
                    head.setFont_(_font(15, bold=True))
                head_h = 48.0 if fullscreen else 26.0
                head.setFrame_(NSMakeRect(x, y, col_w, head_h))
                self.cards_document.addSubview_(head)
                y += head_h + 4
            divider_key = tracker.divider_before(card.key for card in group)
            for card in group:
                if card.key == divider_key:
                    since = self._label(tracker.divider_label.upper(), 11 if not fullscreen else 13, utility=True)
                    since.setTextColor_(_color("A23F32"))
                    since.setFrame_(NSMakeRect(x, y + 6, col_w, 20))
                    self.cards_document.addSubview_(since)
                    if self._divider_y is None or y < self._divider_y:
                        self._divider_y = y
                    y += 32
                y += self._add_point(view, mode, card, index_of[card.key], x, y, col_w, metrics) + (12 if view == "internal" else 0)
            total = max(total, y)
        self.cards_document.setFrame_(NSMakeRect(0, 0, width, max(viewport_h, total)))
        self._apply_scroll("cards", float(old_origin.y))
        self._update_marker()

    def _add_point(self, view: str, mode: str, card: LiveCard, index: int, x: float, y: float, width: float, metrics) -> float:
        """Draw one point at (x, y) in the flipped document; returns its height."""
        body = card.clean_text.strip() or card.verbatim_text
        fullscreen = mode == "fullscreen"
        if view == "meeting":
            size = metrics["point"]
            font = _font(size)
            body_h = _text_height(body, width, font)
            pad = 18 if fullscreen else 12
            rule = _TopView.alloc().initWithFrame_(NSMakeRect(x, y, width, 1))
            rule.setWantsLayer_(True)
            rule.layer().setBackgroundColor_(_color("DEE3DC").CGColor())
            kind = self._label(f"{KIND_LABELS.get(card.kind, 'KARTE')} · {format_timestamp(card.start_seconds)}",
                               13 if fullscreen else 10.5, utility=True)
            kind.setTextColor_(_color(KIND_COLORS.get(card.kind, "59666A")))
            kind.setFrame_(NSMakeRect(x, y + pad, width, 18))
            label = self._label(body, size, wrap=True)
            label.setFont_(font)
            label.setFrame_(NSMakeRect(x, y + pad + 20, width, body_h))
            for sub in (rule, kind, label):
                self.cards_document.addSubview_(sub)
            return pad + 20 + body_h + pad
        if view == "ambient":
            size = 20 if fullscreen else 14
            font = _font(size)
            time_w = 70.0 if fullscreen else 52.0
            body_h = _text_height(body, width - time_w - 8, font)
            pad = 12 if fullscreen else 8
            rule = _TopView.alloc().initWithFrame_(NSMakeRect(x, y, width, 1))
            rule.setWantsLayer_(True)
            rule.layer().setBackgroundColor_(_color("DEE3DC").CGColor())
            stamp = self._label(format_timestamp(card.start_seconds), 14 if fullscreen else 11.5, utility=True)
            stamp.setTextColor_(_color("59666A"))
            stamp.setFrame_(NSMakeRect(x, y + pad + 3, time_w, 18))
            label = self._label(body, size, wrap=True)
            label.setFont_(font)
            label.setFrame_(NSMakeRect(x + time_w + 8, y + pad - 2, width - time_w - 8, body_h))
            for sub in (rule, stamp, label):
                self.cards_document.addSubview_(sub)
            return pad * 2 + body_h - 4
        return self._add_card(card, index, x, y, width, metrics)

    def _add_card(self, card: LiveCard, index: int, x: float, y: float, width: float, metrics) -> float:
        """Intern: the full card with Bezug and Kopieren."""
        body_font = _font(metrics["card"])
        inner_w = max(80, width - 32)
        body = card.clean_text.strip() or card.verbatim_text
        body_h = _text_height(body, inner_w, body_font)
        expanded = card.key in self._expanded_keys
        source = "\n\n".join(f"[{format_timestamp(line.seconds)}] {line.speaker}\n{line.text}" for line in card.lines)
        source_h = _text_height(source, inner_w, _font(13)) if expanded else 0
        height = 82 + body_h + (source_h + 16 if expanded else 0)
        row = _TopView.alloc().initWithFrame_(NSMakeRect(x, y, width, height))
        row.setWantsLayer_(True)
        row.layer().setBackgroundColor_(_color("FFFFFF").CGColor())
        row.layer().setCornerRadius_(12)
        row.layer().setBorderWidth_(1)
        row.layer().setBorderColor_(_color("DEE3DC").CGColor())
        if card.kind.startswith("suggestion_"):
            row.layer().setBackgroundColor_(_color("EDF4F7").CGColor())
        audience_label = " · PRIVAT" if card.audience == "internal" else ""
        meta = self._label(f"{KIND_LABELS.get(card.kind, 'KARTE')}{audience_label} · {format_timestamp(card.start_seconds)}", 10, utility=True)
        meta.setFrame_(NSMakeRect(16, 12, inner_w, 20))
        label = self._label(body, 14, wrap=True)
        label.setFont_(body_font)
        label.setFrame_(NSMakeRect(16, 38, inner_w, body_h))
        source_button = self._button("Bezug ausblenden" if expanded else "Bezug im Transkript", "showSource:", "Zeigt die ausgewählten Originalstellen und markiert sie im Transkript. Bei KI-Ideen ist dies der Anlass, keine wörtliche Aussage.")
        source_button.setTag_(index)
        source_button.setFont_(_font(11))
        source_button.setFrame_(NSMakeRect(10, 42 + body_h, min(156, width - 112), 26))
        copy_button = self._button("Kopieren", "copyCard:", "Kopiert den Kartentext. KI-Ideen sind als Vorschläge gekennzeichnet.")
        copy_button.setTag_(index)
        copy_button.setFont_(_font(11))
        copy_button.setFrame_(NSMakeRect(width - 100, 42 + body_h, 88, 26))
        for sub in (meta, label, source_button, copy_button):
            row.addSubview_(sub)
        if expanded:
            original = self._label(source, 13, wrap=True)
            original.setTextColor_(_color("59666A"))
            original.setFrame_(NSMakeRect(16, 82 + body_h, inner_w, source_h))
            row.addSubview_(original)
        self.cards_document.addSubview_(row)
        self._card_rows.append(row)
        return height

    def _button(self, title: str, action: str, tip: str):
        button = A.NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 28))
        button.setTitle_(title)
        button.setFont_(_font(12))
        button.setBezelStyle_(_constant(A, "NSBezelStyleRounded", "NSRoundedBezelStyle", 1))
        button.setTarget_(self._target)
        button.setAction_(action)
        button.setToolTip_(tip)
        return button

    @staticmethod
    def _label(value: str, size: float, *, display=False, utility=False, wrap=False):
        field = A.NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 24))
        field.setEditable_(False)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setSelectable_(wrap)
        field.setStringValue_(value)
        field.setFont_(_font(size, display=display, utility=utility))
        field.setTextColor_(_color("22323B"))
        field.cell().setWraps_(wrap)
        field.cell().setScrollable_(not wrap)
        if wrap:
            field.cell().setLineBreakMode_(A.NSLineBreakByWordWrapping)
        return field

    @staticmethod
    def _scroll():
        scroll = _FollowScroll.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(A.NSNoBorder)
        scroll.setDrawsBackground_(False)
        return scroll

    @classmethod
    def _scroll_text(cls):
        scroll = cls._scroll()
        text = A.NSTextView.alloc().initWithFrame_(scroll.contentView().bounds())
        text.setEditable_(False)
        text.setRichText_(False)
        text.setFont_(_font(14))
        text.setTextColor_(_color("22323B"))
        text.setDrawsBackground_(False)
        text.setTextContainerInset_(NSMakeSize(0, 4))
        text.setVerticallyResizable_(True)
        text.setHorizontallyResizable_(False)
        text.setAutoresizingMask_(A.NSViewWidthSizable)
        text.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(text)
        return scroll, text

    @staticmethod
    def _fit_text_view(scroll, text):
        width = max(1, scroll.contentView().bounds().size.width)
        text.setFrameSize_(NSMakeSize(width, max(scroll.contentView().bounds().size.height,
                                              _text_height(str(text.string()), width - 10, text.font()) + 12)))
        text.textContainer().setContainerSize_(NSMakeSize(width, 10000000))
