"""One native recorder window: compact work surface or presentation backdrop.

Closing hides the window. Mode changes never touch capture or live workers.
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
from .panel_layout import COMPACT_SIZE, MINIMUM_SIZE, PANEL_HELP, panel_mode, surface_rect, visible_cards, transcript_content, source_ranges, transcript_copy_text, utf16_length
from .transcript import format_timestamp

NONACTIVATING_PANEL_MASK = 1 << 7
FLOATING_WINDOW_LEVEL = 3
BACKGROUND_PATH = Path(__file__).with_name("assets") / "meeting-background-lailix-logo-raised.png"

try:
    import AppKit as A
    import Quartz  # Register CoreGraphics metadata used by CALayer colours.
    from Foundation import NSObject, NSMakeRect, NSMakeSize, NSMakeRange, NSString
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


class _BackdropView(A.NSView if A is not None else object):
    picture = None

    def drawRect_(self, _dirty):
        bounds = self.bounds()
        _color("EAECE7").setFill()
        A.NSRectFill(bounds)
        if self.picture is None:
            return
        size = self.picture.size()
        scale = max(bounds.size.width / size.width, bounds.size.height / size.height)
        # Keep the robot on the left; crop the far right first on narrower screens.
        frame = NSMakeRect(0, (bounds.size.height - size.height * scale) / 2,
                           size.width * scale, size.height * scale)
        self.picture.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
            frame, NSMakeRect(0, 0, size.width, size.height), A.NSCompositingOperationSourceOver,
            1.0, False, None,
        )


class _PanelTarget(NSObject):
    panel = None

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

    def changeMode_(self, sender):
        self.panel.set_mode("compact" if sender.selectedSegment() == 0 else "presentation")

    def changeAudience_(self, sender):
        self.panel.set_audience("client" if sender.selectedSegment() == 0 else "internal")

    def toggleFullscreen_(self, _sender):
        self.panel.toggle_fullscreen()

    def toggleRecording_(self, _sender):
        self.panel.toggle_recording()

    def toggleHelp_(self, _sender):
        self.panel.show_help(not self.panel._help_open)

    def windowDidResize_(self, _notification):
        self.panel._layout()

    def windowDidEnterFullScreen_(self, _notification):
        self.panel._fullscreen = True
        self.panel._layout()

    def windowDidExitFullScreen_(self, _notification):
        self.panel._fullscreen = False
        self.panel.panel.setCollectionBehavior_(panel_collection_behavior(A))
        self.panel.panel.setLevel_(FLOATING_WINDOW_LEVEL)
        if self.panel._compact_after_fullscreen:
            self.panel._compact_after_fullscreen = False
            self.panel.set_mode("compact")
        else:
            self.panel._layout()

    def windowDidFailToEnterFullScreen_(self, _notification):
        self.windowDidExitFullScreen_(_notification)

    def windowShouldClose_(self, _sender):
        self.panel.hide()
        return False


class LivePanel:
    def __init__(self, *, on_copy_clip: Callable[[str], None], on_copy_card: Callable[[str], None],
                 on_toggle_recording: Callable[[], None] | None = None):
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
        self._compact_after_fullscreen = False
        self._compact_frame = None
        self._help_open = False
        self._expanded_keys: set[str] = set()
        self._view_mode = "compact"
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

    @property
    def view_mode(self):
        return self._view_mode

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
        self._backdrop = _BackdropView.alloc().initWithFrame_(content.bounds())
        self._backdrop.picture = A.NSImage.alloc().initWithContentsOfFile_(str(BACKGROUND_PATH))
        self._backdrop.setAutoresizingMask_(A.NSViewWidthSizable | A.NSViewHeightSizable)
        content.addSubview_(self._backdrop)
        self._surface = A.NSView.alloc().initWithFrame_(content.bounds())
        self._surface.setWantsLayer_(True)
        self._surface.layer().setBackgroundColor_(_color("F9FAF6", 0.97).CGColor())
        content.addSubview_(self._surface)

        self._title = self._label("Meeting", 26, display=True)
        self._state_label = self._label("", 11, utility=True)
        self._mode_control = A.NSSegmentedControl.alloc().initWithFrame_(NSMakeRect(0, 0, 222, 26))
        self._mode_control.setSegmentCount_(2)
        self._mode_control.setLabel_forSegment_("Kompakt", 0)
        self._mode_control.setLabel_forSegment_("Präsentation", 1)
        self._mode_control.setWidth_forSegment_(88, 0)
        self._mode_control.setWidth_forSegment_(126, 1)
        self._mode_control.setTarget_(self._target)
        self._mode_control.setAction_("changeMode:")
        self._mode_control.setToolTip_("Ändert nur die Fensteransicht. Aufnahme, Transkript und Aufgaben bleiben erhalten.")
        self._fullscreen_button = self._button("Vollbild", "toggleFullscreen:", "Öffnet oder verlässt das native macOS-Vollbild.")
        self._help_button = self._button("?", "toggleHelp:", "Die Recorder-Funktionen kurz erklärt.")
        self._transcript_label = self._label("TRANSKRIPT", 11, utility=True)
        self._copy_transcript_button = self._button("Transkript kopieren", "copyTranscript:", "Kopiert das ganze bisherige Transkript, auch nach dem Stoppen.")
        self._copy_transcript_button.setFont_(_font(11))
        self._cards_label = self._label("KARTEN & KI-IDEEN", 11, utility=True)
        self._audience_control = A.NSSegmentedControl.alloc().initWithFrame_(NSMakeRect(0, 0, 300, 26))
        self._audience_control.setSegmentCount_(2)
        for i, title in enumerate(("Kundenansicht", "Intern · privat")):
            self._audience_control.setLabel_forSegment_(title, i)
            self._audience_control.setWidth_forSegment_(145, i)
        self._audience_control.setSelectedSegment_(0)
        self._audience_control.setTarget_(self._target)
        self._audience_control.setAction_("changeAudience:")
        self._audience_control.setToolTip_("Kundenansicht blendet interne KI-Hinweise sofort aus. Unabhängig von Kompakt oder Präsentation.")
        self._transcript_scroll, self.transcript_text = self._scroll_text()
        self.cards_scroll = self._scroll()
        self.cards_document = _TopView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
        self.cards_scroll.setDocumentView_(self.cards_document)
        self._help_scroll, self._help_text = self._scroll_text()
        self._help_text.setString_(PANEL_HELP)
        self._help_text.setFont_(_font(14))
        self.status_field = self._label("", 11, wrap=True)
        self.status_field.setTextColor_(_color("59666A"))
        self.speakers_field = A.NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 26))
        self.speakers_field.setPlaceholderString_("Teilnehmende, mit Komma getrennt")
        self.speakers_field.setFont_(_font(13))
        self.speakers_field.setStringValue_(self._speakers_text)
        self.speakers_field.cell().setSendsActionOnEndEditing_(True)
        self.speakers_field.setTarget_(self._target)
        self.speakers_field.setAction_("saveSpeakers:")
        self.speakers_field.setToolTip_("Wer spricht in diesem Meeting? Enter speichert. Das finale Transkript ordnet die Stimmen diesen Namen zu.")
        self._speakers_label = self._label("TEILNEHMENDE", 11, utility=True)
        self.topic_field = A.NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 26))
        self.topic_field.setPlaceholderString_("Thema für einen Textauszug")
        self.topic_field.setFont_(_font(13))
        self.topic_field.setToolTip_("Kopiert passende Stellen aus dem Transkript dieser Aufnahme.")
        self._clip_button = self._button("Auszug kopieren", "copyClip:", "Sucht passende Textstellen zu deinem Thema und kopiert sie.")
        self._record_button = self._button("Aufnahme starten", "toggleRecording:", "Start oder Stop ist immer eine ausdrückliche Aktion.")
        self._consent_hint = self._label("Vorher alle Beteiligten informieren.", 10)
        self._consent_hint.setTextColor_(_color("59666A"))
        for view in (self._title, self._state_label, self._mode_control, self._fullscreen_button,
                     self._help_button, self._audience_control, self._transcript_label, self._copy_transcript_button,
                     self._cards_label, self._transcript_scroll, self.cards_scroll, self._help_scroll, self.status_field,
                     self.topic_field, self._speakers_label, self.speakers_field, self._clip_button, self._record_button, self._consent_hint):
            self._surface.addSubview_(view)
        self._built = True
        self._layout()
        self._render_transcript()
        self._render_state()
        # Opening a panel must not steal keyboard focus from the user's work.
        panel.orderFrontRegardless()

    def _layout(self) -> None:
        if not self._built or self._closed or self._laying_out:
            return
        self._laying_out = True
        try:
            bounds = self.panel.contentView().bounds()
            width, height = bounds.size.width, bounds.size.height
            self._view_mode = panel_mode(width, height, fullscreen=self._fullscreen)
            presentation = self._view_mode == "presentation"
            self._backdrop.setHidden_(not presentation)
            self._backdrop.setNeedsDisplay_(True)
            frame = surface_rect(width, height, fullscreen=self._fullscreen)
            self._surface.setFrame_(NSMakeRect(*frame))
            self._surface.layer().setCornerRadius_(22 if presentation else 0)
            self._surface.layer().setBorderWidth_(1 if presentation else 0)
            self._surface.layer().setBorderColor_(_color("D5DBD5").CGColor())
            w, h = frame[2], frame[3]
            self._title.setFrame_(NSMakeRect(20, h - 48, 160, 34))
            self._state_label.setFrame_(NSMakeRect(w - 235, h - 39, 215, 20))
            self._state_label.setAlignment_(A.NSTextAlignmentRight)
            self._mode_control.setFrame_(NSMakeRect(18, h - 82, 222, 28))
            self._mode_control.setSelectedSegment_(1 if presentation else 0)
            self._fullscreen_button.setFrame_(NSMakeRect(w - 126, h - 82, 80, 28))
            self._help_button.setFrame_(NSMakeRect(w - 48, h - 82, 32, 28))
            self._audience_control.setFrame_(NSMakeRect(18, h - 115, 300, 26))
            self._speakers_label.setFrame_(NSMakeRect(20, h - 146, 110, 18))
            self.speakers_field.setFrame_(NSMakeRect(130, h - 150, w - 150, 26))
            body_top, bottom = h - 172, 116.0
            if presentation:
                gap = 20.0
                col = (w - 40 - gap) / 2
                body_h = max(70, body_top - bottom - 22)
                self._transcript_label.setFrame_(NSMakeRect(20, body_top - 18, col - 150, 18))
                self._copy_transcript_button.setFrame_(NSMakeRect(20 + col - 140, body_top - 22, 140, 24))
                self._cards_label.setFrame_(NSMakeRect(20 + col + gap, body_top - 18, col, 18))
                self._transcript_scroll.setFrame_(NSMakeRect(20, bottom, col, body_h))
                self.cards_scroll.setFrame_(NSMakeRect(20 + col + gap, bottom, col, body_h))
            else:
                available = body_top - bottom
                transcript_h = min(220, max(125, available * 0.36))
                self._transcript_label.setFrame_(NSMakeRect(20, body_top - 18, w - 190, 18))
                self._copy_transcript_button.setFrame_(NSMakeRect(w - 160, body_top - 22, 140, 24))
                self._transcript_scroll.setFrame_(NSMakeRect(20, body_top - 24 - transcript_h, w - 40, transcript_h))
                cards_top = body_top - transcript_h - 42
                self._cards_label.setFrame_(NSMakeRect(20, cards_top - 18, w - 40, 18))
                self.cards_scroll.setFrame_(NSMakeRect(20, bottom, w - 40, max(70, cards_top - bottom - 24)))
            self._help_scroll.setFrame_(NSMakeRect(20, bottom, w - 40, max(70, body_top - bottom)))
            for view in (self._transcript_scroll, self.cards_scroll, self._transcript_label,
                         self._copy_transcript_button, self._cards_label):
                view.setHidden_(self._help_open)
            self._help_scroll.setHidden_(not self._help_open)
            self.status_field.setFrame_(NSMakeRect(20, 78, w - 40, 32))
            self.topic_field.setFrame_(NSMakeRect(20, 44, w - 178, 26))
            self._clip_button.setFrame_(NSMakeRect(w - 152, 43, 132, 28))
            self._record_button.setFrame_(NSMakeRect(18, 10, 172, 28))
            self._consent_hint.setFrame_(NSMakeRect(200, 14, w - 220, 18))
            self.transcript_text.setFont_(_font(15 if presentation else 14))
            self._fit_text_view(self._transcript_scroll, self.transcript_text)
            self._fit_text_view(self._help_scroll, self._help_text)
            self._render_cards()
        finally:
            self._laying_out = False

    def set_mode(self, mode: str) -> None:
        if mode not in {"compact", "presentation"}:
            raise ValueError("unknown recorder view")
        if self.panel is None or self._closed:
            return
        if mode == "compact" and self._fullscreen:
            self._compact_after_fullscreen = True
            self.panel.toggleFullScreen_(None)
            return
        if mode == "presentation":
            if self.view_mode == "compact":
                self._compact_frame = self.panel.frame()
            screen = self.panel.screen() or A.NSScreen.mainScreen()
            if screen is not None:
                self.panel.setFrame_display_(screen.visibleFrame(), True)
        else:
            screen = self.panel.screen() or A.NSScreen.mainScreen()
            if self._compact_frame is not None:
                frame = self._compact_frame
            else:
                old = self.panel.frame()
                frame = NSMakeRect(old.origin.x, old.origin.y + old.size.height - COMPACT_SIZE[1], *COMPACT_SIZE)
            if screen is not None:
                area = screen.visibleFrame()
                frame = NSMakeRect(max(area.origin.x, min(frame.origin.x, area.origin.x + area.size.width - frame.size.width)),
                                   max(area.origin.y, min(frame.origin.y, area.origin.y + area.size.height - frame.size.height)),
                                   frame.size.width, frame.size.height)
            self.panel.setFrame_display_(frame, True)
        self._layout()

    def toggle_fullscreen(self) -> None:
        if self.panel is None or self._closed:
            return
        if not self._fullscreen:
            self.panel.setCollectionBehavior_(A.NSWindowCollectionBehaviorFullScreenPrimary)
            self.panel.setLevel_(A.NSNormalWindowLevel)
            # Focus changes only on this explicit user action, never at Start.
            self.panel.makeKeyAndOrderFront_(None)
        self.panel.toggleFullScreen_(None)

    def show_help(self, visible: bool = True) -> None:
        self._help_open = visible
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
        if self._built:
            self._audience_control.setSelectedSegment_(0)
        self._expanded_keys.clear()
        self._highlighted_key = None
        self._on_speakers = None
        self._speakers_unconfirmed = False
        self.set_speakers("")
        self.apply_update((), (), "")

    @property
    def displayed_cards(self):
        return visible_cards(self._cards, self.audience)

    def set_diagnostics_callback(self, callback) -> None:
        self._on_diagnostics = callback

    def set_audience_callback(self, callback) -> None:
        self._on_audience = callback

    def set_audience(self, audience: str) -> None:
        if audience not in {"client", "internal"}:
            raise ValueError("unknown audience")
        self.audience = audience
        self._highlighted_key = None
        self._expanded_keys.intersection_update(card.key for card in self.displayed_cards)
        if self._built:
            self._audience_control.setSelectedSegment_(0 if audience == "client" else 1)
            self._render_transcript()
            self._render_cards()
        if self._on_audience is not None:
            self._on_audience(audience)

    def toggle_recording(self) -> None:
        if self._on_toggle_recording is not None:
            self._on_toggle_recording()

    def copy_clip(self) -> None:
        topic = str(self.topic_field.stringValue() or "") if self.topic_field is not None else ""
        threading.Thread(target=self._on_copy_clip, args=(topic,), name="meeting-sidecar-live-clip", daemon=True).start()

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
        self._status = f"Transkript kopiert ({len(self._lines)} Abschnitte)."
        if self._built:
            self._render_state()

    def copy_card(self, index: int) -> None:
        if 0 <= index < len(self.displayed_cards):
            text = self.displayed_cards[index].copy_text
            threading.Thread(target=self._on_copy_card, args=(text,), name="meeting-sidecar-live-card", daemon=True).start()

    def toggle_source(self, index: int) -> None:
        if not 0 <= index < len(self.displayed_cards):
            return
        key = self.displayed_cards[index].key
        if key in self._expanded_keys:
            self._expanded_keys.remove(key)
        else:
            self._expanded_keys.add(key)
        self._highlighted_key = key if key in self._expanded_keys else None
        self._render_transcript()
        self._render_cards()

    def apply_update(self, lines: Sequence[LiveLine], cards: Sequence[LiveCard], status: str) -> None:
        snapshot_lines, snapshot_cards = tuple(lines), tuple(cards)
        generation = self._session_generation
        queued_at = time.monotonic()
        def render():
            if self._closed or generation != self._session_generation:
                return
            self._lines, self._cards, self._status = snapshot_lines, snapshot_cards, status
            self._expanded_keys.intersection_update(card.key for card in self.displayed_cards)
            if self._built:
                self._render_transcript()
                self._render_cards()
                self._render_state()
                if self._on_diagnostics is not None:
                    self._on_diagnostics({"stage": "ui_publish", "elapsed_seconds": time.monotonic() - queued_at,
                                          "line_count": len(snapshot_lines), "card_count": len(self.displayed_cards)})
        _dispatch_main(render)

    def _render_state(self) -> None:
        labels = {"idle": "Bereit", "recording": "● Aufnahme läuft", "stopped": "Aufnahme beendet", "preview": "Vorschau · keine Aufnahme"}
        self._state_label.setStringValue_(labels[self._recording_state])
        self._state_label.setTextColor_(_color("A23F32" if self._recording_state == "recording" else "59666A"))
        self._record_button.setTitle_("Aufnahme beenden" if self._recording_state == "recording" else "Aufnahme starten")
        self._record_button.setEnabled_(self._on_toggle_recording is not None and self._recording_state != "preview")
        self._copy_transcript_button.setEnabled_(bool(self._lines))
        has_committed = any(not line.provisional for line in self._lines)
        self._clip_button.setEnabled_(has_committed)
        self.topic_field.setEnabled_(has_committed)
        hint = "Beispieldaten · Mikrofon ist aus." if self._recording_state == "preview" else "Schliessen versteckt nur dieses Fenster."
        self.status_field.setStringValue_(self._status or hint)

    def _render_transcript(self) -> None:
        visible = self.transcript_text.visibleRect()
        old_height = self.transcript_text.frame().size.height
        follow = visible.origin.y + visible.size.height >= old_height - 30
        value, _offsets = transcript_content(self._lines)
        if not value:
            value = ("Sobald du sprichst, erscheint hier das Transkript." if self._recording_state == "recording"
                     else "Starte die Aufnahme, wenn alle Beteiligten einverstanden sind.")
        self.transcript_text.setString_(value)
        self.transcript_text.setTextColor_(_color("22323B" if self._lines else "6D777A"))
        self._fit_text_view(self._transcript_scroll, self.transcript_text)
        selected = next((card for card in self.displayed_cards if card.key == self._highlighted_key), None)
        ranges = source_ranges(self._lines, selected.spans) if selected else []
        storage = self.transcript_text.textStorage()
        for start, length in ranges:
            storage.addAttribute_value_range_(A.NSBackgroundColorAttributeName, _color("F5E6AE"), NSMakeRange(start, length))
        if ranges:
            self.transcript_text.scrollRangeToVisible_(NSMakeRange(*ranges[0]))
        elif follow and self._lines:
            self.transcript_text.scrollRangeToVisible_(NSMakeRange(utf16_length(value), 0))

    def _render_cards(self) -> None:
        if self.cards_document is None:
            return
        old_origin = self.cards_scroll.contentView().bounds().origin
        for view in list(self.cards_document.subviews()):
            view.removeFromSuperview()
        self._card_rows = []
        width = float(self.cards_scroll.contentView().bounds().size.width)
        viewport_h = self.cards_scroll.contentView().bounds().size.height
        body_font = _font(15 if self.view_mode == "presentation" else 14)
        source_font = _font(13)
        y = 0.0
        displayed = self.displayed_cards
        if not displayed:
            text = self._label("Noch keine Karte.\nAnfragen, Entscheidungen und nächste Schritte erscheinen hier. KI-Ideen sind separat beschriftet.", 14, wrap=True)
            text.setTextColor_(_color("6D777A"))
            text.setFrame_(NSMakeRect(12, 14, max(30, width - 24), _text_height(str(text.stringValue()), width - 24, text.font())))
            self.cards_document.addSubview_(text)
        for index, card in enumerate(displayed):
            inner_w = max(80, width - 32)
            body = card.clean_text.strip() or card.verbatim_text
            body_h = _text_height(body, inner_w, body_font)
            expanded = card.key in self._expanded_keys
            source = "\n\n".join(f"[{format_timestamp(line.seconds)}] {line.speaker}\n{line.text}" for line in card.lines)
            source_h = _text_height(source, inner_w, source_font) if expanded else 0
            height = 82 + body_h + (source_h + 16 if expanded else 0)
            row = _TopView.alloc().initWithFrame_(NSMakeRect(0, y, width, height))
            row.setWantsLayer_(True)
            row.layer().setBackgroundColor_(_color("FFFFFF").CGColor())
            row.layer().setCornerRadius_(12)
            row.layer().setBorderWidth_(1)
            row.layer().setBorderColor_(_color("DEE3DC").CGColor())
            labels = {"request": "ANFRAGE", "decision": "ENTSCHEIDUNG", "action": "NÄCHSTER SCHRITT", "suggestion_question": "KI-FRAGE", "suggestion_idea": "KI-IDEE"}
            suggestion = card.kind.startswith("suggestion_")
            audience_label = " · PRIVAT" if card.audience == "internal" else ""
            meta = self._label(f"{labels.get(card.kind, 'KARTE')}{audience_label} · {format_timestamp(card.start_seconds)}", 10, utility=True)
            if suggestion:
                row.layer().setBackgroundColor_(_color("EDF4F7").CGColor())
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
            for view in (meta, label, source_button, copy_button):
                row.addSubview_(view)
            if expanded:
                original = self._label(source, 13, wrap=True)
                original.setTextColor_(_color("59666A"))
                original.setFrame_(NSMakeRect(16, 82 + body_h, inner_w, source_h))
                row.addSubview_(original)
            self.cards_document.addSubview_(row)
            self._card_rows.append(row)
            y += height + 12
        self.cards_document.setFrame_(NSMakeRect(0, 0, width, max(viewport_h, y)))
        self.cards_scroll.contentView().scrollToPoint_((0, min(old_origin.y, max(0, y - viewport_h))))
        self.cards_scroll.reflectScrolledClipView_(self.cards_scroll.contentView())

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
        scroll = A.NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
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
