"""Floating non-activating side window for the live sidecar.

The panel stays above other windows, does not become key when it appears, and
is never shown with a modal loop. Closing it does not stop recording. Stopping
the recording does not close the panel.
"""

from __future__ import annotations

import subprocess
import threading
from typing import Callable, Sequence

from .live import LiveCard, LiveLine
from .transcript import format_timestamp


# Apple's NSWindowStyleMaskNonactivatingPanel / NSNonactivatingPanelMask.
NONACTIVATING_PANEL_MASK = 1 << 7
# NSFloatingWindowLevel.
FLOATING_WINDOW_LEVEL = 3

try:
    from Foundation import NSObject
except ImportError:  # pragma: no cover - macOS only
    NSObject = object  # type: ignore[misc, assignment]


def _constant(module: object, modern: str, legacy: str, fallback: int) -> int:
    return int(getattr(module, modern, getattr(module, legacy, fallback)))


def panel_style_mask(appkit: object) -> int:
    """Titled, closable, resizable utility panel that does not activate the app."""
    return (
        _constant(appkit, "NSWindowStyleMaskTitled", "NSTitledWindowMask", 1 << 0)
        | _constant(appkit, "NSWindowStyleMaskClosable", "NSClosableWindowMask", 1 << 1)
        | _constant(appkit, "NSWindowStyleMaskResizable", "NSResizableWindowMask", 1 << 3)
        | _constant(appkit, "NSWindowStyleMaskUtilityWindow", "NSUtilityWindowMask", 1 << 4)
        | _constant(
            appkit,
            "NSWindowStyleMaskNonactivatingPanel",
            "NSNonactivatingPanelMask",
            NONACTIVATING_PANEL_MASK,
        )
    )


def panel_collection_behavior(appkit: object) -> int:
    """Stay on the active space, including full screen, without following focus."""
    return (
        _constant(appkit, "NSWindowCollectionBehaviorCanJoinAllSpaces", "", 1 << 0)
        | _constant(appkit, "NSWindowCollectionBehaviorStationary", "", 1 << 4)
        | _constant(appkit, "NSWindowCollectionBehaviorFullScreenAuxiliary", "", 1 << 8)
    )


def copy_to_clipboard(text: str) -> None:
    result = subprocess.run(
        ["pbcopy"],
        input=text,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pbcopy failed")


def _dispatch_main(callback: Callable[[], None]) -> None:
    try:
        from PyObjCTools import AppHelper

        AppHelper.callAfter(callback)
    except Exception:
        callback()


class _PanelTarget(NSObject):
    """Button target. Holds the panel so clicks do not need the app to be key first."""

    panel = None

    def copyClip_(self, _sender) -> None:
        if self.panel is not None:
            self.panel.copy_clip()

    def copyCard_(self, sender) -> None:
        if self.panel is not None:
            self.panel.copy_card(int(sender.tag()))


class LivePanel:
    """Small NSPanel. Updates are marshalled to the main thread."""

    def __init__(
        self,
        *,
        on_copy_clip: Callable[[str], None],
        on_copy_card: Callable[[str], None],
    ):
        self._on_copy_clip = on_copy_clip
        self._on_copy_card = on_copy_card
        self._cards: tuple[LiveCard, ...] = ()
        self._lines: tuple[LiveLine, ...] = ()
        self._status = ""
        self._built = False
        self.panel = None
        self.transcript_text = None
        self.cards_scroll = None
        self.cards_document = None
        self.status_field = None
        self.topic_field = None
        self._target = None
        self._closed = False

    def close(self) -> None:
        """Hide the panel. Recording is unchanged. Safe to call more than once."""
        self._closed = True
        panel = self.panel
        self.panel = None
        if panel is None:
            return

        def hide() -> None:
            panel.orderOut_(None)
            panel.close()

        _dispatch_main(hide)

    def order_front(self) -> None:
        """Show the panel again without activating the app or running a modal."""
        panel = self.panel
        if panel is None or self._closed:
            return

        def show() -> None:
            panel.orderFrontRegardless()

        _dispatch_main(show)

    def show(self) -> None:
        """Order the panel front without activating the app or running a modal."""
        import AppKit
        from Foundation import NSMakeRect

        appkit = AppKit
        style = panel_style_mask(appkit)
        width, height = 400.0, 560.0
        panel = appkit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height),
            style,
            appkit.NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Meeting")
        panel.setFloatingPanel_(True)
        panel.setLevel_(getattr(appkit, "NSFloatingWindowLevel", FLOATING_WINDOW_LEVEL))
        panel.setHidesOnDeactivate_(False)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setWorksWhenModal_(False)
        panel.setCollectionBehavior_(panel_collection_behavior(appkit))
        screen = appkit.NSScreen.mainScreen()
        if screen is not None:
            visible = screen.visibleFrame()
            origin_x = visible.origin.x + visible.size.width - width - 16
            origin_y = visible.origin.y + max(16, visible.size.height - height - 16)
            panel.setFrame_display_(NSMakeRect(origin_x, origin_y, width, height), False)

        content = panel.contentView()
        bounds = content.bounds()
        inner_width = float(bounds.size.width)
        inner_height = float(bounds.size.height)
        self.status_field = self._label(appkit, NSMakeRect(12, inner_height - 28, inner_width - 24, 18))
        transcript_scroll, self.transcript_text = self._scroll_text(
            appkit, NSMakeRect(12, inner_height - 250, inner_width - 24, 210)
        )
        self.cards_scroll, self.cards_document = self._card_scroll(
            appkit, NSMakeRect(12, 52, inner_width - 24, inner_height - 310)
        )
        self.topic_field = appkit.NSTextField.alloc().initWithFrame_(
            NSMakeRect(12, 14, inner_width - 150, 26)
        )
        self.topic_field.setPlaceholderString_("Thema")
        self._target = _PanelTarget.alloc().init()
        self._target.panel = self
        button = appkit.NSButton.alloc().initWithFrame_(NSMakeRect(inner_width - 130, 12, 118, 28))
        button.setTitle_("Clip kopieren")
        button.setBezelStyle_(_constant(appkit, "NSBezelStyleRounded", "NSRoundedBezelStyle", 1))
        button.setTarget_(self._target)
        button.setAction_("copyClip:")
        for view in (self.status_field, transcript_scroll, self.cards_scroll, self.topic_field, button):
            content.addSubview_(view)
        self.panel = panel
        self._built = True
        # orderFrontRegardless shows the panel without making it key.
        panel.orderFrontRegardless()

    def copy_clip(self) -> None:
        topic = ""
        if self.topic_field is not None:
            topic = str(self.topic_field.stringValue() or "")
        threading.Thread(
            target=self._on_copy_clip, args=(topic,), name="meeting-sidecar-live-clip", daemon=True
        ).start()

    def copy_card(self, index: int) -> None:
        if not 0 <= index < len(self._cards):
            return
        text = self._cards[index].copy_text
        threading.Thread(
            target=self._on_copy_card, args=(text,), name="meeting-sidecar-live-card", daemon=True
        ).start()

    def apply_update(
        self, lines: Sequence[LiveLine], cards: Sequence[LiveCard], status: str
    ) -> None:
        snapshot_lines = tuple(lines)
        snapshot_cards = tuple(cards)

        def render() -> None:
            if self._closed:
                return
            self._lines = snapshot_lines
            self._cards = snapshot_cards
            self._status = status
            if not self._built:
                return
            if self.status_field is not None:
                self.status_field.setStringValue_(status)
            if self.transcript_text is not None:
                self.transcript_text.setString_(
                    "\n".join(
                        f"[{format_timestamp(line.start)}] {line.speaker}: {line.text}"
                        for line in snapshot_lines
                    )
                )
            self._render_cards()

        _dispatch_main(render)

    def _render_cards(self) -> None:
        if self.cards_document is None or self.cards_scroll is None:
            return
        import AppKit
        from Foundation import NSMakeRect

        for subview in list(self.cards_document.subviews()):
            subview.removeFromSuperview()
        width = float(self.cards_scroll.contentView().bounds().size.width)
        row_height = 92.0
        total = max(row_height, row_height * len(self._cards))
        self.cards_document.setFrame_(NSMakeRect(0, 0, width, total))
        for index, card in enumerate(self._cards):
            top = total - (index + 1) * row_height
            row = AppKit.NSView.alloc().initWithFrame_(NSMakeRect(0, top, width, row_height))
            clean = self._wrapping_label(
                AppKit, NSMakeRect(8, 32, width - 112, 52), card.clean_text.strip() or "(Bereinigung ausstehend)"
            )
            verbatim = self._wrapping_label(
                AppKit,
                NSMakeRect(8, 4, width - 112, 28),
                "\n".join(
                    f"[{format_timestamp(line.seconds)}] {line.speaker}: {line.text}" for line in card.lines
                ),
            )
            button = AppKit.NSButton.alloc().initWithFrame_(NSMakeRect(width - 96, 32, 88, 24))
            button.setTitle_("Kopieren")
            button.setTag_(index)
            button.setTarget_(self._target)
            button.setAction_("copyCard:")
            row.addSubview_(clean)
            row.addSubview_(verbatim)
            row.addSubview_(button)
            self.cards_document.addSubview_(row)

    @staticmethod
    def _label(appkit, frame):
        field = appkit.NSTextField.alloc().initWithFrame_(frame)
        field.setEditable_(False)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setSelectable_(False)
        return field

    @staticmethod
    def _wrapping_label(appkit, frame, value: str):
        field = appkit.NSTextField.alloc().initWithFrame_(frame)
        field.setEditable_(False)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setSelectable_(True)
        field.setStringValue_(value)
        cell = field.cell()
        if cell is not None and hasattr(cell, "setWraps_"):
            cell.setWraps_(True)
            cell.setScrollable_(False)
        return field

    @staticmethod
    def _scroll_text(appkit, frame):
        scroll = appkit.NSScrollView.alloc().initWithFrame_(frame)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(1)
        text = appkit.NSTextView.alloc().initWithFrame_(scroll.contentView().bounds())
        text.setEditable_(False)
        text.setRichText_(False)
        text.setFont_(appkit.NSFont.systemFontOfSize_(13))
        text.setAutoresizingMask_(appkit.NSViewWidthSizable)
        scroll.setDocumentView_(text)
        return scroll, text

    @staticmethod
    def _card_scroll(appkit, frame):
        from Foundation import NSMakeRect

        scroll = appkit.NSScrollView.alloc().initWithFrame_(frame)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(1)
        document = appkit.NSView.alloc().initWithFrame_(NSMakeRect(0, 0, frame.size.width, frame.size.height))
        scroll.setDocumentView_(document)
        return scroll, document
