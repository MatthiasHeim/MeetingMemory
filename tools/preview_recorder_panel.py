"""Native synthetic preview of the three recorder views. Never opens an audio device.

Shows the real LivePanel with sample data in the 'preview' recording state, writes one PNG
per view for the normal window and for native full screen, and a checks.json. Preferences go
to a temporary file and the jump request is a stub, so nothing outside --out is touched.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
import AppKit as A
import Quartz  # Registers CGColor bridge metadata before layer setters.
from PyObjCTools import AppHelper
from sidecar.live import LiveCard, LiveLine
from sidecar.card_semantics import split_sources
from sidecar.live_window import LivePanel
from sidecar.panel_layout import VIEWS
from sidecar.transcript import TranscriptLine, format_timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prefix', default='three-views')
    parser.add_argument('--no-fullscreen', action='store_true', help='skip native full screen')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    A.NSApplication.sharedApplication().setActivationPolicy_(A.NSApplicationActivationPolicyRegular)
    lines = [LiveLine(52, 58, 'Ich', 'Die Oberfläche des Recorders insgesamt verbessern.'),
             LiveLine(75, 82, 'Ich', 'In der neuen Zeile soll Pagination sein, wie unten.'),
             LiveLine(104, 110, 'Ich', 'Screencast und gesprochene Notizen gemeinsam durchgehen.'),
             LiveLine(160, 166, 'Andere', 'Wir haben uns geeinigt: Markus stellt jetzt die Fragen.'),
             LiveLine(288, 296, 'Ich', 'Pagination in die neue Zeile einbauen, statt Pfeile zum Weiterblättern.'),
             LiveLine(310, 318, 'Ich', 'Ich bringe den Testbericht am Donnerstag mit.'),
             LiveLine(330, 338, 'Ich', 'Actually the meeting mode would decide how it is presented.', provisional=True)]
    spans = split_sources(lines)
    def card(key, kind, indexes, text, audience='all'):
        selected = tuple(s for s in spans if s.line_index in indexes)
        evidence = tuple(TranscriptLine(s.line_index, s.seconds, format_timestamp(s.seconds), s.speaker, s.text) for s in selected)
        return LiveCard(key, evidence, text, kind, selected, audience)
    first = [card('r1', 'request', [0], 'Die Oberfläche insgesamt verbessern.'),
             card('r2', 'request', [1], 'In der neuen Zeile soll Pagination sein, wie unten.'),
             card('a1', 'action', [2], 'Screencast und gesprochene Notizen gemeinsam durchgehen.'),
             card('d1', 'decision', [3], 'Markus stellt jetzt die Fragen.'),
             card('ai', 'suggestion_idea', [0], 'Aufgezeichneten Screencast und Notizen gemeinsam sichten.', 'client'),
             card('private', 'suggestion_idea', [0], 'INTERNAL_SENTINEL · Intern eine mögliche Anschlussberatung prüfen.', 'internal')]
    later = first + [card('a2', 'action', [4], 'Pagination in die neue Zeile einbauen, statt Pfeile zum Weiterblättern.'),
                     card('a3', 'action', [5], 'Testbericht am Donnerstag mitbringen.')]
    checks = {}
    posted = []
    prefs = Path(tempfile.mkdtemp(prefix='recorder-preview-')) / 'prefs.json'
    panel = LivePanel(on_copy_clip=lambda _: None, on_copy_card=lambda _: None, preferences_path=prefs,
                      jump_post=lambda url, payload, timeout: posted.append(payload) or {'ok': True},
                      jump_token_path=prefs.with_name('no-token'))
    panel.set_recording_state('preview'); panel.show()
    panel.apply_update(lines[:5], first, 'Synthetische Vorschau · keine Aufnahme')
    panel.apply_update(lines, later, 'Synthetische Vorschau · keine Aufnahme')   # two points now wait behind the marker
    panel.panel.setFrame_display_(A.NSMakeRect(40, 80, 480, 760), True)

    def capture(name):
        view = panel.panel.contentView(); view.displayIfNeeded()
        bitmap = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
        view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), bitmap)
        data = bitmap.representationUsingType_properties_(A.NSBitmapImageFileTypePNG, {})
        data.writeToFile_atomically_(str(args.out / f'{args.prefix}-{name}.png'), True)

    def window_views():
        for view in VIEWS:
            panel.set_window_mode(view, persist=False)
            checks[f'{view}_window_mode'] = panel.view_mode in {'narrow', 'wide'}
            checks[f'{view}_marker_waiting'] = panel.pending_count == 2
            capture(f'{view}-window')
        panel.set_window_mode('ambient', persist=False)
        panel.reveal_pending(); capture('ambient-window-revealed')
        checks['reveal_enters_points'] = panel.pending_count == 0 and len(panel.rendered_cards) == 6 - 2 + 2
        checks['private_hidden_in_meeting_and_ambient'] = all(c.key != 'private' for c in panel.displayed_cards)
        panel.set_window_mode('meeting', persist=False)
        panel.apply_update(lines, later, '')
        checks['preview_capture_disabled'] = not panel._record_button.isEnabled()
        if args.no_fullscreen:
            return finish()
        panel.toggle_fullscreen()
        AppHelper.callLater(3, fullscreen_views)

    def fullscreen_views():
        checks['native_fullscreen'] = panel._fullscreen and bool(panel.panel.styleMask() & A.NSWindowStyleMaskFullScreen)
        checks['fullscreen_mode_value'] = panel.view_mode == 'fullscreen'
        for view in VIEWS:
            panel.set_window_mode(view, persist=False)
            if view != 'ambient':
                for key, tracker in list(panel._reveals.items()):
                    tracker.reveal('11:02') if key[0] == view else None
                panel._layout()
            capture(f'{view}-fullscreen')
        if panel._fullscreen:
            panel.toggle_fullscreen()
        AppHelper.callLater(3, finish)

    def finish():
        checks['window_restored'] = panel.view_mode in {'narrow', 'wide'}
        (args.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps(checks), flush=True)
        panel.close(); AppHelper.stopEventLoop()
    AppHelper.callLater(.7, window_views)
    AppHelper.runEventLoop()

if __name__ == '__main__': main()
