"""Native synthetic preview and state checks. Never opens an audio device."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import AppKit as A
import Quartz  # Registers CGColor bridge metadata before layer setters.
from Foundation import NSMakeRect
from PyObjCTools import AppHelper
from sidecar.live import LiveCard, LiveLine
from sidecar.card_semantics import split_sources
from sidecar.live_window import LivePanel
from sidecar.transcript import TranscriptLine, format_timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--background', type=Path)
    args=parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.background:
        import sidecar.live_window as window_module
        window_module.BACKGROUND_PATH=args.background.resolve()
    A.NSApplication.sharedApplication().setActivationPolicy_(A.NSApplicationActivationPolicyRegular)
    lines=[LiveLine(0,6,'Ich','Bitte fasse zusammengehörige Anfragen zu einer Karte zusammen.'),
           LiveLine(7,12,'Ich','Und markiere die gewählten Originalstellen im Transkript.'),
           LiveLine(15,20,'Andere','Wir haben uns für Donnerstag entschieden.'),
           LiveLine(22,27,'Ich','Ich bringe den Testbericht am Donnerstag mit.')]
    spans=split_sources(lines)
    def card(key,kind,indexes,text,audience='all'):
        selected=tuple(spans[i] for i in indexes)
        evidence=tuple(TranscriptLine(s.line_index,s.seconds,format_timestamp(s.seconds),s.speaker,s.text) for s in selected)
        return LiveCard(key,evidence,text,kind,selected,audience)
    cards=[card('request','request',[0,1],'Zusammengehörige Anfragen bündeln und die ausgewählten Originalstellen im Transkript markieren.'),
           card('decision','decision',[2],'Der Termin ist Donnerstag.'),
           card('action','action',[3],'Ich bringe den Testbericht am Donnerstag mit.'),
           card('public','suggestion_question',[0,1],'Woran erkennen wir, dass eine Anfrage vollständig erfasst ist?','client'),
           card('private','suggestion_idea',[0],'INTERNAL_SENTINEL · Intern eine mögliche Anschlussberatung prüfen.','internal')]
    copied=[]; checks={}
    panel=LivePanel(on_copy_clip=lambda _:None,on_copy_card=copied.append)
    panel.set_recording_state('preview');panel.show();panel.apply_update(lines,cards,'Synthetische Vorschau · keine Aufnahme')
    original_window=panel.panel
    def capture(name):
        view=panel.panel.contentView();view.displayIfNeeded()
        bitmap=view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
        view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(),bitmap)
        data=bitmap.representationUsingType_properties_(A.NSBitmapImageFileTypePNG,{})
        data.writeToFile_atomically_(str(args.out/f'{name}.png'),True)
    def compact():
        panel.toggle_source(0);capture('compact-source')
        checks['source_selected']=panel._highlighted_key=='request'
        panel.panel.performClose_(None)
        checks['close_hides']=not panel.panel.isVisible()
        panel.order_front()
        checks['reopen_same_window']=panel.panel is original_window and len(panel._lines)==4 and panel.panel.isVisible()
        panel.set_audience('internal');panel.toggle_source(4)
        checks['internal_visible']=any(c.key=='private' for c in panel.displayed_cards)
        panel.set_audience('client')
        checks['client_removes_internal']=all(c.key!='private' for c in panel.displayed_cards) and 'private' not in panel._expanded_keys and panel._highlighted_key is None
        checks['client_preserves_session']=len(panel._lines)==4 and len(panel._cards)==5
        panel.set_mode('presentation')
        AppHelper.callLater(.7,presentation)
    def presentation():
        capture('presentation')
        checks['presentation_mode']=panel.view_mode=='presentation'
        checks['mode_preserves_session']=len(panel._lines)==4 and len(panel._cards)==5
        panel.toggle_fullscreen()
        AppHelper.callLater(3,fullscreen)
    def fullscreen():
        checks['native_fullscreen']=panel._fullscreen and bool(panel.panel.styleMask() & A.NSWindowStyleMaskFullScreen)
        capture('fullscreen')
        if panel._fullscreen:
            panel.toggle_fullscreen()
        AppHelper.callLater(3,finish)
    def finish():
        panel.set_mode('compact')
        checks['compact_restored']=panel.view_mode=='compact'
        checks['preview_capture_disabled']=not panel._record_button.isEnabled()
        checks['source_cards_preserved']=len(panel._cards)==5
        (args.out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps(checks),flush=True)
        panel.close();AppHelper.stopEventLoop()
    AppHelper.callLater(.7,compact)
    AppHelper.runEventLoop()

if __name__=='__main__':main()
