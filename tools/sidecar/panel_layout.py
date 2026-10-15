"""Geometry and user-facing help for the native recorder panel."""

from __future__ import annotations


COMPACT_SIZE = (480.0, 760.0)
MINIMUM_SIZE = (420.0, 640.0)
PRESENTATION_MINIMUM = (960.0, 640.0)


def panel_mode(width: float, height: float, *, fullscreen: bool = False) -> str:
    return "presentation" if fullscreen or (
        width >= PRESENTATION_MINIMUM[0] and height >= PRESENTATION_MINIMUM[1]
    ) else "compact"


def surface_rect(width: float, height: float, *, fullscreen: bool = False) -> tuple[float, float, float, float]:
    """Bottom-origin native coordinates; preserve the robot and wall logo."""
    if panel_mode(width, height, fullscreen=fullscreen) == "compact":
        return (0.0, 0.0, width, height)
    margin = max(20.0, min(40.0, width * 0.025))
    surface_width = min(1020.0, max(480.0, width * 0.51))
    surface_height = min(height - margin * 2, max(480.0, height * 0.57))
    return (width - surface_width - margin, margin, surface_width, surface_height)


PANEL_HELP = (
    "Aufnahme starten / beenden\n"
    "Startet Mikrofon und Systemton oder speichert die laufende Aufnahme. "
    "Informiere alle Beteiligten, bevor du startest.\n\n"
    "Prompt-Stelle markieren\n"
    "Merkt die aktuelle Stelle einer laufenden Aufnahme, damit du die diktierte Aufgabe später wiederfindest.\n\n"
    "Transkript kopieren\n"
    "Kopiert das ganze bisherige Transkript dieser Aufnahme, während und nach der Aufnahme.\n\n"
    "Themenauszug kopieren …\n"
    "Wählt ein gespeichertes Transkript und kopiert passende Textstellen zu deinem Thema. "
    "Während einer Aufnahme nutzt du das Themenfeld in diesem Fenster.\n\n"
    "Diktierte Aufgaben kopieren …\n"
    "Sucht markierte oder erkannte Aufgaben in einem gespeicherten Transkript. Du wählst aus, was kopiert wird.\n\n"
    "Fenster und Ansichten\n"
    "Kompakt und Präsentation zeigen dieselbe Aufnahme. Grossziehen zeigt den Hintergrund; "
    "Kleinziehen blendet ihn aus. Vollbild ist separat verfügbar. "
    "Schliessen versteckt nur das Fenster. «Recorder-Fenster öffnen» im Menü bringt es zurück.\n\n"
    "Kundenansicht / Intern · privat\n"
    "Kundenansicht zeigt allgemeine KI-Ideen und blendet interne Hinweise sofort aus. "
    "Intern kann private Beratung enthalten. Diese Wahl gilt unabhängig von Fenstergrösse und Vollbild. "
    "Der Gesprächstranskript bleibt sichtbar. KI-Fragen und KI-Ideen sind Vorschläge, keine Aussagen oder Beschlüsse.\n\n"
    "Decider lokal verwenden\n"
    "Wählt das lokale Modell für Aufgaben-Erkennung und Themenauszüge neuer Aufnahmen. "
    "Transkription und Prompt-Bereinigung verwenden weiterhin Gemini."
)


MENU_HELP = {
    "start": "Startet Mikrofon und Systemton. Bitte informiere vorher alle Beteiligten.",
    "stop": "Beendet und speichert die laufende Aufnahme. Das Fenster und seine Texte bleiben erhalten.",
    "mark": "Merkt die aktuelle Stelle für eine diktierte Aufgabe. Nur während einer Aufnahme verfügbar.",
    "clip": "Wählt ein gespeichertes Transkript und kopiert passende Stellen zu einem Thema. Während der Aufnahme ist das Themenfeld im Recorder-Fenster verfügbar.",
    "prompts": "Sucht diktierte Aufgaben in einem gespeicherten Transkript und lässt dich eine zum Kopieren auswählen. Nach der Aufnahme verfügbar.",
    "window": "Öffnet das Recorder-Fenster oder bringt dieselbe Aufnahme zurück. Startet keine Aufnahme.",
    "help": "Erklärt die Aufnahme, Markierungen, Themenauszüge und Fensteransichten.",
    "decider": "Verwendet das lokale Entscheidungsmodell für neue Aufnahmen. Transkription und Prompt-Bereinigung bleiben bei Gemini.",
    "devices": "Zeigt verfügbare Audioeingänge. Während einer Aufnahme nicht verfügbar.",
    "recordings": "Öffnet den Ordner mit gespeicherten Audioaufnahmen.",
    "transcripts": "Öffnet den Ordner mit gespeicherten Transkripten.",
    "preferences": "Öffnet die Konfigurationsdatei des Recorders.",
    "quit": "Beendet den Recorder. Eine laufende Aufnahme wird zuvor gespeichert.",
}


def set_menu_help(item: object, text: str) -> bool:
    """rumps wraps a native NSMenuItem; leave unsupported hosts unchanged."""
    native = getattr(item, "_menuitem", None)
    setter = getattr(native, "setToolTip_", None)
    if callable(setter):
        setter(text)
        return True
    return False


def visible_cards(cards, audience: str):
    """Fail closed on unknown audiences. Layout never grants private visibility."""
    allowed = {'all', 'client', 'internal'} if audience == 'internal' else {'all', 'client'}
    return tuple(card for card in cards if card.audience in allowed)


def utf16_length(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def transcript_content(lines):
    from .transcript import format_timestamp
    text = ''
    offsets = []
    for line in lines:
        if text:
            text += '\n\n'
        suffix = ' · wird ergänzt' if line.provisional else ''
        text += f'[{format_timestamp(line.start)}] {line.speaker}{suffix}\n'
        offsets.append(len(text))
        text += line.text
    return text, offsets


def transcript_copy_text(lines):
    """Plain clipboard text of every line so far, without the on-screen pending marker."""
    from .transcript import format_timestamp
    return '\n\n'.join(f'[{format_timestamp(line.start)}] {line.speaker}\n{line.text}' for line in lines)


def source_ranges(lines, spans):
    """Native NSTextStorage ranges use UTF-16, not Python character offsets."""
    text, offsets = transcript_content(lines)
    result = []
    for span in spans:
        if not 0 <= span.line_index < len(lines):
            continue
        line = lines[span.line_index]
        if line.start != span.seconds or line.speaker != span.speaker or line.text[span.start_char:span.end_char] != span.text:
            continue
        start = offsets[span.line_index] + span.start_char
        result.append((utf16_length(text[:start]), utf16_length(span.text)))
    return result
