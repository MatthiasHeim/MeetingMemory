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


HEADER_HEIGHT = 44.0
MEETING_COLUMNS_MIN_WIDTH = 720.0
# Meeting mode shows only the transcript, the cards and what is needed to start or stop.
MEETING_CONTROLS = frozenset({"title", "state", "window_mode", "record", "transcript", "cards", "jump"})
INTERNAL_CONTROLS = MEETING_CONTROLS | frozenset({
    "view_size", "fullscreen", "help", "audience", "speakers", "topic", "clip", "copy_transcript", "status", "consent"})


def visible_controls(window_mode: str) -> frozenset:
    """Unknown modes fail to the quiet, screen-share-safe set."""
    return INTERNAL_CONTROLS if window_mode == "internal" else MEETING_CONTROLS


def meeting_layout(width: float, height: float) -> dict[str, tuple[float, float, float, float]]:
    """Bottom-origin rects (x, y, w, h) using the whole window, no margin or backdrop."""
    pad = 14.0
    top = height - HEADER_HEIGHT
    label_h = 18.0
    rects = {"title": (pad, height - 46, 160, 38), "window_mode": (width - 140 - pad, height - 36, 140, 26)}
    rects["record"] = (rects["window_mode"][0] - 150 - 8, height - 37, 150, 28)
    if width >= 640:  # on a narrow window the record button already says what is happening
        rects["state"] = (rects["record"][0] - 170, height - 30, 160, 18)
    if width >= MEETING_COLUMNS_MIN_WIDTH:
        gap = 16.0
        left = round((width - 2 * pad - gap) * 0.56)
        right = width - 2 * pad - gap - left
        body = max(60.0, top - pad - label_h)
        rects["transcript_label"] = (pad, top - label_h, left, label_h)
        rects["cards_label"] = (pad + left + gap, top - label_h, right, label_h)
        rects["transcript"] = (pad, pad, left, body)
        rects["cards"] = (pad + left + gap, pad, right, body)
    else:
        usable = max(120.0, top - pad - 2 * label_h - 10)
        transcript_h = usable * 0.55
        cards_h = usable - transcript_h
        rects["transcript_label"] = (pad, top - label_h, width - 2 * pad, label_h)
        rects["transcript"] = (pad, top - label_h - transcript_h, width - 2 * pad, transcript_h)
        cards_label_y = top - label_h - transcript_h - 10 - label_h
        rects["cards_label"] = (pad, cards_label_y, width - 2 * pad, label_h)
        rects["cards"] = (pad, pad, width - 2 * pad, max(40.0, cards_label_y - pad))
    return rects


FOLLOW_SLACK = 24.0


def is_at_bottom(offset: float, viewport: float, content: float, slack: float = FOLLOW_SLACK) -> bool:
    """True when the visible area reaches the end of the content (or nothing scrolls)."""
    return offset + viewport >= content - slack


def scroll_offset_after_update(follow: bool, old_offset: float, viewport: float, new_content: float) -> float:
    """Follow the end only while following; otherwise stay exactly where the reader is."""
    furthest = max(0.0, new_content - viewport)
    return furthest if follow else min(max(0.0, old_offset), furthest)


PANEL_HELP = (
    "Aufnahme starten / beenden\n"
    "Startet Mikrofon und Systemton oder speichert die laufende Aufnahme. "
    "Informiere alle Beteiligten, bevor du startest.\n\n"
    "Prompt-Stelle markieren\n"
    "Merkt die aktuelle Stelle einer laufenden Aufnahme, damit du die diktierte Aufgabe später wiederfindest.\n\n"
    "Teilnehmende\n"
    "Namen mit Komma trennen, Enter speichert. Vorschläge kommen aus der Kalendereinladung. "
    "Das finale Transkript ordnet die Stimmen diesen Namen zu; nicht genannte Eingeladene werden ignoriert. "
    "Am besten vor dem Stoppen eintragen.\n\n"
    "Transkript kopieren\n"
    "Kopiert das ganze bisherige Transkript dieser Aufnahme, während und nach der Aufnahme.\n\n"
    "Themenauszug kopieren …\n"
    "Wählt ein gespeichertes Transkript und kopiert passende Textstellen zu deinem Thema. "
    "Während einer Aufnahme nutzt du das Themenfeld in diesem Fenster.\n\n"
    "Diktierte Aufgaben kopieren …\n"
    "Sucht markierte oder erkannte Aufgaben in einem gespeicherten Transkript. Du wählst aus, was kopiert wird.\n\n"
    "Meeting / Intern\n"
    "Meeting ist die ruhige Ansicht zum Teilen: Transkript, Karten und Aufnahme-Knopf, immer ohne private Hinweise. "
    "Intern zeigt alle Schalter (Teilnehmende, Auszug, Kompakt/Präsentation, Vollbild, Hilfe). "
    "Die Wahl bleibt gespeichert. Schliessen versteckt nur das Fenster. «Recorder-Fenster öffnen» im Menü bringt es zurück.\n\n"
    "Scrollen\n"
    "Das Fenster folgt dem Neuesten nur, solange du am Ende bist. Scrollst du nach oben, bleibt die Ansicht stehen; "
    "«↓ Zum Neuesten» springt zurück und folgt wieder.\n\n"
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
