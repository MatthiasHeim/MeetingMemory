"""Geometry, view rules and help text for the native recorder panel (pure, unit-tested)."""

from __future__ import annotations


COMPACT_SIZE = (480.0, 760.0)
MINIMUM_SIZE = (420.0, 640.0)
WIDE_MIN_WIDTH = 720.0

VIEWS = ("meeting", "ambient", "internal")
DEFAULT_VIEW = "meeting"
VIEW_LABELS = {"meeting": "Meeting", "ambient": "Ambient", "internal": "Intern"}


def normalize_view(value: object) -> str:
    """Unknown or old values fall back to the quiet, screen-share-safe view."""
    return value if value in VIEWS else DEFAULT_VIEW


def panel_mode(width: float, height: float, *, fullscreen: bool = False) -> str:
    """'fullscreen' (native full screen), 'wide' (columns) or 'narrow' (one column)."""
    if fullscreen:
        return "fullscreen"
    return "wide" if width >= WIDE_MIN_WIDTH else "narrow"


# Type and spacing per mode. Full screen is roomier on purpose.
METRICS = {
    "narrow": {"pad": 16.0, "header": 52.0, "button": 30.0, "point": 18, "card": 15, "transcript": 14, "small": 11, "gap": 16.0},
    "wide": {"pad": 20.0, "header": 56.0, "button": 30.0, "point": 19, "card": 15, "transcript": 15, "small": 11, "gap": 20.0},
    "fullscreen": {"pad": 44.0, "header": 76.0, "button": 36.0, "point": 27, "card": 20, "transcript": 17, "small": 13, "gap": 40.0},
}

# What each view shows. A name missing here is hidden (fail closed).
_BASE = frozenset({"window_mode", "state", "record", "cards", "marker", "jump"})
MEETING_CONTROLS = _BASE | {"recent"}
AMBIENT_CONTROLS = _BASE | {"cards_label", "status", "ctx_label", "ctx_range", "ctx_copy", "ctx_claude"}
INTERNAL_CONTROLS = _BASE | {"cards_label", "transcript", "transcript_label", "copy_transcript", "audience", "help",
                             "speakers", "topic", "clip", "status"}
_FULLSCREEN_EXTRA = {"ambient": frozenset({"transcript", "transcript_label"}),
                     "internal": frozenset({"speakers_label", "topic_label", "rail"})}


def visible_controls(view: str, *, fullscreen: bool = False) -> frozenset:
    view = normalize_view(view)
    base = {"meeting": MEETING_CONTROLS, "ambient": AMBIENT_CONTROLS, "internal": INTERNAL_CONTROLS}[view]
    return base | _FULLSCREEN_EXTRA.get(view, frozenset()) if fullscreen else base


# ---- which points each view may show -------------------------------------

POINT_KINDS = ("request", "decision", "action")
CLIENT_AUDIENCES = frozenset({"all", "client"})


def cards_for_view(cards, view: str, show_private: bool = False):
    """Meeting and Ambient: plain points for clients only, never suggestions or private cards.
    Intern: everything, private ones only while the tick box is on."""
    view = normalize_view(view)
    if view == "internal":
        return visible_cards(cards, "internal" if show_private else "client")
    if view == "meeting":
        return tuple(c for c in cards if c.kind in POINT_KINDS and c.audience in CLIENT_AUDIENCES)
    return tuple(c for c in cards if not c.kind.startswith("suggestion_") and c.audience in CLIENT_AUDIENCES)


def card_columns(view: str, mode: str):
    """((title or None, kinds or None), ...). Meeting groups by kind once there is room."""
    if normalize_view(view) == "meeting":
        if mode == "fullscreen":
            return (("Entscheidungen", ("decision",)), ("Nächste Schritte", ("action",)), ("Anfragen", ("request",)))
        if mode == "wide":
            return (("Entscheidungen & Nächste Schritte", ("decision", "action")), ("Anfragen", ("request",)))
    return ((None, None),)


def split_columns(cards, columns):
    """Cards per column, each keeping the given order."""
    return [[c for c in cards if kinds is None or c.kind in kinds] for _title, kinds in columns]


# ---- context for Claude (Ambient) ------------------------------------------

CONTEXT_RANGES = (("last10", "Letzte 10 Min"), ("all", "Ganze Sitzung"))
LAST_WINDOW_SECONDS = 600.0


def context_lines(lines, range_key: str):
    lines = tuple(lines)
    if range_key != "last10" or not lines:
        return lines
    cutoff = max(line.end for line in lines) - LAST_WINDOW_SECONDS
    return tuple(line for line in lines if line.end >= cutoff)


def context_range_label(range_key: str) -> str:
    return dict(CONTEXT_RANGES).get(range_key, CONTEXT_RANGES[1][1])


def context_text(lines, range_key: str) -> str:
    return transcript_copy_text(context_lines(lines, range_key))


def latest_line_text(lines) -> str:
    """The one quiet 'Zuletzt' line of Meeting view."""
    for line in reversed(tuple(lines)):
        if line.text.strip():
            return " ".join(line.text.split())
    return ""


# ---- geometry (bottom-origin native coordinates: x, y, w, h) -------------------

MARKER_SIZE = (190.0, 32.0)


def _header(rects, view, w, h, m):
    pad, hh, bh = m["pad"], m["header"], m["button"]
    y = h - hh + (hh - bh) / 2
    fs = m is METRICS["fullscreen"]
    rects["window_mode"] = (pad, y, 300.0 if fs else 216.0, bh)
    rec_w = 150.0 if fs else 88.0
    right = w - pad - rec_w
    rects["record"] = (right, y, rec_w, bh)
    if view == "internal":
        right -= 8 + 34
        rects["help"] = (right, y, 34.0, bh)
    state_w = 170.0 if fs else 84.0
    x = right - 8 - state_w
    if x >= rects["window_mode"][0] + rects["window_mode"][2] + 10:
        rects["state"] = (x, y + (bh - 18) / 2, state_w, 18.0)
    return h - hh


def _marker(rects, x, w, y):
    rects["marker"] = (x + (w - MARKER_SIZE[0]) / 2, y, *MARKER_SIZE)


def view_layout(view: str, width: float, height: float, *, fullscreen: bool = False) -> dict:
    """Rects for every control of the view. Narrow window = one column, wide = columns,
    native full screen = its own roomier layout."""
    view = normalize_view(view)
    mode = panel_mode(width, height, fullscreen=fullscreen)
    m = METRICS[mode]
    pad, gap = m["pad"], m["gap"]
    rects: dict[str, tuple[float, float, float, float]] = {}
    top = _header(rects, view, width, height, m)
    inner = width - 2 * pad
    band = 56.0 if mode == "fullscreen" else 44.0
    label_h = 18.0
    if view == "meeting":
        recent_h = 52.0 if mode == "fullscreen" else 34.0
        rects["recent"] = (pad, 0.0, inner, recent_h)
        _marker(rects, pad, inner, recent_h + (band - MARKER_SIZE[1]) / 2)
        rects["cards"] = (pad, recent_h + band, inner, max(60.0, top - recent_h - band - 6))
        return rects
    if view == "ambient":
        foot = 84.0 if mode == "narrow" else (76.0 if mode == "fullscreen" else 56.0)
        rects["ctx_label"] = (pad, foot - 28 if mode == "narrow" else (foot - 18) / 2, 150.0, label_h)
        if mode == "narrow":
            rects["status"] = (pad + 154, foot - 28, inner - 154, label_h)
            row_y, x = 10.0, pad
        else:
            row_y, x = (foot - m["button"]) / 2, pad + 164
        rects["ctx_range"] = (x, row_y, 140.0, m["button"])
        rects["ctx_copy"] = (x + 148, row_y, 90.0, m["button"])
        rects["ctx_claude"] = (x + 246, row_y, 130.0, m["button"])
        if mode != "narrow":
            sx = x + 246 + 130 + 14
            rects["status"] = (sx, (foot - label_h) / 2, max(40.0, width - pad - sx), label_h)
        list_w = inner if mode != "fullscreen" else round((inner - gap) * 0.58)
        rects["cards_label"] = (pad, top - label_h - 4, list_w, label_h)
        _marker(rects, pad, list_w, foot + (band - MARKER_SIZE[1]) / 2)
        rects["cards"] = (pad, foot + band, list_w, max(60.0, top - label_h - 12 - foot - band))
        if mode == "fullscreen":
            tx = pad + list_w + gap
            tw = width - pad - tx
            rects["transcript_label"] = (tx, top - label_h - 4, tw, label_h)
            rects["transcript"] = (tx, foot + 8, tw, max(60.0, top - label_h - 12 - foot - 8))
        return rects
    # internal
    btn_h = 26.0
    help_rect = (pad, 0.0, inner, 0.0)
    if mode == "fullscreen":
        rail_w = 340.0
        rest = width - 2 * pad - rail_w - 2 * gap
        left = round(rest * 0.56)
        right_w = rest - left
        bottom = 28.0
        col_top = top - 4
        rects["transcript_label"] = (pad, col_top - label_h, left - 160, label_h)
        rects["copy_transcript"] = (pad + left - 150, col_top - btn_h + 2, 150.0, btn_h)
        rects["transcript"] = (pad, bottom, left, col_top - label_h - 10 - bottom)
        cx = pad + left + gap
        rects["cards_label"] = (cx, col_top - label_h, right_w - 200, label_h)
        rects["audience"] = (cx + right_w - 200, col_top - btn_h + 2, 200.0, btn_h)
        _marker(rects, cx, right_w, bottom)
        rects["cards"] = (cx, bottom + band - 8, right_w, col_top - label_h - 10 - bottom - band + 8)
        rx = cx + right_w + gap
        rects["rail"] = (rx, bottom, rail_w, col_top - bottom)
        ry = col_top - 18
        ix, iw = rx + 18, rail_w - 36
        rects["speakers_label"] = (ix, ry - label_h, iw, label_h)
        rects["speakers"] = (ix, ry - label_h - 8 - 30, iw, 30.0)
        rects["topic_label"] = (ix, ry - label_h - 8 - 30 - 20 - label_h, iw, label_h)
        rects["topic"] = (ix, ry - label_h - 8 - 30 - 20 - label_h - 8 - 30, iw, 30.0)
        rects["clip"] = (ix, rects["topic"][1] - 12 - 32, 150.0, 32.0)
        rects["status"] = (ix, bottom + 14, iw, max(60.0, rects["clip"][1] - 20 - bottom - 14))
        help_rect = (pad, bottom, width - 2 * pad, col_top - bottom)
    else:
        foot = 66.0
        field_w = (inner - 118 - 16) / 2
        rects["speakers"] = (pad, 10.0, field_w, 28.0)
        rects["topic"] = (pad + field_w + 8, 10.0, field_w, 28.0)
        rects["clip"] = (pad + 2 * field_w + 16, 9.0, 118.0, 30.0)
        rects["status"] = (pad, 44.0, inner, label_h)
        body_bottom = foot
        if mode == "wide":
            left = round((inner - gap) * 0.56)
            right_w = inner - gap - left
            col_top = top - 4
            rects["transcript_label"] = (pad, col_top - label_h, left - 160, label_h)
            rects["copy_transcript"] = (pad + left - 150, col_top - btn_h + 2, 150.0, btn_h)
            rects["transcript"] = (pad, body_bottom + 4, left, col_top - label_h - 10 - body_bottom - 4)
            cx = pad + left + gap
            rects["cards_label"] = (cx, col_top - label_h, right_w - 200, label_h)
            rects["audience"] = (cx + right_w - 200, col_top - btn_h + 2, 200.0, btn_h)
            _marker(rects, cx, right_w, body_bottom + 4)
            rects["cards"] = (cx, body_bottom + band - 4, right_w, col_top - label_h - 10 - body_bottom - band + 4)
        else:
            col_top = top - 4
            usable = col_top - body_bottom - 2 * (label_h + 10) - band - 8
            transcript_h = max(90.0, usable * 0.42)
            cards_h = max(60.0, usable - transcript_h)
            rects["transcript_label"] = (pad, col_top - label_h, inner - 160, label_h)
            rects["copy_transcript"] = (pad + inner - 150, col_top - btn_h + 2, 150.0, btn_h)
            ty = col_top - label_h - 10 - transcript_h
            rects["transcript"] = (pad, ty, inner, transcript_h)
            cl_y = ty - 12 - label_h
            rects["cards_label"] = (pad, cl_y, inner - 200, label_h)
            rects["audience"] = (pad + inner - 200, cl_y - 4, 200.0, btn_h)
            rects["cards"] = (pad, body_bottom + band, inner, max(60.0, cl_y - 10 - body_bottom - band))
            _marker(rects, pad, inner, body_bottom + (band - MARKER_SIZE[1]) / 2)
        help_rect = (pad, body_bottom + 4, inner, top - body_bottom - 8)
    rects["help_body"] = help_rect
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
    "Drei Ansichten\n"
    "Der Schalter oben links wählt die Ansicht. Die Wahl bleibt gespeichert. "
    "Meeting ist die ruhige Ansicht zum Teilen: nur Entscheidungen, nächste Schritte und Anfragen, "
    "ohne KI-Ideen, ohne private Hinweise, ohne Knöpfe an den Punkten. Das Transkript ist nur eine leise Zeile «Zuletzt». "
    "Ambient läuft nebenher, wenn du allein arbeitest: eine Zeile pro Punkt und unten «Kontext für Claude» "
    "(letzte 10 Minuten oder ganze Sitzung). «Kopieren» legt diesen Ausschnitt in die Zwischenablage, "
    "«Claude starten» öffnet eine Claude-Sitzung, die den Ausschnitt nur als Hintergrund bekommt und zuerst nachfragt. "
    "Intern zeigt das ganze Transkript, alle Karten samt KI-Ideen und die Werkzeuge.\n\n"
    "Fenstergrösse\n"
    "Das Layout folgt dem Fenster: schmal eine Spalte, breit mehrere Spalten. "
    "Der grüne Fensterknopf öffnet das native Vollbild mit eigenem, grosszügigerem Layout. "
    "Schliessen versteckt nur das Fenster. «Recorder-Fenster öffnen» im Menü bringt es zurück.\n\n"
    "Neue Punkte\n"
    "Was schon dasteht, bewegt sich nicht und verschwindet nicht. Neue Punkte warten hinter «N neue Punkte ↓»; "
    "ein Klick holt sie in die Liste, darüber steht «Neu seit …».\n\n"
    "Prompt-Stelle markieren\n"
    "Merkt die aktuelle Stelle einer laufenden Aufnahme, damit du die diktierte Aufgabe später wiederfindest.\n\n"
    "Teilnehmende (Intern)\n"
    "Namen mit Komma trennen, Enter speichert. Vorschläge kommen aus der Kalendereinladung. "
    "Das finale Transkript ordnet die Stimmen diesen Namen zu; nicht genannte Eingeladene werden ignoriert. "
    "Am besten vor dem Stoppen eintragen.\n\n"
    "Transkript kopieren (Intern)\n"
    "Kopiert das ganze bisherige Transkript dieser Aufnahme, während und nach der Aufnahme.\n\n"
    "Themenauszug (Intern)\n"
    "Das Themenfeld und «Auszug kopieren» kopieren passende Textstellen der laufenden Aufnahme. "
    "«Themenauszug kopieren …» im Menü wählt ein gespeichertes Transkript.\n\n"
    "Diktierte Aufgaben kopieren …\n"
    "Sucht markierte oder erkannte Aufgaben in einem gespeicherten Transkript. Du wählst aus, was kopiert wird.\n\n"
    "Private Hinweise zeigen (Intern)\n"
    "Mit Häkchen erscheinen auch private Beratungshinweise. Meeting und Ambient zeigen sie nie. "
    "KI-Fragen und KI-Ideen sind Vorschläge, keine Aussagen oder Beschlüsse.\n\n"
    "Scrollen\n"
    "Das Transkript folgt dem Neuesten nur, solange du am Ende bist. Scrollst du nach oben, bleibt die Ansicht stehen; "
    "«↓ Zum Neuesten» springt zurück und folgt wieder.\n\n"
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
    "help": "Erklärt die Aufnahme, Markierungen, Themenauszüge und die drei Ansichten.",
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
