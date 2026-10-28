"""Glue between the menu-bar recorder and the live engine plus side window.

The session starts after capture is already running. ``stop`` only signals the
worker; the panel stays open with whatever it has shown.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable

from .calibration_policy import OWNER_APPROVED_MODEL, OWNER_APPROVED_PROMPT_THRESHOLD
from .judges import DeciderJudge, GeminiJudge, Judge
from .live import (
    GeminiLiveTranscriber,
    LiveEngine,
    snapshot_recording_tails,
)
from .live import default_clean_prompt
from .decider_policy import DECIDER_PROMPT_THRESHOLD, clip_thresholds
from .prompts import PromptCalibrationError, calibrated_prompt_threshold
from .selection import select_clip
from .card_assistant import CardAssistant
from .recording_files import write_live_transcript


class RecordingLiveSession:
    """One recording's live transcript, cards, and floating window."""

    def __init__(
        self,
        recorder: Any,
        *,
        api_key: str | None,
        copy_text: Callable[[str], None],
        open_window: bool = True,
        transcriber: GeminiLiveTranscriber | None = None,
        judge: Judge | None = None,
        decision_stem: str | None = None,
        recordings_root=None,
        prompt_threshold: float | None = None,
        on_unavailable: Callable[[str], None] | None = None,
        panel=None,
        stem: str | None = None,
    ):
        self.recorder = recorder
        # Without a stem (tests, previews) nothing is written to disk.
        self._stem = stem
        self._recordings_root = recordings_root
        self._saved_lines: tuple | None = None
        self._saved_cards: tuple | None = None
        self._recording_active = True
        self._copy_text = copy_text
        self._open_window = open_window
        self.panel = panel
        self._api_key = api_key
        self._judge_factory = (
            (lambda: DeciderJudge(stem=decision_stem, recordings_root=recordings_root))
            if decision_stem else (lambda: _maybe_judge(api_key))
        )
        # Clients and the calibration read happen on the worker's first tick.
        # Building them here used to block Start for about half a second.
        assistant = CardAssistant(api_key)
        self.engine = LiveEngine(
            snapshot=lambda tail: snapshot_recording_tails(recorder, tail),
            transcriber=transcriber,
            transcriber_factory=None
            if transcriber is not None
            else (lambda: GeminiLiveTranscriber(api_key=api_key)),
            judge=judge,
            judge_factory=None if judge is not None else self._judge_factory,
            threshold_factory=None if prompt_threshold is not None else (
                (lambda: DECIDER_PROMPT_THRESHOLD) if decision_stem else _live_prompt_threshold
            ),
            contextual_cards=True,
            clean_card=assistant.clean,
            suggest_cards=assistant.suggest,
            on_update=self._on_update,
            on_unavailable=on_unavailable,
            on_diagnostics=_log_live_diagnostics,
            prompt_threshold=prompt_threshold
            if prompt_threshold is not None
            else OWNER_APPROVED_PROMPT_THRESHOLD,
        )
        self._clip_judge = judge

    def start(self) -> None:
        if self._open_window:
            from .live_window import LivePanel

            if self.panel is None:
                self.panel = LivePanel(on_copy_clip=self.copy_clip, on_copy_card=self.copy_card)
            self.panel.reset_session()
            self.panel.set_recording_state("recording")
            self.panel.set_diagnostics_callback(_log_live_diagnostics)
            self.panel.set_audience_callback(self.engine.set_audience)
            self.engine.set_audience(self.panel.audience)
            self.panel.show()
        self.engine.start()

    def stop(self) -> None:
        """Signal the worker. The window stays until the user closes it."""
        self.engine.stop()
        self._recording_active = False
        # Committed lines only; cards stay as last shown. The closing tick may
        # still publish (and so rewrite) once more.
        self._save_live_transcript(self.engine.lines, force=True)
        if self.panel is not None:
            self.panel.set_recording_state("stopped")

    def retire(self) -> None:
        self.engine.retire()
        self.panel = None

    def _on_update(self, lines, cards, status: str) -> None:
        self._save_live_transcript(lines, cards)
        if self.panel is not None:
            self.panel.apply_update(lines, cards, status)

    def _save_live_transcript(self, lines, cards=None, *, force: bool = False) -> None:
        """Keep ``<stem>.live.json`` current; a failed write never touches recording.

        The only writer of that file. Lines and cards are written whenever
        either changes, and once more at stop with ``recording_active: false``.
        """
        if self._stem is None:
            return
        if cards is None:
            cards = self._saved_cards if self._saved_cards is not None else ()
        snapshot = (tuple(lines), tuple(cards))
        if not force and snapshot == (self._saved_lines, self._saved_cards):
            return
        try:
            write_live_transcript(self._stem, snapshot[0], root=self._recordings_root,
                                  model=OWNER_APPROVED_MODEL, recording_active=self._recording_active,
                                  cards=snapshot[1])
            self._saved_lines, self._saved_cards = snapshot
        except Exception as exc:
            print(f"Live transcript not saved: {type(exc).__name__}: {exc}", file=sys.stderr)

    def copy_card(self, text: str) -> None:
        if not text.strip():
            return
        self._copy_text(text)

    def copy_clip(self, topic: str) -> None:
        topic = topic.strip()
        if not topic:
            self.engine.status = "Thema fehlt."
            self._on_update(self.engine.lines, self.engine.cards, self.engine.status)
            return
        lines = [
            line.as_transcript_line(index) for index, line in enumerate(self.engine.lines)
        ]
        if not lines:
            self.engine.status = "Noch kein Live-Transkript."
            self._on_update(self.engine.lines, self.engine.cards, self.engine.status)
            return
        try:
            judge = self._clip_judge or self.engine._judge
            if judge is None:
                judge = self._judge_factory()
                if judge is None:
                    raise RuntimeError("Decision model unavailable")
                self._clip_judge = judge
            clip = select_clip(
                lines,
                judge.judge(
                    lines,
                    {
                        "relevant": (
                            "Is the current transcript line substantively relevant to this topic: "
                            f"{topic!r}? Score semantic content about the topic, not a generic "
                            "acknowledgement or a passing word overlap."
                        )
                    },
                )["relevant"],
                **clip_thresholds(judge.backend),
            )
            self._copy_text(clip.text)
            message = (
                f"{clip.line_count} Zeilen kopiert."
                if clip.lines
                else "Nichts Passendes gefunden; Top-Kandidaten wurden kopiert."
            )
            self.engine.status = message
        except Exception as exc:
            self.engine.status = f"Clip fehlgeschlagen ({type(exc).__name__}). Die Aufnahme läuft weiter."
        self._on_update(self.engine.lines, self.engine.cards, self.engine.status)


def _live_prompt_threshold() -> float:
    try:
        return calibrated_prompt_threshold(OWNER_APPROVED_MODEL)
    except PromptCalibrationError:
        return OWNER_APPROVED_PROMPT_THRESHOLD


def _log_live_diagnostics(event: dict[str, Any]) -> None:
    """Operational evidence without recording content or exception messages."""
    print("LIVE_SIDECAR " + json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(), **event,
    }, sort_keys=True), file=sys.stderr, flush=True)


def _maybe_judge(api_key: str | None) -> GeminiJudge | None:
    try:
        return GeminiJudge(api_key=api_key, model=OWNER_APPROVED_MODEL)
    except Exception:
        return None

