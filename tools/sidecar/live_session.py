"""Glue between the menu-bar recorder and the live engine plus side window.

The session starts after capture is already running. ``stop`` only signals the
worker; the panel stays open with whatever it has shown.
"""

from __future__ import annotations

from typing import Any, Callable

from .calibration_policy import OWNER_APPROVED_MODEL, OWNER_APPROVED_PROMPT_THRESHOLD
from .judges import GeminiJudge
from .live import (
    GeminiLiveTranscriber,
    LiveEngine,
    snapshot_recording_tails,
)
from .live import default_clean_prompt
from .prompts import PromptCalibrationError, calibrated_prompt_threshold
from .selection import select_clip


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
        judge: GeminiJudge | None = None,
        prompt_threshold: float | None = None,
    ):
        self.recorder = recorder
        self._copy_text = copy_text
        self._open_window = open_window
        self.panel = None
        self._api_key = api_key
        threshold = prompt_threshold if prompt_threshold is not None else _live_prompt_threshold()
        self.engine = LiveEngine(
            snapshot=lambda tail: snapshot_recording_tails(recorder, tail),
            transcriber=transcriber or GeminiLiveTranscriber(api_key=api_key),
            judge=judge if judge is not None else _maybe_judge(api_key),
            clean_prompt=default_clean_prompt(api_key),
            on_update=self._on_update,
            prompt_threshold=threshold,
        )
        self._clip_judge = judge if judge is not None else None

    def start(self) -> None:
        if self._open_window:
            from .live_window import LivePanel

            self.panel = LivePanel(on_copy_clip=self.copy_clip, on_copy_card=self.copy_card)
            self.panel.show()
        self.engine.start()

    def stop(self) -> None:
        """Signal the worker. The window stays until the user closes it."""
        self.engine.stop()

    def _on_update(self, lines, cards, status: str) -> None:
        if self.panel is not None:
            self.panel.apply_update(lines, cards, status)

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
            judge = self._clip_judge or GeminiJudge(api_key=self._api_key, model=OWNER_APPROVED_MODEL)
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


def _maybe_judge(api_key: str | None) -> GeminiJudge | None:
    try:
        return GeminiJudge(api_key=api_key, model=OWNER_APPROVED_MODEL)
    except Exception:
        return None

