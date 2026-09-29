"""Offline meeting-sidecar clerk primitives.

The package deliberately works only with completed transcript JSON files.  It
does not participate in audio capture, the watcher, or transcription.
"""

from .gate import JudgeGateError, judge_backend_for
from .selection import ClipResult, select_clip
from .transcript import Transcript, TranscriptLine, load_transcript

__all__ = [
    "ClipResult",
    "JudgeGateError",
    "Transcript",
    "TranscriptLine",
    "judge_backend_for",
    "load_transcript",
    "select_clip",
]
