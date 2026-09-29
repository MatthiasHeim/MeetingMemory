"""Local probability cache; it stores hashes and scores, never transcript text."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .transcript import TranscriptLine, safe_stem


DEFAULT_CACHE_DIR = Path.home() / ".local" / "share" / "meeting-sidecar" / "cache"


def line_fingerprint(lines: Sequence[TranscriptLine], context: int) -> str:
    payload = {
        "context": context,
        "lines": [
            {
                "i": line.index,
                "t": line.timestamp,
                "s": line.speaker,
                "x": line.text,
            }
            for line in lines
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def question_fingerprint(question_id: str, instructions: Any) -> str:
    encoded = json.dumps(
        {"id": question_id, "instructions": instructions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def model_fingerprint(model: str | None) -> str:
    """Keep differently configured backends in separate cache paths."""
    return hashlib.sha256((model or "<none>").encode("utf-8")).hexdigest()


class ProbabilityCache:
    """Score-only disk cache partitioned by stem, backend and question."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or DEFAULT_CACHE_DIR).expanduser()

    def _path(
        self,
        stem: str,
        backend: str,
        question_id: str,
        instructions: Any,
        model: str | None,
    ) -> Path:
        # Question ids are used only as an informative prefix; the digest avoids
        # both unsafe filenames and accidental cache collisions on prompt edits.
        digest = question_fingerprint(question_id, instructions)
        safe_question = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in question_id)[:48]
        return self.root / safe_stem(stem) / backend / f"{safe_question}-{digest[:16]}-{model_fingerprint(model)[:12]}.json"

    def get(
        self,
        stem: str,
        backend: str,
        question_id: str,
        instructions: Any,
        *,
        input_fingerprint: str,
        model: str | None,
        line_count: int,
    ) -> list[float] | None:
        path = self._path(stem, backend, question_id, instructions, model)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        if value.get("input_sha256") != input_fingerprint or value.get("model") != model:
            return None
        scores = value.get("scores")
        if not isinstance(scores, list) or len(scores) != line_count:
            return None
        try:
            normalised = [float(score) for score in scores]
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(score) or score < 0 or score > 1 for score in normalised):
            return None
        return normalised

    def put(
        self,
        stem: str,
        backend: str,
        question_id: str,
        instructions: Any,
        *,
        input_fingerprint: str,
        model: str | None,
        scores: Sequence[float],
    ) -> None:
        path = self._path(stem, backend, question_id, instructions, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalised_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) or score < 0 or score > 1 for score in normalised_scores):
            raise ValueError("cache scores must be finite probabilities in 0..1")
        payload = {
            "schema_version": 2,
            "backend": backend,
            "model": model,
            "question_id": question_id,
            "question_sha256": question_fingerprint(question_id, instructions),
            "input_sha256": input_fingerprint,
            "scores": [round(score, 6) for score in normalised_scores],
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
