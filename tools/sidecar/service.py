"""High-level offline clerk operations shared by CLI and menu-bar UI."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .cache import ProbabilityCache
from .gate import judge_backend_for, load_recording_sidecar
from .judges import CachedJudge, GeminiJudge, JevJudge, Judge
from .prompts import PROMPT_THRESHOLD, PromptResult, calibrated_prompt_threshold, prompts_from_marks
from .selection import ClipResult, select_clip
from .transcript import Transcript, load_transcript


def relevance_question(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        raise ValueError("topic cannot be empty")
    return (
        "Is the current transcript line substantively relevant to this topic: "
        f"{topic!r}? Score semantic content about the topic, not a generic acknowledgement, "
        "unrelated implementation chatter, or a passing ambiguous word overlap."
    )


def create_judge(
    stem: str,
    *,
    requested: str | None = None,
    recordings_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    gemini_model: str = "gemini-3.8-flash",
    gemini_api_key: str | None = None,
    transcript_path: str | Path | None = None,
    gemini_factory: type[GeminiJudge] = GeminiJudge,
    jev_factory: type[JevJudge] = JevJudge,
) -> Judge:
    """Resolve the gate before constructing an optional Jev client.

    The order here is intentional and testable: a denied Jev choice never even
    creates a Jev adapter, so no OpenRouter-capable object exists on that path.
    """
    backend = judge_backend_for(
        stem,
        requested,
        recordings_root=recordings_root,
        transcript_path=transcript_path,
    )
    if backend == "gemini":
        base: Judge = gemini_factory(api_key=gemini_api_key, model=gemini_model)
    else:
        # JevJudge independently repeats this gate before construction and
        # before each helper call; pass the recording identity through rather
        # than treating this factory as a capability by itself.
        base = jev_factory(stem=stem, recordings_root=recordings_root)
    return CachedJudge(base, ProbabilityCache(cache_root), stem)


def clip_for_transcript(
    transcript: Transcript,
    topic: str,
    *,
    judge: Judge,
    widen: bool = False,
    timestamps: bool = True,
) -> ClipResult:
    probabilities = judge.judge(transcript.lines, {"relevant": relevance_question(topic)})["relevant"]
    return select_clip(transcript.lines, probabilities, widen=widen, timestamps=timestamps)


def clip_for_stem(
    stem: str,
    topic: str,
    *,
    requested_judge: str | None = None,
    transcripts_root: str | Path | None = None,
    recordings_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    gemini_model: str = "gemini-3.8-flash",
    gemini_api_key: str | None = None,
    widen: bool = False,
    timestamps: bool = True,
) -> ClipResult:
    transcript = load_transcript(stem, transcripts_root)
    judge = create_judge(
        stem,
        requested=requested_judge,
        recordings_root=recordings_root,
        cache_root=cache_root,
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        transcript_path=transcript.path,
    )
    return clip_for_transcript(transcript, topic, judge=judge, widen=widen, timestamps=timestamps)


def prompts_for_stem(
    stem: str,
    *,
    requested_judge: str | None = None,
    transcripts_root: str | Path | None = None,
    recordings_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    gemini_model: str = "gemini-3.8-flash",
    gemini_api_key: str | None = None,
    clean_prompt: Callable[[str], str] | None = None,
) -> list[PromptResult]:
    transcript = load_transcript(stem, transcripts_root)
    settings = load_recording_sidecar(stem, recordings_root)
    judge = create_judge(
        stem,
        requested=requested_judge,
        recordings_root=recordings_root,
        cache_root=cache_root,
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        transcript_path=transcript.path,
    )
    # A contract-covered Gemini call is only useful here when its exact model
    # has a threshold selected by a passing calibration. The explicitly gated
    # Jev path retains the historical baseline threshold.
    threshold = (
        calibrated_prompt_threshold(judge.model)
        if judge.backend == "gemini"
        else PROMPT_THRESHOLD
    )
    results = prompts_from_marks(
        transcript.lines,
        settings.marks,
        judge=judge,
        prompt_threshold=threshold,
        channel_lag_seconds=_channel_alignment_lag_seconds(transcript.payload),
        mic_origin_delay_seconds=settings.mic_first_sample_offset_seconds,
    )
    return _attach_clean_prompts(results, clean_prompt=clean_prompt, api_key=gemini_api_key)


def _attach_clean_prompts(
    results: list[PromptResult],
    *,
    clean_prompt: Callable[[str], str] | None,
    api_key: str | None,
) -> list[PromptResult]:
    """Add a ready-to-paste prompt beside each verbatim card.

    The cleaner is Gemini even when the detector was Jev: rewriting stays on
    the contract-covered route. A failure leaves ``clean_text`` empty so the
    card can still show the verbatim lines.
    """
    if clean_prompt is None:
        from .clean_prompt import clean_prompt_text

        def clean_prompt(verbatim: str, _key: str | None = api_key) -> str:
            return clean_prompt_text(verbatim, api_key=_key)

    attached: list[PromptResult] = []
    for result in results:
        clean = ""
        if result.text.strip():
            try:
                clean = clean_prompt(result.text).strip()
            except Exception:
                clean = ""
        attached.append(replace(result, clean_text=clean))
    return attached


def _channel_alignment_lag_seconds(payload: dict[str, Any]) -> float | None:
    """Read only the watcher's additive applied-alignment value."""
    metadata = payload.get("_meta")
    if not isinstance(metadata, dict):
        return None
    alignment = metadata.get("channel_alignment")
    if not isinstance(alignment, dict):
        return None
    value = alignment.get("lag_seconds")
    if isinstance(value, bool):
        return None
    try:
        lag = float(value)
    except (TypeError, ValueError):
        return None
    return lag if math.isfinite(lag) else None
