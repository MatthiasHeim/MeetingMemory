"""Deterministic clip selection over judge probabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .transcript import TranscriptLine


CLIP_HEADER = "Das haben wir im Meeting besprochen:"
DEFAULT_SEED_THRESHOLD = 0.60
DEFAULT_GROW_THRESHOLD = 0.25
WIDE_SEED_THRESHOLD = 0.50
WIDE_GROW_THRESHOLD = 0.15

# Short acknowledgements carry no stand-alone meeting substance. This list is
# intentionally conservative; it does not strip words from a substantive line.
_FILLER = re.compile(
    r"^(?:m+h+m*|mm+|äh+m*|uh+m*|okay\.?|ok\.?|ja\.?|jep\.?|"
    r"yeah\.?|yes\.?|yep\.?|right\.?|genau\.?|super\.?|danke\.?|"
    r"bitte\.?|\[.*\])$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClipResult:
    """A ready-to-copy clip plus the deterministic decisions behind it."""

    text: str
    lines: tuple[TranscriptLine, ...]
    indices: tuple[int, ...]
    seed_threshold: float
    grow_threshold: float
    widened: bool

    @property
    def line_count(self) -> int:
        return len(self.lines)


def is_filler(line: TranscriptLine) -> bool:
    return bool(_FILLER.fullmatch(line.text.strip()))


def _selected_mask(
    probabilities: Sequence[float],
    lines: Sequence[TranscriptLine],
    *,
    seed_threshold: float = DEFAULT_SEED_THRESHOLD,
    grow_threshold: float = DEFAULT_GROW_THRESHOLD,
    min_seeds: int = 2,
    bridge: int = 2,
) -> tuple[bool, ...]:
    """Apply the §6.1 seed/grow/bridge/min-seed hysteresis rule exactly.

    The mask intentionally retains filler membership. Rendering removes pure
    filler turns, but needs to know whether those turns belonged to a kept
    stretch so it does not render a false clip-gap marker.
    """
    if len(probabilities) != len(lines):
        raise ValueError("probabilities and transcript lines must have equal length")
    if not 0 <= grow_threshold <= seed_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= grow <= seed <= 1")
    values = [max(0.0, min(1.0, float(value))) for value in probabilities]
    # An acknowledgement can grow/bridge a substantive cluster, but it must
    # never itself be one of the two required seeds.
    keep = [
        value >= seed_threshold and not is_filler(lines[index])
        for index, value in enumerate(values)
    ]

    # Grow outward from every existing kept line until no eligible neighbour
    # remains. A loop, rather than a one-pass expansion, preserves chains.
    changed = True
    while changed:
        changed = False
        for index, value in enumerate(values):
            if keep[index] or value < grow_threshold:
                continue
            left = index > 0 and keep[index - 1]
            right = index + 1 < len(keep) and keep[index + 1]
            if left or right:
                keep[index] = True
                changed = True

    # Bridge gaps of at most two omitted transcript lines.
    for index, included in enumerate(keep):
        if included:
            continue
        left = next(
            (candidate for candidate in range(index - 1, max(-1, index - bridge - 1), -1) if keep[candidate]),
            None,
        )
        right = next(
            (candidate for candidate in range(index + 1, min(len(keep), index + bridge + 1)) if keep[candidate]),
            None,
        )
        if left is not None and right is not None and right - left <= bridge + 1:
            keep[index] = True

    # A cluster must contain at least two high-confidence seeds. Do this after
    # bridging so a bridge cannot turn two weak islands into a false clip.
    index = 0
    while index < len(keep):
        if not keep[index]:
            index += 1
            continue
        end = index
        while end < len(keep) and keep[end]:
            end += 1
        if sum(
            values[candidate] >= seed_threshold and not is_filler(lines[candidate])
            for candidate in range(index, end)
        ) < min_seeds:
            for candidate in range(index, end):
                keep[candidate] = False
        index = end

    return tuple(keep)


def selected_indices(
    probabilities: Sequence[float],
    lines: Sequence[TranscriptLine],
    *,
    seed_threshold: float = DEFAULT_SEED_THRESHOLD,
    grow_threshold: float = DEFAULT_GROW_THRESHOLD,
    min_seeds: int = 2,
    bridge: int = 2,
) -> tuple[int, ...]:
    """Return the substantive lines from the hysteresis selection mask."""
    keep = _selected_mask(
        probabilities,
        lines,
        seed_threshold=seed_threshold,
        grow_threshold=grow_threshold,
        min_seeds=min_seeds,
        bridge=bridge,
    )
    # Pure filler lines are omitted from the copied text and never count as a
    # seed, even when a model assigns them a high relevance probability.
    return tuple(index for index, included in enumerate(keep) if included and not is_filler(lines[index]))


def _format_indices(
    lines: Sequence[TranscriptLine], indices: Iterable[int], *, timestamps: bool,
    selected_mask: Sequence[bool] | None = None,
) -> str:
    chosen = tuple(indices)
    if not chosen:
        return CLIP_HEADER + "\n"
    output: list[str] = [CLIP_HEADER, ""]
    previous: int | None = None
    for index in chosen:
        # A marker represents omitted substantive content between distinct
        # selected clusters. Pure filler turns may be omitted from the copied
        # text even within one selected stretch, and must not manufacture a
        # false gap such as ``line, […], line``.
        omitted_substance = (
            previous is not None
            and any(
                not is_filler(lines[candidate])
                and (selected_mask is None or not selected_mask[candidate])
                for candidate in range(previous + 1, index)
            )
        )
        if omitted_substance:
            output.append("[…]")
        output.append(lines[index].display(timestamps=timestamps))
        previous = index
    return "\n".join(output)


def select_clip(
    lines: Sequence[TranscriptLine],
    probabilities: Sequence[float],
    *,
    widen: bool = False,
    timestamps: bool = True,
) -> ClipResult:
    """Select and render a verbatim clip. No model is allowed to rewrite text."""
    seed = WIDE_SEED_THRESHOLD if widen else DEFAULT_SEED_THRESHOLD
    grow = WIDE_GROW_THRESHOLD if widen else DEFAULT_GROW_THRESHOLD
    keep = _selected_mask(probabilities, lines, seed_threshold=seed, grow_threshold=grow)
    indices = tuple(index for index, included in enumerate(keep) if included and not is_filler(lines[index]))
    selected = tuple(lines[index] for index in indices)
    return ClipResult(
        text=_format_indices(lines, indices, timestamps=timestamps, selected_mask=keep),
        lines=selected,
        indices=indices,
        seed_threshold=seed,
        grow_threshold=grow,
        widened=widen,
    )
