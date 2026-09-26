from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sidecar.selection import CLIP_HEADER, select_clip, selected_indices  # noqa: E402
from sidecar.transcript import (  # noqa: E402
    TranscriptLine,
    format_timestamp,
    load_transcript,
    parse_transcript_lines,
    timestamp_seconds,
)


def _line(index: int, text: str, seconds: int | None = None) -> TranscriptLine:
    seconds = index * 10 if seconds is None else seconds
    return TranscriptLine(index, float(seconds), f"{seconds // 60:02d}:{seconds % 60:02d}", "Alex", text)


def test_loads_finished_recorder_json_and_ignores_non_turn_bookkeeping(tmp_path):
    (tmp_path / "2026-09-08_14-31-16.json").write_text(
        json.dumps(
            {
                "title": "Synthetic design review",
                "transcript": "[00:01] Alex: First sentence.\n\n[01:02] Bea: Second sentence.\nnot a turn",
            }
        ),
        encoding="utf-8",
    )

    transcript = load_transcript("2026-09-08_14-31-16", tmp_path)

    assert transcript.title == "Synthetic design review"
    assert [line.text for line in transcript.lines] == ["First sentence.", "Second sentence."]
    assert [line.seconds for line in transcript.lines] == [1.0, 62.0]


def test_timestamp_helpers_support_hours_and_reject_bad_values():
    assert timestamp_seconds("43:05") == 2585
    assert timestamp_seconds("1:03:05") == 3785
    assert format_timestamp(2585) == "43:05"
    assert format_timestamp(3785) == "01:03:05"
    with pytest.raises(ValueError):
        timestamp_seconds("03:99")


def test_hysteresis_grows_bridges_requires_two_seeds_and_removes_pure_fillers():
    lines = [
        _line(0, "role overview"),
        _line(1, "mhm"),
        _line(2, "admin permissions"),
        _line(3, "unrelated interlude"),
        _line(4, "onboarding flow"),
        _line(5, "client access"),
        _line(6, "one isolated mention"),
    ]
    probabilities = [0.72, 0.28, 0.69, 0.01, 0.71, 0.71, 0.01]

    # The first cluster grows through the filler but no filler is copied; the
    # second has two seeds and is retained. The final singleton is dropped.
    assert selected_indices(probabilities, lines) == (0, 2, 3, 4, 5)
    clip = select_clip(lines, probabilities)

    assert clip.text.startswith(CLIP_HEADER + "\n\n")
    assert "mhm" not in clip.text
    assert "[…]" not in clip.text
    assert "one isolated mention" not in clip.text


def test_clip_omits_filler_inside_a_kept_stretch_without_a_gap_marker():
    """A copied clip must not claim omitted substance where only filler was removed."""
    lines = [
        _line(0, "Synthetic roles overview."),
        _line(1, "Yeah."),
        _line(2, "Okay."),
        _line(3, "Synthetic permissions detail."),
    ]

    clip = select_clip(lines, [0.9, 0.3, 0.3, 0.9])

    assert clip.indices == (0, 3)
    assert "Yeah." not in clip.text and "Okay." not in clip.text
    assert "[…]" not in clip.text


def test_clip_marks_a_gap_between_clusters_separated_by_omitted_substance():
    lines = [
        _line(0, "Synthetic roles overview."),
        _line(1, "Yeah."),
        _line(2, "Synthetic permissions detail."),
        _line(3, "Synthetic unrelated discussion."),
        _line(4, "Synthetic unrelated continuation."),
        _line(5, "Synthetic unrelated conclusion."),
        _line(6, "Synthetic access control detail."),
        _line(7, "Synthetic access control decision."),
    ]

    clip = select_clip(lines, [0.9, 0.3, 0.9, 0.01, 0.01, 0.01, 0.9, 0.9])

    assert clip.indices == (0, 2, 6, 7)
    assert clip.text.count("[…]") == 1


def test_widen_uses_the_explicit_lower_threshold_pair():
    lines = [_line(0, "roles"), _line(1, "permissions"), _line(2, "onboarding")]
    probabilities = [0.55, 0.16, 0.56]

    default = select_clip(lines, probabilities)
    wide = select_clip(lines, probabilities, widen=True)

    assert default.line_count == 0
    assert wide.indices == (0, 1, 2)
    assert (wide.seed_threshold, wide.grow_threshold) == (0.50, 0.15)
