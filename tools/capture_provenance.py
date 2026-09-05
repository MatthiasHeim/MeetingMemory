"""Keep independently clocked capture inputs and their timing uncertainty."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from speaker_integrity import atomic_json


def merge_filter(duration_seconds: float) -> str:
    # Padding, not cropping: amerge otherwise stops at the shortest input.
    duration = f"{duration_seconds:.9f}"
    return (f"[0:a]apad=whole_dur={duration}[mic];"
            f"[1:a]apad=whole_dur={duration}[sys];"
            "[mic][sys]amerge=inputs=2,pan=3.0|c0=c2|c1=c0|c2=c1[a]")


def archive_capture(root: Path, stem: str, paths: list[Path], metadata: dict) -> Path:
    dest = root / stem
    if dest.exists():
        dest = root / f"{stem}-{time.time_ns()}"
    dest.mkdir(parents=True)
    retained = []
    for path in paths:
        if path and path.exists():
            target = dest / path.name
            shutil.move(str(path), target)
            retained.append(str(target))
    atomic_json(dest / "capture.json", {**metadata, "retained_files": retained,
                "timeline": "sample_zero_merge_with_padded_ends",
                "synchronization": "independent_clocks_not_yet_aligned",
                "system_sample_timestamps": "unavailable_in_current_tap_binary"})
    return dest
