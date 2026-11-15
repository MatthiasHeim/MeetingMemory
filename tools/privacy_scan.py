"""Find private passages (family life, a person's health) in a meeting, locally.

Shadow tool: it changes nothing. It reads either the audio (local Whisper on MLX) or an
existing transcript, scores ~30 s windows with the local Decider, and writes the spans
it would cut before audio goes to Gemini. Output holds timestamps and probabilities
only, never transcript text, so the report itself can be shown anywhere.

    python3 tools/privacy_scan.py audio Recordings/2026-11-09_21-29-23.wav -o scan.json
    python3 tools/privacy_scan.py transcript Transcripts/2026-11-09_21-29-23.json -o scan.json

Uses /opt/homebrew/bin/python3 (mlx_whisper is installed there, not in the venv).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DECIDER_HELPER = Path.home() / ".agents" / "skills" / "decider" / "scripts" / "decider_decide.py"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
WINDOW_SEC = 30.0
PAD_SEC = 15.0
THRESHOLD = 0.5
MAX_WINDOW_CHARS = 6000  # keeps state + question well under the Decider's 8,192 tokens

QUESTIONS = {
    "private_family": {
        "type": "noul",
        "instructions": (
            "Do the speakers in `window` talk about their own private or family life: children, "
            "partner, parenting, family problems, relatives, home life? `previous` is only context."
        ),
        "criteria": {
            "true": "Private or family matters of a speaker or of people close to them",
            "false": "Business, projects, products, clients, or brief small talk (weather, travel, scheduling, a greeting)",
        },
    },
    "personal_health": {
        "type": "noul",
        "instructions": (
            "Does `window` discuss the health of a specific person: illness, diagnosis, therapy, "
            "development, medication, doctor or hospital visits? `previous` is only context."
        ),
        "criteria": {
            "true": "Health or medical care of an identifiable person (a speaker, a child, a relative, a patient)",
            "false": "No individual's health; business talk about healthcare products, clinics or patient-data systems is false",
        },
    },
}


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    text: str


def parse_timestamp(stamp: str) -> float:
    parts = [int(p) for p in stamp.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


LINE_RE = re.compile(r"^\[(\d{1,2}(?::\d{2}){1,2})\]\s*(.*)$")


def transcript_segments(transcript: str, duration: float | None = None) -> list[Segment]:
    """`[MM:SS] Speaker: text` lines -> segments; a line ends where the next begins."""
    rows = []
    for line in transcript.splitlines():
        match = LINE_RE.match(line.strip())
        if match:
            rows.append((parse_timestamp(match.group(1)), match.group(2)))
    out = []
    for index, (start, text) in enumerate(rows):
        end = rows[index + 1][0] if index + 1 < len(rows) else (duration or start + 10.0)
        out.append(Segment(start, max(end, start), text))
    return out


def windows(segments: Sequence[Segment], size: float = WINDOW_SEC) -> list[Window]:
    """Group consecutive segments into windows of about `size` seconds."""
    out: list[Window] = []
    bucket: list[Segment] = []
    for segment in segments:
        if bucket and segment.end - bucket[0].start > size:
            out.append(Window(bucket[0].start, bucket[-1].end, " ".join(s.text for s in bucket)[:MAX_WINDOW_CHARS]))
            bucket = []
        bucket.append(segment)
    if bucket:
        out.append(Window(bucket[0].start, bucket[-1].end, " ".join(s.text for s in bucket)[:MAX_WINDOW_CHARS]))
    return out


def spans(windows_: Sequence[Window], scores: Sequence[dict[str, float]], *, threshold: float = THRESHOLD,
          pad: float = PAD_SEC, duration: float | None = None) -> list[dict]:
    """Flagged windows -> padded, merged spans: what would be cut."""
    raw = []
    for window, score in zip(windows_, scores):
        hits = {key: value for key, value in score.items() if value >= threshold}
        if hits:
            raw.append([max(0.0, window.start - pad), window.end + pad, max(hits.values()), sorted(hits)])
    merged: list[list] = []
    for start, end, p, reasons in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], p)
            merged[-1][3] = sorted(set(merged[-1][3]) | set(reasons))
        else:
            merged.append([start, end, p, reasons])
    limit = duration if duration else float("inf")
    return [{"start": round(s, 1), "end": round(min(e, limit), 1), "max_p": round(p, 3), "reasons": r} for s, e, p, r in merged]


def decide(window: Window, previous: Window | None, helper: Path = DECIDER_HELPER) -> dict[str, float]:
    body = {"state": {"previous": previous.text[-1500:] if previous else "", "window": window.text}, "questions": QUESTIONS}
    process = subprocess.run(
        [sys.executable, str(helper), "--classification", "confidential", "--request", "-", "--answers-only"],
        input=json.dumps(body, ensure_ascii=False), capture_output=True, text=True, timeout=300,
    )
    if process.returncode:
        raise RuntimeError(f"Decider failed ({process.returncode}); check `curl -s 127.0.0.1:8092/health`")
    answers = json.loads(process.stdout)
    return {key: float(answers[key]["noul"] if isinstance(answers[key], dict) else answers[key]) for key in QUESTIONS}


def score_windows(windows_: Sequence[Window]) -> list[dict[str, float]]:
    with ThreadPoolExecutor(max_workers=4) as pool:  # the Decider server batches concurrent requests
        return list(pool.map(lambda i: decide(windows_[i], windows_[i - 1] if i else None), range(len(windows_))))


def whisper_segments(audio: Path, language: str = "de") -> list[Segment]:
    import mlx_whisper  # local only; imported lazily so tests need no MLX

    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=WHISPER_MODEL, language=language,
                                    condition_on_previous_text=False, verbose=None)
    return [Segment(float(s["start"]), float(s["end"]), s["text"].strip()) for s in result["segments"] if s["text"].strip()]


def audio_duration(audio: Path) -> float | None:
    try:
        out = subprocess.run(["afinfo", str(audio)], capture_output=True, text=True, timeout=30).stdout
        match = re.search(r"estimated duration: ([\d.]+)", out)
        return float(match.group(1)) if match else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run(mode: str, path: Path) -> dict:
    started = time.time()
    timings = {}
    if mode == "audio":
        duration = audio_duration(path)
        segments = whisper_segments(path)
        timings["asr_s"] = round(time.time() - started, 1)
    else:
        data = json.loads(path.read_text())
        segments = transcript_segments(data["transcript"])
        duration = segments[-1].end if segments else None
    wins = windows(segments)
    t = time.time()
    scores = score_windows(wins)
    timings["decider_s"] = round(time.time() - t, 1)
    flagged = spans(wins, scores, duration=duration)
    return {
        "source": path.name, "mode": mode, "duration_s": duration, "windows": len(wins),
        "threshold": THRESHOLD, "pad_s": PAD_SEC, "timings": timings,
        "spans": flagged, "cut_s": round(sum(s["end"] - s["start"] for s in flagged), 1),
        "scores": [{"start": round(w.start, 1), "end": round(w.end, 1), **{k: round(v, 3) for k, v in s.items()}}
                   for w, s in zip(wins, scores)],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["audio", "transcript"])
    parser.add_argument("path", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run(args.mode, args.path)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("source", "mode", "windows", "timings", "spans", "cut_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
