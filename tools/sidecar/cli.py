"""Command-line entry points for finished-transcript clips and prompt cards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import DEFAULT_EVAL_DIR, calibrate, markdown_table, write_calibration_report
from .gate import JudgeGateError
from .service import clip_for_stem, prompts_for_stem


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transcripts-dir", type=Path, help="override Transcripts directory")
    parser.add_argument("--recordings-dir", type=Path, help="override Recordings directory")
    parser.add_argument("--cache-dir", type=Path, help="override local sidecar probability cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finished-transcript meeting sidecar")
    subcommands = parser.add_subparsers(dest="command", required=True)

    clip = subcommands.add_parser("clip", help="copy-ready verbatim topic clip")
    clip.add_argument("--stem", required=True, help="recorder filename stem")
    clip.add_argument("--topic", required=True, help="free-text meeting topic")
    clip.add_argument("--widen", action="store_true", help="lower clip seed/grow thresholds one calibrated step")
    clip.add_argument("--judge", choices=("gemini", "jev"), default="gemini")
    clip.add_argument("--model", default="gemini-3.8-flash", help="Gemini model when using Gemini")
    clip.add_argument("--without-timestamps", action="store_true", help="omit [MM:SS] prefixes")
    _common_paths(clip)

    prompts = subcommands.add_parser("prompts", help="marked and suggested dictated prompts")
    prompts.add_argument("--stem", required=True, help="recorder filename stem")
    prompts.add_argument("--judge", choices=("gemini", "jev"), default="gemini")
    prompts.add_argument("--model", default="gemini-3.8-flash", help="Gemini model when using Gemini")
    _common_paths(prompts)

    calibration = subcommands.add_parser("calibrate", help="run local-only Gemini calibration")
    calibration.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    calibration.add_argument(
        "--models",
        nargs="+",
        default=("gemini-3.8-flash", "gemini-3.1-flash-lite"),
        help="Gemini model ids to compare",
    )
    calibration.add_argument("--report", type=Path, help="aggregate-only report destination")
    return parser


def _run_clip(args: argparse.Namespace) -> int:
    result = clip_for_stem(
        args.stem,
        args.topic,
        requested_judge=args.judge,
        transcripts_root=args.transcripts_dir,
        recordings_root=args.recordings_dir,
        cache_root=args.cache_dir,
        gemini_model=args.model,
        widen=args.widen,
        timestamps=not args.without_timestamps,
    )
    print(result.text)
    return 0


def _run_prompts(args: argparse.Namespace) -> int:
    results = prompts_for_stem(
        args.stem,
        requested_judge=args.judge,
        transcripts_root=args.transcripts_dir,
        recordings_root=args.recordings_dir,
        cache_root=args.cache_dir,
        gemini_model=args.model,
    )
    if not results:
        print("No marked or suggested prompts found.")
        return 0
    for index, prompt in enumerate(results, start=1):
        label = "marked" if prompt.source == "mark" else "suggested"
        offset = "" if prompt.mark_seconds is None else f" at mark {prompt.mark_seconds:.1f}s"
        print(f"--- Prompt {index} ({label}{offset}) ---")
        print(prompt.text)
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    report = calibrate(eval_dir=args.eval_dir, models=args.models)
    destination = write_calibration_report(report, args.report)
    print(markdown_table(report))
    print(f"\nAggregate calibration report: {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "clip":
            return _run_clip(args)
        if args.command == "prompts":
            return _run_prompts(args)
        if args.command == "calibrate":
            return _run_calibrate(args)
    except (JudgeGateError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"sidecar: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")
