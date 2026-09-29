"""Command-line entry points for finished-transcript clips and prompt cards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import (
    DEFAULT_EVAL_DIR,
    DEFAULT_REPORT_DIR,
    apply_owner_override,
    calibrate,
    markdown_table,
    read_calibration_report,
    write_calibration_report,
)
from .calibration_policy import OWNER_APPROVED_MODEL
from .gate import JudgeGateError
from .live import LIVE_TAIL_SECONDS, LIVE_TICK_SECONDS, replay_wav
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
    clip.add_argument(
        "--model", choices=(OWNER_APPROVED_MODEL,), default=OWNER_APPROVED_MODEL,
        help="the single owner-approved Gemini judge",
    )
    clip.add_argument("--without-timestamps", action="store_true", help="omit [MM:SS] prefixes")
    _common_paths(clip)

    prompts = subcommands.add_parser("prompts", help="marked and suggested dictated prompts")
    prompts.add_argument("--stem", required=True, help="recorder filename stem")
    prompts.add_argument("--judge", choices=("gemini", "jev"), default="gemini")
    prompts.add_argument(
        "--model", choices=(OWNER_APPROVED_MODEL,), default=OWNER_APPROVED_MODEL,
        help="the single owner-approved Gemini judge",
    )
    _common_paths(prompts)

    calibration = subcommands.add_parser("calibrate", help="run local-only Gemini calibration")
    calibration.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    calibration.add_argument(
        "--models",
        nargs="+",
        choices=(OWNER_APPROVED_MODEL,),
        default=(OWNER_APPROVED_MODEL,),
        help="the single owner-approved Gemini judge",
    )
    calibration.add_argument("--report", type=Path, help="aggregate-only report destination")
    calibration.add_argument(
        "--apply-owner-override",
        action="store_true",
        help="amend an existing aggregate-only report without calling a provider",
    )

    replay = subcommands.add_parser(
        "replay",
        help="run a WAV through the live pipeline on a simulated clock, without a window",
    )
    replay.add_argument("--wav", type=Path, required=True, help="microphone or merged recorder WAV")
    replay.add_argument("--tick", type=float, default=LIVE_TICK_SECONDS, help="seconds between tails")
    replay.add_argument("--tail", type=float, default=LIVE_TAIL_SECONDS, help="seconds of audio per tail")
    replay.add_argument("--from-seconds", type=float, default=0.0, help="first audio second to reveal")
    replay.add_argument("--to-seconds", type=float, default=None, help="last audio second to reveal")
    replay.add_argument(
        "--model",
        choices=(OWNER_APPROVED_MODEL,),
        default=OWNER_APPROVED_MODEL,
        help="the pinned Gemini model for live transcription",
    )
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
        uncertainty = " — Zuordnung unsicher" if prompt.association_label else ""
        print(f"--- Prompt {index} ({label}{offset}{uncertainty}) ---")
        if prompt.clean_text.strip():
            print(prompt.clean_text.strip())
            print("--- verbatim ---")
        print(prompt.text)
    return 0


def _load_repo_env() -> None:
    """Load the repository .env the same way the menu-bar app does, without printing it.

    A git worktree does not share the main checkout's untracked ``.env``. When
    this checkout has none, the main checkout's file is used if it is present.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for env_path in _env_candidates():
        if env_path.is_file():
            load_dotenv(env_path)
            return


def _env_candidates() -> list[Path]:
    repo = Path(__file__).resolve().parents[2]
    candidates = [repo / ".env"]
    git = repo / ".git"
    if git.is_file():
        try:
            line = git.read_text(encoding="utf-8").strip()
        except OSError:
            line = ""
        if line.startswith("gitdir:"):
            gitdir = Path(line.split(":", 1)[1].strip())
            candidates.append(gitdir.parents[2] / ".env")
    return candidates


def _run_replay(args: argparse.Namespace) -> int:
    _load_repo_env()
    from .judges import GeminiJudge
    from .live import GeminiLiveTranscriber, default_clean_prompt
    from .prompts import PromptCalibrationError, calibrated_prompt_threshold

    try:
        threshold = calibrated_prompt_threshold(args.model)
    except PromptCalibrationError:
        from .calibration_policy import OWNER_APPROVED_PROMPT_THRESHOLD

        threshold = OWNER_APPROVED_PROMPT_THRESHOLD
    replay_wav(
        args.wav,
        transcriber=GeminiLiveTranscriber(model=args.model),
        judge=GeminiJudge(model=args.model),
        clean_prompt=default_clean_prompt(),
        tick=args.tick,
        tail=args.tail,
        prompt_threshold=threshold,
        start_seconds=args.from_seconds,
        end_seconds=args.to_seconds,
    )
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    if args.apply_owner_override:
        destination = args.report or DEFAULT_REPORT_DIR / "latest.json"
        report = apply_owner_override(read_calibration_report(destination))
    else:
        report = calibrate(eval_dir=args.eval_dir, models=args.models)
        destination = args.report
    destination = write_calibration_report(report, destination)
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
        if args.command == "replay":
            return _run_replay(args)
    except (JudgeGateError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"sidecar: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")
