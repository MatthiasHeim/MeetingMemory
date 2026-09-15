#!/usr/bin/env python3
"""Run the standalone durable source-transcription worker.

This command intentionally reads only its input manifest and audio files.  It
does not load MeetingRecorder configuration, start a watcher, call a webhook,
or write to a database.  A Gemini request is made only for pending durable
clips and only when ``GEMINI_API_KEY`` is available to the invoked process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transcription_jobs import SourceTranscriptionPipeline, atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Durably transcribe bounded, source-separated audio clips."
    )
    parser.add_argument(
        "input_manifest",
        nargs="?",
        type=Path,
        help="JSON manifest containing {sources: [...], _meta: {capture_gaps: [...]}}.",
    )
    parser.add_argument(
        "--input-manifest",
        dest="input_manifest_option",
        type=Path,
        help="Named form of the input manifest path.",
    )
    parser.add_argument("--state-dir", required=True, type=Path, help="Durable job state directory.")
    parser.add_argument("--output", required=True, type=Path, help="Transcript JSON output path.")
    parser.add_argument("--model", required=True, help="Gemini model ID for this durable job.")
    parser.add_argument("--session-id", help="Opaque session identifier; defaults to manifest or file stem.")
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--chunk-seconds", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=2, help="Attempts per failed region per run.")
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="Inspect/write existing durable state without retrying prior failed regions.",
    )
    return parser


def _resolve_manifest_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    positional = args.input_manifest
    named = args.input_manifest_option
    if positional and named and positional != named:
        parser.error("provide input manifest once, positional or --input-manifest")
    path = positional or named
    if path is None:
        parser.error("an input manifest is required")
    return path


def _load_input_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read input manifest {path}: {exc}") from exc
    if isinstance(payload, list):
        return payload, [], None
    if not isinstance(payload, dict):
        raise ValueError("input manifest must be a source list or JSON object")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("input manifest must contain a sources list")
    metadata = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
    capture_gaps = payload.get("capture_gaps", metadata.get("capture_gaps", []))
    if not isinstance(capture_gaps, list):
        raise ValueError("capture_gaps must be a list when present")
    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise ValueError("session_id must be a string when present")
    return sources, capture_gaps, session_id


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    manifest_path = _resolve_manifest_path(args, parser)
    try:
        sources, capture_gaps, manifest_session_id = _load_input_manifest(manifest_path)
        session_id = args.session_id or manifest_session_id or manifest_path.stem
        pipeline = SourceTranscriptionPipeline(
            args.state_dir,
            args.model,
            request_timeout_seconds=args.request_timeout_seconds,
            job_timeout_seconds=args.job_timeout_seconds,
            chunk_seconds=args.chunk_seconds,
            max_attempts=args.max_attempts,
        )
        payload = pipeline.run(
            sources,
            session_id=session_id,
            retry_failed=not args.no_retry_failed,
            capture_gaps=capture_gaps,
        )
        atomic_write_json(args.output, payload)
    except Exception as exc:
        print(f"transcribe_sources: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    status = payload["_meta"]["durable_job_status"]
    print(f"durable transcription status: {status}", file=sys.stderr)
    # The output remains useful for a review-needed result.  A retryable
    # failure gets a distinct status for unattended callers that want to
    # schedule the next bounded resume.
    return 2 if status == "retryable_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
