#!/usr/bin/env python3
"""Standalone verification for docs/RELIABILITY_PLAN_2026-07.md Phases 1+3.

Re-transcribes a real meeting MP3 through the FIXED pipeline (gemini-2.5-pro
+ transcript_validator coverage gate + escalation ladder + drift-proof
chunking) exactly as _process_with_gemini would, WITHOUT touching the live
watcher, launchd, or config.yaml. Reuses TranscribeWatcher._validate_and_escalate
directly (not a reimplementation) so this exercises the real code path.

Usage:
    python3 tools/verify_reliability_fix.py <mp3_path> [--wav original.wav] [--out out.json]

Prints coverage% / duplicate-span / repetition / last-timestamp before and
after escalation, and writes the accepted transcript JSON to --out (default:
<mp3_stem>.fixed.json next to the mp3) for manual inspection or a follow-up
DB UPDATE. Never writes to the database itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from audio_converter import (  # noqa: E402
    TOPOLOGY_MULTI_SOURCE_GENUINE,
    classify_source_topology,
    get_audio_duration,
)
from channel_vad import compute_channel_vad  # noqa: E402
from gemini_processor import GeminiAudioProcessor  # noqa: E402
import transcribe_watcher as tw  # noqa: E402
from transcribe_watcher import _validate_gemini_result  # noqa: E402


def _report(label: str, validation, result=None) -> None:
    print(f"\n=== {label} ===")
    print(f"Coverage:          {validation.coverage_pct:.1f}%")
    print(f"Last timestamp:    {validation.last_timestamp_sec:.0f}s "
          f"({validation.last_timestamp_sec / 60:.1f}min)")
    print(f"Chunk-fail marker: {validation.has_chunk_failure_marker}")
    print(f"Repetition loop:   {validation.has_repetition_loop}")
    print(f"Duplicate span:    {validation.has_duplicate_span}")
    print(f"PASSED:            {validation.passed}")
    if validation.reasons:
        print("Reasons:")
        for r in validation.reasons:
            print(f"  - {r}")
    if result is not None:
        print(f"Chunked:           {result.chunked} (chunk_count={result.chunk_count})")
        if result.missing_time_ranges:
            print(f"Missing ranges:    {result.missing_time_ranges}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mp3_path", type=Path)
    parser.add_argument("--wav", type=Path, default=None,
                         help="Original 3-channel WAV, for channel_vad ground truth")
    parser.add_argument("--out", type=Path, default=None,
                         help="Where to write the accepted transcript JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("verify_reliability_fix")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set (checked env + .env)", file=sys.stderr)
        sys.exit(1)

    mp3_path = args.mp3_path
    if not mp3_path.exists():
        print(f"Not found: {mp3_path}", file=sys.stderr)
        sys.exit(1)

    audio_duration = get_audio_duration(mp3_path)
    print(f"File: {mp3_path.name}")
    print(f"Duration: {audio_duration:.1f}s ({audio_duration / 60:.1f}min)")

    channel_segments = None
    if args.wav and args.wav.exists():
        topology = classify_source_topology(args.wav)
        print(f"Topology: {topology.topology} (active_channels={topology.active_channels})")
        if topology.topology == TOPOLOGY_MULTI_SOURCE_GENUINE:
            vad = compute_channel_vad(args.wav)
            if vad:
                channel_segments = vad.segments
                print(f"channel_vad: {len(vad.segments)} segments over {vad.duration_sec / 60:.1f}min")
    elif args.wav:
        print(f"WAV not found, proceeding without channel_vad: {args.wav}")

    processor = GeminiAudioProcessor(api_key=api_key, model="gemini-2.5-pro",
                                      max_output_tokens=32768)

    print("\nRunning initial process_audio() (gemini-2.5-pro)...")
    result = processor.process_audio(mp3_path, channel_segments=channel_segments)
    if result.error:
        print(f"Initial call errored: {result.error}")

    initial_validation = _validate_gemini_result(result, audio_duration)
    _report("INITIAL RESULT", initial_validation, result)

    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = logger
    watcher.gemini_processor = processor
    watcher._notify_telegram_partial = lambda audio_file, validation: print(
        f"\n[TELEGRAM ALERT WOULD FIRE] partial transcript, coverage "
        f"{validation.coverage_pct:.1f}%: {'; '.join(validation.reasons)}"
    )

    final_result, final_validation, partial = watcher._validate_and_escalate(
        mp3_path, mp3_path, audio_duration, result,
        known_attendees=None, channel_segments=channel_segments,
        diarization_segments=None,
    )
    final_result.validation_report = final_validation.to_dict()
    final_result.partial = partial

    _report("FINAL RESULT (post-escalation)", final_validation, final_result)
    print(f"\nPARTIAL: {partial}")

    out_path = args.out or mp3_path.with_suffix(".fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_result.parsed_response, f, indent=2, ensure_ascii=False)
    print(f"\nSaved accepted transcript to: {out_path}")


if __name__ == "__main__":
    main()
