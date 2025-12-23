#!/usr/bin/env python3
"""Reprocess all transcripts through the n8n workflow."""

import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

WEBHOOK_URL = "http://localhost:5678/webhook/transcript-mvp"
TRANSCRIPTS_DIR = Path.home() / "Documents/MeetingRecorder/Transcripts"

def parse_timestamp_from_filename(filename: str) -> str:
    """Parse timestamp from filename format: YYYY-MM-DD_HH-MM-SS.html"""
    try:
        stem = Path(filename).stem
        date_part, time_part = stem.split('_')
        year, month, day = date_part.split('-')
        hour, minute, second = time_part.split('-')
        local_dt = datetime(int(year), int(month), int(day),
                           int(hour), int(minute), int(second))
        utc_dt = local_dt.astimezone(timezone.utc)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, IndexError):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def estimate_duration(file_path: Path) -> int:
    """Estimate audio duration based on transcript file size."""
    # Rough estimate: 1KB of HTML = ~30 seconds of audio
    size_kb = file_path.stat().st_size / 1024
    return max(60, int(size_kb * 30))

def process_transcript(transcript_path: Path, index: int, total: int) -> bool:
    """Process a single transcript through the webhook."""
    print(f"\n[{index}/{total}] Processing: {transcript_path.name}")

    try:
        transcript_html = transcript_path.read_text(encoding='utf-8')
        started_at = parse_timestamp_from_filename(transcript_path.name)
        duration = estimate_duration(transcript_path)

        payload = {
            "transcript_path": str(transcript_path),
            "transcript_html": transcript_html,
            "started_at": started_at,
            "audio_duration_seconds": duration
        }

        start_time = time.time()
        response = requests.post(WEBHOOK_URL, json=payload, timeout=180)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"    OK ({elapsed:.1f}s)")
            return True
        else:
            print(f"    FAILED: HTTP {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"    TIMEOUT after 180s")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def main():
    transcripts = sorted(TRANSCRIPTS_DIR.glob("*.html"))
    total = len(transcripts)

    print(f"Found {total} transcripts to process")
    print(f"Estimated time: ~{total * 30 // 60} minutes")
    print("=" * 50)

    success = 0
    failed = 0

    for i, transcript in enumerate(transcripts, 1):
        if process_transcript(transcript, i, total):
            success += 1
        else:
            failed += 1

        # Small delay between requests to avoid overwhelming the server
        if i < total:
            time.sleep(2)

    print("\n" + "=" * 50)
    print(f"COMPLETE: {success} succeeded, {failed} failed")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
