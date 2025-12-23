#!/usr/bin/env python3
"""Process markdown transcripts through the n8n workflow."""

import re
import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
import markdown

WEBHOOK_URL = "http://localhost:5678/webhook/transcript-mvp"

def parse_date_from_content(content: str) -> str:
    """Extract date from 'Date/Time:' line in content."""
    match = re.search(r'Date/Time:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})', content)
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_date_from_filename(filename: str) -> str:
    """Try to parse date from filename like '25.08.21 Description.md'."""
    match = re.match(r'(\d{2})\.(\d{2})\.(\d{2})\s', filename)
    if match:
        year, month, day = match.groups()
        try:
            dt = datetime(2000 + int(year), int(month), int(day), 12, 0, 0)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    return None

def markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML."""
    return markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

def estimate_duration(content: str) -> int:
    """Estimate duration from transcript content."""
    # Look for timestamp patterns like "42:26" at the end
    timestamps = re.findall(r'\b(\d{1,2}):(\d{2})\b', content)
    if timestamps:
        # Get the largest timestamp
        max_seconds = 0
        for mins, secs in timestamps:
            total = int(mins) * 60 + int(secs)
            max_seconds = max(max_seconds, total)
        if max_seconds > 0:
            return max_seconds

    # Fallback: estimate from word count (~150 words/minute)
    words = len(content.split())
    return max(60, (words // 150) * 60)

def process_transcript(file_path: Path, index: int, total: int) -> bool:
    """Process a single markdown transcript."""
    print(f"\n[{index}/{total}] Processing: {file_path.name}")

    try:
        content = file_path.read_text(encoding='utf-8')

        # Parse date from content first, then filename
        started_at = parse_date_from_content(content)
        if started_at == datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"):
            filename_date = parse_date_from_filename(file_path.name)
            if filename_date:
                started_at = filename_date

        # Convert markdown to HTML
        html_content = markdown_to_html(content)

        # Wrap in basic HTML structure
        full_html = f"""<!DOCTYPE html>
<html>
<head><title>{file_path.stem}</title></head>
<body>
{html_content}
</body>
</html>"""

        duration = estimate_duration(content)

        payload = {
            "transcript_path": str(file_path),
            "transcript_html": full_html,
            "started_at": started_at,
            "audio_duration_seconds": duration,
            "source_tool": "Markdown Import"
        }

        start_time = time.time()
        response = requests.post(WEBHOOK_URL, json=payload, timeout=180)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"    OK ({elapsed:.1f}s) - Date: {started_at[:10]}, Duration: {duration//60}m")
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
    transcripts_dir = Path("/Users/Matthias/Desktop/Financial Aid/Meetings")

    # Get all markdown files
    transcripts = sorted(transcripts_dir.glob("*.md"))

    # Also check for files without extension
    for f in transcripts_dir.iterdir():
        if f.is_file() and f.suffix == '' and f not in transcripts:
            transcripts.append(f)

    transcripts = sorted(transcripts)
    total = len(transcripts)

    print(f"Found {total} markdown transcripts to process")
    print("=" * 60)

    success = 0
    failed = 0

    for i, transcript in enumerate(transcripts, 1):
        if process_transcript(transcript, i, total):
            success += 1
        else:
            failed += 1

        if i < total:
            time.sleep(2)

    print("\n" + "=" * 60)
    print(f"COMPLETE: {success} succeeded, {failed} failed")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
