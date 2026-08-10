#!/usr/bin/env python3
"""
update_source_content_text — repair sources.content_text (and its content
revision fingerprint) from a transcript JSON already on disk.

The watcher seeds sources.content_text once, at first transcription
(neon_insert.insert_source). When a transcript is repaired AFTER that —
a speaker-attribution backfill, a corrected diarization pass, anything
that rewrites the JSON's `transcript` field in place — nothing pushes the
fix into the DB: re-extraction (`/meeting-actions`) reads whatever
content_text already says, so a repaired JSON on disk with a stale DB row
silently re-extracts from the OLD text. This is the gap
docs/BACKFILL-speaker-attribution-2026-08-07.md's "nothing here has been
executed" sweep needed and had no tool for.

Reuses neon_insert's own content_text formatting (`_read_transcript`) and
revision fingerprint (`_compute_content_revision_id`) so a repaired row
ends up byte-for-byte what a fresh `insert_source` would have written for
the same JSON — never a second, drifting implementation of either. Uses
neon_insert's `_get_conn` too, so this targets the exact same DB the
seeding path does (see neon_insert.py's INSIGHTBASE_DATABASE_URL docs).

Only touches content_text and metadata.content_revision_id — never
title/summary/participants/etc; those have their own update paths.

Usage:
    python tools/update_source_content_text.py --source-id 767 \\
        --json ~/Documents/MeetingRecorder/Transcripts/2026-08-07_14-11-24.json
    python tools/update_source_content_text.py --source-id 767 --json PATH --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from neon_insert import _compute_content_revision_id, _get_conn, _read_transcript

logger = logging.getLogger(__name__)


def _first_diff_line(old: str, new: str) -> Optional[tuple[int, str, str]]:
    """Return (1-based line number, old_line, new_line) for the first line
    where `old` and `new` diverge, or None if the texts are identical."""
    if old == new:
        return None
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    for i, (o, n) in enumerate(zip(old_lines, new_lines)):
        if o != n:
            return i + 1, o, n
    # Prefix identical; one text is simply longer (lines appended/removed
    # at the end) — report the first line past the shared prefix.
    shorter_len = min(len(old_lines), len(new_lines))
    longer_lines = new_lines if len(new_lines) > len(old_lines) else old_lines
    tail = longer_lines[shorter_len] if len(longer_lines) > shorter_len else ""
    if longer_lines is new_lines:
        return shorter_len + 1, "", tail
    return shorter_len + 1, tail, ""


def _fetch_current(source_id: int) -> Optional[tuple[str, str]]:
    """Return (content_text or '', metadata->>'content_revision_id' or '')
    for `source_id`, or None if the row doesn't exist."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content_text, metadata->>'content_revision_id' "
                "FROM sources WHERE id = %s",
                (source_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0] or "", row[1] or ""
    finally:
        conn.close()


def _apply_update(source_id: int, content_text: str, revision_id: str) -> None:
    """UPDATE content_text + metadata.content_revision_id for source_id.

    Only that one metadata key is touched (jsonb_set merges it into the
    existing object) — extractor_version, transcript_path, seeded_by all
    survive untouched.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sources
                    SET content_text = %s,
                        metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{content_revision_id}',
                            %s::jsonb
                        )
                    WHERE id = %s
                    """,
                    (content_text, json.dumps(revision_id), source_id),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(
                        f"UPDATE affected {cur.rowcount} row(s) for "
                        f"source_id={source_id} (expected 1)"
                    )
    finally:
        conn.close()


def run(source_id: int, json_path: str, dry_run: bool) -> int:
    p = Path(json_path)
    if not p.exists():
        print(f"Transcript not found: {json_path}", file=sys.stderr)
        return 1

    current = _fetch_current(source_id)
    if current is None:
        print(
            f"source_id={source_id} does not exist in sources — refusing to run.",
            file=sys.stderr,
        )
        return 1
    old_content, old_revision = current

    new_content = _read_transcript(str(p))
    new_revision = _compute_content_revision_id(new_content)

    diff = _first_diff_line(old_content, new_content)
    changed = diff is not None

    print(f"source_id:    {source_id}")
    print(f"json:         {p}")
    print(
        f"content_text: {len(old_content)} chars -> {len(new_content)} chars "
        f"({'CHANGED' if changed else 'unchanged'})"
    )
    print(f"revision_id:  {old_revision or '(none)'} -> {new_revision}")
    if changed:
        line_no, old_line, new_line = diff
        print(f"first diff at line {line_no}:")
        print(f"  - {old_line}")
        print(f"  + {new_line}")

    if dry_run:
        print("\n[dry-run] no write performed.")
        return 0

    _apply_update(source_id, new_content, new_revision)
    print(
        f"\nUpdated sources.content_text + metadata.content_revision_id "
        f"for source_id={source_id}."
    )
    return 0


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild sources.content_text from a (repaired) transcript JSON."
    )
    parser.add_argument("--source-id", type=int, required=True,
                         help="Existing sources.id to update")
    parser.add_argument("--json", required=True, dest="json_path",
                         help="Path to the transcript JSON to rebuild content_text from")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the before/after summary; do not write to the DB")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args.source_id, args.json_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(_main())
