#!/usr/bin/env python3
"""
neon_insert — programmatic InsightBase source insertion for MeetingMemory.

Creates a 'sources' row in the InsightBase Neon DB (project
`solitary-brook-40370942` / insightbase-eu, database `neondb`, configured
via the INSIGHTBASE_DATABASE_URL env var) immediately after
transcription completes, BEFORE the Claude-driven `/meeting-actions`
pipeline runs. This guarantees a persistent record of the meeting even
if downstream processing fails.

The row is seeded with minimal metadata (title, content, language,
started_at). `/meeting-actions` later UPDATEs the same row with enriched
fields (summary, participant_details, calendar_event_id, company) after
it has done Google Calendar resolution in Step 1b.

Idempotency:
  Phase 2 of the rearchitecture plan adds `sources.content_revision_id`.
  Once that column exists, this module should SELECT by
  content_revision_id first and return the existing source_id on match
  (see TODO in `insert_source`). Until then, retries will create
  duplicate rows — the watcher's debounce + single-processing queue
  makes this unlikely in practice.

Environment:
  INSIGHTBASE_DATABASE_URL — required; Postgres connection string for
  the InsightBase Neon DB. Keep distinct from `DATABASE_URL` so we don't
  accidentally write to whatever other DB `DATABASE_URL` points at.

Usage (library):
    from neon_insert import insert_source
    source_id = insert_source(
        transcript_path='/path/to/transcript.html',
        title='2026-04-24_15-30-00',
        started_at=datetime(2026, 4, 24, 15, 30, 0, tzinfo=timezone.utc),
    )

Usage (CLI):
    python neon_insert.py --transcript PATH --title TITLE [--language de]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Optional: load .env from project root so the module Just Works when imported
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

EXTRACTOR_VERSION = 'noscribe-v1'  # bumped when HTML parsing logic changes
DEFAULT_ORIGIN = 'noscribe'
DEFAULT_SENSITIVITY = 'internal'
DEFAULT_SOURCE_TYPE = 'meeting'
DEFAULT_LANGUAGE = 'en'
CONNECT_TIMEOUT = 10


# ── Helpers ───────────────────────────────────────────────────────────

def _html_to_text(html_content: str) -> str:
    """Extract plain text from a noScribe HTML transcript.

    Kept in sync with `transcribe_watcher.html_to_text`. If you change
    one, change the other (or extract to a shared module).
    """
    text = re.sub(r'<[^>]+>', '', html_content)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _read_transcript(transcript_path: str) -> str:
    """Read the transcript file and return its plain-text content.

    Supports .html (noScribe), .txt, .md, and .json (Gemini output).
    For JSON, pulls the 'transcript' field if present, else serialises.
    """
    p = Path(transcript_path)
    raw = p.read_text(encoding='utf-8', errors='replace')

    suffix = p.suffix.lower()
    if suffix == '.html':
        return _html_to_text(raw)
    if suffix == '.json':
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and 'transcript' in data:
                return str(data['transcript'])
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            return raw
    return raw  # .txt, .md, anything else


def _compute_content_revision_id(content: str) -> str:
    """sha256(content + ':' + extractor_version).

    Used for idempotency once Phase 2 adds the column. Computed now so
    we can start storing it in metadata even before it has its own
    column, if desired.
    """
    payload = (content + ':' + EXTRACTOR_VERSION).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _parse_filename_timestamp(transcript_path: str) -> Optional[datetime]:
    """Parse `YYYY-MM-DD_HH-MM-SS` from the filename stem (local→UTC).

    Matches the convention used by `transcribe_watcher._send_webhook`.
    """
    stem = Path(transcript_path).stem
    try:
        date_part, time_part = stem.split('_')
        y, mo, d = date_part.split('-')
        h, mi, s = time_part.split('-')
        local_dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
        return local_dt.astimezone(timezone.utc)
    except (ValueError, IndexError):
        return None


def _get_conn():
    """Open a psycopg2 connection to InsightBase.

    We import psycopg2 lazily so that code paths that never call
    `insert_source` (e.g. tests that only import constants) don't need
    the driver installed.
    """
    import psycopg2  # noqa: WPS433 (local import is intentional)

    dsn = os.environ.get('INSIGHTBASE_DATABASE_URL')
    if not dsn:
        raise RuntimeError(
            'INSIGHTBASE_DATABASE_URL is not set. Add it to '
            'MeetingMemory/.env (Neon project solitary-brook-40370942 / '
            'insightbase-eu, database neondb).'
        )
    return psycopg2.connect(dsn, connect_timeout=CONNECT_TIMEOUT)


# ── Public API ────────────────────────────────────────────────────────

def insert_source(
    transcript_path: str,
    title: str,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
    participant_details: Optional[list[dict]] = None,
    calendar_event_id: Optional[str] = None,
    company: Optional[str] = None,
    language: Optional[str] = None,
    origin: str = DEFAULT_ORIGIN,
    sensitivity_level: str = DEFAULT_SENSITIVITY,
    summary: Optional[str] = None,
    duration_minutes: Optional[int] = None,
) -> int:
    """Insert a 'sources' row for a transcribed meeting; return source_id.

    Idempotent INSERT semantics will be added once Phase 2 ships the
    `content_revision_id` column. Until then, this function always
    INSERTs. See module docstring.

    Args:
        transcript_path: Absolute path to the transcript file
            (.html / .txt / .md / .json). Contents are read into
            `sources.content_text`.
        title: Meeting title. Watcher passes the filename stem; Step 1
            of `/meeting-actions` later UPDATEs with a derived title.
        started_at: Meeting start (UTC). If None, parsed from filename
            `YYYY-MM-DD_HH-MM-SS`.
        ended_at: Meeting end (UTC). Optional.
        participant_details: JSONB list of {name, email, company, role}.
            Usually populated by Step 1b; watcher passes None.
        calendar_event_id: Google Calendar event ID. Usually set later.
        company: Client name. Usually set later.
        language: Detected transcript language (de, en, ...). Falls back
            to DEFAULT_LANGUAGE.
        origin: Capture origin. Defaults to 'noscribe'.
        sensitivity_level: One of open/internal/confidential/restricted.
        summary: Optional short summary. Usually None at insert time.
        duration_minutes: Meeting duration. Optional.

    Returns:
        The `id` of the newly inserted sources row.

    Raises:
        RuntimeError: INSIGHTBASE_DATABASE_URL not set.
        FileNotFoundError: transcript_path does not exist.
        psycopg2.Error: DB errors bubble up for caller to handle.
    """
    p = Path(transcript_path)
    if not p.exists():
        raise FileNotFoundError(f'Transcript not found: {transcript_path}')

    content_text = _read_transcript(transcript_path)
    source_attribution = {}
    if p.suffix.lower() == '.json':
        try:
            parsed = json.loads(p.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            parsed = {}  # Preserve _read_transcript's raw-text fallback.
        report = (parsed.get('_meta') or {}).get('speaker_attribution') if isinstance(parsed, dict) else None
        if report is not None:
            source_attribution['speaker_attribution'] = report
    # TODO (Phase 2): once sources.content_revision_id ships, SELECT by
    #   content_revision_id before INSERT and return the existing id on
    #   match. For now we compute+log it for debugging only.
    content_revision_id = _compute_content_revision_id(content_text)
    logger.debug('content_revision_id=%s (not yet persisted)', content_revision_id)

    if started_at is None:
        started_at = _parse_filename_timestamp(transcript_path)
        # If still None (weird filename), leave it NULL — DB column is nullable.

    # Always source_type='meeting' for watcher-driven inserts. Callers
    # wanting a different type should pass it explicitly if we expand
    # this module later.
    source_type = DEFAULT_SOURCE_TYPE
    source_date = started_at.date() if started_at else datetime.now(timezone.utc).date()

    # `participants` column is TEXT[]. Extract names from
    # participant_details if provided; otherwise empty array.
    participant_names: list[str] = []
    if participant_details:
        for p_detail in participant_details:
            name = p_detail.get('name') if isinstance(p_detail, dict) else None
            if name:
                participant_names.append(str(name))

    participant_details_json = (
        json.dumps(participant_details) if participant_details else None
    )

    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sources (
                        source_type, title, source_date, duration_minutes,
                        participants, origin, summary, content_text,
                        sensitivity_level, started_at, ended_at,
                        calendar_event_id, company, participant_details,
                        language, metadata
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s::jsonb,
                        %s, %s::jsonb
                    )
                    RETURNING id;
                    """,
                    (
                        source_type,
                        title,
                        source_date,
                        duration_minutes,
                        participant_names,
                        origin,
                        summary,
                        content_text,
                        sensitivity_level,
                        started_at,
                        ended_at,
                        calendar_event_id,
                        company,
                        participant_details_json,
                        language or DEFAULT_LANGUAGE,
                        json.dumps({
                            'transcript_path': str(p),
                            'content_revision_id': content_revision_id,
                            'extractor_version': EXTRACTOR_VERSION,
                            'seeded_by': 'transcribe_watcher',
                            **source_attribution,
                        }),
                    ),
                )
                source_id = cur.fetchone()[0]
                logger.info(
                    'Inserted source id=%s title=%r from %s',
                    source_id, title, p.name,
                )
                return source_id
    finally:
        conn.close()


def update_source_with_gemini(
    source_id: int,
    gemini_result: dict,
    duration_seconds: Optional[float] = None,
) -> None:
    """UPDATE a sources row with audio-derived metadata from Gemini, and
    INSERT a sibling meeting_metadata row.

    Called by the watcher's deterministic-ingest phase right after Gemini
    finishes. Title/summary stay untouched here — Claude /meeting-actions
    Step 2 will UPDATE the title (textual reasoning over the transcript)
    and write summary during insight extraction.

    Args:
        source_id: Existing sources.id (seeded by insert_source).
        gemini_result: Either a dict (parsed_response form) or anything that
            exposes the same field names.
        duration_seconds: Audio duration; gets stored as duration_minutes
            on sources for backwards-compatible queries.
    """
    # Accept either the parsed_response dict or a GeminiResult-like object.
    g = gemini_result
    if not isinstance(g, dict):
        g = g.parsed_response  # type: ignore[attr-defined]

    meta = g.get("_meta", {})
    attribution_metadata = {k: meta[k] for k in ("speaker_attribution", "channel_separation") if k in meta}
    attribution_metadata.update({k: g[k] for k in ("speaker_verification", "speaker_coherence") if k in g})

    language = g.get("language") or DEFAULT_LANGUAGE
    participants = g.get("participants") or []
    participant_names: list[str] = [
        p.get("name") for p in participants
        if isinstance(p, dict) and p.get("name")
    ]
    participant_details_json = json.dumps(participants) if participants else None

    duration_minutes = (
        int(duration_seconds // 60) if duration_seconds and duration_seconds > 0 else None
    )

    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # 1) UPDATE sources with audio-derived stable fields. Don't
                #    overwrite title/summary/company/calendar_event_id —
                #    those come from Claude / calendar_resolve later.
                cur.execute(
                    """
                    UPDATE sources SET
                        language = COALESCE(%s, language),
                        participants = %s,
                        participant_details = COALESCE(%s::jsonb, participant_details),
                        duration_minutes = COALESCE(%s, duration_minutes),
                        metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                    WHERE id = %s
                    """,
                    (
                        language,
                        participant_names,
                        participant_details_json,
                        duration_minutes,
                        json.dumps(attribution_metadata),
                        source_id,
                    ),
                )

                # 2) INSERT meeting_metadata. Use ON CONFLICT to make this
                #    idempotent against retries (source_id is the PK).
                cur.execute(
                    """
                    INSERT INTO meeting_metadata (
                        source_id,
                        overall_sentiment, sentiment_intensity,
                        speaker_emotions, speaker_pacing,
                        interruptions, energy_levels,
                        gemini_model, gemini_input_tokens, gemini_output_tokens,
                        chunked, chunk_count, reduce_pass_used
                    ) VALUES (
                        %s,
                        %s, %s,
                        %s::jsonb, %s::jsonb,
                        %s::jsonb, %s::jsonb,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (source_id) DO UPDATE SET
                        overall_sentiment = EXCLUDED.overall_sentiment,
                        sentiment_intensity = EXCLUDED.sentiment_intensity,
                        speaker_emotions = EXCLUDED.speaker_emotions,
                        speaker_pacing = EXCLUDED.speaker_pacing,
                        interruptions = EXCLUDED.interruptions,
                        energy_levels = EXCLUDED.energy_levels,
                        gemini_model = EXCLUDED.gemini_model,
                        gemini_input_tokens = EXCLUDED.gemini_input_tokens,
                        gemini_output_tokens = EXCLUDED.gemini_output_tokens,
                        chunked = EXCLUDED.chunked,
                        chunk_count = EXCLUDED.chunk_count,
                        reduce_pass_used = EXCLUDED.reduce_pass_used,
                        updated_at = NOW();
                    """,
                    (
                        source_id,
                        g.get("overall_sentiment") or "neutral",
                        g.get("sentiment_intensity") or "moderate",
                        json.dumps(g.get("speaker_emotions") or []),
                        json.dumps(g.get("speaker_pacing") or {}),
                        json.dumps(g.get("interruptions") or []),
                        json.dumps(g.get("energy_levels") or {}),
                        meta.get("model"),
                        meta.get("input_tokens"),
                        meta.get("output_tokens"),
                        bool(meta.get("chunked")),
                        meta.get("chunk_count"),
                        bool(meta.get("reduce_pass_used")),
                    ),
                )
                logger.info(
                    "Enriched source id=%s: language=%s, %d participants, "
                    "%d emotion arcs, %d interruptions, chunked=%s, reduce_pass=%s",
                    source_id, language, len(participants),
                    len(g.get("speaker_emotions") or []),
                    len(g.get("interruptions") or []),
                    meta.get("chunked"), meta.get("reduce_pass_used"),
                )
    finally:
        conn.close()


def update_source_calendar_match(
    source_id: int,
    participant_details: list[dict],
    participant_resolution_log: dict,
    calendar_event_id: Optional[str] = None,
    company: Optional[str] = None,
) -> None:
    """UPDATE a sources row with calendar resolution output.

    Called by the watcher after `calendar_resolve.resolve()` returns. Only
    overwrites participant_details / participant_resolution_log /
    calendar_event_id / company. Leaves everything else untouched.
    """
    pd_json = json.dumps(participant_details) if participant_details else None
    prl_json = json.dumps(participant_resolution_log) if participant_resolution_log else None
    participant_names: list[str] = [
        p.get("name") for p in (participant_details or [])
        if isinstance(p, dict) and p.get("name")
    ]

    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sources SET
                        participant_details = COALESCE(%s::jsonb, participant_details),
                        participant_resolution_log = COALESCE(%s::jsonb, participant_resolution_log),
                        calendar_event_id = COALESCE(%s, calendar_event_id),
                        company = COALESCE(%s, company),
                        participants = CASE
                            WHEN cardinality(%s::text[]) > 0 THEN %s::text[]
                            ELSE participants
                        END
                    WHERE id = %s
                    """,
                    (
                        pd_json, prl_json,
                        calendar_event_id, company,
                        participant_names, participant_names,
                        source_id,
                    ),
                )
                logger.info(
                    "Calendar-resolved source id=%s: cal_event=%s, company=%s, %d attendees",
                    source_id, calendar_event_id, company, len(participant_details or []),
                )
    finally:
        conn.close()


# ── CLI ───────────────────────────────────────────────────────────────

def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Insert a sources row into InsightBase for a transcript.'
    )
    parser.add_argument('--transcript', required=True, help='Path to transcript file')
    parser.add_argument('--title', required=True, help='Meeting title')
    parser.add_argument('--language', default=None, help='Transcript language (de/en/...)')
    parser.add_argument('--company', default=None, help='Client/company name')
    parser.add_argument('--calendar-event-id', default=None, help='Google Calendar event ID')
    parser.add_argument('--origin', default=DEFAULT_ORIGIN, help='Origin (noscribe, gemini, ...)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    source_id = insert_source(
        transcript_path=args.transcript,
        title=args.title,
        language=args.language,
        company=args.company,
        calendar_event_id=args.calendar_event_id,
        origin=args.origin,
    )
    print(source_id)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
