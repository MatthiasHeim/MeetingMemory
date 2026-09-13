#!/usr/bin/env python3
"""apply_calendar_candidate — bind one roster event (or unknown) after a collision.

Used by /meeting-actions Step 0 when calendar_search.match_count > 1.
The watcher persisted the 2–3 window events as
participant_resolution_log.calendar_search.candidates and did not publish
the nearest event as identity. This CLI is the only write path that may
turn a candidate into identity.

    python3 tools/apply_calendar_candidate.py --source-id 940 --event-id evt-sarah \
        --confidence high --evidence "named intro + company we"
    python3 tools/apply_calendar_candidate.py --source-id 940 --unknown \
        --evidence "roster names never appear; cannot tell"

Never invents people: --event-id must be on the persisted roster.
Bind requires --confidence high; anything else is refused (use --unknown).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_speaker_resolution import (  # noqa: E402
    _load_env,
    _rewrite_labels,
    fetch_source,
    merge_participant_details,
)
from attribution_gate import HOLD, REQUIRE_TURN_EVIDENCE, calendar_search_of  # noqa: E402

logger = logging.getLogger("apply_calendar_candidate")


def find_candidate(candidates: list, event_id: str | None) -> dict | None:
    if not event_id:
        return None
    wanted = str(event_id)
    for cand in candidates or []:
        if isinstance(cand, dict) and str(cand.get("event_id") or "") == wanted:
            return cand
    return None


def primary_counterpart(candidate: dict | None) -> dict | None:
    for person in (candidate or {}).get("attendees") or []:
        if isinstance(person, dict) and (person.get("role") or "").lower() != "self":
            return person
    return None


def mark_bound(search: dict | None, candidate: dict, confidence: str, evidence: str) -> dict:
    out = dict(search or {})
    out["identity_authoritative"] = True
    out["ambiguous"] = False
    out["chosen_event_id"] = candidate.get("event_id")
    out["chosen_event_title"] = candidate.get("title")
    out["resolution"] = {
        "decision": "event_id",
        "event_id": candidate.get("event_id"),
        "confidence": confidence,
        "evidence": evidence,
    }
    return out


def mark_unknown(search: dict | None, evidence: str, confidence: str = "low") -> dict:
    out = dict(search or {})
    out["identity_authoritative"] = False
    out["ambiguous"] = True
    out["resolution"] = {
        "decision": "unknown",
        "event_id": None,
        "confidence": confidence,
        "evidence": evidence,
    }
    return out


def publication_from_candidate(candidate: dict) -> dict:
    return {
        "calendar_event_id": candidate.get("event_id"),
        "company": candidate.get("company"),
        "participant_details": list(candidate.get("attendees") or []),
    }


def apply_bind_to_transcript(data: dict, candidate: dict, confidence: str = "high") -> dict:
    """Relabel generic speakers to the chosen roster counterpart. Mutates data."""
    counterpart = primary_counterpart(candidate)
    recon = {"rewrote_speakers": 0, "decisions": []}
    if counterpart and counterpart.get("name"):
        recon = _rewrite_labels(data, {
            "name": counterpart["name"],
            "email": counterpart.get("email"),
            "company": counterpart.get("company"),
            "method": "calendar_candidate",
            "confidence": confidence,
        })
    meta = data.setdefault("_meta", {})
    attr = dict(meta.get("speaker_attribution") or {})
    search = mark_bound(
        attr.get("calendar_search") or calendar_search_of(data),
        candidate, confidence, "calendar_candidate bind",
    )
    attr["calendar_search"] = search
    missing = [s for s in (attr.get("missing_stages") or []) if s != "calendar_identity_review"]
    attr["missing_stages"] = missing
    attr["speaker_dependent_actions"] = HOLD if missing else REQUIRE_TURN_EVIDENCE
    attr["status"] = "needs_review" if missing else attr.get("status", "partially_checked")
    meta["speaker_attribution"] = attr
    return recon


def apply_unknown_to_transcript(data: dict, evidence: str, confidence: str = "low") -> dict:
    meta = data.setdefault("_meta", {})
    attr = dict(meta.get("speaker_attribution") or {})
    search = mark_unknown(
        attr.get("calendar_search") or calendar_search_of(data), evidence, confidence
    )
    attr["calendar_search"] = search
    missing = list(attr.get("missing_stages") or [])
    if "calendar_identity_review" not in missing:
        missing.append("calendar_identity_review")
    attr["missing_stages"] = missing
    attr["speaker_dependent_actions"] = HOLD
    attr["status"] = "needs_review"
    meta["speaker_attribution"] = attr
    return search


def _atomic_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply(source_id: int, event_id: str | None = None, unknown: bool = False,
          confidence: str = "high", evidence: str = "", dry_run: bool = False) -> int:
    src = fetch_source(source_id)
    tpath = src["transcript_path"]
    if not tpath or not Path(tpath).exists():
        logger.error(f"transcript file not found: {tpath!r}")
        return 1

    prl = copy.deepcopy(src["participant_resolution_log"] or {})
    search = dict(prl.get("calendar_search") or {})
    candidates = search.get("candidates") or []
    gemini = json.loads(Path(tpath).read_text(encoding="utf-8"))

    if unknown:
        search = mark_unknown(search, evidence, confidence)
        apply_unknown_to_transcript(gemini, evidence, confidence)
        prl["calendar_search"] = search
        payload = {
            "source_id": source_id, "decision": "unknown",
            "identity_authoritative": False,
            "calendar_search": search,
        }
        if dry_run:
            print(json.dumps({"dry_run": True, **payload}, indent=2, ensure_ascii=False))
            return 0
        _atomic_json(Path(tpath), gemini)
        _persist_unknown(source_id, prl)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if confidence != "high":
        logger.error("refusing to bind calendar identity without high confidence")
        return 1
    candidate = find_candidate(candidates, event_id)
    if candidate is None:
        logger.error(
            "event_id %r is not on the persisted roster — refusing to invent",
            event_id,
        )
        return 1

    search = mark_bound(search, candidate, confidence, evidence)
    recon = apply_bind_to_transcript(gemini, candidate, confidence)
    prl["calendar_search"] = search
    published = publication_from_candidate(candidate)
    counterpart = primary_counterpart(candidate)
    details = list(published["participant_details"] or src["participant_details"] or [])
    if counterpart and counterpart.get("name"):
        details = merge_participant_details(details, {
            "name": counterpart["name"],
            "email": counterpart.get("email"),
            "company": counterpart.get("company"),
            "method": "calendar_candidate",
            "confidence": confidence,
            "evidence": evidence,
        })
    payload = {
        "source_id": source_id, "decision": "event_id",
        "event_id": candidate.get("event_id"),
        "rewrote_labels": recon.get("rewrote_speakers", 0),
        "identity_authoritative": True,
    }
    if dry_run:
        print(json.dumps({"dry_run": True, **payload, "participant_details": details},
                         indent=2, ensure_ascii=False))
        return 0
    _atomic_json(Path(tpath), gemini)
    _persist_bind(source_id, published, details, prl, gemini)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _persist_unknown(source_id: int, prl: dict) -> None:
    from neon_insert import _get_conn
    conn = _get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sources SET
                    participant_resolution_log = %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(prl, ensure_ascii=False), source_id),
            )
    finally:
        conn.close()


def _persist_bind(source_id: int, published: dict, details: list,
                  prl: dict, gemini: dict) -> None:
    from neon_insert import _get_conn, update_source_calendar_match
    update_source_calendar_match(
        source_id,
        participant_details=details,
        participant_resolution_log=prl,
        calendar_event_id=published.get("calendar_event_id"),
        company=published.get("company"),
    )
    transcript = gemini.get("transcript") or ""
    names = [p.get("name") for p in details if isinstance(p, dict) and p.get("name")]
    conn = _get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sources SET
                    content_text = CASE WHEN %s <> '' THEN %s ELSE content_text END,
                    participants = %s
                WHERE id = %s
                """,
                (transcript, transcript, names, source_id),
            )
    finally:
        conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-id", type=int, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--event-id", default=None)
    group.add_argument("--unknown", action="store_true")
    ap.add_argument("--confidence", default="high",
                    choices=["high", "medium", "low"])
    ap.add_argument("--evidence", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_env()
    try:
        return apply(
            args.source_id,
            event_id=args.event_id,
            unknown=args.unknown,
            confidence=args.confidence,
            evidence=args.evidence,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
