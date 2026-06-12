#!/usr/bin/env python3
"""apply_speaker_resolution — write an inferred counterpart back into the
transcript JSON and the InsightBase sources row.

Closes the loop that the A/B experiment (2026-06-12) exposed: Brain's
/meeting-actions Step 1c reliably infers WHO the unknown "Speaker B" is
(topic_inference / transcript_named / client_context), but its result only
landed in `sources.participant_details` — the transcript labels stayed
"Speaker B" in the on-disk JSON, `sources.content_text`, and
`sources.participants`, so every downstream consumer (insights, wiki, the
next channel-attribution run) kept the unknown label.

This CLI is invoked by /meeting-actions Step 1d after a successful Step 1c
resolution (and usable manually for historical fixes):

    python3 tools/apply_speaker_resolution.py \
        --source-id 391 --name "Margo Schorta" \
        [--email margo.schorta@bluecare.ch] [--company BlueCare] \
        [--role-title "Qlik Developer"] \
        [--method topic_inference] [--confidence medium] \
        [--evidence "..."] [--dry-run]

What it does:
  1. Loads the transcript JSON (path from sources.metadata->transcript_path).
  2. Runs the existing speaker_reconcile.reconcile() with a pseudo calendar
     resolution carrying the inferred counterpart — the same battle-tested
     rewrite (singleton + singleton_collapse) used on the live path, so
     "Speaker B"/"Speaker 1" labels collapse to the real name across
     transcript, participants, emotions, pacing, interruptions, energy.
  3. Writes the JSON back (atomic) and UPDATEs sources.content_text,
     sources.participants, sources.participant_details, and appends a
     forensic `speaker_writeback` block to participant_resolution_log.

Exit codes: 0 = applied (or clean no-op), 1 = error, 2 = nothing to rewrite
(no generic/unmatched labels found — logged, DB still gets the counterpart
in participant_details if missing there).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import re  # noqa: E402

from speaker_reconcile import (  # noqa: E402
    _is_self_label,
    _label_pattern,
    _looks_like_speaker_name,
    reconcile,
)

_GENERIC_RE = re.compile(r"^speaker\s+[A-Za-z0-9]+$", re.IGNORECASE)
_LABEL_SCAN_RE = re.compile(
    r"(?:^|(?<=[\s\]]))([A-ZÄÖÜ][^\n:]{0,40}?)(?=:)", re.MULTILINE
)

logger = logging.getLogger("apply_speaker_resolution")

SELF_NAME = "Matthias Heim"
SELF_EMAIL = "matthias@lailix.com"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass


def fetch_source(source_id: int) -> dict:
    from neon_insert import _get_conn
    conn = _get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT metadata->>'transcript_path',
                       participant_details, participant_resolution_log
                FROM sources WHERE id = %s
                """,
                (source_id,),
            )
            row = cur.fetchone()
            if not row:
                raise LookupError(f"source id={source_id} not found")
            return {
                "transcript_path": row[0],
                "participant_details": row[1] or [],
                "participant_resolution_log": row[2] or {},
            }
    finally:
        conn.close()


def build_pseudo_calendar(existing_details: list[dict],
                          counterpart: dict) -> dict:
    """A calendar_resolve-shaped dict carrying self + the inferred person.

    speaker_reconcile only needs `participant_details`; restricting the
    list to self + counterpart makes the 1:1 singleton/collapse rules fire,
    which is exactly right — the write-back applies one inferred person.
    """
    self_entry = next(
        (p for p in existing_details
         if isinstance(p, dict) and (p.get("role") or "").lower() == "self"),
        {"name": SELF_NAME, "email": SELF_EMAIL,
         "company": "Lailix", "role": "self"},
    )
    return {"participant_details": [self_entry, {
        "name": counterpart["name"],
        "email": counterpart.get("email"),
        "company": counterpart.get("company"),
        "role": "participant",
    }]}


def merge_participant_details(existing: list[dict],
                              counterpart: dict) -> list[dict]:
    """Add/enrich the counterpart entry; keep all other entries untouched."""
    out = [dict(p) for p in existing if isinstance(p, dict)]
    cp_first = counterpart["name"].split()[0].lower()
    for p in out:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        if (name.lower() == counterpart["name"].lower()
                or name.split()[0].lower() == cp_first):
            # Already present — enrich missing fields, normalise the name.
            p["name"] = counterpart["name"]
            for k in ("email", "company"):
                if counterpart.get(k) and not p.get(k):
                    p[k] = counterpart[k]
            if counterpart.get("role_title") and not p.get("title"):
                p["title"] = counterpart["role_title"]
            return out
    entry = {
        "name": counterpart["name"],
        "email": counterpart.get("email"),
        "company": counterpart.get("company"),
        "role": "counterpart",
        "confidence": counterpart.get("confidence", "medium"),
        "resolution_method": counterpart.get("method", "manual"),
    }
    if counterpart.get("role_title"):
        entry["title"] = counterpart["role_title"]
    # Drop a generic placeholder ("Speaker B" counterpart) if present.
    out = [p for p in out
           if not ((p.get("name") or "").lower().startswith("speaker ")
                   and (p.get("role") or "") in ("counterpart", "participant", "speaker"))]
    out.append(entry)
    return out


def _harvest_labels(transcript: str) -> set[str]:
    """Distinct non-self speaker labels (generic + name-shaped)."""
    labels: set[str] = set()
    for m in _LABEL_SCAN_RE.finditer(transcript):
        n = m.group(1).strip()
        if not n or _is_self_label(n):
            continue
        if _GENERIC_RE.match(n) or _looks_like_speaker_name(n):
            labels.add(n)
    return labels


def _rewrite_labels(gemini_dict: dict, counterpart: dict) -> dict:
    """Rewrite unknown labels to the counterpart's name — collapse-safe.

    reconcile()'s singleton_collapse rewrites EVERY unmatched non-self
    label to the sole calendar attendee. That is correct on the live path
    (a 1:1 calendar invite guarantees one external voice) but dangerous
    here: the write-back's pseudo-calendar always lists exactly one
    person, so a multi-party transcript ("Gina", "Vanessa", "Speaker B")
    would have its REAL names swallowed too. Decide per label set:

    - All non-self labels generic or first-name-matching the counterpart
      → full reconcile() (collapse is safe and rewrites every field).
    - Mixed labels → targeted: rewrite ONLY the generic labels, and only
      when exactly one distinct generic label exists (two generics in a
      multi-party meeting could be two different unknown people).
    """
    transcript = gemini_dict.get("transcript") or ""
    labels = _harvest_labels(transcript)
    cp_first = counterpart["name"].split()[0].lower()
    safe_for_collapse = all(
        _GENERIC_RE.match(lbl) or lbl.split()[0].lower() == cp_first
        for lbl in labels
    )

    if safe_for_collapse:
        pseudo_cal = build_pseudo_calendar([], counterpart)
        return reconcile(gemini_dict, pseudo_cal)

    generic = sorted(lbl for lbl in labels if _GENERIC_RE.match(lbl))
    if len(generic) != 1:
        logger.info(
            "mixed labels %s with %d generic — ambiguous, no label rewrite",
            sorted(labels), len(generic),
        )
        return {"decisions": [], "rewrote_speakers": 0,
                "skipped_ambiguous_multiparty": True}

    old = generic[0]
    new = counterpart["name"]
    gemini_dict["transcript"] = _label_pattern(old).sub(new, transcript)
    for p in gemini_dict.get("participants") or []:
        if isinstance(p, dict) and p.get("name") == old:
            p["name"] = new
    for entry in gemini_dict.get("speaker_emotions") or []:
        if isinstance(entry, dict) and entry.get("speaker") == old:
            entry["speaker"] = new
    for ev in gemini_dict.get("interruptions") or []:
        if isinstance(ev, dict):
            for k in ("interrupter", "interruptee"):
                if ev.get(k) == old:
                    ev[k] = new
    for field in ("speaker_pacing", "energy_levels"):
        d = gemini_dict.get(field)
        if isinstance(d, dict) and old in d:
            d[new] = d.pop(old)
    return {
        "decisions": [{"gemini_name": old, "canonical_name": new,
                       "rule": "targeted_generic",
                       "evidence": "single generic label in multi-party "
                                   "transcript rewritten to inferred "
                                   "counterpart"}],
        "rewrote_speakers": 1,
    }


def apply(source_id: int, counterpart: dict, dry_run: bool = False) -> int:
    src = fetch_source(source_id)
    tpath = src["transcript_path"]
    if not tpath or not Path(tpath).exists():
        logger.error(f"transcript file not found: {tpath!r}")
        return 1

    gemini_dict = json.loads(Path(tpath).read_text(encoding="utf-8"))
    transcript_before = gemini_dict.get("transcript") or ""

    recon_log = _rewrite_labels(gemini_dict, counterpart)
    rewrote = recon_log.get("rewrote_speakers", 0)
    transcript_changed = gemini_dict.get("transcript") != transcript_before

    new_details = merge_participant_details(
        src["participant_details"], counterpart)
    writeback_log = {
        "applied_name": counterpart["name"],
        "method": counterpart.get("method", "manual"),
        "confidence": counterpart.get("confidence", "medium"),
        "evidence": counterpart.get("evidence", ""),
        "reconcile": recon_log,
        "transcript_rewritten": transcript_changed,
    }

    if dry_run:
        print(json.dumps({"dry_run": True, "would_rewrite": rewrote,
                          "participant_details": new_details,
                          "log": writeback_log}, indent=2, ensure_ascii=False))
        return 0 if rewrote else 2

    # 1. Transcript JSON back to disk (atomic rename, same directory).
    if transcript_changed:
        fd, tmp = tempfile.mkstemp(dir=str(Path(tpath).parent),
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(gemini_dict, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, tpath)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info(f"rewrote transcript JSON: {tpath}")

    # 2. DB row: content_text, participants[], participant_details,
    #    participant_resolution_log.speaker_writeback.
    from neon_insert import _get_conn
    participant_names = [p.get("name") for p in new_details if p.get("name")]
    prl = dict(src["participant_resolution_log"])
    prl["speaker_writeback"] = writeback_log
    conn = _get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sources SET
                    content_text = CASE WHEN %s THEN %s ELSE content_text END,
                    participants = %s,
                    participant_details = %s::jsonb,
                    participant_resolution_log = %s::jsonb
                WHERE id = %s
                """,
                (
                    transcript_changed,
                    gemini_dict.get("transcript") or "",
                    participant_names,
                    json.dumps(new_details, ensure_ascii=False),
                    json.dumps(prl, ensure_ascii=False),
                    source_id,
                ),
            )
    finally:
        conn.close()

    logger.info(
        f"source {source_id}: counterpart={counterpart['name']!r}, "
        f"labels rewritten={rewrote}, content_text "
        f"{'updated' if transcript_changed else 'unchanged'}"
    )
    print(json.dumps({"source_id": source_id, "rewrote_labels": rewrote,
                      "transcript_rewritten": transcript_changed,
                      "participants": participant_names},
                     ensure_ascii=False))
    return 0 if (rewrote or transcript_changed) else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write an inferred counterpart back into transcript + DB.")
    ap.add_argument("--source-id", type=int, required=True)
    ap.add_argument("--name", required=True,
                    help="Resolved counterpart full name")
    ap.add_argument("--email", default=None)
    ap.add_argument("--company", default=None)
    ap.add_argument("--role-title", default=None,
                    help="Job title (participant_details.title)")
    ap.add_argument("--method", default="manual",
                    help="Resolution method (topic_inference, transcript_named, "
                         "client_context, direct_address, manual)")
    ap.add_argument("--confidence", default="medium",
                    choices=["high", "medium", "low"])
    ap.add_argument("--evidence", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")
    _load_env()

    counterpart = {
        "name": args.name.strip(),
        "email": args.email,
        "company": args.company,
        "role_title": args.role_title,
        "method": args.method,
        "confidence": args.confidence,
        "evidence": args.evidence,
    }
    try:
        return apply(args.source_id, counterpart, dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
