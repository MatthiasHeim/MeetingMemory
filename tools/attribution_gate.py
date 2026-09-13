#!/usr/bin/env python3
"""Machine-readable gate for downstream speaker-dependent actions."""
import argparse
import copy
import json
from pathlib import Path

RESOLVE_THEN_EXTRACT = "resolve_then_extract"
HOLD = "hold"
REQUIRE_TURN_EVIDENCE = "require_turn_evidence"
_ALLOWED_POLICIES = {HOLD, REQUIRE_TURN_EVIDENCE, RESOLVE_THEN_EXTRACT}

RESOLVE_THEN_EXTRACT_PROMPT = """
CALENDAR IDENTITY IS AMBIGUOUS (match_count > 1). This is Step 0 of
/meeting-actions — do it BEFORE insights, emails, Linear, CRM, or wiki.

1. Read the roster on the source:
   sources.participant_resolution_log.calendar_search.candidates
   (also copied onto the transcript _meta.speaker_attribution.calendar_search).
   Each candidate has event_id, title, start, attendees, company.
2. Using the transcript (named intros, company "we"/"our", Q/A adjacency,
   who is addressed by name), pick exactly one candidate event_id OR unknown.
   Never invent a person who is not on the chosen roster (host is always allowed).
3. Apply the decision with this CLI (do not write identity by hand):
   python3 {apply_cli} --source-id {source_id} --event-id <id> --confidence high --evidence "..."
   or
   python3 {apply_cli} --source-id {source_id} --unknown --evidence "..."
   Bind only at confidence=high. Medium/low must use --unknown.
4. If unknown or not high-confidence: STOP. Do not extract named insights
   or speaker-dependent actions. Leave the source held for human review.
5. If you bound an event at high confidence: you may repair obvious wrong
   speaker labels (Q/A adjacency, company "we", named intros) only within
   that roster, then continue normal /meeting-actions extraction.
"""


def calendar_search_of(data: dict) -> dict:
    """Find calendar_search on a transcript, attribution report, or cal_match."""
    if not isinstance(data, dict):
        return {}
    for candidate in (
        data.get("calendar_search"),
        ((data.get("_meta") or {}).get("speaker_attribution") or {}).get("calendar_search"),
        ((data.get("participant_resolution_log") or {}).get("calendar_search")),
    ):
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def calendar_bind_ambiguous(data: dict | None) -> bool:
    """True when the window collided and identity is not yet decided.

    match_count == 0 is not a collision. match_count == 1 is authoritative.
    After Step 0, a high-confidence bind or an explicit unknown is no
    longer an open collision — the gate then follows speaker_dependent_actions.
    """
    search = calendar_search_of(data or {})
    if search.get("identity_authoritative") is True:
        return False
    if (search.get("resolution") or {}).get("decision") == "unknown":
        return False
    if search.get("ambiguous") is True:
        return True
    try:
        return int(search.get("match_count") or 0) > 1
    except (TypeError, ValueError):
        return False


def apply_calendar_bind_to_attribution(report: dict | None, calendar_search: dict | None) -> dict:
    """Copy the calendar bind onto the attribution report.

    A collision is resolve_then_extract: Claude may disambiguate, but named
    commitments stay blocked until a high-confidence roster bind.
    """
    report = dict(report or {})
    if calendar_search:
        report["calendar_search"] = dict(calendar_search)
    if calendar_bind_ambiguous({"calendar_search": report.get("calendar_search") or {}}):
        report["speaker_dependent_actions"] = RESOLVE_THEN_EXTRACT
        missing = list(report.get("missing_stages") or [])
        if "calendar_identity_review" not in missing:
            missing.append("calendar_identity_review")
        report["missing_stages"] = missing
        if report.get("status") != "no_speech":
            report["status"] = "needs_review"
    return report


def safe_calendar_publication(cal_match: dict | None) -> dict | None:
    """Persist the bind log, not the chosen event's counterpart as identity."""
    if not cal_match:
        return cal_match
    if not calendar_bind_ambiguous(cal_match):
        return cal_match
    out = copy.deepcopy(cal_match)
    out["participant_details"] = [
        p for p in (out.get("participant_details") or [])
        if isinstance(p, dict) and (p.get("role") or "").lower() == "self"
    ]
    out["company"] = None
    out["calendar_event_id"] = None
    return out


def allows_claude_trigger(decision: dict | None) -> bool:
    return (decision or {}).get("speaker_dependent_actions") in (
        REQUIRE_TURN_EVIDENCE, RESOLVE_THEN_EXTRACT,
    )


def claude_prompt_suffix(source_id, apply_cli: str) -> str:
    return RESOLVE_THEN_EXTRACT_PROMPT.format(
        source_id=source_id if source_id is not None else "UNKNOWN",
        apply_cli=apply_cli,
    )


def gate(data: dict) -> dict:
    status = (data.get("_meta") or {}).get("speaker_attribution") or {}
    policy = status.get("speaker_dependent_actions", HOLD)
    if policy not in _ALLOWED_POLICIES:
        policy = HOLD
    if "missing_stages" in status:
        missing = list(status.get("missing_stages") or [])
    else:
        missing = ["acoustic_identity_review"]
    if calendar_bind_ambiguous(data):
        # Collision is a review stage, not a dead end: Claude may
        # disambiguate. Named commitments stay blocked until a bind.
        policy = RESOLVE_THEN_EXTRACT
        if "calendar_identity_review" not in missing:
            missing.append("calendar_identity_review")
    return {"speaker_dependent_actions": policy, "status": status.get("status", "legacy_unverified"),
            "allow_named_commitments": False, "allow_content_only_extraction": True,
            "require_turn_evidence": True,
            "trigger_claude": allows_claude_trigger({"speaker_dependent_actions": policy}),
            "missing_stages": missing}


def safe_enrichment(data: dict) -> dict:
    """Persist status and whole-meeting signals, not inferred named identities."""
    if gate(data)["speaker_dependent_actions"] == REQUIRE_TURN_EVIDENCE:
        return data
    out = copy.deepcopy(data)
    for field in ("participants", "speaker_emotions", "interruptions"):
        out[field] = []
    for field in ("speaker_pacing", "energy_levels"):
        out[field] = {}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    print(json.dumps(gate(json.loads(args.transcript.read_text())), indent=2))
