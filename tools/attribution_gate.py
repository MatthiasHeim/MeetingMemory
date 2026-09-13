#!/usr/bin/env python3
"""Machine-readable gate for downstream speaker-dependent actions."""
import argparse
import copy
import json
from pathlib import Path


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
    """True when calendar resolution collided (match_count > 1).

    match_count == 0 is not a collision — there is simply no calendar
    identity. match_count == 1 is an authoritative bind.
    """
    search = calendar_search_of(data or {})
    if search.get("ambiguous") is True:
        return True
    try:
        return int(search.get("match_count") or 0) > 1
    except (TypeError, ValueError):
        return False


def apply_calendar_bind_to_attribution(report: dict | None, calendar_search: dict | None) -> dict:
    """Copy the calendar bind onto the attribution report and hold if ambiguous."""
    report = dict(report or {})
    if calendar_search:
        report["calendar_search"] = dict(calendar_search)
    if calendar_bind_ambiguous({"calendar_search": report.get("calendar_search") or {}}):
        report["speaker_dependent_actions"] = "hold"
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


def gate(data: dict) -> dict:
    status = (data.get("_meta") or {}).get("speaker_attribution") or {}
    policy = status.get("speaker_dependent_actions", "hold")
    if policy not in ("hold", "require_turn_evidence"):
        policy = "hold"
    if "missing_stages" in status:
        missing = list(status.get("missing_stages") or [])
    else:
        missing = ["acoustic_identity_review"]
    if calendar_bind_ambiguous(data):
        policy = "hold"
        if "calendar_identity_review" not in missing:
            missing.append("calendar_identity_review")
    return {"speaker_dependent_actions": policy, "status": status.get("status", "legacy_unverified"),
            "allow_named_commitments": False, "allow_content_only_extraction": True,
            "require_turn_evidence": True,
            "missing_stages": missing}


def safe_enrichment(data: dict) -> dict:
    """Persist status and whole-meeting signals, not inferred named identities."""
    if gate(data)["speaker_dependent_actions"] != "hold":
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
