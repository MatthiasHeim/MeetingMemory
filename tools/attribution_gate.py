#!/usr/bin/env python3
"""Machine-readable gate for downstream speaker-dependent actions."""
import argparse
import copy
import json
from pathlib import Path


def gate(data: dict) -> dict:
    status = (data.get("_meta") or {}).get("speaker_attribution") or {}
    policy = status.get("speaker_dependent_actions", "hold")
    if policy not in ("hold", "require_turn_evidence"):
        policy = "hold"
    return {"speaker_dependent_actions": policy, "status": status.get("status", "legacy_unverified"),
            "allow_named_commitments": False, "allow_content_only_extraction": True,
            "require_turn_evidence": True,
            "missing_stages": status.get("missing_stages", ["acoustic_identity_review"])}


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
