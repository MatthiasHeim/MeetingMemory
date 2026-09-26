"""The recorded owner decision for Slice 1's Gemini calibration.

This contains aggregate configuration only.  It must never include evaluation
or meeting transcript text.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


OWNER_OVERRIDE_STATUS = "owner_approved_override"
OWNER_APPROVED_MODEL = "gemini-3.8-flash"
OWNER_APPROVED_PROMPT_THRESHOLD = 0.90
OWNER_APPROVED_CLIP_SEED_THRESHOLD = 0.60
OWNER_APPROVED_CLIP_GROW_THRESHOLD = 0.25

# Matthias's explicit final decision is intentionally machine-readable so a
# failed B reference-set cell cannot silently re-disable the accepted clerk.
OWNER_OVERRIDE: dict[str, Any] = {
    "owner": "Matthias",
    "date": "2026-09-26",
    "reason": (
        "Swiss German relevance labels are noisy and the repeated Gemini "
        "variance is larger than the calibration gap; use gemini-3.8-flash "
        "as the single Slice 1 judge until a human-checked reference set exists."
    ),
    "model": OWNER_APPROVED_MODEL,
    "prompt_threshold": OWNER_APPROVED_PROMPT_THRESHOLD,
    "clip_hysteresis": {
        "seed_threshold": OWNER_APPROVED_CLIP_SEED_THRESHOLD,
        "grow_threshold": OWNER_APPROVED_CLIP_GROW_THRESHOLD,
    },
    "swiss_german_relevance_auc_range": [0.788, 0.884],
    "scope": "acceptance_criterion_2_swiss_german_relevance_only",
}


def owner_override_payload() -> dict[str, Any]:
    """Return a writable copy for an aggregate-only local report."""
    return deepcopy(OWNER_OVERRIDE)


def valid_owner_override(report: Mapping[str, Any]) -> bool:
    """Check the exact, dated decision before enabling an overridden report."""
    if report.get("status") != OWNER_OVERRIDE_STATUS:
        return False
    override = report.get("owner_override")
    chosen = report.get("chosen")
    if not isinstance(override, Mapping) or not isinstance(chosen, Mapping):
        return False
    expected = OWNER_OVERRIDE
    if any(override.get(key) != expected[key] for key in ("owner", "date", "model", "scope")):
        return False
    if override.get("prompt_threshold") != OWNER_APPROVED_PROMPT_THRESHOLD:
        return False
    if override.get("clip_hysteresis") != expected["clip_hysteresis"]:
        return False
    if override.get("swiss_german_relevance_auc_range") != expected["swiss_german_relevance_auc_range"]:
        return False
    return (
        chosen.get("model") == OWNER_APPROVED_MODEL
        and chosen.get("prompt_threshold") == OWNER_APPROVED_PROMPT_THRESHOLD
        and chosen.get("relevance_seed_threshold") == OWNER_APPROVED_CLIP_SEED_THRESHOLD
        and chosen.get("relevance_grow_threshold") == OWNER_APPROVED_CLIP_GROW_THRESHOLD
    )
