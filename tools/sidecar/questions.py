"""Canonical binary question definitions shared by runtime and calibration."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


# Prompt scores are model-prompt-sensitive. Keep the exact runtime wording and
# criteria in one immutable definition so calibration cannot score a paraphrase.
DICTATING_PROMPT_QUESTION: Mapping[str, object] = MappingProxyType(
    {
        "instructions": (
            "Is the current line itself part of a reusable prompt/instruction being dictated for later copying? "
            "Prompts may be English inside Swiss German. Exclude mere discussion of prompting and lead-ins."
        ),
        "criteria": MappingProxyType(
            {
                "true": "The current line is prompt text being dictated for reuse.",
                "false": "The line is ordinary meeting discussion, a lead-in, or an explanation around a prompt.",
            }
        ),
    }
)

PROMPT_CONTENT_QUESTION: Mapping[str, object] = MappingProxyType(
    {
        "instructions": (
            "Is this whole sentence itself part of the reusable prompt text being dictated, "
            "rather than a lead-in, recording aside, or explanation?"
        ),
        "criteria": MappingProxyType(
            {
                "true": "The full sentence belongs verbatim in the reusable prompt.",
                "false": "The sentence is surrounding talk, a lead-in, or an explanation rather than prompt text.",
            }
        ),
    }
)
