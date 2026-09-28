"""Canonical binary question definitions shared by runtime and calibration."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


# Prompt scores are model-prompt-sensitive. Keep the exact runtime wording and
# criteria in one immutable definition so calibration cannot score a paraphrase.
DICTATING_PROMPT_QUESTION: Mapping[str, object] = MappingProxyType(
    {
        "instructions": (
            "Is the current line part of an instruction being dictated for an AI, agent, or coding assistant to carry out? "
            "Count direct dictation and indirect instructions in Swiss German or Hochdeutsch, for example "
            "'ich würd em Claude säge, er söll …', 'mir müessted em Agent säge …', or "
            "'ich würde Claude sagen, er soll …'. "
            "Count English dictation inside dialect. "
            "Do not count talk that merely mentions an AI without stating a task."
        ),
        "criteria": MappingProxyType(
            {
                "true": (
                    "The line states, even indirectly, a task, constraint, or question "
                    "an AI or agent should carry out."
                ),
                "false": (
                    "The line has no task aimed at an AI or agent. "
                    "Recording asides and ordinary discussion do not count."
                ),
            }
        ),
    }
)

PROMPT_CONTENT_QUESTION: Mapping[str, object] = MappingProxyType(
    {
        "instructions": (
            "Is this whole sentence part of the instruction being given to an AI, agent, or coding assistant, "
            "including an indirect frame such as telling Claude or an agent what it should do? "
            "A recording aside or unrelated chatter is not prompt content."
        ),
        "criteria": MappingProxyType(
            {
                "true": (
                    "The sentence carries the task, constraint, or question for the AI, "
                    "including an indirect instruction frame."
                ),
                "false": (
                    "The sentence is a recording aside, filler, or unrelated conversation "
                    "rather than the instruction."
                ),
            }
        ),
    }
)
