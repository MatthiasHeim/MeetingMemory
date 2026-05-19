#!/usr/bin/env python3
"""speaker_reconcile — canonicalise Gemini-guessed speaker names against the
calendar attendee list BEFORE the transcript is persisted.

Gemini's `participants[].name` is a guess from audio. When it's wrong (e.g.
"Nadine Maricic" instead of the actual attendee "Ladina Walicki-Kasper"),
the wrong name flows into sources.title, sources.participants,
sources.content_text, every extracted insight, the CRM, and the wiki.

This module rewrites Gemini's labels to canonical names taken from the
Google Calendar attendee list (resolved by `calendar_resolve.resolve()`).
It runs in `transcribe_watcher.py:_process_with_gemini` between the Gemini
call and the JSON write, so every downstream consumer sees canonical names.

Decision rules per Gemini-guessed name (highest precedence first):
  - "Matthias" / "Matthias Heim" / "host" → SELF (always anchored).
  - First+last token both match a calendar attendee → confident, rewrite.
  - 1:1 meeting (calendar has exactly one non-self attendee AND Gemini
    surfaced exactly one non-self speaker) → singleton, rewrite to the
    sole calendar attendee. This catches the dangerous case where Gemini
    hallucinates a totally wrong name (e.g. "Nadine Maricic" instead of
    "Ladina Walicki-Kasper").
  - First-name-only match AND exactly one non-self attendee shares it
    → fuzzy, rewrite.
  - Otherwise → keep Gemini guess (logged as no-match for review).

Second pass — singleton_collapse: after the per-name loop, if calendar
has exactly ONE non-self attendee but multiple non-self gemini labels
remained unmatched (chunk drift produced 2-3 different labels for one
physical speaker, e.g. "Speaker 1" / "Speaker 2" / "Vivienne" all for
Antonella), collapse ALL unmatched non-self labels to that sole canonical.
Rationale: in a 1:1, every non-host voice must be that one attendee —
chunk drift cannot create real additional speakers.

When the calendar lookup returned no external attendees (solo event, no
match), reconciliation is skipped entirely — only Matthias is anchored,
and there's no canonical to compare guesses against.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

SELF_NAME = "Matthias Heim"
SELF_FIRST = "Matthias"


def _tokenize(name: str) -> list[str]:
    return [t for t in re.split(r"\s+", (name or "").strip().lower()) if t]


def _canonical_attendees(participant_details: list[dict]) -> list[dict]:
    """Flatten participant_details into match-friendly records."""
    out: list[dict] = []
    for p in participant_details or []:
        if not isinstance(p, dict):
            continue
        full = (p.get("name") or "").strip()
        if not full:
            continue
        toks = _tokenize(full)
        out.append({
            "full": full,
            "first": toks[0] if toks else "",
            "last": toks[-1] if len(toks) > 1 else "",
            "role": p.get("role") or "participant",
        })
    return out


def _is_self_label(guess: str) -> bool:
    g_toks = _tokenize(guess)
    if not g_toks:
        return False
    return (g_toks[0] == SELF_FIRST.lower()
            or guess.strip().lower() == SELF_NAME.lower()
            or g_toks == ["host"])


def _match(guess: str, canonicals: list[dict],
           singleton_external: Optional[dict]) -> tuple[Optional[dict], str]:
    """Return (canonical_record_or_None, rule).

    rule ∈ {'self', 'confident', 'singleton', 'fuzzy', 'none'}.
    """
    g = (guess or "").strip()
    if not g:
        return None, "none"
    g_toks = _tokenize(g)

    # Self anchor — Gemini prompt labels host as "Matthias:". We anchor the
    # decision to 'self' but return the GUESS as canonical so we don't rewrite
    # "Matthias:" labels to "Matthias Heim:" (the convention is the short form).
    if _is_self_label(g):
        return {"full": g, "first": g_toks[0], "last": "", "role": "self"}, "self"

    # Confident: first AND last token match a non-self canonical.
    if len(g_toks) >= 2:
        for c in canonicals:
            if c["role"] == "self":
                continue
            if c["first"] and c["last"] and \
                    c["first"] == g_toks[0] and c["last"] == g_toks[-1]:
                return c, "confident"

    # Singleton: 1:1 meeting and Gemini surfaced exactly one non-self speaker.
    # This is the rule that catches the "wrong-name hallucination" case where
    # Gemini's guess shares no tokens with the actual attendee.
    if singleton_external is not None:
        return singleton_external, "singleton"

    # Fuzzy: first-name match, and exactly one non-self canonical shares it.
    matches = [c for c in canonicals
               if c["role"] != "self" and c["first"] == g_toks[0]]
    if len(matches) == 1:
        return matches[0], "fuzzy"

    return None, "none"


# Regex that matches a speaker label `Name:` when preceded by line start,
# whitespace, or `]` (handles both `\nLadina:` and `[00:00] Ladina:`).
def _label_pattern(name: str) -> re.Pattern:
    return re.compile(r"(?:^|(?<=[\s\]]))" + re.escape(name) + r"(?=:)",
                      re.MULTILINE)


# Tight name-shape check used to filter speaker candidates harvested from the
# transcript. 1-3 tokens, each starting uppercase; allows umlauts, hyphens
# (Walicki-Kasper), apostrophes (O'Brien). Rejects fragments with digits,
# brackets, or sentence punctuation — those are mid-paragraph captures, not
# real speaker labels.
_NAME_RE = re.compile(
    r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'\-]*"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'\-]*){0,2}$"
)


def _looks_like_speaker_name(s: str) -> bool:
    return bool(_NAME_RE.match((s or "").strip()))


def reconcile(gemini_dict: dict, calendar_resolution: Optional[dict]) -> dict:
    """Rewrite Gemini speaker names in `gemini_dict` (mutates in place).

    Args:
        gemini_dict: Gemini parsed_response — has transcript, participants,
            speaker_emotions, speaker_pacing, interruptions, energy_levels.
        calendar_resolution: output of `calendar_resolve.resolve()`. Uses
            participant_details. Pass None to skip.

    Returns:
        Forensic log to attach to participant_resolution_log.speaker_reconciliation:
        {
            "decisions": [{"gemini_name", "canonical_name", "rule", "evidence"}],
            "rewrote_speakers": int,
            "skipped_no_calendar": bool,
        }
    """
    log = {
        "decisions": [],
        "rewrote_speakers": 0,
        "collapsed_labels": 0,
        "skipped_no_calendar": False,
    }

    cal_attendees = (calendar_resolution or {}).get("participant_details") or []
    canonicals = _canonical_attendees(cal_attendees)
    if not any(c["role"] != "self" for c in canonicals):
        log["skipped_no_calendar"] = True
        logger.info("speaker_reconcile: no external calendar attendees; skipping rewrite")
        return log

    # Collect Gemini-guessed speaker names from participants[] and from the
    # transcript labels themselves (Gemini sometimes labels speakers without
    # listing them in participants). Generic "Speaker N" / "Speaker A" labels
    # ARE collected — when calendar shows a 1:1 they collapse to the sole
    # attendee in the second pass below.
    gemini_names: list[str] = []
    for p in gemini_dict.get("participants") or []:
        if isinstance(p, dict) and p.get("name"):
            n = p["name"].strip()
            if n and n not in gemini_names:
                gemini_names.append(n)
    transcript = gemini_dict.get("transcript") or ""
    _generic_speaker_re = re.compile(r"^speaker\s+[A-Za-z0-9]+$", re.IGNORECASE)
    for m in re.finditer(
        r"(?:^|(?<=[\s\]]))([A-ZÄÖÜ][^\n:]{0,40}?)(?=:)",
        transcript, flags=re.MULTILINE,
    ):
        n = m.group(1).strip()
        if not n or n in gemini_names:
            continue
        # Generic "Speaker N"/"Speaker A" labels are valid candidates (for
        # the collapse pass); skip mid-paragraph captures with non-name shape.
        if _generic_speaker_re.match(n) or _looks_like_speaker_name(n):
            gemini_names.append(n)

    # Compute strict singleton: calendar has exactly one non-self attendee
    # AND Gemini surfaced exactly one non-self speaker label.
    non_self_canonicals = [c for c in canonicals if c["role"] != "self"]
    non_self_gemini = [n for n in gemini_names if not _is_self_label(n)]
    singleton_external = (
        non_self_canonicals[0]
        if len(non_self_canonicals) == 1 and len(non_self_gemini) == 1
        else None
    )

    rename: dict[str, str] = {}
    for g in gemini_names:
        canonical, rule = _match(g, canonicals, singleton_external)
        decision = {
            "gemini_name": g,
            "canonical_name": canonical["full"] if canonical else g,
            "rule": rule,
            "evidence": (
                f"calendar attendee match ({rule})" if canonical
                else "no calendar match — kept Gemini guess"
            ),
        }
        log["decisions"].append(decision)
        if canonical and canonical["full"] != g:
            rename[g] = canonical["full"]

    # Second pass — singleton_collapse. In a 1:1 calendar meeting where
    # strict singleton couldn't fire (multiple drifted gemini labels) AND
    # the sole canonical wasn't already confidently/fuzzy matched, every
    # unmatched non-self label must be the sole canonical. Rewrite each
    # 'none'-ruled decision in place. Self labels are excluded. The
    # "no prior confident/fuzzy match for the canonical" guard preserves
    # the ad-hoc-joiner case: when Ladina is correctly identified and a
    # second person also appears, the second person is NOT collapsed.
    if len(non_self_canonicals) == 1 and singleton_external is None:
        sole = non_self_canonicals[0]
        canonical_already_matched = any(
            d.get("rule") in ("confident", "fuzzy")
            and d.get("canonical_name") == sole["full"]
            for d in log["decisions"]
        )
        if not canonical_already_matched:
            unmatched_non_self = [
                d for d in log["decisions"]
                if d["rule"] == "none" and not _is_self_label(d["gemini_name"])
            ]
            collapse_count = 0
            for d in unmatched_non_self:
                if sole["full"] == d["gemini_name"]:
                    continue  # already canonical
                rename[d["gemini_name"]] = sole["full"]
                d["canonical_name"] = sole["full"]
                d["rule"] = "singleton_collapse"
                d["evidence"] = (
                    f"1:1 calendar with {len(unmatched_non_self)} drifted "
                    f"labels — collapsed to sole attendee"
                )
                collapse_count += 1
            if collapse_count:
                log["collapsed_labels"] = collapse_count
                logger.info(
                    "speaker_reconcile: singleton_collapse rewrote %d "
                    "drifted label(s) to %r",
                    collapse_count, sole["full"],
                )

    if not rename:
        return log
    log["rewrote_speakers"] = len(rename)

    # Apply rewrites.
    new_transcript = transcript
    for old, new in rename.items():
        new_transcript = _label_pattern(old).sub(new, new_transcript)
    gemini_dict["transcript"] = new_transcript

    for p in gemini_dict.get("participants") or []:
        if isinstance(p, dict) and p.get("name") in rename:
            p["name"] = rename[p["name"]]

    for entry in gemini_dict.get("speaker_emotions") or []:
        if isinstance(entry, dict) and entry.get("speaker") in rename:
            entry["speaker"] = rename[entry["speaker"]]

    for ev in gemini_dict.get("interruptions") or []:
        if isinstance(ev, dict):
            for k in ("interrupter", "interruptee"):
                if ev.get(k) in rename:
                    ev[k] = rename[ev[k]]

    for key in ("speaker_pacing", "energy_levels"):
        d = gemini_dict.get(key) or {}
        if isinstance(d, dict):
            gemini_dict[key] = {rename.get(k, k): v for k, v in d.items()}

    logger.info(
        "speaker_reconcile: rewrote %d speakers — %s",
        len(rename),
        ", ".join(f"{old!r}→{new!r}" for old, new in rename.items()),
    )
    return log
