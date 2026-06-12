#!/usr/bin/env python3
"""speaker_hints — infer the meeting counterpart from the transcript itself,
validated against the Brain ClientContext stakeholder directories.

Used when the calendar lookup returned no external attendees (BlueCare
meetings run on their Teams/M365 calendar, not Matthias's Google Calendar —
~37% of recorded meetings have no calendar match). In that case
speaker_reconcile has no canonical names to rewrite "Speaker B" to, and the
unknown label leaks into InsightBase, insights, and the wiki.

Empirical grounding (A/B test 2026-06-12 on 6 ground-truth meetings):

- Raw name-mention counting picks the WRONG person: in 1:1s the counterpart's
  own name is rarely spoken, while absent colleagues are named constantly
  ("Gina" 40x in a meeting whose counterpart was Florian).
- In Swiss German, third-person references carry an article — "de Gina",
  "d'Gina", "vo de Gina", "em Roger" — while direct address never does
  ("Sali Natascha", "Merci dir, Philipp"). The article is a reliable
  third-person marker.
- Gemini often starts a speaker as "Speaker 1" and drifts to the real name
  as a label mid-transcript once someone says it. That label IS the
  counterpart with high precision — it just never reaches participants[].

Signals, all validated against a known-person directory (never trust a bare
transcript name — the Stefan-bias lesson):

1. label: a non-self transcript speaker label matches a directory person.
2. vocative: the HOST's lines address a directory person by first name
   without a Swiss German article in front.

Both signals exclude names that the counterpart speaks about (their lines
talking ABOUT colleagues), and a candidate is returned only when the
evidence is one-sided.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BRAIN_REPO = Path(os.environ.get("BRAIN_REPO", os.path.expanduser("~/Repos/Brain")))
CLIENTS_DIR = BRAIN_REPO / "Projects" / "clients"

SELF_FIRST = "matthias"
SELF_TOKENS = {"matthias", "heim", "lailix", "host"}

# Swiss German / German articles + prepositions that mark a following name
# as THIRD PERSON ("de Gina", "d'Gina", "em Roger", "vom Stefan", "die Anna").
_ARTICLE_BEFORE = re.compile(
    r"(?:\b(?:de|dr|em|en|vo|vom|zum|zur|bi|bim|mit em|für de|für d|de[mnr]|die|das|"
    r"d|s)['’]?|['’])\s*$",
    re.IGNORECASE,
)

# Vocative cues in the host's speech: greeting/thanks/closing + Name, or a
# name set off by a comma at a clause boundary ("..., Natascha?").
_GREETING_WORDS = (
    r"(?:hi|hoi|sali|salü|hallo|hey|guete morge|morge|merci|danke|tschüss|"
    r"ciao|adieu|gell|oder|genau|exactly|thanks|thank you)"
)

_GENERIC_SPEAKER_RE = re.compile(r"^speaker\s+[A-Za-z0-9]+$", re.IGNORECASE)
_LABEL_RE = re.compile(
    r"(?:^|(?<=[\s\]]))([A-ZÄÖÜ][^\n:]{0,40}?)(?=:)", re.MULTILINE
)
# Unicode-aware name shape: `[^\W\d_]` matches any letter, so names like
# "Natascha Bekrić" or "Małgorzata" survive (the ASCII-umlaut-only class
# used elsewhere drops them).
_NAME_SHAPE_RE = re.compile(
    r"^[A-ZÄÖÜ][^\W\d_'’\-]*(?:['’\-][^\W\d_]+)*"
    r"(?:\s+[A-ZÄÖÜ][^\W\d_'’\-]*(?:['’\-][^\W\d_]+)*){0,2}$",
    re.UNICODE,
)


# ── Brain ClientContext directory ─────────────────────────────────────

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HEADER_ROW_RE = re.compile(r"^\|\s*\**\s*Name\s*\**\s*\|", re.IGNORECASE)


def load_client_directories(clients_dir: Path = CLIENTS_DIR) -> list[dict]:
    """Parse every ClientContext.md stakeholder/contact table.

    Returns [{name, first, company, role, email}], deduped by (first, company).
    Table rows live under '## … Contact Directory' / '## … Stakeholder' headers
    with a Name column first. Robust to bold markers and parenthetical notes.
    """
    people: list[dict] = []
    if not clients_dir.is_dir():
        logger.warning(f"speaker_hints: clients dir not found: {clients_dir}")
        return people
    for ctx in sorted(clients_dir.glob("*/*_ClientContext.md")):
        company = ctx.name.replace("_ClientContext.md", "")
        in_section = False
        try:
            lines = ctx.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning(f"speaker_hints: cannot read {ctx}: {e}")
            continue
        for line in lines:
            if line.startswith("## "):
                in_section = bool(re.search(
                    r"contact directory|stakeholder", line, re.IGNORECASE))
                continue
            if not in_section or not line.startswith("|") or "---" in line:
                continue
            if _HEADER_ROW_RE.match(line):
                continue
            cells = [c.strip().strip("*").strip() for c in line.split("|")[1:-1]]
            if not cells or not cells[0]:
                continue
            raw_name = re.sub(r"\s*[(\[].*$", "", cells[0]).strip().strip("*").strip()
            if not _NAME_SHAPE_RE.match(raw_name):
                continue
            first = raw_name.split()[0].lower()
            if first in SELF_TOKENS or len(first) < 3:
                continue
            email_m = _EMAIL_RE.search(line)
            people.append({
                "name": raw_name,
                "first": first,
                "company": company,
                "role": cells[1][:80] if len(cells) > 1 else "",
                "email": email_m.group(0).lower() if email_m else None,
            })
    seen: set[tuple] = set()
    out = []
    for p in people:
        key = (p["first"], p["company"].lower())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ── Transcript analysis ───────────────────────────────────────────────

def _split_turns(transcript: str) -> list[tuple[str, str]]:
    """[(speaker_label, text)] in order. Text up to the next label."""
    turns = []
    matches = [m for m in _LABEL_RE.finditer(transcript)
               if _GENERIC_SPEAKER_RE.match(m.group(1).strip())
               or _NAME_SHAPE_RE.match(m.group(1).strip())]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
        turns.append((m.group(1).strip(), transcript[m.end() + 1:end]))
    return turns


def _is_self(label: str) -> bool:
    toks = label.lower().split()
    return bool(toks) and toks[0] in SELF_TOKENS


def _vocative_hits(host_text: str, first: str) -> int:
    """Count direct-address uses of `first` in the host's speech.

    A hit needs (a) no Swiss German article directly before the name, and
    (b) a vocative shape: greeting word before it, or comma right before it,
    or sentence punctuation right after it.
    """
    hits = 0
    for m in re.finditer(r"\b" + re.escape(first) + r"\b", host_text,
                         re.IGNORECASE):
        before = host_text[max(0, m.start() - 24):m.start()]
        after = host_text[m.end():m.end() + 3]
        if _ARTICLE_BEFORE.search(before):
            continue  # "de Gina" — third person
        vocative_shape = (
            re.search(_GREETING_WORDS + r"[\s,]*$", before, re.IGNORECASE)
            or before.rstrip().endswith(",")
            or re.match(r"\s*[,.!?]", after)
        )
        if vocative_shape:
            hits += 1
    return hits


def detect_counterpart(gemini_dict: dict,
                       directory: Optional[list[dict]] = None) -> Optional[dict]:
    """Infer the meeting counterpart from the transcript, directory-validated.

    Args:
        gemini_dict: Gemini parsed_response (transcript + participants).
        directory: known-person records (default: Brain ClientContext tables).

    Returns None, or:
        {"name", "email", "company", "role", "method": "transcript_label" |
         "direct_address", "confidence": "high", "evidence": str}

    Only returns a candidate when evidence points at exactly ONE directory
    person — ambiguity returns None (the Brain /meeting-actions layer can
    still resolve those with full context).
    """
    transcript = (gemini_dict or {}).get("transcript") or ""
    if not transcript:
        return None
    if directory is None:
        directory = load_client_directories()
    if not directory:
        return None

    turns = _split_turns(transcript)
    host_text = " ".join(t for lbl, t in turns if _is_self(lbl))
    other_text = " ".join(t for lbl, t in turns if not _is_self(lbl))
    label_counts: dict[str, int] = {}
    for lbl, _ in turns:
        if not _is_self(lbl) and not _GENERIC_SPEAKER_RE.match(lbl):
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    by_first: dict[str, list[dict]] = {}
    for p in directory:
        by_first.setdefault(p["first"], []).append(p)

    # Signal 1 — transcript speaker label matches a directory person.
    # Gemini drifts "Speaker 1" → "Natascha" once the name is audible; a
    # real-name label is the counterpart with high precision — UNLESS the
    # label itself is a Gemini misattribution (the Stefan-bias incident:
    # a meeting whose real counterpart was Lorenz got labeled "Stefan").
    label_candidates: list[tuple[dict, str]] = []
    for label, n_label in label_counts.items():
        if n_label < 2:
            continue  # one-off labels are usually mid-paragraph captures
        first = label.split()[0].lower()
        matches = by_first.get(first, [])
        if not matches:
            continue
        if len(matches) > 1:
            # Several directory people share the first name. A bare
            # first-name label cannot disambiguate; only a multi-token
            # label matching exactly one full directory name may pass.
            full = [p for p in matches
                    if " " in label and p["name"].lower() == label.lower()]
            if len(full) != 1:
                continue
            matches = full
        person = matches[0]
        # Self-reference check: people don't talk about themselves in the
        # third person. If the `label`-labeled speaker's own lines mention
        # that first name WITH a Swiss German article ("de Stefan het
        # gseit..."), the label is misattributed — the real counterpart is
        # someone else talking ABOUT that person. Only article-marked
        # mentions count: bare occurrences inside a turn are usually nested
        # speaker labels that the lazy label regex swallowed in chunked
        # transcripts, not genuine references.
        own_text = " ".join(t for lbl2, t in turns if lbl2 == label)
        self_refs = 0
        for m in re.finditer(r"\b" + re.escape(first) + r"\b", own_text,
                             re.IGNORECASE):
            if _ARTICLE_BEFORE.search(own_text[max(0, m.start() - 24):m.start()]):
                self_refs += 1
        if self_refs >= 2:
            logger.info(
                "speaker_hints: label %r rejected — speaker refers to "
                "%r %dx in their own lines (third-person self-reference "
                "impossible; label likely misattributed)",
                label, first, self_refs,
            )
            continue
        label_candidates.append(
            (person, f"transcript speaker label {label!r} ({n_label} turns) "
                     f"matches {person['name']} ({person['company']})")
        )

    if len({c[0]["first"] for c in label_candidates}) == 1:
        person, evidence = label_candidates[0]
        return {**{k: person.get(k) for k in ("name", "email", "company", "role")},
                "method": "transcript_label", "confidence": "high",
                "evidence": evidence}
    if label_candidates:
        logger.info("speaker_hints: multiple labeled candidates — ambiguous, skipping")
        return None

    # Signal 2 — the host directly addresses a directory person, and the
    # counterpart's own lines don't dominate that name in third person.
    scored: list[tuple[int, dict]] = []
    for first, matches in by_first.items():
        if len(matches) != 1:
            continue  # same first name at several clients — ambiguous
        person = matches[0]
        voc = _vocative_hits(host_text, first)
        if voc < 2:
            continue  # one vocative can be a mis-hear; require repetition
        third_person = len(re.findall(
            r"\b" + re.escape(first) + r"\b", other_text, re.IGNORECASE))
        if third_person > voc:
            continue  # mostly talked ABOUT (by the counterpart) — exclusion
        scored.append((voc, person))

    if len(scored) == 1:
        voc, person = scored[0]
        return {**{k: person.get(k) for k in ("name", "email", "company", "role")},
                "method": "direct_address", "confidence": "high",
                "evidence": f"host addressed {person['first'].title()!r} "
                            f"directly {voc}x (no third-person article)"}
    if len(scored) > 1:
        logger.info(
            "speaker_hints: %d direct-address candidates — ambiguous, skipping",
            len(scored),
        )
    return None


if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python speaker_hints.py <transcript.json>")
        sys.exit(1)
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    result = detect_counterpart(d)
    print(json.dumps(result, indent=2, ensure_ascii=False) if result else "no hint")
