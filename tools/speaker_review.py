#!/usr/bin/env python3
"""speaker_review — suggest who each transcript speaker is, and apply
Matthias's confirmed answer.

Why: a meeting whose speaker identity cannot be verified (bleeding mic, no
calendar roster, relabelled lines) is held by attribution_gate: insights,
tickets and drafts wait. Until now the only way out was a hand-run
/meeting-actions. The Brain Portal /meetings page now shows the proposal and
lets Matthias correct it; this CLI is both halves of that loop.

    # 1. Suggest (the watcher fires this after every seed; the portal can re-run it)
    python3 tools/speaker_review.py propose --source-id 1387 [--dry-run]

    # 2. Apply Matthias's answer (the portal calls this on "Confirm speakers")
    python3 tools/speaker_review.py confirm --source-id 1387 --mapping-json '[
        {"label": "Tomi", "kind": "crm", "name": "Dominic Schelling",
         "company": "BlueCare", "contact_id": 1234},
        {"label": "Matthias", "kind": "self"},
        {"label": "Ruedi", "kind": "unknown"}]' [--dry-run] [--no-trigger]

propose stores `sources.metadata.speaker_review` (status "proposed"): one entry
per transcript label with line share, sample lines, CRM candidates and the
model's suggestion. It never changes the transcript or the gate.

confirm rewrites the labels in the transcript JSON and the sources row in one
pass (swaps and merges are safe), marks attribution `human_confirmed` so the
gate opens, and then hands the meeting on:
  * never processed, no insights  -> /meeting-actions, as the watcher would
  * insights already extracted    -> those insights are flagged for
    re-extraction (memory-consolidation rebuilds them under the right names)
    and /meeting-actions runs with Step 2 skipped, so nothing is duplicated
  * already fully processed       -> labels corrected + insights flagged, no
    second /meeting-actions (it would create its tickets and draft again)

Labels that are not part of the meeting (a child after the goodbye, a
bystander, a pre-meeting aside) map to kind "nonparticipant": confirm cuts
their lines out of the transcript and content_text (raw lines kept in
`_meta.excluded_speech`), so they never reach insights, participants or the
follow-up draft, and they never hold a meeting.

Auto-confirm: after propose, `auto_confirm_decision` (thresholds in
AUTO_CONFIRM below) applies the suggestion without a human when every label
with a meaningful share of the talk is Matthias or an existing CRM contact at
high confidence AND has independent evidence (host channel or voice for
Matthias; calendar e-mail, voice, or the name spoken in the transcript at an
involved company for a contact). It is recorded as attribution status
`auto_confirmed`, never as a human confirmation, and hands on exactly like
confirm. Otherwise the portal asks, and `review.auto.reasons` says why.

    python3 tools/speaker_review.py auto-check --source-id 1572     # dry-run the rule on the stored proposal
    python3 tools/speaker_review.py auto-check --all-proposed       # ... on every meeting still waiting
    python3 tools/speaker_review.py agreement [--days 90]            # proposals vs confirmations

The last stdout line is always one JSON object (the portal reads it).
Exit codes: 0 ok, 1 error, 2 invalid mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_coherence import claude_cli_runner, parse_lines  # noqa: E402
from speaker_reconcile import (  # noqa: E402
    _is_self_label,
    _merge_energy_under_rename,
    _merge_pacing_under_rename,
)

logger = logging.getLogger("speaker_review")

SELF_NAME = "Matthias Heim"
SELF_EMAIL = "matthias@lailix.com"
BRAIN_REPO = Path("/Users/Matthias/Repos/Brain")
FIND_PERSON = BRAIN_REPO / ".claude" / "scripts" / "find_person.py"
FLAG_REEXTRACTION = BRAIN_REPO / ".claude" / "scripts" / "flag_reextraction.py"
HEADLESS_CLAUDE = BRAIN_REPO / ".claude" / "scripts" / "headless_claude.py"
CLIENTS_DIR = BRAIN_REPO / "Projects" / "clients"
LOG_DIR = Path.home() / "Documents" / "MeetingRecorder" / "logs"
AUTOMATION_CONFIG_DIR = Path.home() / ".claude-automation"
TRIGGER_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/Users/Matthias/.local/bin"

KINDS = ("self", "crm", "new", "unknown", "nonparticipant")
CONFIDENCES = ("high", "medium", "low")
MAX_TRANSCRIPT_CHARS = 120_000
SAMPLES_PER_LABEL = 4
SAMPLE_CHARS = 220
MAX_NAME_CHARS = 60          # speaker_coherence._LINE_RE reads labels of at most 60 chars
UNKNOWN_PREFIX = "Unidentified participant"
CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"
SELF_DOMAINS = {"lailix.com", "lailix.ch"}
# Mail domains that say nothing about an employer.
GENERIC_DOMAINS = {"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "hotmail.ch", "live.com",
                   "icloud.com", "me.com", "yahoo.com", "gmx.ch", "gmx.net", "gmx.de", "bluewin.ch",
                   "sunrise.ch", "protonmail.com", "proton.me", "web.de", "group.calendar.google.com"}
LEGAL_SUFFIXES = {"ag", "gmbh", "sa", "sàrl", "sarl", "ltd", "inc", "llc", "corp", "co", "kg", "ab", "bv",
                  "nv", "plc", "se", "oy", "as", "a/s", "limited", "holding"}

# ── Auto-confirm: every threshold of the rule lives here ─────────────────────
# Measured 2026-11-27 over 30 confirmed labels: high-confidence suggestions were
# right 23 of 24 times, every other miss was medium/low. So: high confidence
# plus at least one independent signal, and anything that talks for real must
# pass. `speaker_review.py agreement` tells whether to loosen or tighten this.
AUTO_CONFIRM = {
    "rule": "auto-v1",
    # A label at or below BOTH limits is "minor": it never blocks. If it cannot
    # pass the full test it becomes "Unidentified participant".
    "minor_max_share": 0.05,
    "minor_max_lines": 10,
    # "Not in the meeting" (child, bystander) is applied without asking up to
    # this share at high confidence, or at any confidence while minor.
    "nonparticipant_max_share": 0.15,
    # Fewer labelled lines than this (a garbled or one-line transcript) -> ask.
    "min_total_lines": 20,
    # At least one substantial label must be verified as Matthias.
    "require_self": True,
    # Voice evidence counts only at the voice-match "clear" verdict; the host
    # voice check counts at or above this similarity to the self profile.
    "host_voice_min_similarity": 0.5,
    # Only meetings the gate holds for a human (policy "hold"). Other policies
    # already started /meeting-actions; relabelling under it would race it.
    "eligible_policies": ("hold",),
}

PROPOSE_SYSTEM = (
    "You identify who the speakers in a meeting transcript really are. Answer "
    "with a single JSON object and nothing else. Do not use tools; everything "
    "you need is in the message."
)


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────

def is_speaker_label(label: str) -> bool:
    """A real `[ts] Label:` speaker, not text cut at a colon.

    parse_lines takes everything up to the first colon, so a line with a
    second timestamp ("Nice to see you. [00:12:39] …") or a stage direction
    ("(Background noise … until 02:43)") yields a fake label. Those lines
    are never offered and never rewritten.
    """
    label = (label or "").strip()
    if not label or len(label) > MAX_NAME_CHARS or any(ch in label for ch in "[]()"):
        return False
    return len(label.split()) <= 6 and not label.isdigit()


def label_stats(transcript: str) -> list[dict]:
    """One entry per distinct speaker label: line count, share, sample lines.

    Samples prefer substantive lines (longest first among the first ones seen)
    so the portal shows something that identifies the person, not "Genau.".
    """
    lines = [ln for ln in parse_lines(transcript or "") if is_speaker_label(ln["label"])]
    total = len(lines) or 1
    by_label: dict[str, list[dict]] = {}
    order: list[str] = []
    for ln in lines:
        label = ln["label"]
        if label not in by_label:
            by_label[label] = []
            order.append(label)
        by_label[label].append(ln)
    out = []
    for label in order:
        rows = by_label[label]
        # Longest among the first 40 lines, then back into time order.
        picked = sorted(rows[:40], key=lambda r: -len(r["rest"].strip()))[:SAMPLES_PER_LABEL]
        picked.sort(key=lambda r: r["index"])
        out.append({
            "label": label,
            "lines": len(rows),
            "share": round(len(rows) / total, 3),
            "self_label": _is_self_label(label),
            "samples": [{"ts": r["ts"], "text": _clip(r["rest"].strip(), SAMPLE_CHARS)} for r in picked],
        })
    return out


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ── Names, companies, calendar attendees (pure) ──────────────────────────────

def norm_text(text: Optional[str]) -> str:
    """Casefolded, accent-free, single-spaced: 'Rüede' == 'Ruede'."""
    text = unicodedata.normalize("NFKD", (text or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.replace("-", " ").replace("_", " ").split())


def person_name(raw: Optional[str]) -> str:
    """'Keller, Marc' -> 'Marc Keller' (Outlook/Exchange invite order)."""
    raw = " ".join((raw or "").split())
    if raw.count(",") == 1:
        last, first = (x.strip() for x in raw.split(","))
        if first and last and " " not in first.strip("."):
            return f"{first} {last}"
    return raw


def first_name(name: Optional[str]) -> str:
    toks = norm_text(person_name(name)).split()
    return toks[0] if toks else ""


def same_name(a: Optional[str], b: Optional[str]) -> bool:
    return bool(norm_text(person_name(a))) and norm_text(person_name(a)) == norm_text(person_name(b))


def company_base(name: Optional[str]) -> str:
    """'Example Group (Example Holding)' -> 'Example Group'; 'Acme AG' -> 'Acme'."""
    base = re.sub(r"\([^)]*\)", " ", name or "")
    toks = base.replace(",", " ").split()
    while len(toks) > 1 and toks[-1].casefold().strip(".") in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def company_variants(name: Optional[str]) -> list[str]:
    """Every way a company may be written in a title or transcript: the full
    name, without brackets and legal suffix, and the bracketed long form."""
    out: list[str] = []
    for v in [name or "", company_base(name), *re.findall(r"\(([^)]{3,})\)", name or "")]:
        v = " ".join(v.split())
        if len(v) >= 3 and v.casefold() not in {o.casefold() for o in out}:
            out.append(v)
    return out


def companies_match(a: Optional[str], b: Optional[str]) -> bool:
    """Same company despite suffixes and brackets ('Acme' ~ 'Acme AG')."""
    x, y = norm_text(company_base(a)), norm_text(company_base(b))
    if not x or not y:
        return False
    return x == y or SequenceMatcher(None, x, y).ratio() >= 0.88 or \
        (min(len(x), len(y)) >= 4 and (x.split()[0] == y.split()[0]) and (x in y or y in x))


def email_domain(email: Optional[str]) -> str:
    email = (email or "").strip().casefold()
    return email.rsplit("@", 1)[1] if "@" in email else ""


def calendar_attendees(calendar: dict) -> tuple[list[dict], bool]:
    """(non-host attendees of the bound event, authoritative?).

    Only a single matching event (or the chosen one after a bind) is
    authoritative; a collision offers its candidates as hints only.
    """
    cands = [c for c in (calendar or {}).get("candidates") or [] if isinstance(c, dict)]
    chosen = (calendar or {}).get("chosen_event_id")
    picked = [c for c in cands if chosen and c.get("event_id") == chosen] or (cands if len(cands) == 1 else [])
    authoritative = bool(picked) and not (calendar or {}).get("ambiguous") and (
        (calendar or {}).get("identity_authoritative") is True or int((calendar or {}).get("match_count") or 0) == 1)
    out, seen = [], set()
    for c in (picked or cands):
        for a in c.get("attendees") or []:
            if not isinstance(a, dict) or (a.get("role") or "") == "self":
                continue
            email = (a.get("email") or "").strip().casefold()
            if email_domain(email) in SELF_DOMAINS:
                continue
            key = email or norm_text(a.get("name"))
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({"name": person_name(a.get("name")), "email": email or None,
                        "company": a.get("company")})
    return out, authoritative


def attendee_domains(attendees: list[dict]) -> list[str]:
    """Employer domains of the attendees, most attendees first."""
    counts: dict[str, int] = {}
    for a in attendees:
        d = email_domain(a.get("email"))
        if d and d not in GENERIC_DOMAINS and d not in SELF_DOMAINS:
            counts[d] = counts.get(d, 0) + 1
    return sorted(counts, key=lambda d: -counts[d])


def first_name_matches(label: str, people: list[dict]) -> list[dict]:
    """Contacts whose first name is this one-word label (mishearings allowed:
    'Lukas'~'Lucas'). Used for labels like 'Marc' that name no company."""
    fn = norm_text(label)
    if not fn or " " in fn or _is_self_label(label):
        return []
    out = []
    for p in people:
        pf = first_name(p.get("name"))
        if pf and (pf == fn or (len(fn) >= 4 and SequenceMatcher(None, pf, fn).ratio() >= 0.85)):
            out.append(p)
    return out


def resolve_new_to_crm(proposals: dict[str, dict], *, lookup: Callable[[str], list[dict]],
                       attendee_contacts: list[dict], involved_companies: list[str],
                       company_people: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Turn a proposed "new" person into the CRM contact they already are (in place).

    The source-1583 miss: the model proposed a "new" person who was a CRM
    contact and on the calendar invite. A "new" becomes "crm" when, in order:
      1. a calendar attendee resolved to a CRM contact by e-mail has that name
         (or that first name, uniquely, when only a first name was given);
      2. a one-word name matches exactly one contact at an involved company;
      3. find_person on the full name gives a near-exact match (>= 0.9) whose
         company is consistent (an involved company or the one the model
         named) and no second such match exists.
    Returns {label: [contact rows]} the portal should list as candidates.
    """
    extra: dict[str, list[dict]] = {}
    involved_people = [p for ppl in company_people.values() for p in ppl]
    for label, prop in proposals.items():
        if prop.get("kind") != "new" or not prop.get("name"):
            continue
        name = person_name(prop["name"])
        one_word = len(norm_text(name).split()) == 1
        hit, how = None, ""
        by_name = [c for c in attendee_contacts if same_name(c.get("name"), name)]
        if not by_name and one_word:
            by_name = first_name_matches(name, attendee_contacts)
        if len({c["id"] for c in by_name}) == 1:
            hit, how = by_name[0], "on the calendar invite (matched by e-mail)"
        if hit is None and one_word:
            fm = {c["id"]: c for c in first_name_matches(name, involved_people)}
            if len(fm) == 1:
                hit = next(iter(fm.values()))
                how = f"the only {name} at {hit.get('company') or 'an involved company'}"
        if hit is None and not one_word:
            rows = [c for c in lookup(name) if float(c.get("score") or 0) >= 0.9]
            consistent = [c for c in rows if any(companies_match(c.get("company"), co)
                                                 for co in [*involved_companies, prop.get("company")] if co)]
            if len({c["id"] for c in consistent}) == 1:
                hit, how = consistent[0], f"CRM name match at {consistent[0].get('company')}"
            elif rows:
                extra[label] = rows[:5]
        if hit is None:
            continue
        prop.update(kind="crm", contact_id=int(hit["id"]), name=hit["name"], company=hit.get("company"),
                    email=hit.get("email"), position=hit.get("position"), crm_match=how)
        prop["evidence"] = _clip(f"{prop.get('evidence') or ''} Existing CRM contact #{hit['id']}: {how}.".strip(), 300)
        extra.setdefault(label, []).insert(0, hit)
    return extra


# ── Voice evidence (from _meta.voice_match; names and scores only) ───────────

def _voice_entry_for(voice_log: Optional[dict], label: str) -> Optional[dict]:
    """The voice-match entry for a transcript label. After an auto_apply rename
    the transcript carries the new name while the log is keyed by the old one."""
    labels = (voice_log or {}).get("labels") or {}
    if label in labels:
        return labels[label]
    for r in (voice_log or {}).get("renames") or []:
        if r.get("to") == label and r.get("label") in labels:
            return labels[r["label"]]
    return None


def _same_person(a: str, b: str) -> bool:
    ta, tb = set((a or "").casefold().split()), set((b or "").casefold().split())
    return bool(ta and tb and (ta <= tb or tb <= ta))


def apply_voice_evidence(proposals: dict[str, dict], labels: list[dict],
                         voice_log: Optional[dict]) -> dict[str, list[dict]]:
    """Fold `_meta.voice_match` into the per-label proposals (in place).

    Returns {label: [extra candidates]} for the portal's candidate list.

    Resolution against the transcript-based (model) proposal:
      * clear voice match, model unknown / low / medium confidence  -> the
        voice match becomes the proposal (confidence medium: one new signal
        from a library that is still small; the human confirms anyway);
      * clear voice match that AGREES with the model                -> the
        model's proposal stays, the voice is added to its evidence and the
        confidence is raised to high;
      * clear voice match that CONTRADICTS a HIGH-confidence model proposal
        (an introduction, being addressed by name)                  -> the
        model's proposal stays; the disagreement is stated in the evidence
        and flagged `voice_conflict` so the human decides;
      * possible match -> evidence text and candidate only.
    Host labels and matches to the host are never used.
    """
    extra: dict[str, list[dict]] = {}
    if not isinstance(voice_log, dict) or not voice_log.get("ran"):
        return extra
    for lb in labels:
        label = lb["label"]
        entry = _voice_entry_for(voice_log, label)
        if lb.get("self_label") or not entry or not entry.get("best"):
            continue
        best, verdict = entry["best"], entry.get("verdict")
        if verdict not in ("clear", "possible") or best.get("role") == "self":
            continue
        name, score = best["name"], float(best["score"])
        text = f"Voice matches {name} ({score:.2f})"
        runner = entry.get("runner_up")
        if runner:
            text += f", next closest {runner['name']} ({float(runner['score']):.2f})"
        prop = proposals.get(label) or {"kind": "unknown", "confidence": "low", "evidence": ""}
        voice = {"name": name, "score": round(score, 3), "verdict": verdict}
        if best.get("crm_contact_id") is not None:
            voice["crm_contact_id"] = best["crm_contact_id"]
        if verdict == "clear":
            model_named = prop.get("kind") in ("self", "crm", "new") and prop.get("name")
            agrees = bool(model_named) and _same_person(prop["name"], name) and prop.get("kind") != "self"
            contradicts = bool(model_named) and not agrees and prop.get("confidence") == "high"
            if agrees:
                prop["confidence"] = "high"
            elif contradicts:
                prop["voice_conflict"] = True
                text += f"; this disagrees with the transcript evidence for {prop['name']}"
            else:
                replaced = f" (replaces the transcript-based guess {prop['name']})" if model_named else ""
                prop = {"kind": "crm" if best.get("crm_contact_id") is not None else "new",
                        "name": name, "confidence": "medium",
                        "evidence": prop.get("evidence") or ""}
                if best.get("crm_contact_id") is not None:
                    prop["contact_id"] = best["crm_contact_id"]
                text += replaced
        else:
            text += f" (possible only: {entry.get('reason') or 'not clear'})"
        prop["evidence"] = _clip((text + ". " + (prop.get("evidence") or "")).strip(), 300)
        prop["voice"] = voice
        proposals[label] = prop
        if best.get("crm_contact_id") is not None:
            extra.setdefault(label, []).append(
                {"id": best["crm_contact_id"], "name": name, "company": None, "position": None,
                 "email": None, "voice_score": round(score, 3)})
    return extra


def validate_mapping(mapping: object, labels: list[str]) -> list[dict]:
    """Normalise the portal's mapping; raise ValueError with a readable reason.

    Every transcript label must be answered exactly once. A label maps to
    Matthias (self), a CRM contact, a new person (name required) or unknown.
    """
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("mapping must be a non-empty list")
    seen: set[str] = set()
    out: list[dict] = []
    for raw in mapping:
        if not isinstance(raw, dict):
            raise ValueError("each mapping entry must be an object")
        label = str(raw.get("label") or "").strip()
        kind = str(raw.get("kind") or "").strip()
        if label not in labels:
            raise ValueError(f"unknown speaker label {label!r}")
        if label in seen:
            raise ValueError(f"speaker label {label!r} answered twice")
        seen.add(label)
        if kind not in KINDS:
            raise ValueError(f"{label}: kind must be one of {', '.join(KINDS)}")
        entry: dict = {"label": label, "kind": kind}
        if kind == "nonparticipant" and len(labels) == 1:
            raise ValueError(f"{label}: the only speaker cannot be outside the meeting (use 'Not a meeting')")
        if kind == "self":
            entry.update(name=SELF_NAME, email=SELF_EMAIL, company="Lailix")
        elif kind in ("crm", "new"):
            name = " ".join(str(raw.get("name") or "").split())
            if (not name or len(name) > MAX_NAME_CHARS or not is_speaker_label(name) or ":" in name
                    or name.lower().startswith(UNKNOWN_PREFIX.lower())):
                raise ValueError(f"{label}: a real name is required (at most {MAX_NAME_CHARS} characters, "
                                 "no ':' or brackets)")
            entry["name"] = name
            for key in ("company", "email", "position"):
                val = raw.get(key)
                entry[key] = " ".join(str(val).split())[:120] if val else None
            if kind == "crm":
                try:
                    entry["contact_id"] = int(raw.get("contact_id"))
                except (TypeError, ValueError):
                    raise ValueError(f"{label}: a CRM contact needs its contact_id") from None
        out.append(entry)
    missing = [lbl for lbl in labels if lbl not in seen]
    if missing:
        raise ValueError(f"no answer for speaker(s): {', '.join(missing)}")
    if not any(e["kind"] == "self" for e in out):
        logger.info("no label mapped to Matthias — allowed (e.g. a recording he only listened to)")
    return out


def target_names(mapping: list[dict]) -> dict[str, str]:
    """label -> new transcript label. Unknowns become "Unidentified N" so no
    wrong name survives in the text; two labels for one person merge."""
    rename: dict[str, str] = {}
    n = 0
    for e in mapping:
        if e["kind"] == "nonparticipant":
            continue            # cut out by exclude_labels, never renamed
        if e["kind"] == "unknown":
            n += 1
            rename[e["label"]] = UNKNOWN_PREFIX if _count_unknown(mapping) == 1 else f"{UNKNOWN_PREFIX} {n}"
        else:
            rename[e["label"]] = e["name"]
    return rename


def _count_unknown(mapping: list[dict]) -> int:
    return sum(1 for e in mapping if e["kind"] == "unknown")


def rename_labels(gemini_dict: dict, rename: dict[str, str]) -> int:
    """Rewrite speaker labels everywhere in one pass. Returns lines changed.

    Single-pass on purpose: sequential substitution breaks a swap
    (Tomi->Ruedi then Ruedi->Tomi turns everyone into Tomi). Only the label
    position of `[ts] Label:` lines is touched, never the spoken text.
    """
    rename = {k: v for k, v in rename.items() if k != v}
    if not rename:
        return 0
    changed = 0
    out_lines = []
    by_index = {ln["index"]: ln for ln in parse_lines(gemini_dict.get("transcript") or "")}
    for i, raw in enumerate((gemini_dict.get("transcript") or "").split("\n")):
        ln = by_index.get(i)
        if ln and ln["label"] in rename:
            out_lines.append(f"{ln['prefix']}{rename[ln['label']]}:{ln['rest']}")
            changed += 1
        else:
            out_lines.append(raw)
    gemini_dict["transcript"] = "\n".join(out_lines)

    parts, seen = [], set()
    for p in gemini_dict.get("participants") or []:
        if isinstance(p, dict):
            p = dict(p, name=rename.get(p.get("name"), p.get("name")))
            if p.get("name") in seen:
                continue
            seen.add(p.get("name"))
        parts.append(p)
    gemini_dict["participants"] = parts

    emotions: dict[str, list] = {}
    for entry in gemini_dict.get("speaker_emotions") or []:
        if isinstance(entry, dict) and entry.get("speaker"):
            emotions.setdefault(rename.get(entry["speaker"], entry["speaker"]), []).extend(entry.get("arc") or [])
    if gemini_dict.get("speaker_emotions"):
        gemini_dict["speaker_emotions"] = [{"speaker": s, "arc": a} for s, a in emotions.items()]
    for ev in gemini_dict.get("interruptions") or []:
        if isinstance(ev, dict):
            for k in ("interrupter", "interruptee"):
                if ev.get(k) in rename:
                    ev[k] = rename[ev[k]]
    if isinstance(gemini_dict.get("speaker_pacing"), dict) and gemini_dict["speaker_pacing"]:
        gemini_dict["speaker_pacing"] = _merge_pacing_under_rename(gemini_dict["speaker_pacing"], rename)
    if isinstance(gemini_dict.get("energy_levels"), dict) and gemini_dict["energy_levels"]:
        gemini_dict["energy_levels"] = _merge_energy_under_rename(gemini_dict["energy_levels"], rename)
    return changed


def exclude_labels(gemini_dict: dict, labels: set[str], *, now: str, by: str) -> int:
    """Cut every line of these labels out of the transcript (in place).

    For speech that is not part of the meeting: a child after the goodbye, a
    bystander. It must not reach content_text, insights, participants or the
    draft. The raw lines stay recoverable in `_meta.excluded_speech` (and in
    the .pre-speaker-review.json backup). Returns the number of lines cut.
    """
    if not labels:
        return 0
    lines = (gemini_dict.get("transcript") or "").split("\n")
    cut = {ln["index"]: ln for ln in parse_lines("\n".join(lines)) if ln["label"] in labels}
    if not cut:
        return 0
    gemini_dict["transcript"] = "\n".join(raw for i, raw in enumerate(lines) if i not in cut)
    gemini_dict["participants"] = [p for p in gemini_dict.get("participants") or []
                                   if not (isinstance(p, dict) and p.get("name") in labels)]
    if gemini_dict.get("speaker_emotions"):
        gemini_dict["speaker_emotions"] = [e for e in gemini_dict["speaker_emotions"]
                                           if not (isinstance(e, dict) and e.get("speaker") in labels)]
    if gemini_dict.get("interruptions"):
        gemini_dict["interruptions"] = [e for e in gemini_dict["interruptions"] if not (
            isinstance(e, dict) and (e.get("interrupter") in labels or e.get("interruptee") in labels))]
    for key in ("speaker_pacing", "energy_levels"):
        if isinstance(gemini_dict.get(key), dict):
            gemini_dict[key] = {k: v for k, v in gemini_dict[key].items() if k not in labels}
    meta = gemini_dict.setdefault("_meta", {})
    store = meta.get("excluded_speech") if isinstance(meta.get("excluded_speech"), dict) else {"lines": []}
    store["lines"] = list(store.get("lines") or []) + [
        {"label": ln["label"], "line_index": i, "ts": ln["ts"], "raw": ln["raw"]} for i, ln in sorted(cut.items())]
    store["labels"] = sorted(set(store.get("labels") or []) | labels)
    store.update(excluded_at=now, excluded_by=by,
                 note="Not part of the meeting (kind nonparticipant). line_index is the position in the "
                      "transcript before the cut; insert the raw lines back to restore them.")
    meta["excluded_speech"] = store
    return len(cut)


def participant_details_for(mapping: list[dict], existing: Optional[list] = None,
                            method: str = "human_confirmed") -> list[dict]:
    """sources.participant_details after confirmation: one row per person.

    Speakers come from the mapping; an existing entry with the same name lends
    its email. Calendar attendees who never got a label (silent listeners) are
    kept, minus the old transcript labels and generic "Speaker X" rows.
    """
    existing = [p for p in (existing or []) if isinstance(p, dict) and p.get("name")]
    # Calendar rows arrive as "Keller, Marc": one key per person either way.
    key_of = lambda name: norm_text(person_name(name))  # noqa: E731
    by_name = {key_of(p["name"]): p for p in existing}
    by_email = {(p.get("email") or "").casefold(): p for p in existing if p.get("email")}
    for e in mapping:
        if e["kind"] in ("crm", "new") and not e.get("email") and by_name.get(key_of(e["name"]), {}).get("email"):
            e["email"] = by_name[key_of(e["name"])]["email"]
    out, seen, seen_email = [], set(), set()
    for e in mapping:
        if e["kind"] in ("unknown", "nonparticipant"):
            continue
        key = key_of(e["name"])
        if key in seen:
            continue
        seen.add(key)
        if e.get("email"):
            seen_email.add(e["email"].casefold())
        row = {"name": e["name"], "email": e.get("email"), "company": e.get("company"),
               "role": "self" if e["kind"] == "self" else "participant",
               "confidence": "high", "resolution_method": method}
        if e.get("contact_id"):
            row["crm_contact_id"] = e["contact_id"]
        if e.get("position"):
            row["title"] = e["position"]
        out.append(row)
    old_labels = {key_of(e["label"]) for e in mapping}
    for p in existing:
        name = key_of(p["name"])
        if (name in seen or name in old_labels or name.startswith("speaker ") or (p.get("role") or "") == "self"
                or (p.get("email") or "").casefold() in seen_email):
            continue
        seen.add(name)
        out.append(dict(p))
    return out


def confirmed_attribution(previous: dict, now: str, auto: Optional[dict] = None) -> dict:
    """The _meta.speaker_attribution block after a confirmation.

    require_turn_evidence is the gate's "go" policy: /meeting-actions may
    use named quotes and speaker-dependent actions. The calendar bind is
    marked authoritative so a collision cannot re-hold it. An automatic
    confirmation is `auto_confirmed` with the rule and its evidence, never
    `human_confirmed` (voice enrolment and accuracy figures rely on that).
    """
    out = dict(previous or {})
    out.update(speaker_dependent_actions="require_turn_evidence", missing_stages=[], confirmed_at=now)
    if auto:
        out.update(status="auto_confirmed", identity_basis="auto_rule",
                   confirmed_by=f"speaker_review {auto.get('rule') or AUTO_CONFIRM['rule']}",
                   auto_rule=auto.get("rule") or AUTO_CONFIRM["rule"], auto_evidence=auto.get("evidence") or {})
    else:
        out.update(status="human_confirmed", identity_basis="human", confirmed_by="Matthias (Brain Portal)")
        out.pop("auto_rule", None)
        out.pop("auto_evidence", None)
    cs = out.get("calendar_search")
    if isinstance(cs, dict) and cs:
        out["calendar_search"] = dict(cs, identity_authoritative=True, ambiguous=False)
    return out


def relabelled_review(transcript: str, mapping: list[dict], rename: dict[str, str],
                      evidence: str = "Confirmed by you.") -> list[dict]:
    """Review rows for the transcript as it is now, so a later correction in
    the portal edits the current labels. The confirmed answer is the proposal."""
    by_new = {}
    for e in mapping:
        if e["label"] in rename:
            by_new.setdefault(rename[e["label"]], e)
    rows = []
    for lb in label_stats(transcript):
        e = by_new.get(lb["label"])
        prop = {"kind": e["kind"], "confidence": "high", "evidence": evidence} if e else \
            {"kind": "unknown", "confidence": "low", "evidence": "Not in the confirmed answer."}
        if e and e["kind"] != "unknown":
            prop.update({k: e.get(k) for k in ("name", "company", "email", "position", "contact_id") if e.get(k)})
        rows.append(dict(lb, proposal=prop, candidates=[]))
    return rows


def handoff_plan(metadata: dict, insight_count: int) -> str:
    """What confirm hands on. Never starts a second /meeting-actions:

    - ticket decisions recorded without a speaker hold -> processed: relabel only
    - decisions recorded under --attribution-held -> its held tickets and the
      draft never ran: follow-up again (Step 5 skips already-applied decisions)
    - already handed off (watcher run, an earlier confirm, the follow-up job)
      but not finished -> that run owns it: relabel only
    - otherwise follow-up, without Step 2 when content-only insights exist
    """
    decisions = (metadata or {}).get("task_decisions")
    if isinstance(decisions, dict):
        return "followup_only" if decisions.get("attribution_held") else "corrected"
    if (metadata or {}).get("pipeline_status") == "ready_for_followup":
        return "corrected"
    return "followup_only" if insight_count else "full"


def build_propose_prompt(*, title: str, calendar: dict, labels: list[dict], candidates: dict,
                         company_people: dict, transcript: str, attendee_contacts: Optional[list] = None) -> str:
    cal_lines = []
    for c in (calendar.get("candidates") or [])[:3]:
        att = ", ".join(f"{a.get('name')} <{a.get('email') or '?'}>" for a in c.get("attendees") or [])
        cal_lines.append(f"- {c.get('start')} \"{c.get('title')}\" attendees: {att or 'none listed'}")
    label_block = []
    for lb in labels:
        label_block.append(f"### Label \"{lb['label']}\" ({lb['lines']} lines, {round(lb['share'] * 100)}%)")
        for s in lb["samples"]:
            label_block.append(f"  [{s['ts']}] {s['text']}")
        cands = candidates.get(lb["label"]) or []
        if cands:
            label_block.append("  CRM name matches: " + "; ".join(
                f"#{c['id']} {c['name']} ({c.get('company') or '?'}, {c.get('position') or '?'})" for c in cands))
    invite_block = [f"  #{c['id']} {c['name']} | {c.get('company') or '?'} | {c.get('email')}"
                    for c in attendee_contacts or []]
    people_block = []
    for company, people in company_people.items():
        people_block.append(f"### {company}")
        people_block.extend(f"  #{p['id']} {p['name']} | {p.get('position') or '?'}" for p in people)
    body = transcript if len(transcript) <= MAX_TRANSCRIPT_CHARS else transcript[:MAX_TRANSCRIPT_CHARS] + "\n[…truncated]"
    return f"""A meeting was recorded by Matthias Heim (Lailix, the host; label "Matthias" or similar is usually him).
The speaker labels below come from an audio model that guesses names from what it hears. In Swiss German
recordings it often mishears names (e.g. "Roel" heard as "Ruedi", "Dominic" as "Tomi"). Work out who each
label really is.

Meeting: {title}
Calendar events around the recording:
{chr(10).join(cal_lines) or "- none found"}

## Speaker labels
{chr(10).join(label_block)}

## Calendar attendees who are CRM contacts (matched by e-mail address, certain)
{chr(10).join(invite_block) or "(none)"}

## CRM contacts at companies that look involved
{chr(10).join(people_block) or "(none)"}

## Rules
- kind "self" = Matthias Heim. kind "crm" = one of the CRM contacts above (give its contact_id). kind "new" =
  a real person who is not in the CRM (give the name as said in the meeting, and the company if clear).
  kind "unknown" = you cannot tell. kind "nonparticipant" = speech that is not part of the meeting: a child or
  family member, a bystander, someone talking to Matthias before the call starts or after the goodbye.
- Before you answer "new", check the calendar attendees and the contact lists: a first name heard in the
  meeting that matches exactly one attendee or contact at the involved company is that contact ("crm").
- Use evidence: self-introductions, being addressed by name, roles/teams described, the calendar title,
  the company's contact list. A similar-sounding first name at the right company with a matching role is
  good evidence for a mishearing.
- confidence "high" only with direct evidence (introduction, addressed by name, unique role match).
- Two labels can be the same person. Never invent a contact_id.
- evidence: one short English sentence a human can check.

Return exactly:
{{"speakers": [{{"label": "...", "kind": "self|crm|new|unknown|nonparticipant", "contact_id": 123 or null, "name": "Full Name" or null,
  "company": "..." or null, "confidence": "high|medium|low", "evidence": "..."}}]}}

## Transcript
{body}
"""


def parse_proposal(text: str, labels: list[str], known_ids: dict[int, dict]) -> dict[str, dict]:
    """Model JSON -> {label: proposal}. Invalid entries degrade to unknown."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError("no JSON object in model response")
    data = json.loads(m.group(0))
    out: dict[str, dict] = {}
    for raw in data.get("speakers") or []:
        if not isinstance(raw, dict) or raw.get("label") not in labels or raw["label"] in out:
            continue
        kind = raw.get("kind") if raw.get("kind") in KINDS else "unknown"
        conf = raw.get("confidence") if raw.get("confidence") in CONFIDENCES else "low"
        prop = {"kind": kind, "confidence": conf, "evidence": _clip(str(raw.get("evidence") or ""), 300)}
        if kind == "self":
            prop.update(name=SELF_NAME, company="Lailix")
        elif kind == "crm":
            try:
                contact = known_ids.get(int(raw.get("contact_id")))
            except (TypeError, ValueError):
                contact = None
            if contact:   # only ids we actually showed the model
                prop.update(contact_id=contact["id"], name=contact["name"], company=contact.get("company"),
                            email=contact.get("email"), position=contact.get("position"))
            elif raw.get("name"):
                prop.update(kind="new", name=str(raw["name"])[:MAX_NAME_CHARS], company=raw.get("company"))
            else:
                prop = {"kind": "unknown", "confidence": "low", "evidence": prop["evidence"]}
        elif kind == "new":
            if raw.get("name"):
                prop.update(name=str(raw["name"])[:MAX_NAME_CHARS], company=raw.get("company"))
            else:
                prop["kind"] = "unknown"
        out[raw["label"]] = prop
    for label in labels:
        out.setdefault(label, {"kind": "unknown", "confidence": "low", "evidence": "No suggestion."})
    return out


def companies_to_check(*, title: str, source_company: Optional[str], calendar: dict, transcript: str,
                       client_names: list[str], domain_companies: Optional[list[str]] = None,
                       limit: int = 4) -> list[str]:
    """Companies whose contact list the model should see (max `limit`).

    Order: the source's company, the companies behind the attendees' mail
    domains (resolved in the CRM: the invite of source 1583 carried only
    "Last, First <address>" and no company), calendar title tags, then client
    names found in the title or transcript under any of their spellings (a
    folder "Example Group (Example Holding)" is said as "Example Group").
    """
    found: list[str] = []

    def add(name):
        name = (name or "").strip()
        if name and name.lower() != "lailix" and not any(companies_match(name, f) for f in found):
            found.append(name)

    add(source_company)
    for name in domain_companies or []:
        add(name)
    for c in calendar.get("candidates") or []:
        for m in re.findall(r"\[([^\]]{2,40})\]", c.get("title") or ""):
            add(m)
        for a in c.get("attendees") or []:
            add(a.get("company"))
    for m in re.findall(r"\[([^\]]{2,40})\]", title or ""):
        add(m)
    text = (title or "") + "\n" + (transcript or "")[:40_000]
    for client in client_names:
        for variant in company_variants(client):
            if len(variant) >= 4 and re.search(r"\b" + re.escape(variant) + r"\b", text, re.IGNORECASE):
                add(company_base(client) or client)
                break
    return found[:limit]


# ── Auto-confirm (pure) ──────────────────────────────────────────────────────

def _channel_classes(system_track: Optional[dict], label: str) -> list[tuple[str, int, float]]:
    """(class, window, remote_share) for this label in every window whose
    host/remote orientation the system-track stage could decide."""
    out = []
    st = system_track or {}
    if not st.get("ran"):
        return out
    for w in st.get("windows") or []:
        if not isinstance(w, dict) or w.get("orientation") != "consistent" or w.get("swapped"):
            continue
        info = (w.get("labels") or {}).get(label)
        if isinstance(info, dict) and info.get("class") in ("host", "remote"):
            out.append((info["class"], int(w.get("index") or 0), float(info.get("remote_share") or 0)))
    return out


def self_evidence(label: str, *, self_label: bool, voice_log: Optional[dict],
                  system_track: Optional[dict], rule: dict = AUTO_CONFIRM) -> tuple[list[str], list[str], list[str]]:
    """(support, contradiction, note) that this label is Matthias, beyond the label text.

    Support: the system track measures the label as the host's own mic in a
    window with a decided orientation, or the voice clearly matches the self
    profile, or (host label) the host-voice check agrees with the profile.
    Contradiction: the system track measures it as remote audio, or the voice
    clearly matches someone else. A failed host-voice check alone is noted;
    it blocks only when nothing physical supports the label.
    """
    support, contra, note = [], [], []
    for cls, win, share in _channel_classes(system_track, label):
        if cls == "host":
            support.append(f"host channel (window {win}, remote share {share:.2f})")
        else:
            contra.append(f"measured as remote audio (window {win}, remote share {share:.2f})")
    entry = _voice_entry_for(voice_log, label) if (voice_log or {}).get("ran") else None
    if entry and entry.get("verdict") == "clear" and entry.get("best"):
        if entry["best"].get("role") == "self":
            support.append(f"voice matches Matthias ({float(entry['best']['score']):.2f})")
        else:
            contra.append(f"voice clearly matches {entry['best'].get('name')}")
    hc = (voice_log or {}).get("host_check") or {}
    if self_label and (voice_log or {}).get("ran") and hc.get("ran"):
        sim = hc.get("centroid_similarity")
        if hc.get("reason") is None and sim is not None and float(sim) >= rule["host_voice_min_similarity"]:
            support.append(f"host voice matches Matthias's profile ({float(sim):.2f})")
        elif hc.get("reason") == "host_voice_does_not_match_self_profile":
            note.append(f"host voice check did not match the profile ({sim})")
    if note and not support:
        contra.extend(note)
    return support, contra, note


def _spoken_names(transcript: str, labels: set[str]) -> str:
    """Normalised text spoken by these labels (where a name being said counts)."""
    return " ".join(norm_text(ln["rest"]) for ln in parse_lines(transcript or "") if ln["label"] in labels)


def crm_evidence(contact: dict, *, labels_for_contact: set[str], all_labels: set[str], transcript: str,
                 voice_log: Optional[dict], system_track: Optional[dict], attendee_emails: set[str],
                 calendar_authoritative: bool, involved_companies: list[str],
                 involved_people: list[dict]) -> tuple[list[str], list[str]]:
    """(support, contradiction) that the labels mapped to this CRM contact are them.

    Support (any one):
      * calendar: the contact's CRM e-mail is an attendee of the bound event;
      * voice: a clear voice match to this contact;
      * name + company: someone else in the meeting says the contact's first
        (or last) name, or the audio model labelled them by it, the contact's
        company is one the meeting involves, and no other contact there has
        that first name.
    Contradiction: a clear voice match to someone else (Matthias included), or
    the system track measuring the label as the host's own mic.
    """
    support, contra = [], []
    email = (contact.get("email") or "").strip().casefold()
    if email and calendar_authoritative and email in attendee_emails:
        support.append(f"{email} is on the calendar invite")
    for label in sorted(labels_for_contact):
        entry = _voice_entry_for(voice_log, label) if (voice_log or {}).get("ran") else None
        if entry and entry.get("verdict") == "clear" and entry.get("best"):
            best = entry["best"]
            if best.get("crm_contact_id") == contact.get("id") or (
                    best.get("crm_contact_id") is None and same_name(best.get("name"), contact.get("name"))):
                support.append(f"voice of {label!r} matches {contact.get('name')} ({float(best['score']):.2f})")
            else:
                contra.append(f"voice of {label!r} clearly matches {best.get('name')}")
        for cls, win, share in _channel_classes(system_track, label):
            if cls == "host":
                contra.append(f"{label!r} is measured on Matthias's own mic (window {win})")
    fn = first_name(contact.get("name"))
    ln_ = norm_text(person_name(contact.get("name"))).split()[-1:] or [""]
    company_ok = any(companies_match(contact.get("company"), c) for c in involved_companies)
    same_first = {p.get("id") for p in involved_people if first_name(p.get("name")) == fn}
    unique = same_first <= {contact.get("id")}
    if fn and company_ok and unique:
        said = set(re.findall(r"\w+", _spoken_names(transcript, all_labels - labels_for_contact)))
        named_by_label = any(fn in norm_text(lb).split() for lb in labels_for_contact)
        token = next((t for t in (fn, ln_[0]) if len(t) >= 3 and t in said), None)
        if token or named_by_label:
            how = f"called {token.title()!r} in the meeting" if token else "labelled by first name by the audio model"
            support.append(f"{how}; the only {fn.title()} at {contact.get('company')}, an involved company")
    return support, contra


def auto_confirm_decision(labels: list[dict], proposals: dict[str, dict], *, transcript: str,
                          voice_log: Optional[dict] = None, system_track: Optional[dict] = None,
                          attendee_emails: Optional[set] = None, calendar_authoritative: bool = False,
                          involved_companies: Optional[list] = None, involved_people: Optional[list] = None,
                          gate_policy: Optional[str] = "hold", review_error: Optional[str] = None,
                          rule: dict = AUTO_CONFIRM) -> dict:
    """Decide whether the suggestion is solid enough to apply without Matthias.

    Returns {"auto": bool, "rule", "reasons": [why a human is still needed],
    "evidence": {label: [...]}, "mapping": [confirm-ready entries]}. Every
    label is judged; `reasons` lists each blocker so the portal can show them.
    """
    out = {"auto": False, "rule": rule["rule"], "reasons": [], "evidence": {}, "mapping": []}
    reasons, evidence, mapping = out["reasons"], out["evidence"], out["mapping"]
    if gate_policy not in rule["eligible_policies"]:
        reasons.append(f"not waiting for a human (gate: {gate_policy})")
    if review_error:
        reasons.append("the suggestion itself failed")
    total = sum(lb["lines"] for lb in labels)
    if total < rule["min_total_lines"]:
        reasons.append(f"only {total} labelled lines (need {rule['min_total_lines']}); the transcript may be garbled")
    all_labels = {lb["label"] for lb in labels}
    by_contact: dict[int, set[str]] = {}
    for lb in labels:
        p = proposals.get(lb["label"]) or {}
        if p.get("kind") == "crm" and p.get("contact_id") is not None:
            by_contact.setdefault(int(p["contact_id"]), set()).add(lb["label"])
    self_ok = False
    for lb in labels:
        label, share, lines = lb["label"], float(lb["share"]), int(lb["lines"])
        p = proposals.get(label) or {"kind": "unknown", "confidence": "low"}
        kind, conf = p.get("kind"), p.get("confidence")
        minor = share <= rule["minor_max_share"] and lines <= rule["minor_max_lines"]
        pct = f"{round(share * 100)}%"
        ev: list[str] = []
        verdict: Optional[dict] = None
        blocker = None
        if kind == "nonparticipant":
            if share <= rule["nonparticipant_max_share"] and (conf == "high" or minor):
                verdict = {"label": label, "kind": "nonparticipant"}
                ev.append(f"not part of the meeting ({conf}): {p.get('evidence') or ''}".strip())
            else:
                blocker = f"{label!r} ({pct}) suggested as not in the meeting at {conf} confidence"
        elif conf != "high" or kind not in ("self", "crm"):
            blocker = f"{label!r} ({pct}) is {kind} at {conf} confidence"
        elif p.get("voice_conflict"):
            blocker = f"{label!r} ({pct}): voice and transcript disagree"
        elif kind == "self":
            support, contra, note = self_evidence(label, self_label=bool(lb.get("self_label")),
                                                  voice_log=voice_log, system_track=system_track, rule=rule)
            ev = support + [f"note: {n}" for n in note if n not in contra]
            if contra:
                blocker = f"{label!r} ({pct}) as Matthias is contradicted: {'; '.join(contra)}"
            elif not support:
                blocker = f"{label!r} ({pct}) as Matthias has no host-channel or voice evidence"
            else:
                verdict = {"label": label, "kind": "self"}
        else:
            cid = int(p["contact_id"]) if p.get("contact_id") is not None else None
            if cid is None:
                blocker = f"{label!r} ({pct}) has no CRM contact id"
            else:
                contact = {"id": cid, "name": p.get("name"), "company": p.get("company"), "email": p.get("email")}
                support, contra = crm_evidence(
                    contact, labels_for_contact=by_contact.get(cid, {label}), all_labels=all_labels,
                    transcript=transcript, voice_log=voice_log, system_track=system_track,
                    attendee_emails=set(attendee_emails or ()), calendar_authoritative=calendar_authoritative,
                    involved_companies=list(involved_companies or []), involved_people=list(involved_people or []))
                ev = support
                if contra:
                    blocker = f"{label!r} ({pct}) as {p.get('name')} is contradicted: {'; '.join(contra)}"
                elif not support:
                    blocker = f"{label!r} ({pct}) as {p.get('name')} has no evidence beyond the suggestion"
                else:
                    verdict = {"label": label, "kind": "crm", "contact_id": cid, "name": p.get("name"),
                               "company": p.get("company"), "email": p.get("email"), "position": p.get("position")}
        if blocker and minor:
            ev.append(f"minor ({lines} lines, {pct}): {blocker}; left as unidentified")
            verdict, blocker = {"label": label, "kind": "unknown"}, None
        if blocker:
            reasons.append(blocker)
        if verdict:
            mapping.append(verdict)
            if verdict["kind"] == "self" and not minor:
                self_ok = True
        evidence[label] = ev
    if rule["require_self"] and not self_ok:
        reasons.append("no substantial label is verified as Matthias")
    out["auto"] = not reasons and len(mapping) == len(labels)
    if not out["auto"]:
        out["mapping"] = []
    return out


# ── I/O ──────────────────────────────────────────────────────────────────────

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass


def _conn():
    from neon_insert import _get_conn
    return _get_conn()


def fetch_source(source_id: int) -> dict:
    conn = _conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """SELECT title, company, metadata, participant_details, participant_resolution_log,
                          (SELECT count(*) FROM insights i WHERE i.source_id = s.id
                             AND coalesce(i.lifecycle_status, 'active') = 'active')
                   FROM sources s WHERE id = %s""", (source_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise LookupError(f"source id={source_id} not found")
    meta = row[2] or {}
    return {"title": row[0], "company": row[1], "metadata": meta, "participant_details": row[3] or [],
            "participant_resolution_log": row[4] or {}, "insight_count": int(row[5] or 0),
            "transcript_path": meta.get("transcript_path"), "has_decisions": "task_decisions" in meta}


def _read_transcript(path: Optional[str]) -> dict:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"transcript file not found: {path!r}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:   # not a mapping problem: keep it out of exit code 2
        raise RuntimeError(f"transcript is not readable JSON: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"transcript is not a Gemini JSON object: {path}")
    return data


def _write_json_atomic(path: str, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix=".tmp")
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


def _find_person(*args: str) -> list[dict]:
    try:
        out = subprocess.run([sys.executable, str(FIND_PERSON), *args, "--json"], capture_output=True,
                             text=True, timeout=30, check=True).stdout
        return json.loads(out.strip().splitlines()[-1]) if out.strip() else []
    except Exception as exc:  # CRM down must not kill the proposal
        logger.warning("find_person %s failed: %s", args, exc)
        return []


def _set_metadata(source_id: int, key: str, value: dict) -> None:
    conn = _conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE sources SET metadata = coalesce(metadata, '{}'::jsonb) || jsonb_build_object(%s, %s::jsonb) WHERE id = %s",
                        (key, json.dumps(value, ensure_ascii=False), source_id))
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_runner() -> Callable[[str], str]:
    return claude_cli_runner(claude_path=str(HEADLESS_CLAUDE),
                             config_dir=os.environ.get("CLAUDE_CONFIG_DIR") or str(AUTOMATION_CONFIG_DIR),
                             timeout=600, model="sonnet")


# ── Commands ─────────────────────────────────────────────────────────────────

def gather_context(src: dict, gem: dict, labels: list[dict],
                   find: Callable[..., list[dict]] = None) -> dict:
    """Everything propose and the auto rule look up: calendar attendees resolved
    to CRM contacts by e-mail, the companies involved (attendee mail domains,
    calendar tags, client names in any spelling) and their contacts, and the
    CRM candidates per label (first-name matches at those companies first)."""
    find = find or _find_person
    transcript = gem.get("transcript") or ""
    calendar = ((gem.get("_meta") or {}).get("speaker_attribution") or {}).get("calendar_search") \
        or (src.get("participant_resolution_log") or {}).get("calendar_search") or {}
    attendees, authoritative = calendar_attendees(calendar)
    emails = sorted({a["email"] for a in attendees if a.get("email")})
    attendee_contacts = find("--email", ",".join(emails)) if emails else []
    domain_companies = []
    for domain in attendee_domains(attendees)[:3]:
        rows = find("--domain", domain, "--limit", "60")
        counts: dict[str, int] = {}
        for r in rows:
            if r.get("company"):
                counts[r["company"]] = counts.get(r["company"], 0) + 1
        if counts:
            domain_companies.append(max(counts, key=lambda c: (counts[c], -len(c))))
    clients = [d.name for d in CLIENTS_DIR.iterdir() if d.is_dir()] if CLIENTS_DIR.is_dir() else []
    involved = companies_to_check(title=src.get("title") or "", source_company=src.get("company"), calendar=calendar,
                                  transcript=transcript, client_names=clients, domain_companies=domain_companies,
                                  limit=5)
    company_people = {c: find("--company", c, "--limit", "60", "--threshold", "0.85") for c in involved}
    company_people = {k: v for k, v in company_people.items() if v}
    involved_people = list({p["id"]: p for ppl in company_people.values() for p in ppl}.values())
    pool = list({p["id"]: p for p in attendee_contacts + involved_people}.values())
    candidates = {}
    for lb in labels:
        if lb["self_label"]:
            continue
        rows = first_name_matches(lb["label"], pool)
        seen = {r["id"] for r in rows}
        rows += [r for r in find(lb["label"], "--limit", "5", "--threshold", "0.7") if r["id"] not in seen]
        candidates[lb["label"]] = rows[:8]
    return {"calendar": calendar, "attendees": attendees, "calendar_authoritative": authoritative,
            "attendee_contacts": attendee_contacts, "attendee_emails": set(emails),
            "involved_companies": involved, "company_people": company_people,
            "involved_people": involved_people, "candidates": candidates}


def _gate_policy(src: dict, gem: dict) -> str:
    """The policy the auto rule checks. "hold" only when nothing else owns the
    meeting yet: no pipeline status (handed off, cancelled, no-show), no ticket
    decisions, and both the stored and the transcript's gate say hold (they
    disagree after a /meeting-actions calendar bind)."""
    meta = src.get("metadata") or {}
    if meta.get("pipeline_status"):
        return f"already {meta['pipeline_status']}"
    if "task_decisions" in meta:
        return "already processed"
    from attribution_gate import gate
    stored = (meta.get("speaker_attribution") or {}).get("speaker_dependent_actions") or "hold"
    live = gate(gem)["speaker_dependent_actions"]
    return live if stored == "hold" else stored


def decide(src: dict, gem: dict, labels: list[dict], proposals: dict[str, dict], ctx: dict,
           error: Optional[str] = None) -> dict:
    meta = gem.get("_meta") or {}
    return auto_confirm_decision(
        labels, proposals, transcript=gem.get("transcript") or "", voice_log=meta.get("voice_match"),
        system_track=meta.get("system_track_attribution"), attendee_emails=ctx["attendee_emails"],
        calendar_authoritative=ctx["calendar_authoritative"], involved_companies=ctx["involved_companies"],
        involved_people=ctx["involved_people"] + ctx["attendee_contacts"], gate_policy=_gate_policy(src, gem),
        review_error=error)


def _auto_summary(decision: dict) -> dict:
    return {"rule": decision["rule"], "decision": "auto_confirmed" if decision["auto"] else "ask",
            "reasons": decision["reasons"], "evidence": decision["evidence"], "checked_at": _now()}


def _auto_enabled() -> bool:
    cfg = (_pipeline_config().get("speaker_review") or {})
    return bool(cfg.get("auto_confirm", True))


def propose(source_id: int, *, runner: Optional[Callable[[str], str]] = None, dry_run: bool = False,
            force: bool = False, auto: Optional[bool] = None) -> dict:
    src = fetch_source(source_id)
    existing = (src["metadata"] or {}).get("speaker_review") or {}
    if existing.get("status") == "confirmed" and not force:
        return {"ok": True, "source_id": source_id, "skipped": "already confirmed"}
    gem = _read_transcript(src["transcript_path"])
    transcript = gem.get("transcript") or ""
    labels = label_stats(transcript)
    if not labels:
        review = {"status": "no_speakers", "proposed_at": _now(), "labels": []}
        if not dry_run:
            _set_metadata(source_id, "speaker_review", review)
        return {"ok": True, "source_id": source_id, "review": review}

    ctx = gather_context(src, gem, labels)
    candidates, company_people = ctx["candidates"], ctx["company_people"]
    known = {c["id"]: c for group in [*candidates.values(), *company_people.values(), ctx["attendee_contacts"]]
             for c in group}

    prompt = build_propose_prompt(title=src["title"], calendar=ctx["calendar"], labels=labels, candidates=candidates,
                                  company_people=company_people, transcript=transcript,
                                  attendee_contacts=ctx["attendee_contacts"])
    error = None
    try:
        proposals = parse_proposal((runner or _default_runner())(prompt), [lb["label"] for lb in labels], known)
    except Exception as exc:
        error = str(exc)[:300]
        logger.warning("proposal failed for source %s: %s", source_id, error)
        proposals = {lb["label"]: ({"kind": "self", "name": SELF_NAME, "company": "Lailix", "confidence": "medium",
                                    "evidence": "Label looks like Matthias."} if lb["self_label"]
                                   else {"kind": "unknown", "confidence": "low", "evidence": "No suggestion (model call failed)."})
                     for lb in labels}

    # Voice fingerprint evidence (local; names and scores only) from the watcher.
    voice_extra = apply_voice_evidence(proposals, labels, (gem.get("_meta") or {}).get("voice_match"))
    # Never store an existing CRM contact as "new" (source 1583).
    crm_extra = resolve_new_to_crm(proposals, lookup=lambda n: _find_person(n, "--limit", "5", "--threshold", "0.85"),
                                   attendee_contacts=ctx["attendee_contacts"],
                                   involved_companies=ctx["involved_companies"], company_people=company_people)

    def _cands(label: str) -> list[dict]:
        rows, seen = [], set()
        for c in [*crm_extra.get(label, []), *candidates.get(label, []), *voice_extra.get(label, [])]:
            if c.get("id") in seen:
                continue
            seen.add(c.get("id"))
            row = {k: c.get(k) for k in ("id", "name", "company", "position", "email")}
            if "voice_score" in c:
                row["voice_score"] = c["voice_score"]
            rows.append(row)
        return rows

    decision = decide(src, gem, labels, proposals, ctx, error)
    review = {
        "status": "proposed", "proposed_at": _now(), "model": "sonnet", "error": error, "kinds": list(KINDS),
        "labels": [dict(lb, proposal=proposals[lb["label"]], candidates=_cands(lb["label"]))
                   for lb in labels],
        "auto": _auto_summary(decision),
    }
    result = {"ok": True, "source_id": source_id, "review": review}
    if dry_run:
        return result
    _set_metadata(source_id, "speaker_review", review)
    if decision["auto"] and (auto if auto is not None else _auto_enabled()):
        try:
            result["auto_confirm"] = confirm(source_id, decision["mapping"], auto=decision)
        except Exception as exc:   # the proposal stands; Matthias confirms by hand
            logger.exception("auto-confirm failed for source %s", source_id)
            review["auto"] = dict(review["auto"], decision="ask", reasons=[f"auto-confirm failed: {exc}"[:300]])
            _set_metadata(source_id, "speaker_review", review)
            result["auto_confirm"] = {"ok": False, "error": str(exc)[:300]}
    return result


def auto_check(source_id: int) -> dict:
    """Dry-run the auto rule on the STORED proposal: no model call, no writes."""
    src = fetch_source(source_id)
    review = (src["metadata"] or {}).get("speaker_review") or {}
    if review.get("status") != "proposed":
        return {"source_id": source_id, "title": src["title"], "decision": "skip",
                "reasons": [f"review status {review.get('status')!r}"]}
    gem = _read_transcript(src["transcript_path"])
    labels = label_stats(gem.get("transcript") or "")
    stored = {lb["label"]: json.loads(json.dumps(lb.get("proposal") or {})) for lb in review.get("labels") or []}
    proposals = {lb["label"]: stored.get(lb["label"]) or {"kind": "unknown", "confidence": "low"} for lb in labels}
    ctx = gather_context(src, gem, labels)
    resolve_new_to_crm(proposals, lookup=lambda n: _find_person(n, "--limit", "5", "--threshold", "0.85"),
                       attendee_contacts=ctx["attendee_contacts"], involved_companies=ctx["involved_companies"],
                       company_people=ctx["company_people"])
    d = decide(src, gem, labels, proposals, ctx, review.get("error"))
    return {"source_id": source_id, "title": src["title"], "decision": "auto_confirm" if d["auto"] else "ask",
            "reasons": d["reasons"], "evidence": d["evidence"],
            "mapping": [{k: v for k, v in e.items() if k in ("label", "kind", "name", "contact_id")}
                        for e in d["mapping"]],
            "labels": [{"label": lb["label"], "share": lb["share"], "lines": lb["lines"],
                        "kind": proposals[lb["label"]].get("kind"), "name": proposals[lb["label"]].get("name"),
                        "confidence": proposals[lb["label"]].get("confidence")} for lb in labels]}


def proposed_source_ids() -> list[int]:
    conn = _conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM sources WHERE metadata->'speaker_review'->>'status' = 'proposed' ORDER BY id")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _flag_insights(source_id: int) -> str:
    proc = subprocess.run([sys.executable, str(FLAG_REEXTRACTION), "--source-id", str(source_id)],
                          capture_output=True, text=True, timeout=60, cwd=str(BRAIN_REPO))
    if proc.returncode != 0:
        raise RuntimeError(f"flag_reextraction failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


FOLLOWUP_ONLY_SUFFIX = """
SPEAKERS CONFIRMED BY MATTHIAS (Brain Portal). speaker_attribution.status = human_confirmed:
the speaker labels in content_text and the JSON are authoritative. Skip the Step 1b writeback.
This source ALREADY HAS insights from a content-only pass; they are flagged for re-extraction
under the corrected names. Do NOT insert insights again in Step 2 (only update title/summary if
they are still the filename stem). Run Steps 3-7 normally (ticket decisions, follow-up draft,
ClientContext, CRM, wiki). If metadata.task_decisions was recorded with attribution_held, run
Step 5 again WITHOUT --attribution-held: applied decisions are skipped, held ones now apply.
"""

FULL_SUFFIX = """
SPEAKERS CONFIRMED BY MATTHIAS (Brain Portal). speaker_attribution.status = human_confirmed:
the speaker labels in content_text and the JSON are authoritative. Skip the Step 1b writeback.
"""

AUTO_NOTE = """
SPEAKERS AUTO-CONFIRMED (speaker_review rule {rule}): every label with a real share of the talk is
Matthias or a CRM contact at high confidence with independent evidence (host channel, voice, calendar
e-mail or the name said in the meeting). speaker_attribution.status = auto_confirmed. Treat the labels
like a human confirmation: they are authoritative. Lines not part of the meeting (child, bystander)
were removed before this run; do not look for them.
"""


def handoff_suffix(plan: str, auto: Optional[dict]) -> str:
    base = FOLLOWUP_ONLY_SUFFIX if plan == "followup_only" else FULL_SUFFIX
    if not auto:
        return base
    return AUTO_NOTE.format(rule=auto.get("rule") or AUTO_CONFIRM["rule"]) + base.replace(
        "SPEAKERS CONFIRMED BY MATTHIAS (Brain Portal). speaker_attribution.status = human_confirmed:\n",
        "").replace("SPEAKERS CONFIRMED BY MATTHIAS (Brain Portal). speaker_attribution.status = human_confirmed:", "")


def _pipeline_config() -> dict:
    try:
        import yaml
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


def hand_off(source_id: int, transcript_path: str, suffix: str) -> dict:
    """Same hand-off as the watcher: mark ready_for_followup (the reconciler
    and the follow-up job watch that), then start Claude in watcher mode."""
    from neon_insert import mark_ready_for_followup
    cfg = _pipeline_config()
    mode = ((cfg.get("followup") or {}).get("mode")) or "watcher"
    mark_ready_for_followup(source_id, transcript_path, triggered_by=f"speaker_review:{mode}", prompt_suffix=suffix)
    if mode != "watcher" or not (cfg.get("claude_trigger") or {}).get("enabled", True):
        return {"mode": mode, "started": False}
    return dict(trigger_meeting_actions(source_id, transcript_path, suffix), mode=mode, started=True)


def trigger_meeting_actions(source_id: int, transcript_path: str, suffix: str) -> dict:
    prompt = (f"Read .claude/commands/meeting-actions.md and process transcript: "
              f"{transcript_path} --source-id {source_id}\n{suffix}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"claude-speaker-review-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    env = os.environ.copy()
    env["PATH"] = TRIGGER_PATH
    env.setdefault("CLAUDE_CONFIG_DIR", str(AUTOMATION_CONFIG_DIR))
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(HEADLESS_CLAUDE), "-p", "--model", "sonnet", "--effort", "medium",
             prompt, "--dangerously-skip-permissions"],
            cwd=str(BRAIN_REPO), env=env, stdout=lf, stderr=lf, start_new_session=True)
    return {"pid": proc.pid, "log": str(log_file)}


def enrol_voices(tpath: str, gem: dict, mapping: list[dict], rename: dict[str, str]) -> dict:
    """Enrol each confirmed non-host person's voice from the archived system
    track. Names and counts only; never raises, so it can never fail a confirm.

    The recording is found from the transcript path: `<root>/Transcripts/<stem>.json`
    -> `<root>/CaptureArchive/<stem>/` (then `<root>/Recordings/<stem>.wav`).
    `prune_recordings` deletes both after its retention window; once they are
    gone nothing is enrolled and the report says so (the confirmation itself
    is unaffected). Gated by voice_matching.enabled and .store_others.
    """
    try:
        cfg = (_pipeline_config().get("voice_matching") or {})
        if not cfg.get("enabled", True) or not cfg.get("store_others", True):
            return {"enrolled": [], "skipped": [], "reason": "store_others disabled in config"}
        people = {}
        for e in mapping:
            if e["kind"] in ("crm", "new"):
                who = {"name": e["name"]}
                if e.get("contact_id") is not None:
                    who["crm_contact_id"] = e["contact_id"]
                people[rename[e["label"]]] = who
        if not people:
            return {"enrolled": [], "skipped": [], "reason": "no confirmed non-host person"}
        import voice_match
        from voice_library import VoiceLibrary, default_library_dir
        meta = gem.get("_meta") or {}
        stem = Path(tpath).stem
        return voice_match.enrol_people(
            gem.get("transcript") or "", people, stem=stem, base=Path(tpath).parent.parent,
            duration_sec=meta.get("audio_duration_seconds"),
            library=VoiceLibrary(default_library_dir({"voice_matching": cfg})),
            mic_offset_sec=(meta.get("channel_alignment") or {}).get("lag_seconds") or 0.0,
            participants=[p.get("name") for p in gem.get("participants") or [] if isinstance(p, dict)],
            device=str(cfg.get("device") or ""),
            timeout_seconds=int(cfg.get("timeout_seconds") or voice_match.EMBED_TIMEOUT_SECONDS))
    except Exception as exc:
        logger.warning("voice enrolment failed (confirm unaffected): %s", exc)
        return {"enrolled": [], "skipped": [], "reason": f"{type(exc).__name__}: {exc}"[:200]}


def original_proposal(review: dict) -> Optional[dict]:
    """The model's first answer, kept through every confirmation (measurement)."""
    if review.get("original"):
        return review["original"]
    if review.get("status") != "proposed" or not review.get("labels"):
        return None
    return {"proposed_at": review.get("proposed_at"), "model": review.get("model"), "error": review.get("error"),
            "labels": [{"label": lb["label"], "lines": lb.get("lines"), "share": lb.get("share"),
                        "proposal": lb.get("proposal")} for lb in review["labels"]]}


def _delete_chunks(cur, source_id: int) -> None:
    """Excluded lines were already chunked at seed; drop the chunks so the
    nightly chunk-sweep (--sweep-missing) rebuilds them from the cleaned text."""
    cur.execute("DELETE FROM source_chunks WHERE source_id = %s", (source_id,))


def confirm(source_id: int, mapping_raw: object, *, dry_run: bool = False, trigger: bool = True,
            auto: Optional[dict] = None) -> dict:
    """Apply a speaker mapping. `auto` is the auto_confirm_decision when the
    rule (not Matthias) confirms; it changes only how the result is recorded,
    never what is applied."""
    src = fetch_source(source_id)
    tpath = src["transcript_path"]
    gem = _read_transcript(tpath)
    original_text = Path(tpath).read_text(encoding="utf-8")
    labels = [lb["label"] for lb in label_stats(gem.get("transcript") or "")]
    mapping = validate_mapping(mapping_raw, labels)   # ValueError -> exit 2
    now = _now()
    by = "auto" if auto else "human"
    outside = {e["label"] for e in mapping if e["kind"] == "nonparticipant"}
    excluded = exclude_labels(gem, outside, now=now, by=by)
    rename = target_names(mapping)
    changed = rename_labels(gem, rename)
    meta = gem.setdefault("_meta", {})
    meta["speaker_attribution"] = confirmed_attribution(meta.get("speaker_attribution") or {}, now, auto=auto)
    method = "auto_confirmed" if auto else "human_confirmed"
    details = participant_details_for(mapping, src["participant_details"], method=method)
    plan = handoff_plan(src["metadata"], src["insight_count"])
    previous = dict((src["metadata"] or {}).get("speaker_review") or {})
    history = list(previous.get("history") or [])
    history.append({"at": now, "by": by, "mapping": [dict(e, new_label=rename.get(e["label"])) for e in mapping]})
    review = dict(previous, status="confirmed", confirmed_at=now, confirmed_by=by, handoff=plan,
                  mapping=[dict(e, new_label=rename.get(e["label"])) for e in mapping], history=history,
                  labels=relabelled_review(gem.get("transcript") or "", mapping, rename,
                                           evidence="Auto-confirmed (evidence below)." if auto else "Confirmed by you."))
    review["kinds"] = list(KINDS)
    orig = original_proposal(previous)
    if orig:
        review["original"] = orig
    if outside:
        review["excluded"] = sorted(set(previous.get("excluded") or []) | outside)
    if auto:
        review["auto"] = dict(_auto_summary(auto), decision="auto_confirmed")
    elif isinstance(previous.get("auto"), dict) and previous.get("confirmed_by") == "auto":
        review["auto"] = dict(previous["auto"], overridden_at=now)   # a human corrected the rule
    result = {"ok": True, "source_id": source_id, "lines_relabelled": changed, "lines_excluded": excluded,
              "rename": rename, "participants": [d["name"] for d in details], "handoff": plan,
              "confirmed_by": by, "dry_run": dry_run, "warnings": []}
    if dry_run:
        return result

    # The first confirm keeps the untouched transcript next to it.
    backup = Path(tpath).with_name(Path(tpath).stem + ".pre-speaker-review.json")
    if not backup.exists():
        backup.write_text(original_text, encoding="utf-8")

    prl = dict(src["participant_resolution_log"], speaker_review={"confirmed_at": now, "confirmed_by": by,
                                                                  "rename": rename, "excluded": sorted(outside)})
    if isinstance(prl.get("calendar_search"), dict) and prl["calendar_search"]:
        prl["calendar_search"] = dict(prl["calendar_search"], identity_authoritative=True, ambiguous=False)
    # One unit: the DB row and the JSON change together. The JSON is written
    # inside the open transaction; a failed write rolls the row back, a failed
    # commit puts the original JSON back.
    json_written = False
    conn = _conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE sources SET content_text = %s, participants = %s, participant_details = %s::jsonb,
                       participant_resolution_log = %s::jsonb,
                       metadata = coalesce(metadata, '{}'::jsonb)
                                  || jsonb_build_object('speaker_attribution', %s::jsonb, 'speaker_review', %s::jsonb)
                   WHERE id = %s""",
                (gem.get("transcript") or "", [d["name"] for d in details],
                 json.dumps(details, ensure_ascii=False), json.dumps(prl, ensure_ascii=False),
                 json.dumps(meta["speaker_attribution"], ensure_ascii=False),
                 json.dumps(review, ensure_ascii=False), source_id))
            if excluded:
                _delete_chunks(cur, source_id)
            _write_json_atomic(tpath, gem)
            json_written = True
    except BaseException:
        if json_written:
            Path(tpath).write_text(original_text, encoding="utf-8")
        raise
    finally:
        conn.close()

    # The relabel is done; what follows is reported, never undone.
    if auto:
        # Voice profiles grow only from human answers: enrolling the rule's own
        # output would let one wrong call reinforce itself.
        result["voice_enrolment"] = {"enrolled": [], "skipped": [], "reason": "auto-confirmed: not enrolled"}
    else:
        try:
            result["voice_enrolment"] = enrol_voices(tpath, gem, mapping, rename)
        except Exception as exc:   # enrol_voices already guards itself; this is the second shield
            result["warnings"].append(f"voice enrolment failed: {exc}")
    if src["insight_count"]:
        try:
            result["flagged"] = _flag_insights(source_id)
        except Exception as exc:
            result["warnings"].append(f"insights not flagged for re-extraction: {exc}")
    if trigger and plan != "corrected":
        try:
            result["meeting_actions"] = hand_off(source_id, tpath, handoff_suffix(plan, auto))
        except Exception as exc:
            result["warnings"].append(f"/meeting-actions not started: {exc}")
    return result


# ── Measurement ──────────────────────────────────────────────────────────────

def same_answer(proposal: Optional[dict], answer: Optional[dict]) -> bool:
    """Did the suggestion name the person the final answer names?"""
    p, a = proposal or {}, answer or {}
    if p.get("kind") != a.get("kind"):
        # A suggested new person who turned out to be the CRM contact of that name still names them right.
        return {p.get("kind"), a.get("kind")} == {"new", "crm"} and same_name(p.get("name"), a.get("name"))
    if a.get("kind") == "crm":
        return p.get("contact_id") is not None and int(p["contact_id"]) == int(a.get("contact_id") or -1)
    if a.get("kind") == "new":
        return same_name(p.get("name"), a.get("name"))
    return True


def final_answers(history: list[dict]) -> dict[str, dict]:
    """original label -> the last answer for that person, following relabels
    (an edit after a confirmation answers the renamed label)."""
    if not history:
        return {}
    answers = {e["label"]: e for e in history[0].get("mapping") or []}
    for step in history[1:]:
        by_label = {e["label"]: e for e in step.get("mapping") or []}
        for orig, ans in list(answers.items()):
            nxt = by_label.get(ans.get("new_label") or "")
            if nxt:
                answers[orig] = nxt
    return answers


def agreement_rows(review: dict) -> list[dict]:
    """One row per original label: suggestion vs first and final answer."""
    orig = review.get("original") or {}
    history = review.get("history") or []
    if not orig or not history:
        return []
    first = {e["label"]: e for e in history[0].get("mapping") or []}
    final = final_answers(history)
    rows = []
    for lb in orig.get("labels") or []:
        prop = lb.get("proposal") or {}
        if lb["label"] not in first:
            continue
        rows.append({"label": lb["label"], "share": lb.get("share"), "confidence": prop.get("confidence") or "low",
                     "suggested": prop.get("kind"), "mode": history[0].get("by") or "human",
                     "agree": same_answer(prop, final.get(lb["label"])),
                     "corrected_later": len(history) > 1 and not same_answer(first[lb["label"]], final.get(lb["label"]))})
    return rows


def _rows_from_log(source_id: int, review: dict) -> list[dict]:
    """Legacy confirmations (before `original` was kept): the last proposal in
    the per-source log against the stored mapping."""
    log = LOG_DIR / f"speaker-review-{source_id}.log"
    if not log.exists() or not review.get("mapping"):
        return []
    last = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith('{"ok": true') and '"status": "proposed"' in line:
            try:
                last = json.loads(line).get("review")
            except ValueError:
                continue
    if not last:
        return []
    fake = {"original": original_proposal(last), "history": [{"by": "human", "mapping": review["mapping"]}]}
    return agreement_rows(fake)


def agreement_report(rows: list[dict]) -> dict:
    """Agreement split by confidence and by auto versus human confirmation."""
    def bucket(sel):
        n = len(sel)
        ok = sum(1 for r in sel if r["agree"])
        return {"labels": n, "agree": ok, "rate": round(ok / n, 3) if n else None}
    out = {"all": bucket(rows), "by_confidence": {}, "by_mode": {}, "substantial_only": bucket(
        [r for r in rows if float(r.get("share") or 0) > AUTO_CONFIRM["minor_max_share"]])}
    for c in CONFIDENCES:
        out["by_confidence"][c] = bucket([r for r in rows if r["confidence"] == c])
    for m in ("auto", "human"):
        out["by_mode"][m] = bucket([r for r in rows if r["mode"] == m])
    out["auto_corrected_later"] = sum(1 for r in rows if r["mode"] == "auto" and r["corrected_later"])
    return out


def agreement(days: int = 90, include_logs: bool = True) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    conn = _conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""SELECT id, metadata->'speaker_review' FROM sources
                           WHERE metadata->'speaker_review'->>'status' = 'confirmed' AND created_at >= %s
                           ORDER BY id""", (since,))
            found = cur.fetchall()
    finally:
        conn.close()
    rows, sources = [], {"kept_original": 0, "from_logs": 0, "no_record": 0}
    for sid, review in found:
        r = agreement_rows(review or {})
        if r:
            sources["kept_original"] += 1
        elif include_logs:
            r = _rows_from_log(sid, review or {})
            sources["from_logs" if r else "no_record"] += 1
        else:
            sources["no_record"] += 1
        rows.extend(dict(x, source_id=sid) for x in r)
    misses = [{k: x[k] for k in ("source_id", "label", "confidence", "suggested", "mode")}
              for x in rows if not x["agree"]]
    return {"ok": True, "days": days, "sources": sources, "report": agreement_report(rows), "misses": misses}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--source-id", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-propose even after a confirmation")
    c = sub.add_parser("confirm")
    c.add_argument("--source-id", type=int, required=True)
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--mapping-json")
    g.add_argument("--mapping-file", type=Path)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--no-trigger", action="store_true", help="relabel only; do not start /meeting-actions")
    p.add_argument("--no-auto", action="store_true", help="store the suggestion only; never auto-confirm")
    a = sub.add_parser("auto-check", help="dry-run the auto-confirm rule on stored proposals (no writes)")
    ga = a.add_mutually_exclusive_group(required=True)
    ga.add_argument("--source-id", type=int)
    ga.add_argument("--all-proposed", action="store_true")
    m = sub.add_parser("agreement", help="how often suggestions matched the confirmed answer")
    m.add_argument("--days", type=int, default=90)
    m.add_argument("--no-logs", action="store_true", help="skip reconstructing legacy confirmations from logs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    _load_env()
    try:
        if args.cmd == "propose":
            out = propose(args.source_id, dry_run=args.dry_run, force=args.force,
                          auto=False if args.no_auto else None)
        elif args.cmd == "auto-check":
            ids = proposed_source_ids() if args.all_proposed else [args.source_id]
            results = []
            for sid in ids:
                try:
                    results.append(auto_check(sid))
                except Exception as exc:
                    results.append({"source_id": sid, "decision": "error", "reasons": [str(exc)[:200]]})
            for r in results:
                print(f"#{r['source_id']} {r['decision'].upper():12} {(r.get('title') or '')[:60]}", file=sys.stderr)
                for why in r.get("reasons") or []:
                    print(f"      - {why}", file=sys.stderr)
            out = {"ok": True, "checked": len(results),
                   "auto_confirm": [r["source_id"] for r in results if r["decision"] == "auto_confirm"],
                   "ask": [r["source_id"] for r in results if r["decision"] == "ask"], "results": results}
        elif args.cmd == "agreement":
            out = agreement(args.days, include_logs=not args.no_logs)
        else:
            raw = json.loads(args.mapping_json if args.mapping_json is not None else args.mapping_file.read_text())
            out = confirm(args.source_id, raw, dry_run=args.dry_run, trigger=not args.no_trigger)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    except Exception as exc:
        logger.exception("speaker_review %s failed", args.cmd)
        print(json.dumps({"ok": False, "error": str(exc)[:500]}, ensure_ascii=False))
        return 1
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
