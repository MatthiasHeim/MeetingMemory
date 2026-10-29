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
from datetime import datetime, timezone
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

KINDS = ("self", "crm", "new", "unknown")
CONFIDENCES = ("high", "medium", "low")
MAX_TRANSCRIPT_CHARS = 120_000
SAMPLES_PER_LABEL = 4
SAMPLE_CHARS = 220
MAX_NAME_CHARS = 60          # speaker_coherence._LINE_RE reads labels of at most 60 chars
UNKNOWN_PREFIX = "Unidentified participant"
CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"

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


def participant_details_for(mapping: list[dict], existing: Optional[list] = None) -> list[dict]:
    """sources.participant_details after confirmation: one row per person.

    Speakers come from the mapping; an existing entry with the same name lends
    its email. Calendar attendees who never got a label (silent listeners) are
    kept, minus the old transcript labels and generic "Speaker X" rows.
    """
    existing = [p for p in (existing or []) if isinstance(p, dict) and p.get("name")]
    by_name = {p["name"].lower(): p for p in existing}
    for e in mapping:
        if e["kind"] in ("crm", "new") and not e.get("email") and by_name.get(e["name"].lower(), {}).get("email"):
            e["email"] = by_name[e["name"].lower()]["email"]
    out, seen = [], set()
    for e in mapping:
        if e["kind"] == "unknown":
            continue
        key = e["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        row = {"name": e["name"], "email": e.get("email"), "company": e.get("company"),
               "role": "self" if e["kind"] == "self" else "participant",
               "confidence": "high", "resolution_method": "human_confirmed"}
        if e.get("contact_id"):
            row["crm_contact_id"] = e["contact_id"]
        if e.get("position"):
            row["title"] = e["position"]
        out.append(row)
    old_labels = {e["label"].lower() for e in mapping}
    for p in existing:
        name = p["name"].lower()
        if name in seen or name in old_labels or name.startswith("speaker ") or (p.get("role") or "") == "self":
            continue
        seen.add(name)
        out.append(dict(p))
    return out


def confirmed_attribution(previous: dict, now: str) -> dict:
    """The _meta.speaker_attribution block after a human confirmation.

    require_turn_evidence is the gate's "go" policy: /meeting-actions may
    use named quotes and speaker-dependent actions. The calendar bind is
    marked authoritative so a collision cannot re-hold it.
    """
    out = dict(previous or {})
    out.update(status="human_confirmed", identity_basis="human",
               speaker_dependent_actions="require_turn_evidence", missing_stages=[],
               confirmed_at=now, confirmed_by="Matthias (Brain Portal)")
    cs = out.get("calendar_search")
    if isinstance(cs, dict) and cs:
        out["calendar_search"] = dict(cs, identity_authoritative=True, ambiguous=False)
    return out


def relabelled_review(transcript: str, mapping: list[dict], rename: dict[str, str]) -> list[dict]:
    """Review rows for the transcript as it is now, so a later correction in
    the portal edits the current labels. The confirmed answer is the proposal."""
    by_new = {}
    for e in mapping:
        by_new.setdefault(rename[e["label"]], e)
    rows = []
    for lb in label_stats(transcript):
        e = by_new.get(lb["label"])
        prop = {"kind": e["kind"], "confidence": "high", "evidence": "Confirmed by you."} if e else \
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
                         company_people: dict, transcript: str) -> str:
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

## CRM contacts at companies that look involved
{chr(10).join(people_block) or "(none)"}

## Rules
- kind "self" = Matthias Heim. kind "crm" = one of the CRM contacts above (give its contact_id). kind "new" =
  a real person who is not in the CRM (give the name as said in the meeting, and the company if clear).
  kind "unknown" = you cannot tell.
- Use evidence: self-introductions, being addressed by name, roles/teams described, the calendar title,
  the company's contact list. A similar-sounding first name at the right company with a matching role is
  good evidence for a mishearing.
- confidence "high" only with direct evidence (introduction, addressed by name, unique role match).
- Two labels can be the same person. Never invent a contact_id.
- evidence: one short English sentence a human can check.

Return exactly:
{{"speakers": [{{"label": "...", "kind": "self|crm|new|unknown", "contact_id": 123 or null, "name": "Full Name" or null,
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
                       client_names: list[str]) -> list[str]:
    """Companies whose contact list the model should see (max 4)."""
    found: list[str] = []

    def add(name):
        name = (name or "").strip()
        if name and name.lower() not in {f.lower() for f in found} and name.lower() != "lailix":
            found.append(name)

    add(source_company)
    for c in calendar.get("candidates") or []:
        for m in re.findall(r"\[([^\]]{2,40})\]", c.get("title") or ""):
            add(m)
        for a in c.get("attendees") or []:
            add(a.get("company"))
    for m in re.findall(r"\[([^\]]{2,40})\]", title or ""):
        add(m)
    text = (title or "") + "\n" + (transcript or "")[:40_000]
    for client in client_names:
        if len(client) >= 4 and re.search(r"\b" + re.escape(client) + r"\b", text, re.IGNORECASE):
            add(client)
    return found[:4]


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

def propose(source_id: int, *, runner: Optional[Callable[[str], str]] = None, dry_run: bool = False,
            force: bool = False) -> dict:
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

    calendar = ((gem.get("_meta") or {}).get("speaker_attribution") or {}).get("calendar_search") \
        or (src["participant_resolution_log"] or {}).get("calendar_search") or {}
    candidates = {lb["label"]: _find_person(lb["label"], "--limit", "5", "--threshold", "0.7")
                  for lb in labels if not lb["self_label"]}
    clients = [d.name for d in CLIENTS_DIR.iterdir() if d.is_dir()] if CLIENTS_DIR.is_dir() else []
    company_people = {c: _find_person("--company", c, "--limit", "60", "--threshold", "0.85")
                      for c in companies_to_check(title=src["title"], source_company=src["company"],
                                                  calendar=calendar, transcript=transcript, client_names=clients)}
    company_people = {k: v for k, v in company_people.items() if v}
    known = {c["id"]: c for group in list(candidates.values()) + list(company_people.values()) for c in group}

    prompt = build_propose_prompt(title=src["title"], calendar=calendar, labels=labels, candidates=candidates,
                                  company_people=company_people, transcript=transcript)
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

    review = {
        "status": "proposed", "proposed_at": _now(), "model": "sonnet", "error": error,
        "labels": [dict(lb, proposal=proposals[lb["label"]],
                        candidates=[{k: c.get(k) for k in ("id", "name", "company", "position", "email")}
                                    for c in candidates.get(lb["label"], [])])
                   for lb in labels],
    }
    if not dry_run:
        _set_metadata(source_id, "speaker_review", review)
    return {"ok": True, "source_id": source_id, "review": review}


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


def confirm(source_id: int, mapping_raw: object, *, dry_run: bool = False, trigger: bool = True) -> dict:
    src = fetch_source(source_id)
    tpath = src["transcript_path"]
    gem = _read_transcript(tpath)
    original_text = Path(tpath).read_text(encoding="utf-8")
    labels = [lb["label"] for lb in label_stats(gem.get("transcript") or "")]
    mapping = validate_mapping(mapping_raw, labels)   # ValueError -> exit 2
    rename = target_names(mapping)
    changed = rename_labels(gem, rename)
    now = _now()
    meta = gem.setdefault("_meta", {})
    meta["speaker_attribution"] = confirmed_attribution(meta.get("speaker_attribution") or {}, now)
    details = participant_details_for(mapping, src["participant_details"])
    plan = handoff_plan(src["metadata"], src["insight_count"])
    review = dict((src["metadata"] or {}).get("speaker_review") or {}, status="confirmed", confirmed_at=now,
                  mapping=[dict(e, new_label=rename[e["label"]]) for e in mapping], handoff=plan,
                  labels=relabelled_review(gem.get("transcript") or "", mapping, rename))
    result = {"ok": True, "source_id": source_id, "lines_relabelled": changed, "rename": rename,
              "participants": [d["name"] for d in details], "handoff": plan, "dry_run": dry_run, "warnings": []}
    if dry_run:
        return result

    # The first confirm keeps the untouched transcript next to it.
    backup = Path(tpath).with_name(Path(tpath).stem + ".pre-speaker-review.json")
    if not backup.exists():
        backup.write_text(original_text, encoding="utf-8")

    prl = dict(src["participant_resolution_log"], speaker_review={"confirmed_at": now, "rename": rename})
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
            _write_json_atomic(tpath, gem)
            json_written = True
    except BaseException:
        if json_written:
            Path(tpath).write_text(original_text, encoding="utf-8")
        raise
    finally:
        conn.close()

    # The relabel is done; what follows is reported, never undone.
    if src["insight_count"]:
        try:
            result["flagged"] = _flag_insights(source_id)
        except Exception as exc:
            result["warnings"].append(f"insights not flagged for re-extraction: {exc}")
    if trigger and plan != "corrected":
        try:
            result["meeting_actions"] = hand_off(
                source_id, tpath, FOLLOWUP_ONLY_SUFFIX if plan == "followup_only" else FULL_SUFFIX)
        except Exception as exc:
            result["warnings"].append(f"/meeting-actions not started: {exc}")
    return result


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
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    _load_env()
    try:
        if args.cmd == "propose":
            out = propose(args.source_id, dry_run=args.dry_run, force=args.force)
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
