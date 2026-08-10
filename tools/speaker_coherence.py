#!/usr/bin/env python3
"""speaker_coherence — semantic speaker-attribution repair before extraction.

The acoustic stages can only ask "whose microphone was hot". When that signal
is unavailable or inadmissible (`channel_vad.host_bleed_rate` — most of our
recordings, see docs/SPEC-speaker-attribution-2026-08-07.md) nothing downstream
checks whether the labels make *sense*, and Gemini's own diarization errors
reach InsightBase unchallenged: on source 767 the answer to a question Matthias
asked was attributed to Matthias himself.

That class of error is trivially visible in the text. A transcript is
internally coherent only if:

  - the person who answers a question is not the person who asked it;
  - "we / our / us" about a company is spoken by that company's employee;
  - someone describing their own role, product or biography is that person;
  - someone addressed by name ("Was meinsch, Philipp?") is not the speaker;
  - a speaker does not refer to themselves in the third person.

Claude already reads the whole transcript downstream, so it can apply those
rules before any insight is written. This module runs that check, rewrites the
labels it is confident about, and records everything it changed and why.

Design constraints, all of them learned from prior repairs that went wrong:

  - **Never invent a name.** Rewrites must target a label already present in
    the transcript, the host, or a name from the calendar attendee list.
    (`apply_speaker_resolution`'s `singleton_collapse` merged a third speaker
    into the client counterpart on source 434 by rewriting freely.)
  - **Never trust a stale index.** Each proposed relabel names the line's
    CURRENT label; if it doesn't match, the proposal is discarded rather than
    applied to whatever happens to be at that offset.
  - **Never rewrite the whole meeting.** More than `MAX_RELABEL_FRACTION` of
    lines proposed for relabelling means the model re-diarized from scratch
    instead of repairing — that is refused wholesale, not applied.
  - **Only high confidence edits.** Medium/low proposals are recorded as
    `uncertain_regions` and the text is left alone, so an unresolvable passage
    is visibly flagged rather than silently guessed (the failure mode the
    speaker pipeline has repeatedly produced).
  - **Degrade, never block.** Any failure leaves the transcript untouched and
    reports `ok: False`; transcription must not be lost to a repair stage.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Callable, Optional

logger = logging.getLogger(__name__)

SELF_NAME = "Matthias Heim"
SELF_FIRST = "Matthias"

# A model that wants to relabel more than this share of lines is not repairing
# an attribution, it is replacing it — refuse and keep the original.
MAX_RELABEL_FRACTION = 0.35
MIN_LINES = 6                # below this there is nothing coherent to check

# Audit latency scales badly with transcript length: a 30-line window returns
# in ~4 min, the full 119-line source-767 transcript did not return within 15
# (measured 2026-08-07). So the audit runs over WINDOW_LINES-sized windows,
# each preceded by CONTEXT_LINES of read-only lead-in for conversational
# context. A window may only relabel lines in its OWN range, which makes
# overlapping windows conflict-free by construction rather than by tie-break.
WINDOW_LINES = 30
CONTEXT_LINES = 8
# Concurrent audits. Each one is a headless Claude session, so this is also a
# quota-burst cap: the 2026-07-10 incident was fire-and-forget sessions
# starving on a shared pool (tools/CLAUDE_QUOTA_ISOLATION.md). Keep it small,
# run the gate in its own CLAUDE_CONFIG_DIR, and lower it via
# `speaker_coherence.max_parallel` if audits start hitting session limits.
MAX_PARALLEL_WINDOWS = 3
MAX_LINE_CHARS = 600         # per-line truncation in the prompt
DEFAULT_TIMEOUT_SEC = 900    # per window

# `[MM:SS] Label: text` / `[HH:MM:SS] Label: text`
_LINE_RE = re.compile(r"^(\s*\[\d{1,2}(?::\d{2}){1,2}\]\s*)([^:\n]{1,60}?):(.*)$")


def _first_tok(s: str) -> str:
    toks = [t for t in re.split(r"\s+", (s or "").strip().lower()) if t]
    return toks[0] if toks else ""


def parse_lines(transcript: str) -> list[dict]:
    """Split a transcript into labelled lines, keeping everything else verbatim.

    Returns one entry per `[MM:SS] Label: text` line:
    {index, raw, prefix, label, rest, ts}. Lines that don't match (blank lines,
    continuation text) are not returned and are never touched.
    """
    out: list[dict] = []
    for i, raw in enumerate(transcript.split("\n")):
        m = _LINE_RE.match(raw)
        if not m:
            continue
        ts = re.search(r"\[(\d{1,2}(?::\d{2}){1,2})\]", m.group(1))
        out.append({
            "index": i,
            "raw": raw,
            "prefix": m.group(1),
            "label": m.group(2).strip(),
            "rest": m.group(3),
            "ts": ts.group(1) if ts else "",
        })
    return out


def allowed_labels(lines: list[dict], known_attendees: Optional[list[dict]],
                   participants: Optional[list[dict]]) -> set[str]:
    """The closed universe a relabel may target.

    Existing transcript labels + the host + calendar attendees + participants.
    Anything outside this set is a hallucinated name and is rejected.
    """
    allowed = {ln["label"] for ln in lines}
    allowed |= {SELF_NAME, SELF_FIRST}
    for src in (known_attendees or [], participants or []):
        for p in src:
            if isinstance(p, dict) and (p.get("name") or "").strip():
                name = p["name"].strip()
                allowed.add(name)
                allowed.add(name.split()[0])
    return allowed


def render_numbered(lines: list[dict], max_chars: int = MAX_LINE_CHARS,
                    start: int = 1, editable_from: Optional[int] = None) -> str:
    """Render lines for the prompt with stable 1-based numbers.

    Numbers are GLOBAL (the caller passes `start`) so every window speaks the
    same coordinate system as the merged transcript — a window-local index
    that had to be offset afterwards is exactly the kind of arithmetic that
    silently relabels the wrong line.
    """
    parts = []
    for i, ln in enumerate(lines):
        n = start + i
        text = ln["rest"].strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " …[gekürzt]"
        marker = ""
        if editable_from is not None and n < editable_from:
            marker = " (context only)"
        parts.append(f"{n}\t[{ln['ts']}]\t{ln['label']}{marker}\t{text}")
    return "\n".join(parts)


def plan_windows(n_lines: int, window: int = WINDOW_LINES,
                 context: int = CONTEXT_LINES) -> list[tuple[int, int, int]]:
    """Split `n_lines` into (context_start, editable_start, end) 1-based spans.

    Windows tile the transcript without overlap in their EDITABLE range; the
    context lead-in overlaps the previous window and is read-only.
    """
    if n_lines <= 0:
        return []
    out: list[tuple[int, int, int]] = []
    start = 1
    while start <= n_lines:
        end = min(n_lines, start + window - 1)
        # Absorb a runt tail rather than spending a whole call on 2 lines.
        if n_lines - end < max(2, window // 4):
            end = n_lines
        out.append((max(1, start - context), start, end))
        start = end + 1
    return out


def build_prompt(lines: list[dict], *, known_attendees: Optional[list[dict]],
                 allowed: set[str], start: int = 1,
                 editable_from: Optional[int] = None,
                 total_lines: Optional[int] = None) -> str:
    roster = ""
    non_self = [
        p for p in (known_attendees or [])
        if isinstance(p, dict) and (p.get("name") or "").strip()
        and (p.get("role") or "").lower() != "self"
        and (p.get("name") or "").strip().lower() != SELF_NAME.lower()
    ]
    if non_self:
        roster = (
            "\nCalendar attendees for this meeting:\n"
            + f"- {SELF_NAME} (host)\n"
            + "".join(
                f"- {p['name'].strip()}"
                f"{' (' + p['company'].strip() + ')' if (p.get('company') or '').strip() else ''}\n"
                for p in non_self
            )
        )

    scope = ""
    if editable_from is not None and editable_from > start:
        scope = (
            f"\n\nThis is an excerpt of a longer transcript. Lines "
            f"{start}-{editable_from - 1} are marked `(context only)`: read "
            f"them for conversational context, but do NOT propose relabels "
            f"for them — another pass owns those. Only lines {editable_from} "
            f"and later are yours to correct."
        )
    elif total_lines and total_lines > len(lines):
        scope = (
            f"\n\nThis is the first excerpt of a longer "
            f"{total_lines}-line transcript."
        )

    return f"""You are auditing the speaker attribution of a meeting transcript before it is used to write insights. The transcript was diarized by an audio model that cannot reason about meaning, so some lines are attributed to the wrong person. Your job is to find those lines using the CONTENT alone and correct them.

The host is {SELF_NAME} (labelled "{SELF_FIRST}"). The language is Swiss German.{roster}{scope}

## Method

First work out WHO each label is, using the strongest evidence anywhere in the excerpt. Then check every line against that identity.

Do not rely on local turn-taking alone. Two adjacent lines can BOTH be mislabelled — a question and its answer swapped onto each other's speakers still reads as a perfectly ordinary exchange, and that is the most common failure in this pipeline. A line is wrong whenever its CONTENT does not fit the person named, even if its neighbours look consistent.

## Coherence rules

A speaker attribution is wrong when it violates any of these:

1. **Question/answer.** The person who answers a question is not the person who asked it. Watch for a question and its answer on consecutive lines carrying the same label — one of them is misattributed.
2. **Company voice — first person.** Whoever says "we / our / us / mir / öis / öisi / bi üs" about a company is an employee of that company. {SELF_NAME} runs Lailix and is an external consultant; he is NOT an employee of the client company.
3. **Company voice — second person.** The mirror of rule 2, and just as decisive: whoever addresses a company in the second person ("ihr / euch / euer / bi euch / vo eurer Siite", or a question like "dünd ihr monatlich release?") is NOT part of that company. Someone asking how a company's internal process works is an outsider to it; whoever then explains that process from the inside is the insider. Use this pair to settle swapped question/answer lines.
4. **Self-description.** Whoever describes their own role, team, product, process, release schedule or biography is that person.
5. **Direct address.** Someone addressed by name is not the speaker of that line.
6. **Third person.** A speaker never refers to themselves in the third person.
7. **Continuity.** A single uninterrupted argument usually belongs to one speaker; an abrupt label change mid-argument with no conversational turn is suspect. Conversely, a line that merely opens with agreement ("Ja, genau…") and then continues the PREVIOUS speaker's own argument is often that same speaker carrying on, not the other one agreeing.

## Transcript

Columns are: line number, timestamp, current label, text.

```
{render_numbered(lines, start=start, editable_from=editable_from)}
```

## Output

Return ONLY a JSON object, no prose, no markdown fence:

{{
  "speakers": [
    {{"label": "<current label>", "identity": "<real name or null>", "evidence": "<what in the text identifies them>"}}
  ],
  "relabels": [
    {{"line": <line number>, "from": "<current label, exactly as shown>", "to": "<correct label>", "confidence": "high|medium|low", "reason": "<which rule and the specific textual evidence>"}}
  ],
  "uncertain_regions": [
    {{"from_line": <n>, "to_line": <n>, "reason": "<why the speaker cannot be determined here>"}}
  ]
}}

Rules for your output — these are hard constraints:

- `to` MUST be one of exactly these labels: {", ".join(sorted(repr(a) for a in allowed))}. Never invent a name that is not in that list.
- `from` MUST be the label currently shown for that line. If it doesn't match, your line number is wrong.
- Use `"confidence": "high"` ONLY when the text alone proves the attribution, in a way you could point at. A change of topic, a guess from speaking style, or "it flows better" is NOT high confidence — use "medium" or put the span in `uncertain_regions` instead.
- Only list lines you are actually changing. Do NOT list lines that are already correct.
- If the attribution is coherent throughout, return empty `relabels`. That is a valid and expected answer — do not invent corrections to look useful.
- Prefer flagging an ambiguous passage in `uncertain_regions` over guessing at it.
"""


def parse_response(text: str) -> dict:
    """Extract the JSON object from a model response. Raises ValueError."""
    if not text or not text.strip():
        raise ValueError("empty response")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth, end, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("unterminated JSON object")
    data = json.loads(s[start:end])
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    return data


def _screen(proposals, lines: list[dict], allowed: set[str]) -> tuple[list, list]:
    """Split proposals into applicable relabels and rejected ones (with cause)."""
    applied, rejected = [], []
    seen: set[int] = set()
    for p in proposals or []:
        if not isinstance(p, dict):
            rejected.append({"proposal": p, "cause": "not_an_object"})
            continue
        rec = {
            "line": p.get("line"),
            "from": (p.get("from") or "").strip(),
            "to": (p.get("to") or "").strip(),
            "confidence": (p.get("confidence") or "").strip().lower(),
            "reason": (p.get("reason") or "").strip(),
        }
        n = rec["line"]
        if not isinstance(n, int) or not (1 <= n <= len(lines)):
            rejected.append({**rec, "cause": "line_out_of_range"})
            continue
        ln = lines[n - 1]
        rec["ts"] = ln["ts"]
        if rec["confidence"] != "high":
            rejected.append({**rec, "cause": "not_high_confidence"})
            continue
        # Stale-index guard: the model must be looking at the line we are.
        if rec["from"] != ln["label"]:
            rejected.append({**rec, "cause": "from_label_mismatch",
                             "actual_label": ln["label"]})
            continue
        if not rec["to"]:
            rejected.append({**rec, "cause": "empty_target"})
            continue
        if rec["to"] not in allowed:
            rejected.append({**rec, "cause": "target_not_allowed"})
            continue
        if rec["to"] == ln["label"]:
            rejected.append({**rec, "cause": "no_op"})
            continue
        if n in seen:
            rejected.append({**rec, "cause": "duplicate_line"})
            continue
        seen.add(n)
        applied.append(rec)
    return applied, rejected


def canonical_label_map(known_attendees: Optional[list[dict]]) -> dict[str, str]:
    """Map every known person's name forms onto ONE transcript label.

    The allowed universe deliberately accepts both "Philipp Baltensperger" and
    "Philipp", so a relabel and an identity binding can legitimately pick
    different forms for the same person — which is how source 767's first
    end-to-end run ended up with `Philipp:` and `Philipp Baltensperger:` as two
    separate speakers in one transcript. Everything downstream counts labels,
    so that split is a real defect, not cosmetics.

    Canonical form is the first name (the convention `speaker_verify` already
    uses). A first name shared by two known people is NOT canonicalised —
    merging two speakers is worse than leaving both full names intact.
    """
    # Dedupe by name: the host is usually in known_attendees too, and counting
    # him twice would make him his own namesake and disable canonicalisation.
    names: list[str] = []
    for n in [SELF_NAME] + [
        (p.get("name") or "").strip() for p in (known_attendees or [])
        if isinstance(p, dict)
    ]:
        if n and n.lower() not in {x.lower() for x in names}:
            names.append(n)
    first_counts: dict[str, int] = {}
    for n in names:
        first_counts[_first_tok(n)] = first_counts.get(_first_tok(n), 0) + 1
    out: dict[str, str] = {}
    for full in names:
        first = full.split()[0]
        if first_counts[first.lower()] > 1:
            continue  # namesakes: keep them distinguishable
        out[full] = first
        out[first] = first
    return out


def _bind_identities(speakers, lines: list[dict], allowed: set[str],
                     known_attendees: Optional[list[dict]]) -> list[dict]:
    """Generic label -> real name, but ONLY for names the calendar confirms.

    Fixes the `participants: []` case (source 767: the counterpart stayed
    "Speaker 2" through the whole pipeline). Restricted to calendar attendees
    so an identity the model inferred from the audio alone can never introduce
    a name nobody verified.
    """
    if not known_attendees:
        return []
    attendee_names = {
        p["name"].strip() for p in known_attendees
        if isinstance(p, dict) and (p.get("name") or "").strip()
    }
    generic = re.compile(r"^speaker\s+[A-Za-z0-9]+$", re.IGNORECASE)
    present = {ln["label"] for ln in lines}
    bindings = []
    for s in speakers or []:
        if not isinstance(s, dict):
            continue
        label = (s.get("label") or "").strip()
        identity = (s.get("identity") or "").strip()
        if not label or not identity or label not in present:
            continue
        if not generic.match(label):
            continue
        match = next(
            (a for a in attendee_names
             if a.lower() == identity.lower()
             or _first_tok(a) == _first_tok(identity)), None
        )
        if not match:
            continue
        bindings.append({"label": label, "to": match.split()[0],
                         "full_name": match,
                         "evidence": (s.get("evidence") or "").strip()})
    return bindings


def claude_cli_runner(claude_path: str = "/Users/Matthias/.local/bin/claude",
                      config_dir: Optional[str] = None,
                      timeout: int = DEFAULT_TIMEOUT_SEC,
                      model: Optional[str] = None) -> Callable[[str], str]:
    """Headless `claude -p` runner.

    Uses the same quota-isolated CLAUDE_CONFIG_DIR as the watcher's
    fire-and-forget sessions (tools/CLAUDE_QUOTA_ISOLATION.md) so a repair
    never competes with interactive usage for one pool.
    """
    def run(prompt: str) -> str:
        env = os.environ.copy()
        env["PATH"] = ("/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:"
                       "/Users/Matthias/.local/bin")
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        # Prompt goes on stdin, not argv: a 60-min transcript is ~60 kB and
        # would crowd ARG_MAX alongside the environment.
        cmd = [claude_path, "-p", "--append-system-prompt",
               "You are auditing speaker attribution in a transcript. Answer "
               "with a single JSON object and nothing else — no preamble, no "
               "explanation outside the JSON. Do not use tools; everything you "
               "need is in the message."]
        if model:
            cmd += ["--model", model]
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=timeout, env=env,
                              cwd=os.path.expanduser("~"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude exited {proc.returncode}: {proc.stderr.strip()[:300]}"
            )
        return proc.stdout
    return run


def check_and_repair(gemini_dict: dict, *,
                     known_attendees: Optional[list[dict]] = None,
                     runner: Optional[Callable[[str], str]] = None,
                     bind_identities: bool = True,
                     max_parallel: int = MAX_PARALLEL_WINDOWS) -> dict:
    """Audit and repair speaker attribution in `gemini_dict` (mutates in place).

    Args:
        gemini_dict: parsed Gemini response — uses/rewrites `transcript`,
            reads `participants`.
        known_attendees: calendar attendee records; enables identity binding
            and widens the allowed label universe.
        runner: callable(prompt) -> response text. Defaults to the headless
            Claude CLI. Injected in tests so no network is required.
        bind_identities: rename generic labels to calendar-confirmed names.

    Returns a forensic log (stored as `speaker_coherence` in the transcript
    JSON and in participant_resolution_log).
    """
    log: dict = {
        "ok": False,
        "ran": False,
        "changed": False,
        "lines_total": 0,
        "relabels_proposed": 0,
        "relabels_applied": [],
        "relabels_rejected": [],
        "identity_bindings": [],
        "uncertain_regions": [],
        "speakers": [],
        "refused_runaway": False,
        "windows": 0,
        "failed_windows": [],
        "error": None,
    }
    transcript = (gemini_dict or {}).get("transcript") or ""
    if not transcript.strip():
        log["error"] = "empty transcript"
        return log

    lines = parse_lines(transcript)
    log["lines_total"] = len(lines)
    if len(lines) < MIN_LINES:
        log["ok"] = True
        log["error"] = f"only {len(lines)} labelled lines — nothing to check"
        return log

    allowed = allowed_labels(lines, known_attendees,
                             (gemini_dict or {}).get("participants"))
    run = runner or claude_cli_runner()
    windows = plan_windows(len(lines))
    log["windows"] = len(windows)

    def audit(win: tuple[int, int, int]) -> dict:
        ctx_start, edit_start, end = win
        prompt = build_prompt(
            lines[ctx_start - 1:end], known_attendees=known_attendees,
            allowed=allowed, start=ctx_start,
            editable_from=edit_start if edit_start > ctx_start else None,
            total_lines=len(lines),
        )
        return parse_response(run(prompt))

    results: list[tuple[tuple[int, int, int], Optional[dict], Optional[str]]] = []
    if len(windows) == 1:
        try:
            results.append((windows[0], audit(windows[0]), None))
        except Exception as e:
            results.append((windows[0], None, f"{type(e).__name__}: {e}"))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
            futures = [(w, pool.submit(audit, w)) for w in windows]
            for w, fut in futures:
                try:
                    results.append((w, fut.result(), None))
                except Exception as e:
                    results.append((w, None, f"{type(e).__name__}: {e}"))

    failures = [(w, err) for w, _d, err in results if err]
    if len(failures) == len(results):
        log["error"] = failures[0][1]
        logger.warning(
            "speaker_coherence: all %d window(s) failed (%s) — transcript left "
            "unchanged; speaker attribution for this meeting is UNVERIFIED",
            len(results), log["error"],
        )
        return log

    log["ran"] = True
    if failures:
        # Partial coverage is not silent success: name the unaudited ranges.
        log["failed_windows"] = [
            {"lines": f"{w[1]}-{w[2]}", "error": err} for w, err in failures
        ]
        log["error"] = (
            f"{len(failures)}/{len(results)} window(s) failed; lines "
            + ", ".join(f"{w[1]}-{w[2]}" for w, _ in failures)
            + " were not audited"
        )
        logger.warning("speaker_coherence: %s", log["error"])

    proposals: list[dict] = []
    seen_speakers: set[tuple] = set()
    for (_ctx, edit_start, end), data, err in results:
        if err or not data:
            continue
        for p in (data.get("relabels") or []):
            # A window may only speak for its own range. Anything else is
            # either the context lead-in (owned by the previous window) or a
            # stray index; both are dropped here rather than tie-broken.
            if isinstance(p, dict) and isinstance(p.get("line"), int) \
                    and edit_start <= p["line"] <= end:
                proposals.append(p)
        for s in (data.get("speakers") or []):
            key = (s.get("label"), s.get("identity")) if isinstance(s, dict) \
                else (str(s),)
            if key not in seen_speakers:
                seen_speakers.add(key)
                log["speakers"].append(s)
        log["uncertain_regions"].extend(data.get("uncertain_regions") or [])

    log["relabels_proposed"] = len(proposals)

    applied, rejected = _screen(proposals, lines, allowed)
    log["relabels_rejected"] = rejected

    # Runaway guard: a repair touches a handful of lines. Re-diarizing the
    # meeting from the text is a different, unvalidated operation.
    if applied and len(applied) > max(1, int(len(lines) * MAX_RELABEL_FRACTION)):
        log["refused_runaway"] = True
        log["ok"] = True
        logger.warning(
            "speaker_coherence: refused %d relabels over %d lines (> %.0f%%) — "
            "that is a re-diarization, not a repair; transcript unchanged",
            len(applied), len(lines), MAX_RELABEL_FRACTION * 100,
        )
        return log

    bindings = (_bind_identities(log["speakers"], lines, allowed,
                                 known_attendees)
                if bind_identities else [])
    log["identity_bindings"] = bindings

    if not applied and not bindings:
        log["ok"] = True
        logger.info(
            "speaker_coherence: attribution coherent over %d lines "
            "(%d proposal(s) rejected, %d uncertain region(s))",
            len(lines), len(rejected), len(log["uncertain_regions"]),
        )
        return log

    # Rewrite. Per-line relabels first (indices refer to pre-rename labels),
    # then the global generic->name binding, then canonicalisation so one
    # person cannot end up under two spellings.
    canonical = canonical_label_map(known_attendees)
    by_index = {a["line"]: a for a in applied}
    out = transcript.split("\n")
    for n, ln in enumerate(lines, start=1):
        new_label = None
        if n in by_index:
            new_label = by_index[n]["to"]
        label_now = new_label or ln["label"]
        for b in bindings:
            if label_now == b["label"]:
                new_label = b["to"]
                break
        if new_label:
            new_label = canonical.get(new_label, new_label)
        if new_label and new_label != ln["label"]:
            out[ln["index"]] = f"{ln['prefix']}{new_label}:{ln['rest']}"
    gemini_dict["transcript"] = "\n".join(out)
    # Keep the log honest about what was actually written.
    for a in applied:
        a["to"] = canonical.get(a["to"], a["to"])

    for b in bindings:
        for p in (gemini_dict.get("participants") or []):
            if isinstance(p, dict) and (p.get("name") or "").strip() == b["label"]:
                p["name"] = b["full_name"]

    log["relabels_applied"] = applied
    log["changed"] = True
    log["ok"] = True
    logger.info(
        "speaker_coherence: repaired %d line(s)%s — %s",
        len(applied),
        f", bound {len(bindings)} identity/identities" if bindings else "",
        "; ".join(f"[{a['ts']}] {a['from']!r}→{a['to']!r} ({a['reason'][:80]})"
                  for a in applied) or "identity binding only",
    )
    return log


def revert_verify_flips(gemini_dict: dict) -> int:
    """Undo `speaker_verify`'s flips using its own decision log. Returns count.

    Needed by the backfill (docs/BACKFILL-speaker-attribution-2026-08-07.md):
    transcripts stored before the admissibility gate carry flips applied
    against a bleeding oracle. The log records `time_span`/`from`/`to` per
    flip, so the revert is exact — and it is skipped for any line whose label
    is no longer the flip's `to`, since something else has since rewritten it.
    """
    log = (gemini_dict or {}).get("speaker_verification") or {}
    flips = log.get("flips") or []
    if not flips or not (gemini_dict or {}).get("transcript"):
        return 0
    by_ts: dict[str, dict] = {}
    for f in flips:
        m = re.match(r"\[(\d{1,2}(?::\d{2}){1,2})\]", f.get("time_span") or "")
        if m:
            by_ts[m.group(1)] = f
    out, n = [], 0
    for raw in gemini_dict["transcript"].split("\n"):
        m = _LINE_RE.match(raw)
        if m:
            ts = re.search(r"\[(\d{1,2}(?::\d{2}){1,2})\]", m.group(1))
            f = by_ts.get(ts.group(1)) if ts else None
            if f and m.group(2).strip() == (f.get("to") or "").strip():
                raw = f"{m.group(1)}{f['from']}:{m.group(3)}"
                n += 1
        out.append(raw)
    gemini_dict["transcript"] = "\n".join(out)
    return n


def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Audit and repair speaker attribution in a transcript JSON."
    )
    ap.add_argument("json_path", help="Transcripts/*.json produced by the watcher")
    ap.add_argument("--attendee", action="append", default=[],
                    metavar="NAME[,COMPANY]",
                    help="non-host attendee; repeatable. Enables identity binding.")
    ap.add_argument("--revert-flips", action="store_true",
                    help="undo speaker_verify's flips first (bleeding-oracle "
                         "transcripts — see the backfill doc)")
    ap.add_argument("--write", action="store_true",
                    help="write the repaired transcript back (default: dry run)")
    ap.add_argument("--out", help="write to this path instead of in place")
    ap.add_argument("--model")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    path = os.path.abspath(args.json_path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    attendees = [{"name": SELF_NAME, "role": "self"}]
    for a in args.attendee:
        name, _, company = a.partition(",")
        attendees.append({"name": name.strip(), "role": "attendee",
                          "company": company.strip()})

    if args.revert_flips:
        n = revert_verify_flips(data)
        print(f"reverted {n} speaker_verify flip(s)")

    log = check_and_repair(data, known_attendees=attendees if args.attendee else None,
                           runner=claude_cli_runner(model=args.model))
    data["speaker_coherence"] = log

    for a in log["relabels_applied"]:
        print(f"  [{a['ts']}] {a['from']!r} -> {a['to']!r}: {a['reason']}")
    for b in log["identity_bindings"]:
        print(f"  identity: {b['label']!r} -> {b['full_name']!r}")
    for u in log["uncertain_regions"]:
        print(f"  uncertain: lines {u.get('from_line')}-{u.get('to_line')}: "
              f"{u.get('reason')}")
    print(f"\nok={log['ok']} changed={log['changed']} "
          f"windows={log['windows']} failed={len(log['failed_windows'])} "
          f"applied={len(log['relabels_applied'])} "
          f"rejected={len(log['relabels_rejected'])}")
    if log.get("error"):
        print(f"ERROR: {log['error']}")

    if args.write or args.out:
        dest = args.out or path
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"written: {dest}")
    else:
        print("(dry run — pass --write or --out to persist)")
    return 0 if log["ok"] and not log.get("error") else 1


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(_main(_sys.argv[1:]))
