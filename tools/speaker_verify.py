#!/usr/bin/env python3
"""speaker_verify — flip confidently-misattributed transcript turns using the
channel-VAD ground truth from the hybrid recording's separate mic channel.

Gemini diarizes the mono MP3 by voice alone and sometimes flips speakers
mid-meeting (verified 2026-06-11: a host monologue section where the mic
channel carried ~87% of speech energy was attributed mostly to "Speaker B").
`speaker_reconcile` canonicalises NAMES but cannot fix turn-level
misattribution. This module runs AFTER reconcile, parses per-turn timestamps
from the transcript (the prompt now requires a [MM:SS] at every speaker
change), computes host-share per turn from the channel VAD, and flips turns
where physics and label confidently disagree:

  - Turn ≥ 8 s labeled as the host (Matthias) with host-share < 15 %
    → relabel to the meeting's dominant remote speaker.
  - Turn ≥ 8 s labeled non-host with host-share > 85 %
    → relabel to Matthias.

Conservatism, because Gemini timestamps are only ±few-seconds accurate and
~40-50 % of windows carry overlapped ('both') speech:

  - Turns shorter than MIN_TURN_SEC are never touched.
  - Turns whose VAD interval holds < MIN_SPEECH_SEC of detected speech are
    skipped (no evidence).
  - Turns where 'both' overlap dominates (> MAX_BOTH_SHARE of speech time)
    are skipped — the mic channel can't arbitrate overlapped speech.
  - host_share counts 'both' windows toward the host (his mic was active),
    which biases AGAINST flipping in both directions.

Writes a forensic decision log (same pattern as speaker_reconciliation) that
the watcher attaches to participant_resolution_log.speaker_verification.
After flips, participants[].speaking_pct / total_seconds are recomputed from
the (post-flip) turn durations.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

SELF_NAME = "Matthias Heim"
SELF_FIRST = "Matthias"

MIN_TURN_SEC = 8.0          # Gemini timestamps are ±few-seconds; short turns
                            # can't be judged against them
MIN_SPEECH_SEC = 4.0        # need this much VAD-detected speech in the turn
FLIP_TO_REMOTE_MAX_HOST_SHARE = 0.15
FLIP_TO_HOST_MIN_HOST_SHARE = 0.85
MAX_BOTH_SHARE = 0.40       # overlap-dominated turns are not arbitrated


def _is_self_label(name: str) -> bool:
    toks = [t for t in re.split(r"\s+", (name or "").strip().lower()) if t]
    if not toks:
        return False
    return (toks[0] == SELF_FIRST.lower()
            or (name or "").strip().lower() == SELF_NAME.lower()
            or toks == ["host"])


# Candidate-label shape rules (matching speaker_reconcile's harvest scan): a
# label is `Name:` at line start / after whitespace / after `]`, where Name
# is either a generic "Speaker N" or a 1-3 token capitalised name. German
# capitalises all nouns, so verbatim speech like "Mein Vorschlag: ..."
# matches too — those false turns are tolerable as BOUNDARIES (they carry no
# timestamp, so the adjacent intervals are skipped as untimed — conservative)
# but must never be FLIPPED: flips are restricted to the known label
# universe (self / generic Speaker N / participants[] names), see
# `_known_labels`.
_GENERIC_SPEAKER_RE = re.compile(r"^speaker\s+[A-Za-z0-9]+$", re.IGNORECASE)
_NAME_RE = re.compile(
    r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'\-]*"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'\-]*){0,2}$"
)
_LABEL_RE = re.compile(
    r"(?:^|(?<=[\s\]]))([A-ZÄÖÜ][^\n:]{0,40}?)(?=:)", re.MULTILINE
)
_TS_BEFORE_RE = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2})\]\s*$")


def _ts_to_sec(ts: str) -> float:
    parts = [int(x) for x in ts.split(":")]
    return float(sum(p * 60 ** (len(parts) - 1 - i) for i, p in enumerate(parts)))


def _parse_turns(transcript: str) -> list[dict]:
    """Extract speaker turns with char spans and (optional) start times.

    A turn's start time is the `[MM:SS]` timestamp immediately preceding its
    speaker label, if any. End time is filled in later from the next turn's
    start (turns without a start leave the previous turn's end unknown —
    those intervals are skipped rather than guessed).
    """
    turns: list[dict] = []
    for m in _LABEL_RE.finditer(transcript):
        name = m.group(1).strip()
        if not name:
            continue
        if not (_GENERIC_SPEAKER_RE.match(name) or _NAME_RE.match(name)):
            continue
        # A directly-preceding "[HH:MM:SS] " is at most ~12 chars back.
        lookback = transcript[max(0, m.start() - 16): m.start()]
        ts_match = _TS_BEFORE_RE.search(lookback)
        t_start = _ts_to_sec(ts_match.group(1)) if ts_match else None
        turns.append({
            "speaker": name,
            "label_start": m.start(1),
            "label_end": m.end(1),
            "t_start": t_start,
            "t_end": None,
        })
    for i in range(len(turns) - 1):
        turns[i]["t_end"] = turns[i + 1]["t_start"]
    return turns


def _known_labels(gemini_dict: dict, turns: list[dict]) -> set[str]:
    """Labels eligible for flipping: self forms, generic 'Speaker N' turns,
    and names listed in participants[]. Arbitrary capitalised words that
    the turn scan picked up out of verbatim German speech are excluded —
    rewriting those would corrupt transcript text, not fix attribution."""
    known: set[str] = set()
    for p in gemini_dict.get("participants") or []:
        if isinstance(p, dict) and (p.get("name") or "").strip():
            known.add(p["name"].strip())
    for t in turns:
        if _GENERIC_SPEAKER_RE.match(t["speaker"]) or _is_self_label(t["speaker"]):
            known.add(t["speaker"])
    return known


def _remote_speakers(turns: list[dict], known: set[str]) -> list[str]:
    """Distinct known non-self speakers, ordered by timed turn-seconds
    (turn count as fallback when nothing is timed)."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for t in turns:
        spk = t["speaker"]
        if _is_self_label(spk) or spk not in known:
            continue
        counts[spk] = counts.get(spk, 0) + 1
        if t["t_start"] is not None and t["t_end"] is not None:
            dur = max(0.0, t["t_end"] - t["t_start"])
            totals[spk] = totals.get(spk, 0.0) + dur
    ranking = totals if totals else counts
    return sorted(ranking, key=lambda k: ranking[k], reverse=True)


def _fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _recompute_speaking_stats(gemini_dict: dict, turns: list[dict]) -> bool:
    """Recompute participants[].speaking_pct / total_seconds from timed turns.

    Only updates participant entries whose name matches a turn speaker
    (exact or first-token match); never invents new entries. Returns True
    if anything was updated.
    """
    per_speaker: dict[str, float] = {}
    for t in turns:
        if t["t_start"] is None or t["t_end"] is None:
            continue
        dur = max(0.0, t["t_end"] - t["t_start"])
        per_speaker[t["speaker"]] = per_speaker.get(t["speaker"], 0.0) + dur
    total = sum(per_speaker.values())
    if total <= 0:
        return False

    def _first_tok(s: str) -> str:
        toks = [x for x in re.split(r"\s+", (s or "").strip().lower()) if x]
        return toks[0] if toks else ""

    updated = False
    for p in gemini_dict.get("participants") or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        name = p["name"].strip()
        secs = per_speaker.get(name)
        if secs is None:
            matches = [v for k, v in per_speaker.items()
                       if _first_tok(k) == _first_tok(name)]
            secs = matches[0] if len(matches) == 1 else None
        if secs is None:
            continue
        p["speaking_pct"] = round(100.0 * secs / total)
        p["total_seconds"] = round(secs)
        updated = True
    return updated


def verify(gemini_dict: dict, vad) -> dict:
    """Flip confidently-misattributed turns in `gemini_dict` (mutates in place).

    Args:
        gemini_dict: Gemini parsed_response (post speaker_reconcile) — uses
            `transcript` and `participants`.
        vad: channel_vad.ChannelVAD (or any object with a
            `shares(t0, t1) -> {'host_only','remote_only','both','speech'}`
            method and a `duration_sec` attribute). Pass None to skip.

    Returns:
        Forensic log for participant_resolution_log.speaker_verification:
        {
            "flips": [{time_span, from, to, duration_sec, host_share,
                       both_share, rule}],
            "turns_total": int, "turns_checked": int,
            "skipped": {untimed, short, no_speech, overlap_dominated},
            "dominant_remote": str|None,
            "flip_to_remote_suppressed_multiparty": bool,
            "speaking_stats_recomputed": bool,
            "speaking_stats_method": "turn_gap",
            "skipped_no_vad": bool,
        }
    """
    log: dict = {
        "flips": [],
        "turns_total": 0,
        "turns_checked": 0,
        "skipped": {"untimed": 0, "short": 0, "no_speech": 0,
                    "overlap_dominated": 0},
        "dominant_remote": None,
        "flip_to_remote_suppressed_multiparty": False,
        "speaking_stats_recomputed": False,
        # Recomputed totals measure timestamp-to-timestamp turn gaps (incl.
        # intra-turn silence), not Gemini's speech-time estimates — slightly
        # inflated seconds, but consistent across speakers post-flip.
        "speaking_stats_method": "turn_gap",
        "skipped_no_vad": False,
    }
    transcript = (gemini_dict or {}).get("transcript") or ""
    if vad is None or not transcript:
        log["skipped_no_vad"] = True
        return log

    turns = _parse_turns(transcript)
    log["turns_total"] = len(turns)
    if not turns:
        return log
    # Last turn: end at recording end if it has a start.
    if turns[-1]["t_end"] is None and turns[-1]["t_start"] is not None:
        duration = getattr(vad, "duration_sec", None)
        if duration and duration > turns[-1]["t_start"]:
            turns[-1]["t_end"] = float(duration)

    known = _known_labels(gemini_dict, turns)
    remotes = _remote_speakers(turns, known)
    dominant_remote = remotes[0] if remotes else None
    log["dominant_remote"] = dominant_remote
    # The mic channel only proves "not the host" — it cannot say WHICH
    # remote participant spoke. Flipping a host label to a remote name is
    # therefore only safe when there is exactly one remote speaker in the
    # meeting; with several, a confident flip to the wrong person is worse
    # than the original error.
    flip_to_remote_ok = len(remotes) == 1
    log["flip_to_remote_suppressed_multiparty"] = (
        not flip_to_remote_ok and len(remotes) > 1
    )

    flips: list[dict] = []
    for turn in turns:
        if turn["speaker"] not in known and not _is_self_label(turn["speaker"]):
            continue  # boundary-only label (e.g. German-noun false positive)
        t0, t1 = turn["t_start"], turn["t_end"]
        if t0 is None or t1 is None:
            log["skipped"]["untimed"] += 1
            continue
        dur = t1 - t0
        if dur < MIN_TURN_SEC:
            log["skipped"]["short"] += 1
            continue
        shares = vad.shares(t0, t1)
        speech = shares["speech"]
        if speech < MIN_SPEECH_SEC:
            log["skipped"]["no_speech"] += 1
            continue
        both_share = shares["both"] / speech
        if both_share > MAX_BOTH_SHARE:
            log["skipped"]["overlap_dominated"] += 1
            continue
        log["turns_checked"] += 1
        # 'both' counts toward the host: his mic carried speech. This biases
        # against flipping in BOTH directions (see module docstring).
        host_share = (shares["host_only"] + shares["both"]) / speech

        is_self = _is_self_label(turn["speaker"])
        new_speaker = None
        rule = None
        if is_self and host_share < FLIP_TO_REMOTE_MAX_HOST_SHARE:
            if dominant_remote and flip_to_remote_ok:
                new_speaker = dominant_remote
                rule = "host_label_but_mic_silent"
        elif not is_self and host_share > FLIP_TO_HOST_MIN_HOST_SHARE:
            new_speaker = SELF_FIRST  # transcript convention is the short form
            rule = "remote_label_but_mic_dominant"
        if new_speaker is None or new_speaker == turn["speaker"]:
            continue

        flips.append({
            "turn": turn,
            "decision": {
                "time_span": f"[{_fmt_ts(t0)}]-[{_fmt_ts(t1)}]",
                "from": turn["speaker"],
                "to": new_speaker,
                "duration_sec": round(dur, 1),
                "host_share": round(host_share, 3),
                "both_share": round(both_share, 3),
                "rule": rule,
            },
        })

    if not flips:
        return log

    # Apply label rewrites back-to-front so char offsets stay valid.
    new_transcript = transcript
    for f in sorted(flips, key=lambda f: f["turn"]["label_start"], reverse=True):
        turn = f["turn"]
        new_transcript = (
            new_transcript[: turn["label_start"]]
            + f["decision"]["to"]
            + new_transcript[turn["label_end"]:]
        )
        turn["speaker"] = f["decision"]["to"]
        log["flips"].append(f["decision"])
    log["flips"].reverse()  # chronological order in the forensic log
    gemini_dict["transcript"] = new_transcript

    log["speaking_stats_recomputed"] = _recompute_speaking_stats(
        gemini_dict, turns
    )

    logger.info(
        "speaker_verify: flipped %d turn(s) — %s",
        len(log["flips"]),
        "; ".join(f"{d['time_span']} {d['from']!r}→{d['to']!r} "
                  f"(host_share={d['host_share']})" for d in log["flips"]),
    )
    return log
