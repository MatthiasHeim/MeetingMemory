#!/usr/bin/env python3
"""system_track_attribution — deterministic host/remote speaker attribution
from the SYSTEM track of a hybrid recording. No model calls.

Why this exists
---------------
The recorder captures the host microphone (ch0) and the system audio (ch1+2,
the remote participants) as separate tracks. When the mic hears the
loudspeakers ("bleeding", the majority of recordings) the MIC track stops
discriminating host from remote, and `channel_vad.separation_report()` says
`admissible: False`. Until now that verdict discarded ALL channel evidence.

That conflates two different things. Bleed corrupts only the mic. The system
track is digital: it carries nothing but what the remote side played, so it
still says exactly when a remote participant speaks (`remote` + `both`
segments = remote active), bleed or not. This module uses that one reliable
signal and nothing else.

What it does
------------
For each Gemini chunk window (Gemini labels each 15-minute chunk
independently, so inversions happen per chunk, e.g. recording
2026-11-12_16-30-49: chunk 1 had "Matthias" on the remote speaker) it measures,
per label, the duration-weighted share of speaking time during which the
system track is active (`remote_share`) over scorable turns, then:

  * Two-label window (one host label + one other label): if the host label
    measures remote and the other label measures host, with enough evidence on
    both, the two labels are swapped for every line in the window.
  * Two-party windows only: a long turn whose own share strongly contradicts
    its label is relabelled; a weaker contradiction is recorded as an
    uncertain region and left alone.
  * More than one non-host label: remote labels are NEVER merged or renamed
    among themselves. Only host-vs-remote is acted on. A host-labelled turn
    that measures remote is relabelled only when both neighbouring turns carry
    the same single remote label (the name is then already on the page);
    otherwise it is flagged uncertain, never guessed.

It never invents a name (only labels already in the transcript), leaves short
backchannels alone, and degrades instead of blocking: any exception returns
the transcript untouched.

What it does NOT establish
--------------------------
WHICH remote person a remote-measured turn belongs to. `names_verified` is
always False in the log. Naming stays with the calendar roster and the
semantic coherence audit.

CLI (read-only)
---------------
    python tools/system_track_attribution.py --stem 2026-11-12_16-30-49 --dry-run
    python tools/system_track_attribution.py --all --dry-run

reads `CaptureArchive/<stem>/<stem>.sys.wav` (or a 3-channel WAV via --wav) and
`Transcripts/<stem>.json`, prints the log and writes nothing.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_verify import _GENERIC_SPEAKER_RE, _is_self_label  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# A turn is scorable when it is long enough that Gemini's +-few-second timestamp
# noise cannot dominate, and carries enough words to be real speech rather than
# a backchannel ("Mhm", "Ja").
MIN_SCORABLE_TURN_SEC = 3.0
MIN_SCORABLE_TEXT_CHARS = 40

# Label classification from its duration-weighted remote_share.
REMOTE_LABEL_MIN_SHARE = 0.6
HOST_LABEL_MAX_SHARE = 0.25

# Swap needs real evidence on BOTH labels.
MIN_LABEL_SCORABLE_SEC = 30.0

# A host label between HOST_LABEL_MAX_SHARE and this still counts as
# "verified host-side" when a remote-classified label sits clearly above it
# (an inverted window cannot look like host 0.28 / remote 0.92). Not used to
# swap or relabel anything, only to credit coverage and enable turn checks.
HOST_VERIFIED_MAX_SHARE = 0.40
HOST_VERIFIED_MIN_MARGIN = 0.40

# Per-turn correction. Turns shorter than this are never touched.
MIN_TURN_ACTION_SEC = 8.0
TURN_HOST_LABEL_RELABEL_MIN_SHARE = 0.8    # host label, system track active
TURN_REMOTE_LABEL_RELABEL_MAX_SHARE = 0.1  # remote label, system track silent

# Less system-track speech than this: nothing to measure against.
MIN_REMOTE_SPEECH_SEC = 60.0

# Must equal gemini_processor.GeminiAudioProcessor.CHUNK_DURATION_SEC; the
# watcher passes the live value, this is only the CLI / fallback default.
DEFAULT_CHUNK_DURATION_SEC = 15 * 60

_TURN_RE = re.compile(
    r"^(\[(?:(\d+):)?(\d{1,3}):(\d{2})\]\s*)([^:\n]{1,60}?)(:\s*)(.*)$"
)


# ---------------------------------------------------------------------------
# Remote (system-track) activity
# ---------------------------------------------------------------------------

class RemoteActivity:
    """Sorted, merged intervals during which the system track is active."""

    def __init__(self, intervals: Iterable[tuple[float, float]],
                 duration_sec: Optional[float] = None):
        merged: list[list[float]] = []
        for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self._starts = [m[0] for m in merged]
        self._ends = [m[1] for m in merged]
        self.duration_sec = duration_sec
        self.total_seconds = sum(e - s for s, e in zip(self._starts, self._ends))

    @classmethod
    def from_segments(cls, segments, duration_sec: Optional[float] = None):
        """Remote active = `remote` or `both` segments (channel_vad labels).

        Valid regardless of mic bleed: both labels are defined by the digital
        system track being active.
        """
        return cls(
            ((t0, t1) for t0, t1, label in segments or []
             if label in ("remote", "both")),
            duration_sec,
        )

    @classmethod
    def from_windows(cls, active, window_sec: float,
                     duration_sec: Optional[float] = None):
        runs: list[tuple[float, float]] = []
        start = None
        n = 0
        for i, flag in enumerate(active):
            n = i + 1
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                runs.append((start * window_sec, i * window_sec))
                start = None
        if start is not None:
            runs.append((start * window_sec, n * window_sec))
        return cls(runs, duration_sec if duration_sec is not None
                   else n * window_sec)

    @classmethod
    def from_vad(cls, vad):
        """Build from a `channel_vad.ChannelVAD` (or a stub with `segments`).

        Prefers the raw per-window `remote_active` array (no bridging); falls
        back to segments. Returns None when neither is available.
        """
        if vad is None:
            return None
        duration = getattr(vad, "duration_sec", None)
        active = getattr(vad, "remote_active", None)
        window = getattr(vad, "window_sec", None)
        if active is not None and window:
            return cls.from_windows(active, float(window), duration)
        segments = getattr(vad, "segments", None)
        if segments is None:
            return None
        return cls.from_segments(segments, duration)

    def active_seconds(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        i = max(0, bisect.bisect_right(self._ends, t0))
        total = 0.0
        while i < len(self._starts) and self._starts[i] < t1:
            total += max(0.0, min(t1, self._ends[i]) - max(t0, self._starts[i]))
            i += 1
        return total

    def share(self, t0: float, t1: float) -> Optional[float]:
        if t1 <= t0:
            return None
        return self.active_seconds(t0, t1) / (t1 - t0)


def remote_activity_from_system_wav(path: Path) -> Optional[RemoteActivity]:
    """System-track activity from a separate mono/stereo system WAV.

    Same band-pass, window and adaptive threshold `channel_vad` applies to the
    system channels of the 3-channel file (strongest of the two channels).
    """
    import numpy as np
    import channel_vad as cv

    channels = cv._channel_count(Path(path))
    if channels < 1:
        return None
    frames = cv._decode_bandpassed(Path(path), channels)
    win = int(cv.WINDOW_SEC * cv.TARGET_SR)
    if len(frames) < win:
        return None
    env = np.max(np.abs(frames), axis=1)
    db = cv._window_rms_db(env, win)
    active = cv._activity(db, cv._noise_floor_threshold(db), "remote")
    return RemoteActivity.from_windows(active, cv.WINDOW_SEC)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def _parse_turns(lines: list[str], participants: Optional[Iterable[str]],
                 duration_sec: Optional[float]) -> list[dict]:
    """One turn per `[MM:SS] Label: text` line; untimed lines continue the
    previous turn's text. Labels that are not plausibly speakers (a single
    German "Hinweis:" at line start) are rejected."""
    raw: list[dict] = []
    for idx, line in enumerate(lines):
        m = _TURN_RE.match(line)
        if m:
            h = int(m.group(2) or 0)
            t = h * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            raw.append({
                "line": idx, "t_start": float(t),
                "label": m.group(5).strip(),
                "label_span": (m.start(5), m.end(5)),
                "text": m.group(7),
            })
        elif raw and line.strip():
            raw[-1]["text"] += " " + line.strip()
    counts: dict[str, int] = {}
    for t in raw:
        counts[t["label"]] = counts.get(t["label"], 0) + 1
    known = {p.strip() for p in participants or [] if p and p.strip()}
    turns = [
        t for t in raw
        if (_GENERIC_SPEAKER_RE.match(t["label"]) or _is_self_label(t["label"])
            or t["label"] in known or counts[t["label"]] >= 2)
    ]
    for i, t in enumerate(turns):
        nxt = turns[i + 1]["t_start"] if i + 1 < len(turns) else duration_sec
        t["t_end"] = nxt if nxt is not None and nxt > t["t_start"] else None
    return turns


def _windows(chunk_offsets: Optional[list[float]]) -> list[tuple[float, float]]:
    offs = sorted(set(float(o) for o in (chunk_offsets or [0.0])))
    if not offs or offs[0] != 0.0:
        offs = [0.0] + offs
    return [(o, offs[i + 1] if i + 1 < len(offs) else float("inf"))
            for i, o in enumerate(offs)]


def chunk_offsets_for(meta: Optional[dict], duration_sec: Optional[float],
                      chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC
                      ) -> list[float]:
    """Chunk window starts from transcript `_meta` (zero-overlap chunks)."""
    meta = meta or {}
    try:
        n = int(meta.get("chunk_count") or 1)
    except (TypeError, ValueError):
        n = 1
    if not meta.get("chunked") or n <= 1:
        return [0.0]
    return [i * float(chunk_duration_sec) for i in range(n)]


def _fmt_ts(sec: float) -> str:
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _classify(share: Optional[float], sec: float) -> Optional[str]:
    if share is None or sec <= 0:
        return None
    if share >= REMOTE_LABEL_MIN_SHARE:
        return "remote"
    if share <= HOST_LABEL_MAX_SHARE:
        return "host"
    return "undecided"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _empty_log(reason: Optional[str], ran: bool = False, ok: bool = False
               ) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "ran": ran, "ok": ok, "reason": reason,
        "thresholds": {
            "min_scorable_turn_sec": MIN_SCORABLE_TURN_SEC,
            "min_scorable_text_chars": MIN_SCORABLE_TEXT_CHARS,
            "remote_label_min_share": REMOTE_LABEL_MIN_SHARE,
            "host_label_max_share": HOST_LABEL_MAX_SHARE,
            "min_label_scorable_sec": MIN_LABEL_SCORABLE_SEC,
            "host_verified_max_share": HOST_VERIFIED_MAX_SHARE,
            "host_verified_min_margin": HOST_VERIFIED_MIN_MARGIN,
            "min_turn_action_sec": MIN_TURN_ACTION_SEC,
            "turn_host_label_relabel_min_share": TURN_HOST_LABEL_RELABEL_MIN_SHARE,
            "turn_remote_label_relabel_max_share": TURN_REMOTE_LABEL_RELABEL_MAX_SHARE,
            "min_remote_speech_sec": MIN_REMOTE_SPEECH_SEC,
        },
        "windows": [], "swaps": [], "relabels": [], "uncertain_regions": [],
        "turns_total": 0, "turns_checked": 0, "seconds_checked": 0.0,
        "seconds_total": 0.0, "coverage": 0.0,
        "seconds_verified": 0.0, "verified_coverage": 0.0,
        "unresolved": 0, "labels_changed": False,
        # This stage proves host-vs-remote only, never WHICH remote person.
        "names_verified": False,
    }


def _label_stats(turns: list[dict], labels: dict[int, str]) -> dict[str, dict]:
    """label -> {sec, weighted, share, turns} over SCORABLE turns."""
    out: dict[str, dict] = {}
    for t in turns:
        if not t["scorable"]:
            continue
        s = out.setdefault(labels[t["line"]],
                           {"sec": 0.0, "weighted": 0.0, "turns": 0})
        s["sec"] += t["dur"]
        s["weighted"] += t["share"] * t["dur"]
        s["turns"] += 1
    for s in out.values():
        s["share"] = s["weighted"] / s["sec"] if s["sec"] > 0 else None
    return out


def run(transcript: str, remote: Optional[RemoteActivity], *,
        duration_sec: Optional[float] = None,
        chunk_offsets: Optional[list[float]] = None,
        participants: Optional[Iterable[str]] = None,
        namesake_remote: bool = False,
        skip_reason: str = "no_three_channel_layout") -> tuple[str, dict]:
    """Return (transcript, log). The returned text is the input object itself
    when nothing changed; on any exception it is the untouched input."""
    try:
        return _run(transcript, remote, duration_sec, chunk_offsets,
                    participants, namesake_remote, skip_reason)
    except Exception as e:  # degrade, never block
        logger.warning(f"system_track_attribution failed: {e}", exc_info=True)
        return transcript, _empty_log(f"error: {type(e).__name__}: {e}")


def _run(transcript, remote, duration_sec, chunk_offsets, participants,
         namesake_remote, skip_reason):
    if namesake_remote:
        return transcript, _empty_log("host_namesake_remote_attendee")
    if remote is None:
        return transcript, _empty_log(skip_reason)
    if not isinstance(transcript, str) or not transcript.strip():
        return transcript, _empty_log("empty_transcript")

    lines = transcript.split("\n")
    turns = _parse_turns(lines, participants, duration_sec)
    if not turns:
        return transcript, _empty_log("no_timed_turns")
    if not any(_is_self_label(t["label"]) for t in turns):
        return transcript, _empty_log("no_self_label")
    if remote.total_seconds < MIN_REMOTE_SPEECH_SEC:
        log = _empty_log("insufficient_remote_speech")
        log["remote_speech_sec"] = round(remote.total_seconds, 1)
        return transcript, log

    windows = _windows(chunk_offsets)
    log = _empty_log(None, ran=True, ok=True)
    log["remote_speech_sec"] = round(remote.total_seconds, 1)
    log["turns_total"] = len(turns)

    # Measure every turn once.
    for t in turns:
        w_end = next(e for s, e in windows if s <= t["t_start"] < e) \
            if t["t_start"] >= 0 else float("inf")
        end = t["t_end"]
        if end is not None:
            end = min(end, w_end)
        t["dur"] = (end - t["t_start"]) if end is not None else 0.0
        t["end"] = end
        t["share"] = (remote.share(t["t_start"], end)
                      if end is not None and t["dur"] > 0 else None)
        t["scorable"] = bool(
            t["share"] is not None and t["dur"] >= MIN_SCORABLE_TURN_SEC
            and len(t["text"].strip()) >= MIN_SCORABLE_TEXT_CHARS)
        if t["end"] is not None:
            log["seconds_total"] += t["dur"]
        if t["scorable"]:
            log["turns_checked"] += 1
            log["seconds_checked"] += t["dur"]

    labels: dict[int, str] = {t["line"]: t["label"] for t in turns}
    original = dict(labels)
    verified_sec = 0.0

    for wi, (w0, w1) in enumerate(windows):
        wturns = [t for t in turns if w0 <= t["t_start"] < w1]
        wlog = {"index": wi, "start": w0, "end": None if w1 == float("inf") else w1,
                "labels": {}, "orientation": "no_turns", "swapped": False}
        log["windows"].append(wlog)
        if not wturns:
            continue
        distinct = sorted({labels[t["line"]] for t in wturns})
        host_labels = [l for l in distinct if _is_self_label(l)]
        other_labels = [l for l in distinct if not _is_self_label(l)]

        stats = _label_stats(wturns, labels)
        host_label = host_labels[0] if len(host_labels) == 1 else None

        def host_group(stats_):
            sec = sum(stats_[l]["sec"] for l in host_labels if l in stats_)
            wsum = sum(stats_[l]["weighted"] for l in host_labels if l in stats_)
            return (wsum / sec if sec > 0 else None), sec

        h_share, h_sec = host_group(stats)

        # --- Two-label swap (the incident fix) -----------------------------
        if host_label and len(other_labels) == 1:
            o_label = other_labels[0]
            o = stats.get(o_label, {"sec": 0.0, "share": None})
            if (h_share is not None and o["share"] is not None
                    and h_share >= REMOTE_LABEL_MIN_SHARE
                    and o["share"] <= HOST_LABEL_MAX_SHARE
                    and h_sec >= MIN_LABEL_SCORABLE_SEC
                    and o["sec"] >= MIN_LABEL_SCORABLE_SEC):
                swap = {host_label: o_label, o_label: host_label}
                n = 0
                for t in wturns:
                    labels[t["line"]] = swap[labels[t["line"]]]
                    n += 1
                wlog["swapped"] = True
                log["swaps"].append({
                    "window": wi, "labels": [host_label, o_label],
                    "from_time": _fmt_ts(w0),
                    "measured_shares": {host_label: round(h_share, 3),
                                        o_label: round(o["share"], 3)},
                    "lines_changed": n,
                })
                stats = _label_stats(wturns, labels)
                h_share, h_sec = host_group(stats)

        # --- Per-label classification (after any swap) ---------------------
        classes: dict[str, Optional[str]] = {}
        for l in distinct:
            s = stats.get(l)
            lab_cls = _classify(s["share"], s["sec"]) if s else None
            if s and s["sec"] < MIN_LABEL_SCORABLE_SEC:
                lab_cls = "insufficient"
            classes[l] = lab_cls
            wlog["labels"][l] = {
                "role": "host" if _is_self_label(l) else "remote",
                "remote_share": None if not s or s["share"] is None
                else round(s["share"], 3),
                "scorable_sec": round(s["sec"], 1) if s else 0.0,
                "turns": s["turns"] if s else 0,
                "class": lab_cls,
            }
        host_cls = _classify(h_share, h_sec) if h_share is not None else None
        if h_share is not None and h_sec < MIN_LABEL_SCORABLE_SEC:
            host_cls = "insufficient"
        two_party = bool(host_label) and len(other_labels) == 1

        decided_remote = [l for l in other_labels if classes.get(l) == "remote"]
        host_leaning = bool(
            host_cls == "host"
            or (h_share is not None and host_cls != "insufficient"
                and h_share <= HOST_VERIFIED_MAX_SHARE
                and any(stats[l]["share"] - h_share >= HOST_VERIFIED_MIN_MARGIN
                        for l in decided_remote)))
        for l in distinct:
            s_ = stats.get(l)
            if not s_:
                continue
            if _is_self_label(l):
                if host_leaning:
                    verified_sec += s_["sec"]
            elif classes.get(l) == "remote":
                verified_sec += s_["sec"]
        if wlog["swapped"]:
            wlog["orientation"] = "swapped"
        elif host_leaning and decided_remote:
            wlog["orientation"] = "consistent"
        elif host_cls == "remote":
            wlog["orientation"] = "contradiction"
        elif host_cls is None and not stats:
            wlog["orientation"] = "no_scorable_turns"
        else:
            wlog["orientation"] = "undecided"

        # Host label measures remote and no swap was possible: the whole
        # window's host identity is in doubt. Do not guess; flag.
        if host_cls == "remote" and not wlog["swapped"]:
            log["uncertain_regions"].append({
                "window": wi, "scope": "window",
                "from_line": min(t["line"] for t in wturns) + 1,
                "to_line": max(t["line"] for t in wturns) + 1,
                "time": f"{_fmt_ts(w0)}-" + ("end" if w1 == float("inf")
                                             else _fmt_ts(w1)),
                "label": host_label or ", ".join(host_labels),
                "remote_share": round(h_share, 3),
                "reason": ("host label measures remote on the system track and "
                           "no single other label to swap with"),
                "source": "system_track",
            })
            continue

        # --- Per-turn correction / flags -----------------------------------
        snapshot = dict(labels)
        order = [t for t in turns if w0 <= t["t_start"] < w1]
        for pos, t in enumerate(order):
            if not t["scorable"] or t["dur"] < MIN_TURN_ACTION_SEC:
                continue
            cur = snapshot[t["line"]]
            s = t["share"]
            is_host = _is_self_label(cur)

            def neighbours_same_remote():
                prev = snapshot[order[pos - 1]["line"]] if pos > 0 else None
                nxt = snapshot[order[pos + 1]["line"]] if pos + 1 < len(order) else None
                if (prev and nxt and prev == nxt and not _is_self_label(prev)):
                    return prev
                return None

            def record_uncertain(reason):
                log["uncertain_regions"].append({
                    "window": wi, "scope": "turn",
                    "from_line": t["line"] + 1, "to_line": t["line"] + 1,
                    "time": f"{_fmt_ts(t['t_start'])}-{_fmt_ts(t['end'])}",
                    "label": cur, "remote_share": round(s, 3),
                    "seconds": round(t["dur"], 1), "reason": reason,
                    "source": "system_track",
                })

            def relabel(new, reason):
                labels[t["line"]] = new
                log["relabels"].append({
                    "window": wi, "line": t["line"] + 1,
                    "time": f"{_fmt_ts(t['t_start'])}-{_fmt_ts(t['end'])}",
                    "from": cur, "to": new, "remote_share": round(s, 3),
                    "seconds": round(t["dur"], 1), "reason": reason,
                })

            if is_host:
                # Only meaningful when the host label as a whole measures host.
                if not host_leaning:
                    continue
                if s >= TURN_HOST_LABEL_RELABEL_MIN_SHARE:
                    if two_party:
                        relabel(other_labels[0],
                                "host label, system track active")
                    else:
                        adj = neighbours_same_remote()
                        if adj:
                            relabel(adj, "host label, system track active; "
                                    "both neighbours carry this remote label")
                        else:
                            record_uncertain(
                                "host-labelled turn measures remote; remote "
                                "name unknown (several remote labels)")
                elif s >= REMOTE_LABEL_MIN_SHARE:
                    record_uncertain("host-labelled turn partly measures remote")
            else:
                # Only meaningful when this remote label as a whole is remote.
                if classes.get(cur) != "remote":
                    continue
                if s <= TURN_REMOTE_LABEL_RELABEL_MAX_SHARE:
                    if two_party and host_label:
                        relabel(host_label,
                                "remote label, system track silent")
                    else:
                        record_uncertain(
                            "remote-labelled turn measures host; "
                            "several remote labels, left unchanged")
                elif s <= HOST_LABEL_MAX_SHARE:
                    record_uncertain("remote-labelled turn mostly measures host")

    # --- Assemble ----------------------------------------------------------
    log["seconds_checked"] = round(log["seconds_checked"], 1)
    log["seconds_total"] = round(log["seconds_total"], 1)
    log["coverage"] = round(log["seconds_checked"] / log["seconds_total"], 3) \
        if log["seconds_total"] > 0 else 0.0
    log["seconds_verified"] = round(verified_sec, 1)
    log["verified_coverage"] = round(verified_sec / log["seconds_total"], 3) \
        if log["seconds_total"] > 0 else 0.0
    log["unresolved"] = len(log["uncertain_regions"])

    changed = {i: new for i, new in labels.items() if new != original[i]}
    if not changed:
        return transcript, log
    by_line = {t["line"]: t for t in turns}
    for i, new in changed.items():
        a, b = by_line[i]["label_span"]
        lines[i] = lines[i][:a] + new + lines[i][b:]
    log["labels_changed"] = True
    return "\n".join(lines), log


def apply_to_result_dict(data: dict, remote: Optional[RemoteActivity], **kw
                         ) -> dict:
    """Run on a transcript dict (`transcript`, `participants`, `_meta`) in
    place; returns the log. Used by the watcher via GeminiResult.parsed_response."""
    names = [p.get("name") for p in data.get("participants") or []
             if isinstance(p, dict)]
    new_text, log = run(data.get("transcript") or "", remote,
                        participants=names, **kw)
    if log.get("labels_changed"):
        data["transcript"] = new_text
    return log


# ---------------------------------------------------------------------------
# CLI (read-only)
# ---------------------------------------------------------------------------

DEFAULT_BASE = Path.home() / "Documents" / "MeetingRecorder"


def _load_stem(stem: str, base: Path, wav: Optional[Path]):
    tr = json.loads((base / "Transcripts" / f"{stem}.json").read_text())
    if wav is not None:
        import channel_vad as cv
        if cv._channel_count(wav) >= 3:
            vad = cv.compute_channel_vad(wav)
            remote = RemoteActivity.from_vad(vad)
        else:
            remote = remote_activity_from_system_wav(wav)
    else:
        sys_wav = base / "CaptureArchive" / stem / f"{stem}.sys.wav"
        remote = remote_activity_from_system_wav(sys_wav) if sys_wav.exists() else None
    return tr, remote


def evaluate_stem(stem: str, base: Path = DEFAULT_BASE,
                  wav: Optional[Path] = None) -> tuple[dict, dict]:
    """Run the stage against an archived recording; writes nothing."""
    tr, remote = _load_stem(stem, base, wav)
    meta = tr.get("_meta") or {}
    duration = meta.get("audio_duration_seconds")
    offsets = chunk_offsets_for(meta, duration)
    names = [p.get("name") for p in tr.get("participants") or []
             if isinstance(p, dict)]
    _, log = run(tr.get("transcript") or "", remote, duration_sec=duration,
                 chunk_offsets=offsets, participants=names,
                 skip_reason="system_track_missing_or_empty")
    log["stem"] = stem
    return tr, log


def _print_log(log: dict) -> None:
    print(f"== {log['stem']}  ran={log['ran']} ok={log['ok']} "
          f"reason={log['reason']}")
    if not log["ran"]:
        return
    print(f"   remote_speech={log['remote_speech_sec']}s  "
          f"turns {log['turns_checked']}/{log['turns_total']} scored, "
          f"{log['seconds_checked']}s of {log['seconds_total']}s "
          f"(coverage {log['coverage']}, verified {log['verified_coverage']})")
    for w in log["windows"]:
        end = "end" if w["end"] is None else _fmt_ts(w["end"])
        shares = "; ".join(
            f"{l}[{v['role']}] share={v['remote_share']} "
            f"({v['scorable_sec']}s, {v['class']})"
            for l, v in w["labels"].items())
        print(f"   window {w['index']} {_fmt_ts(w['start'])}-{end} "
              f"{w['orientation']}: {shares}")
    for s in log["swaps"]:
        print(f"   SWAP window {s['window']}: {s['labels']} "
              f"({s['lines_changed']} lines) {s['measured_shares']}")
    for r in log["relabels"]:
        print(f"   RELABEL line {r['line']} {r['time']} {r['from']} -> "
              f"{r['to']} share={r['remote_share']} ({r['seconds']}s)")
    for u in log["uncertain_regions"]:
        print(f"   UNCERTAIN {u['scope']} win {u['window']} {u['time']} "
              f"{u['label']} share={u['remote_share']}: {u['reason']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stem", action="append", default=[])
    ap.add_argument("--all", action="store_true",
                    help="every CaptureArchive stem that has a transcript JSON")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help="MeetingRecorder directory")
    ap.add_argument("--wav", type=Path,
                    help="3-channel WAV (or a system-only WAV) instead of the "
                         "archived <stem>.sys.wav")
    ap.add_argument("--dry-run", action="store_true",
                    help="accepted for explicitness; this CLI never writes")
    ap.add_argument("--json", action="store_true", help="print raw logs")
    args = ap.parse_args(argv)

    stems = list(args.stem)
    if args.all:
        arch = args.base / "CaptureArchive"
        stems += sorted(p.name for p in arch.iterdir()
                        if (args.base / "Transcripts" / f"{p.name}.json").exists())
    if not stems:
        ap.error("give --stem or --all")
    rc = 0
    for stem in stems:
        try:
            _, log = evaluate_stem(stem, args.base, args.wav)
        except Exception as e:
            print(f"== {stem}  ERROR {type(e).__name__}: {e}")
            rc = 1
            continue
        if args.json:
            print(json.dumps(log, indent=1, ensure_ascii=False))
        else:
            _print_log(log)
    return rc


if __name__ == "__main__":
    sys.exit(main())
