#!/usr/bin/env python3
"""voice_match — WHICH person speaks, from the voice itself. Local only.

The system-track stage (`system_track_attribution`) decides host vs remote.
It cannot say which remote person is which. This stage measures that with a
local speaker-embedding model (`pyannote/embedding`, on this machine, no
network, no API) and compares each transcript label against a library of
voice fingerprints (`voice_library`, biometric, never leaves the machine).

Pipeline
--------
1. ``plan_segments``      transcript turns >= 7 s; remote speakers are cut from
                          the SYSTEM track where it is active >= 70 % of the
                          turn, the host from the MIC track where the system
                          track is active <= 12 %; 1 s in from each edge, at
                          most 25 s, at most 14 segments per label.
2. ``embed_segments``     decode + embed in a spawned worker with a timeout
                          (same pattern as `diarize`); never raises.
3. ``match_labels``       per non-host label: best person, runner-up, verdict
                          clear / possible / none.
4. ``host_check``         host-labelled mic-only turns vs the self profile.
5. ``find_conflicts``     two labels -> one person, a label split between two
                          people, a label that contradicts its own name.
                          Reported, never auto-resolved.

Nothing here edits a transcript by itself. ``process`` returns a log (names and
similarity scores only, no vectors) and, only with ``auto_apply: true``, a
rename map for GENERIC labels with a clear, conflict-free match.

Thresholds (named constants below) come from the 10-recording feasibility run
on this machine: the same remote speaker scores 0.79-0.96 against their own
meeting centroid (5th-95th pct), different speakers -0.07-0.42, the same
person two months apart 0.94 vs 0.42 for the next closest person.

CLI (dry runs write nothing outside the directory you give them)::

    python tools/voice_match.py evaluate --all --library-dir /tmp/vm-eval
    python tools/voice_match.py enrol-confirmed --library-dir DIR [--use-db]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import queue as pyqueue
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_verify import _is_self_label  # noqa: E402
from system_track_attribution import (  # noqa: E402
    RemoteActivity,
    _parse_turns,
    remote_activity_from_system_wav,
)
from voice_library import (  # noqa: E402
    BASIS_HUMAN,
    BASIS_SELF,
    LibraryError,
    PersonConflict,
    Profile,
    ROLE_OTHER,
    ROLE_SELF,
    SELF_NAME,
    VoiceLibrary,
    default_library_dir,
    load_config,
    normalise_name,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
SAMPLE_RATE = 16000
MODEL_PATH = Path(__file__).resolve().parent.parent / "pyannote" / "embedding" / "pytorch_model.bin"
MODEL_NAME = "pyannote/embedding"

# --- segment selection (validated in the feasibility run) -------------------
MIN_TURN_SEC = 7.0
EDGE_SEC = 1.0                   # cut this far in from each turn edge
MAX_SEGMENT_SEC = 25.0
MIN_SEGMENT_SEC = 4.0
MAX_SEGMENTS_PER_LABEL = 14
REMOTE_MIN_SYSTEM_SHARE = 0.70   # remote speaker: system track active this much
HOST_MAX_SYSTEM_SHARE = 0.12     # host: system track (almost) silent

# --- matching ---------------------------------------------------------------
# Same speaker scored 0.79-0.96 (5th-95th pct) per segment and ~0.94 across
# meetings; different speakers topped out at 0.42. 0.70 keeps a wide gap on
# both sides. The 0.20 margin to the next different person guards against a
# library that holds the same voice under two entries or two similar voices.
CLEAR_MIN_SCORE = 0.70
CLEAR_MIN_MARGIN = 0.20
POSSIBLE_MIN_SCORE = 0.55        # below this a label is simply "none"
MIN_LABEL_SEGMENTS = 3
SPLIT_MIN_SEGMENT_SCORE = 0.60   # a segment "votes" for a person above this
SPLIT_MIN_VOTES = 3              # ... and a label splits when two people each
SPLIT_MIN_VOTE_SHARE = 0.25      # get this many and this share of its votes

# --- host --------------------------------------------------------------------
# Host centroids across meetings: median pairwise 0.74, 5th pct 0.46. A host
# turn near zero is a different voice, not a bad meeting.
HOST_SUSPICIOUS_MAX_SIM = 0.10
HOST_CENTROID_MISMATCH_MAX_SIM = 0.30
MIN_HOST_ENROL_SEGMENTS = 5
HOST_ENROL_MIN_COHERENCE = 0.50  # mean similarity of the segments to their centroid

# --- enrolment from a confirmation -----------------------------------------
MIN_ENROL_SEGMENTS = 3

EMBED_TIMEOUT_SECONDS = 600

_GENERIC_LABEL_RE = re.compile(
    r"^(speaker|sprecher|sprecherin|teilnehmer|teilnehmerin|participant|unidentified participant|"
    r"unbekannt|unknown)(\s+[A-Za-z0-9]+)?$|^(andere|others?|remote)$",
    re.IGNORECASE,
)


def is_generic_label(label: str) -> bool:
    """"Speaker 1", "Speaker A", "Andere", "Unidentified participant 2" style."""
    return bool(_GENERIC_LABEL_RE.match((label or "").strip()))


def _fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ===========================================================================
# Sources (where the audio comes from)
# ===========================================================================

def three_channel_source(path: Path | str) -> dict:
    """The watcher's WAV (ch0 host mic, ch1+2 system). Pass the ALIGNED file."""
    return {"layout": "three_channel", "path": str(path), "mic_offset_sec": 0.0}


def tracks_source(mic: Path | str, sys_wav: Path | str, mic_offset_sec: float = 0.0) -> dict:
    """Archived `<stem>.mic.wav` + `<stem>.sys.wav`. The raw mic can lag the
    system track (`_meta.channel_alignment.lag_seconds`); host cuts shift by it."""
    return {"layout": "tracks", "mic": str(mic), "sys": str(sys_wav),
            "mic_offset_sec": float(mic_offset_sec or 0.0)}


def locate_source(base: Path | str, stem: str, mic_offset_sec: float = 0.0,
                  need_mic: bool = True) -> Optional[dict]:
    """Archived tracks first, then the raw 3-channel recording. None when both
    have been pruned (`prune_recordings` deletes them after the retention)."""
    base = Path(base)
    arch = base / "CaptureArchive" / stem
    mic, sys_wav = arch / f"{stem}.mic.wav", arch / f"{stem}.sys.wav"
    if sys_wav.exists() and (mic.exists() or not need_mic):
        return tracks_source(mic if mic.exists() else sys_wav, sys_wav, mic_offset_sec)
    raw = base / "Recordings" / f"{stem}.wav"
    if raw.exists():
        src = three_channel_source(raw)
        src["mic_offset_sec"] = float(mic_offset_sec or 0.0)
        return src
    return None


def remote_activity_for_source(source: dict) -> Optional[RemoteActivity]:
    import channel_vad as cv
    if source["layout"] == "three_channel":
        return RemoteActivity.from_vad(cv.compute_channel_vad(Path(source["path"])))
    return remote_activity_from_system_wav(Path(source["sys"]))


# ===========================================================================
# Segment planning (pure)
# ===========================================================================

@dataclass
class SegmentPlan:
    label: str
    role: str            # "remote" (cut from the system track) | "host" (mic)
    start: float         # seconds, transcript timeline
    end: float
    line: int
    share: float         # system-track share of the whole turn


def plan_segments(transcript: str, remote: Optional[RemoteActivity], *,
                  duration_sec: Optional[float],
                  participants: Optional[Iterable[str]] = None,
                  labels: Optional[Iterable[str]] = None,
                  roles: tuple[str, ...] = ("remote", "host"),
                  max_per_label: int = MAX_SEGMENTS_PER_LABEL) -> list[SegmentPlan]:
    """Which stretches of audio to embed. Host label -> mic-only turns, other
    labels -> system-track turns. Spread evenly over the meeting when a label
    has more than `max_per_label` candidates."""
    if remote is None or not transcript:
        return []
    turns = _parse_turns(transcript.split("\n"), participants, duration_sec)
    wanted = set(labels) if labels is not None else None
    cands: dict[tuple[str, str], list[SegmentPlan]] = {}
    for t in turns:
        label = t["label"]
        if wanted is not None and label not in wanted:
            continue
        t0, t1 = t["t_start"], t.get("t_end")
        if t1 is None or t1 - t0 < MIN_TURN_SEC:
            continue
        share = remote.share(t0, t1)
        if share is None:
            continue
        is_host = _is_self_label(label)
        if is_host and share <= HOST_MAX_SYSTEM_SHARE:
            role = "host"
        elif not is_host and share >= REMOTE_MIN_SYSTEM_SHARE:
            role = "remote"
        else:
            continue
        if role not in roles:
            continue
        a0 = t0 + EDGE_SEC
        a1 = min(t1 - EDGE_SEC, a0 + MAX_SEGMENT_SEC)
        if a1 - a0 < MIN_SEGMENT_SEC:
            continue
        cands.setdefault((label, role), []).append(
            SegmentPlan(label, role, a0, a1, t["line"], round(share, 3)))
    out: list[SegmentPlan] = []
    for plans in cands.values():
        if len(plans) > max_per_label:
            idx = np.unique(np.linspace(0, len(plans) - 1, max_per_label).round().astype(int))
            plans = [plans[i] for i in idx]
        out.extend(plans)
    return out


# ===========================================================================
# Embedding (worker process)
# ===========================================================================

def embedding_available() -> tuple[bool, Optional[str]]:
    """Cheap check that never imports torch into the caller."""
    for mod in ("torch", "pyannote.audio"):
        try:
            if importlib.util.find_spec(mod) is None:
                return False, f"{mod} not installed"
        except Exception as exc:
            return False, f"{mod}: {type(exc).__name__}"
    if not MODEL_PATH.exists():
        return False, f"model file missing: {MODEL_PATH.name}"
    return True, None


def _ffmpeg() -> str:
    return FFMPEG_PATH if Path(FFMPEG_PATH).exists() else (shutil.which("ffmpeg") or "ffmpeg")


def decode_track(source: dict, which: str) -> np.ndarray:
    """16 kHz mono int16 of the host mic ("mic") or the system track ("sys").

    ffmpeg only: torchcodec does not load on this machine and pyannote is
    never given a file path.
    """
    if source["layout"] == "three_channel":
        path = source["path"]
        pan = "pan=mono|c0=c0" if which == "mic" else "pan=mono|c0=0.5*c1+0.5*c2"
        cmd = [_ffmpeg(), "-v", "error", "-i", path, "-af", pan,
               "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    else:
        cmd = [_ffmpeg(), "-v", "error", "-i", source["mic" if which == "mic" else "sys"],
               "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(r.stdout, dtype=np.int16)


def _embed_child(args: dict, q) -> None:
    """Spawn target. Loads the model, embeds every segment, puts vectors on the
    queue (same machine, in memory only). Any exception becomes ok=False."""
    try:
        import warnings
        warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")   # we never pass file paths
        t0 = time.time()
        import torch
        from pyannote.audio import Inference, Model

        device = args.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
        model = Model.from_pretrained(str(args["model_path"]))
        inference = Inference(model, window="whole", device=torch.device(device))
        load_sec = time.time() - t0

        source = args["source"]
        audio: dict[str, np.ndarray] = {}
        vectors: list[Optional[np.ndarray]] = []
        failed = 0
        t1 = time.time()
        for seg in args["segments"]:
            which = "sys" if seg["role"] == "remote" else "mic"
            if which not in audio:
                audio[which] = decode_track(source, which)
            shift = source.get("mic_offset_sec", 0.0) if which == "mic" else 0.0
            a = int(max(0.0, seg["start"] + shift) * SAMPLE_RATE)
            b = int(max(0.0, seg["end"] + shift) * SAMPLE_RATE)
            clip = audio[which][a:b]
            if len(clip) < MIN_SEGMENT_SEC * SAMPLE_RATE * 0.9:
                vectors.append(None)
                failed += 1
                continue
            wave = torch.from_numpy(clip.astype(np.float32) / 32768.0)[None]
            try:
                emb = np.asarray(inference({"waveform": wave, "sample_rate": SAMPLE_RATE})).reshape(-1)
                emb = (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)
                vectors.append(emb if np.isfinite(emb).all() else None)
            except Exception:
                vectors.append(None)
            if vectors[-1] is None:
                failed += 1
        q.put({"type": "result", "ok": True, "vectors": vectors, "device": device,
               "load_sec": round(load_sec, 2), "embed_sec": round(time.time() - t1, 2),
               "failed": failed})
    except Exception as exc:
        try:
            q.put({"type": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
        except Exception:
            pass


def embed_segments(source: dict, plans: list[SegmentPlan], *, device: str = "",
                   timeout_seconds: int = EMBED_TIMEOUT_SECONDS,
                   model_path: Path | str = MODEL_PATH) -> tuple[Optional[list], dict]:
    """Embed `plans` in a worker process. Returns (vectors | None, info).

    A vector is None for a segment that could not be embedded. The whole result
    is None on any failure (model missing, worker crash, timeout); never raises.
    """
    info: dict = {"ok": False}
    if not plans:
        return [], {"ok": True, "segments": 0}
    ok, why = embedding_available()
    if not ok:
        return None, {"ok": False, "error": why or "embedding unavailable"}
    proc = q = None
    try:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        args = {"source": source, "model_path": str(model_path), "device": device,
                "segments": [{"role": p.role, "start": p.start, "end": p.end} for p in plans]}
        proc = ctx.Process(target=_embed_child, args=(args, q))
        proc.start()
        deadline = time.time() + timeout_seconds
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"voice embedding exceeded {timeout_seconds}s")
            try:
                msg = q.get(timeout=0.25)
            except pyqueue.Empty:
                if not proc.is_alive():
                    raise RuntimeError(f"voice embedding worker exited (code {proc.exitcode})")
                continue
            if isinstance(msg, dict) and msg.get("type") == "result":
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error") or "voice embedding failed")
                info = {k: msg.get(k) for k in ("device", "load_sec", "embed_sec", "failed")}
                info.update(ok=True, segments=len(plans))
                return msg["vectors"], info
    except Exception as exc:
        logger.warning("voice embedding failed: %s", exc)
        return None, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        if proc is not None:
            try:
                proc.join(timeout=0.2)
                if proc.is_alive():
                    proc.terminate()
                proc.close()
            except Exception:
                pass
        if q is not None:
            try:
                q.close()
            except Exception:
                pass


# ===========================================================================
# Label embeddings + extraction
# ===========================================================================

@dataclass
class LabelEmbeddings:
    label: str
    role: str                        # "remote" | "host"
    vectors: np.ndarray              # (n, dim), unit rows
    plans: list[SegmentPlan] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def seconds(self) -> float:
        return round(sum(p.end - p.start for p in self.plans), 1)

    def centroid(self) -> np.ndarray:
        return _unit(self.vectors.mean(axis=0))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def extract_label_embeddings(transcript: str, source: dict, remote: Optional[RemoteActivity], *,
                             duration_sec: Optional[float],
                             participants: Optional[Iterable[str]] = None,
                             labels: Optional[Iterable[str]] = None,
                             roles: tuple[str, ...] = ("remote", "host"),
                             device: str = "",
                             timeout_seconds: int = EMBED_TIMEOUT_SECONDS,
                             ) -> tuple[dict[tuple[str, str], LabelEmbeddings], dict]:
    """Turns + system-track activity + audio -> embeddings per (label, role).

    Works from the watcher's 3-channel WAV or the archived mic/sys tracks
    (`source`). Returns ({}, info{ok: False, ...}) on any failure.
    """
    plans = plan_segments(transcript, remote, duration_sec=duration_sec,
                          participants=participants, labels=labels, roles=roles)
    info: dict = {"ok": True, "planned": len(plans)}
    if not plans:
        return {}, info
    vectors, einfo = embed_segments(source, plans, device=device, timeout_seconds=timeout_seconds)
    info.update(einfo)
    if vectors is None:
        info["ok"] = False
        return {}, info
    grouped: dict[tuple[str, str], tuple[list, list]] = {}
    for plan, vec in zip(plans, vectors):
        if vec is None:
            continue
        vs, ps = grouped.setdefault((plan.label, plan.role), ([], []))
        vs.append(np.asarray(vec, dtype=np.float32))
        ps.append(plan)
    out = {k: LabelEmbeddings(k[0], k[1], np.vstack(vs), ps) for k, (vs, ps) in grouped.items()}
    return out, info


# ===========================================================================
# Matching (pure, numpy only)
# ===========================================================================

def _person_scores(centroid: np.ndarray, profiles: list[Profile]) -> list[tuple[float, Profile]]:
    """Per person: best similarity over their per-meeting centroids."""
    out = []
    for p in profiles:
        out.append((float((p.centroids @ centroid).max()), p))
    out.sort(key=lambda x: -x[0])
    return out


def _person_ref(score: float, p: Profile) -> dict:
    ref = {"name": p.name, "score": round(score, 3), "role": p.role}
    if p.crm_contact_id is not None:
        ref["crm_contact_id"] = p.crm_contact_id
    return ref


def match_label(emb: LabelEmbeddings, profiles: list[Profile]) -> dict:
    """Verdict for one non-host label. Names and scores only."""
    entry: dict = {"segments": emb.n, "seconds": emb.seconds, "best": None,
                   "runner_up": None, "verdict": "none", "reason": None}
    if not profiles or emb.n == 0:
        entry["reason"] = "empty_library" if not profiles else "no_segments"
        return entry
    ranked = _person_scores(emb.centroid(), profiles)
    best_score, best = ranked[0]
    entry["best"] = _person_ref(best_score, best)
    if len(ranked) > 1:
        entry["runner_up"] = _person_ref(*ranked[1])
    margin = best_score - (ranked[1][0] if len(ranked) > 1 else -1.0)
    entry["margin"] = round(margin, 3) if len(ranked) > 1 else None

    if best_score < POSSIBLE_MIN_SCORE:
        entry["reason"] = "below_possible_threshold"
    elif emb.n < MIN_LABEL_SEGMENTS:
        entry.update(verdict="possible", reason="too_few_segments")
    elif best_score < CLEAR_MIN_SCORE:
        entry.update(verdict="possible", reason="score_below_clear_threshold")
    elif len(ranked) > 1 and margin < CLEAR_MIN_MARGIN:
        entry.update(verdict="possible", reason="margin_below_threshold")
    else:
        entry.update(verdict="clear", reason=None)

    # Segment votes: do this label's segments agree on one person?
    votes: dict[str, int] = {}
    for vec in emb.vectors:
        sc = [(float((p.centroids @ vec).max()), p) for p in profiles]
        s, p = max(sc, key=lambda x: x[0])
        if s >= SPLIT_MIN_SEGMENT_SCORE:
            votes[p.name] = votes.get(p.name, 0) + 1
    entry["segment_votes"] = dict(sorted(votes.items(), key=lambda kv: -kv[1]))
    return entry


def match_labels(label_embs: dict[str, LabelEmbeddings], profiles: list[Profile]) -> dict[str, dict]:
    return {label: match_label(e, profiles) for label, e in label_embs.items()}


def _names_agree(label: str, person: str) -> bool:
    a, b = set(normalise_name(label).split()), set(normalise_name(person).split())
    return bool(a and b and (a <= b or b <= a))


def find_conflicts(matches: dict[str, dict]) -> list[dict]:
    """Report-only. Two labels -> one person; one label split between two
    people; a remote label matching the host; a named label whose voice
    belongs to someone else."""
    out: list[dict] = []
    clear = {l: m for l, m in matches.items() if m["verdict"] == "clear" and m["best"]}
    by_person: dict[str, list[str]] = {}
    for label, m in clear.items():
        by_person.setdefault(m["best"]["name"], []).append(label)
    for person, labels in by_person.items():
        if len(labels) > 1:
            out.append({"type": "same_person_two_labels", "person": person, "labels": sorted(labels)})
    for label, m in matches.items():
        votes = m.get("segment_votes") or {}
        total = sum(votes.values())
        strong = {p: n for p, n in votes.items()
                  if n >= SPLIT_MIN_VOTES and total and n / total >= SPLIT_MIN_VOTE_SHARE}
        if len(strong) >= 2:
            out.append({"type": "label_split_between_people", "label": label, "votes": strong})
    for label, m in clear.items():
        if m["best"].get("role") == ROLE_SELF:
            out.append({"type": "remote_label_matches_host", "label": label,
                        "score": m["best"]["score"]})
        elif not is_generic_label(label) and not _names_agree(label, m["best"]["name"]):
            out.append({"type": "name_disagrees_with_voice", "label": label,
                        "voice_name": m["best"]["name"], "score": m["best"]["score"]})
    return out


def host_check(host: Optional[LabelEmbeddings], self_profile: Optional[Profile]) -> dict:
    """Do the host-labelled mic-only turns sound like the self profile?"""
    out: dict = {"ran": False, "reason": None, "segments_checked": 0,
                 "centroid_similarity": None, "suspicious": []}
    if host is None or host.n == 0:
        out["reason"] = "no_host_segments"
        return out
    if self_profile is None:
        out["reason"] = "no_self_profile"
        return out
    out.update(ran=True, segments_checked=host.n)
    centroid_sim = float((self_profile.centroids @ host.centroid()).max())
    out["centroid_similarity"] = round(centroid_sim, 3)
    if centroid_sim < HOST_CENTROID_MISMATCH_MAX_SIM:
        out["reason"] = "host_voice_does_not_match_self_profile"
    for plan, vec in zip(host.plans, host.vectors):
        sim = float((self_profile.centroids @ vec).max())
        if sim < HOST_SUSPICIOUS_MAX_SIM:
            out["suspicious"].append({"line": plan.line, "time": _fmt_ts(plan.start),
                                      "label": plan.label, "similarity": round(sim, 3)})
    return out


def host_enrolment(host: Optional[LabelEmbeddings], self_profile: Optional[Profile]
                   ) -> tuple[Optional[np.ndarray], str]:
    """The host centroid to add to the self profile, or (None, reason)."""
    if host is None or host.n < MIN_HOST_ENROL_SEGMENTS:
        return None, "too_few_host_segments"
    c = host.centroid()
    if float((host.vectors @ c).mean()) < HOST_ENROL_MIN_COHERENCE:
        return None, "host_segments_not_coherent"
    if self_profile is not None and float((self_profile.centroids @ c).max()) < HOST_CENTROID_MISMATCH_MAX_SIM:
        return None, "host_voice_does_not_match_self_profile"
    return c, "ok"


def plan_renames(transcript_labels: Iterable[str], matches: dict[str, dict],
                 conflicts: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """auto_apply: GENERIC labels with a clear match to a non-host person, only
    when the meeting has no conflict, never onto a name another label already
    carries, never two labels onto one name."""
    present = set(transcript_labels)
    if conflicts:
        return {}, [{"label": l, "skipped": "meeting_has_conflicts"}
                    for l, m in matches.items() if m["verdict"] == "clear" and is_generic_label(l)]
    rename: dict[str, str] = {}
    log: list[dict] = []
    targets: dict[str, str] = {}
    for label, m in matches.items():
        if m["verdict"] != "clear" or not m["best"] or m["best"].get("role") == ROLE_SELF:
            continue
        if not is_generic_label(label):
            continue                                 # never touch a real name
        name = m["best"]["name"]
        if any(normalise_name(name) == normalise_name(p) for p in present if p != label):
            log.append({"label": label, "skipped": "name_already_in_transcript", "to": name})
            continue
        if normalise_name(name) in targets:
            log.append({"label": label, "skipped": "two_labels_one_name", "to": name})
            continue
        targets[normalise_name(name)] = label
        rename[label] = name
        log.append({"label": label, "to": name, "score": m["best"]["score"],
                    "margin": m.get("margin")})
    return rename, log


# ===========================================================================
# Orchestration for one recording (watcher)
# ===========================================================================

def _empty_log(reason: Optional[str], ran: bool = False, ok: bool = False) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "ran": ran, "ok": ok, "reason": reason,
        "model": MODEL_NAME,
        "thresholds": {
            "clear_min_score": CLEAR_MIN_SCORE, "clear_min_margin": CLEAR_MIN_MARGIN,
            "possible_min_score": POSSIBLE_MIN_SCORE, "min_label_segments": MIN_LABEL_SEGMENTS,
            "min_turn_sec": MIN_TURN_SEC, "remote_min_system_share": REMOTE_MIN_SYSTEM_SHARE,
            "host_max_system_share": HOST_MAX_SYSTEM_SHARE,
        },
        "labels": {}, "host_check": None, "conflicts": [], "renames": [],
        "self_enrolment": None, "auto_apply": False, "library_people": 0,
        "runtime_seconds": None,
    }


def voice_config(config: Optional[dict]) -> dict:
    """`voice_matching` block with safe defaults when keys are absent."""
    raw = (config or {}).get("voice_matching")
    raw = raw if isinstance(raw, dict) else {}
    return {"enabled": bool(raw.get("enabled", True)),
            "store_others": bool(raw.get("store_others", True)),
            "auto_apply": bool(raw.get("auto_apply", False)),
            "device": str(raw.get("device") or ""),
            "timeout_seconds": int(raw.get("timeout_seconds") or EMBED_TIMEOUT_SECONDS)}


def process(transcript: str, **kw) -> tuple[dict, dict[str, str]]:
    """See `_process`. Adds the runtime to the log; never raises."""
    t_start = time.time()
    try:
        log, rename = _process(transcript, **kw)
    except Exception as exc:                          # degrade, never block
        logger.warning("voice_match failed: %s", exc, exc_info=True)
        log, rename = _empty_log(f"error: {type(exc).__name__}: {exc}"), {}
    log["runtime_seconds"] = round(time.time() - t_start, 1)
    return log, rename


def _process(transcript: str, *, source: Optional[dict], remote: Optional[RemoteActivity],
            duration_sec: Optional[float], participants: Optional[Iterable[str]], stem: str,
            config: Optional[dict], host_verified: bool = False,
            library: Optional[VoiceLibrary] = None) -> tuple[dict, dict[str, str]]:
    """One recording: embed, match, check, optionally enrol the host.

    Returns (log, rename_map). The transcript is never edited here; the rename
    map is empty unless `auto_apply` and a clear, conflict-free match exist.
    """
    log = _empty_log(None)
    cfg = voice_config(config)
    log["auto_apply"] = cfg["auto_apply"]
    if not cfg["enabled"]:
        log["reason"] = "disabled_in_config"
        return log, {}
    if source is None or remote is None:
        log["reason"] = "single_source_or_no_system_track"
        return log, {}
    if not isinstance(transcript, str) or not transcript.strip():
        log["reason"] = "empty_transcript"
        return log, {}
    lib = library or VoiceLibrary(default_library_dir(config))
    try:
        profiles = lib.profiles(exclude_stem=stem)
    except LibraryError as exc:
        log["reason"] = f"library_unreadable: {exc}"
        return log, {}
    self_profile = next((p for p in profiles if p.role == ROLE_SELF), None)
    others = [p for p in profiles if p.role != ROLE_SELF]
    log["library_people"] = len(profiles)

    roles = ("host", "remote") if others else ("host",)
    embs, info = extract_label_embeddings(
        transcript, source, remote, duration_sec=duration_sec, participants=participants,
        roles=roles, device=cfg["device"], timeout_seconds=cfg["timeout_seconds"])
    log["embedding"] = {k: v for k, v in info.items() if k != "error"}
    if not info.get("ok"):
        log["reason"] = f"embedding_failed: {info.get('error')}"
        return log, {}
    log["ran"], log["ok"] = True, True
    if not others:
        log["reason"] = "library_has_no_other_people"

    remote_embs = {l: e for (l, r), e in embs.items() if r == "remote"}
    host_emb = next((e for (l, r), e in embs.items() if r == "host"), None)
    if len([1 for (l, r) in embs if r == "host"]) > 1:      # two host-style labels: pool them
        hosts = [e for (l, r), e in embs.items() if r == "host"]
        host_emb = LabelEmbeddings(hosts[0].label, "host", np.vstack([h.vectors for h in hosts]),
                                   [p for h in hosts for p in h.plans])

    matches = match_labels(remote_embs, others + ([self_profile] if self_profile else []))
    log["labels"] = matches
    log["host_check"] = host_check(host_emb, self_profile)
    log["conflicts"] = find_conflicts(matches)

    rename: dict[str, str] = {}
    if cfg["auto_apply"]:
        labels_in_text = {t["label"] for t in _parse_turns(transcript.split("\n"), participants, duration_sec)}
        rename, log["renames"] = plan_renames(labels_in_text, matches, log["conflicts"])

    # Self enrolment: only where the system-track stage verified the host
    # label, and only a coherent centroid that fits an existing profile.
    if host_verified:
        vec, why = host_enrolment(host_emb, self_profile)
        enrol = {"enrolled": False, "reason": why}
        if vec is not None:
            try:
                res = lib.add_sample(SELF_NAME, vec, stem=stem, segments=host_emb.n,
                                     basis=BASIS_SELF, role=ROLE_SELF)
                enrol = {"enrolled": True, "reason": "ok", "segments": host_emb.n,
                         "samples": res["samples"]}
            except Exception as exc:
                enrol = {"enrolled": False, "reason": f"library_write_failed: {type(exc).__name__}"}
        log["self_enrolment"] = enrol
    else:
        log["self_enrolment"] = {"enrolled": False, "reason": "host_label_not_verified"}
    return log, rename


# ===========================================================================
# Enrolment from a human confirmation (speaker_review.confirm, backfill)
# ===========================================================================

def enrol_people(transcript: str, people: dict[str, dict], *, stem: str, base: Path | str,
                 duration_sec: Optional[float], library: VoiceLibrary, mic_offset_sec: float = 0.0,
                 participants: Optional[Iterable[str]] = None, device: str = "",
                 source: Optional[dict] = None,
                 timeout_seconds: int = EMBED_TIMEOUT_SECONDS) -> dict:
    """Enrol confirmed non-host people from their system-track segments.

    `people`: transcript label (as written in `transcript`) -> {"name", optional
    "crm_contact_id"}. Never raises; returns a report with names and counts.
    """
    report: dict = {"enrolled": [], "skipped": [], "reason": None}
    try:
        src = source or locate_source(base, stem, mic_offset_sec, need_mic=False)
        if src is None:
            report["reason"] = "archived tracks no longer on disk"
            return report
        remote = remote_activity_for_source(src)
        known = list(participants or []) + list(people)   # a confirmed name seen once must still count
        embs, info = extract_label_embeddings(
            transcript, src, remote, duration_sec=duration_sec, participants=known,
            labels=list(people), roles=("remote",), device=device, timeout_seconds=timeout_seconds)
        if not info.get("ok"):
            report["reason"] = f"embedding_failed: {info.get('error')}"
            return report
        try:
            profiles = library.profiles(exclude_stem=stem)
        except LibraryError:
            profiles = []
        for label, who in people.items():
            name = who["name"]
            emb = embs.get((label, "remote"))
            if emb is None or emb.n < MIN_ENROL_SEGMENTS:
                report["skipped"].append({"name": name, "reason": "too_few_system_track_segments"})
                continue
            # A confirmation that disagrees with a different person's clear
            # voice match is not enrolled: a wrong sample would poison the
            # profile for every later meeting.
            cid, norm = who.get("crm_contact_id"), normalise_name(name)
            others = [p for p in profiles
                      if not ((cid is not None and p.crm_contact_id == cid) or normalise_name(p.name) == norm)]
            m = match_label(emb, others)
            if m["verdict"] == "clear":
                report["skipped"].append({"name": name, "reason": f"voice_matches_{m['best']['name']}_clearly",
                                          "score": m["best"]["score"]})
                continue
            try:
                res = library.add_sample(name, emb.centroid(), stem=stem, segments=emb.n,
                                         basis=BASIS_HUMAN, role=ROLE_OTHER,
                                         crm_contact_id=who.get("crm_contact_id"))
                report["enrolled"].append({"name": name, "segments": emb.n, "created": res["created"]})
            except (PersonConflict, ValueError, LibraryError) as exc:
                report["skipped"].append({"name": name, "reason": f"{type(exc).__name__}: {exc}"[:120]})
        return report
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        return report


# ===========================================================================
# CLI: evaluate (leave-one-meeting-out) and enrol-confirmed
# ===========================================================================

DEFAULT_BASE = Path.home() / "Documents" / "MeetingRecorder"


def _load_meeting(base: Path, stem: str) -> Optional[dict]:
    tr_path = base / "Transcripts" / f"{stem}.json"
    if not tr_path.exists():
        return None
    tr = json.loads(tr_path.read_text())
    meta = tr.get("_meta") or {}
    lag = (meta.get("channel_alignment") or {}).get("lag_seconds") or 0.0
    return {"stem": stem, "transcript": tr.get("transcript") or "", "meta": meta,
            "participants": [p.get("name") for p in tr.get("participants") or [] if isinstance(p, dict)],
            "duration": meta.get("audio_duration_seconds"), "lag": float(lag)}


def _synthetic_library(root: Path, stems_embs: dict[str, dict]) -> VoiceLibrary:
    """Temp library: each (meeting, remote label) is its own person; every
    meeting's host centroid feeds the self profile."""
    lib = VoiceLibrary(root)
    for stem, embs in stems_embs.items():
        for (label, role), e in embs.items():
            if role == "host":
                if e.n >= MIN_HOST_ENROL_SEGMENTS:
                    lib.add_sample(SELF_NAME, e.centroid(), stem=stem, segments=e.n,
                                   basis=BASIS_SELF, role=ROLE_SELF)
            elif e.n >= MIN_LABEL_SEGMENTS:
                lib.add_sample(f"{stem[:10]} {label}", e.centroid(), stem=stem, segments=e.n,
                               basis=BASIS_HUMAN, role=ROLE_OTHER)
    return lib


def evaluate(base: Path, stems: list[str], library_dir: Path, *, device: str = "") -> int:
    print(f"extracting embeddings for {len(stems)} recording(s) (this runs the local model)")
    meetings: dict[str, dict] = {}
    embs: dict[str, dict] = {}
    runtime: dict[str, float] = {}
    for stem in stems:
        m = _load_meeting(base, stem)
        src = locate_source(base, stem, (m or {}).get("lag", 0.0)) if m else None
        if not m or not src:
            print(f"  {stem}: skipped (no transcript or no tracks)")
            continue
        t0 = time.time()
        remote = remote_activity_for_source(src)
        e, info = extract_label_embeddings(m["transcript"], src, remote, duration_sec=m["duration"],
                                           participants=m["participants"], device=device)
        runtime[stem] = round(time.time() - t0, 1)
        if not info.get("ok"):
            print(f"  {stem}: FAILED {info.get('error')}")
            continue
        meetings[stem], embs[stem] = m, e
        print(f"  {stem}: {runtime[stem]:>5}s  "
              + ", ".join(f"{l}/{r[0]}={v.n}" for (l, r), v in sorted(embs[stem].items())))
    if len(embs) < 2:
        print("need at least two recordings")
        return 1

    lib = _synthetic_library(library_dir, embs)
    print(f"\ntemp library: {lib.stats()['people']} people, {lib.stats()['samples']} samples ({lib.root})\n")
    rows, false_clear = [], []
    host_rows = []
    for stem, e in embs.items():
        profiles = lib.profiles(exclude_stem=stem)
        self_p = next((p for p in profiles if p.role == ROLE_SELF), None)
        others = [p for p in profiles if p.role != ROLE_SELF]
        remote_embs = {l: v for (l, r), v in e.items() if r == "remote"}
        matches = match_labels(remote_embs, others + ([self_p] if self_p else []))
        conflicts = find_conflicts(matches)
        for label, mt in matches.items():
            b, r = mt["best"], mt["runner_up"]
            rows.append((stem, label, mt["segments"], b, r, mt["verdict"], mt.get("reason")))
            if mt["verdict"] == "clear" and b:
                src_label = b["name"].split(" ", 1)[1] if b["role"] != ROLE_SELF else None
                if (src_label and not is_generic_label(label) and not is_generic_label(src_label)
                        and not _names_agree(label, src_label)):
                    false_clear.append((stem, label, b["name"], b["score"]))
        host = next((v for (l, r_), v in e.items() if r_ == "host"), None)
        hc = host_check(host, self_p)
        coh = round(float((host.vectors @ host.centroid()).mean()), 2) if host and host.n else None
        host_rows.append((stem, host.n if host else 0, hc, coh))
        for c in conflicts:
            print(f"  conflict in {stem}: {c}")

    print(f"{'recording':<21}{'label':<16}{'n':>3}  {'best match (score)':<40}{'runner-up (score)':<36}verdict")
    for stem, label, n, b, r, verdict, reason in rows:
        bs = f"{b['name']} ({b['score']:.2f})" if b else "-"
        rs = f"{r['name']} ({r['score']:.2f})" if r else "-"
        print(f"{stem:<21}{label[:15]:<16}{n:>3}  {bs[:39]:<40}{rs[:35]:<36}{verdict}"
              + (f" [{reason}]" if reason and verdict != "clear" else ""))
    print("\nhost vs self profile (leave-one-meeting-out)")
    for stem, n, hc, coh in host_rows:
        print(f"  {stem}: host segments={n:>2} coherence={coh}  "
              + (f"centroid sim={hc['centroid_similarity']}  suspicious turns={len(hc['suspicious'])}"
                 if hc["ran"] else f"not checked ({hc['reason']})"))
        for sus in hc["suspicious"]:
            print(f"      suspicious: line {sus['line']} at {sus['time']} {sus['label']} sim={sus['similarity']}")
    clear = [r for r in rows if r[5] == "clear"]
    print(f"\nclear matches: {len(clear)}   possible: {sum(1 for r in rows if r[5] == 'possible')}   "
          f"none: {sum(1 for r in rows if r[5] == 'none')}")
    print(f"FALSE clear matches (named label vs differently named person): {len(false_clear)}")
    for f in false_clear:
        print(f"  {f}")
    print("per-recording runtime (decode + embed, seconds): "
          + ", ".join(f"{s}={t}" for s, t in runtime.items()))
    return 0


def _db_contact_ids(transcript_paths: list[str]) -> dict[str, dict[str, int]]:
    """Read-only: transcript_path -> {confirmed name -> CRM contact id} from
    sources.metadata.speaker_review.mapping. Only SELECTs; nothing is written."""
    from speaker_review import _conn  # same connection helper the tools use
    out: dict[str, dict[str, int]] = {}
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata->>'transcript_path', metadata->'speaker_review'->'mapping' "
                        "FROM sources WHERE metadata->>'transcript_path' = ANY(%s) "
                        "AND metadata->'speaker_review'->>'status' = 'confirmed'", (transcript_paths,))
            for path, mapping in cur.fetchall():
                ids = {}
                for e in mapping or []:
                    if isinstance(e, dict) and e.get("kind") == "crm" and e.get("contact_id") is not None:
                        ids[normalise_name(e.get("name") or "")] = int(e["contact_id"])
                out[path] = ids
    finally:
        conn.rollback()
        conn.close()
    return out


def enrol_confirmed(base: Path, library_dir: Path, *, use_db: bool = False, device: str = "") -> int:
    """Enrol people from recordings a human already confirmed and whose tracks
    still exist. Disk-only by default: `_meta.speaker_attribution.status ==
    "human_confirmed"` in the transcript JSON, labels that are real names.
    `--use-db` adds CRM contact ids from the confirmation mapping (SELECT only)."""
    lib = VoiceLibrary(library_dir)
    paths = sorted((base / "Transcripts").glob("*.json"))
    candidates = []
    for p in paths:
        if p.name.count(".") != 1:                  # skip *.pre-speaker-review.json etc.
            continue
        try:
            tr = json.loads(p.read_text())
        except Exception:
            continue
        sa = (tr.get("_meta") or {}).get("speaker_attribution") or {}
        if sa.get("status") != "human_confirmed" or sa.get("identity_basis") != "human":
            continue
        candidates.append((p, tr))
    ids = _db_contact_ids([str(p) for p, _ in candidates]) if use_db and candidates else {}
    total = 0
    for p, tr in candidates:
        stem = p.stem
        m = _load_meeting(base, stem)
        labels = {}
        for lab in {t["label"] for t in _parse_turns(m["transcript"].split("\n"), m["participants"], m["duration"])}:
            if _is_self_label(lab) or is_generic_label(lab) or lab.lower().startswith("unidentified"):
                continue
            who = {"name": lab}
            cid = (ids.get(str(p)) or {}).get(normalise_name(lab))
            if cid is not None:
                who["crm_contact_id"] = cid
            labels[lab] = who
        if not labels:
            print(f"{stem}: no named non-host labels")
            continue
        rep = enrol_people(m["transcript"], labels, stem=stem, base=base, duration_sec=m["duration"],
                           library=lib, mic_offset_sec=m["lag"], participants=m["participants"], device=device)
        total += len(rep["enrolled"])
        print(f"{stem}: enrolled {[e['name'] for e in rep['enrolled']]} "
              f"skipped {rep['skipped']} {('(' + rep['reason'] + ')') if rep['reason'] else ''}")
    print(f"enrolled {total} sample(s) into {lib.root}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="leave-one-meeting-out dry run over archived recordings")
    ev.add_argument("--all", action="store_true")
    ev.add_argument("--stem", action="append", default=[])
    ev.add_argument("--base", type=Path, default=DEFAULT_BASE, help="MeetingRecorder directory (read only)")
    ev.add_argument("--library-dir", type=Path, required=True, help="temp directory for the synthetic library")
    ev.add_argument("--device", default="")
    en = sub.add_parser("enrol-confirmed", help="enrol people from human-confirmed recordings with archived tracks")
    en.add_argument("--base", type=Path, default=DEFAULT_BASE)
    en.add_argument("--library-dir", type=Path, required=True)
    en.add_argument("--use-db", action="store_true", help="read CRM contact ids (SELECT only)")
    en.add_argument("--device", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(levelname)s %(message)s")
    if args.cmd == "evaluate":
        stems = list(args.stem)
        if args.all:
            arch = args.base / "CaptureArchive"
            stems += sorted(p.name for p in arch.iterdir()
                            if (args.base / "Transcripts" / f"{p.name}.json").exists())
        if not stems:
            ap.error("give --all or --stem")
        return evaluate(args.base, stems, args.library_dir, device=args.device)
    return enrol_confirmed(args.base, args.library_dir, use_db=args.use_db, device=args.device)


if __name__ == "__main__":
    sys.exit(main())
