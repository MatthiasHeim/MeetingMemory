#!/usr/bin/env python3
"""speech_trim — don't pay Gemini for audio nobody speaks in.

Recordings where the stop button was forgotten are billed in full: 11.10.
87.8 min at 6 words/min, 15.10. 56.6 min at 15 wpm and 126.2 min at 9 wpm
(normal meetings: 140-200 wpm). A dBFS threshold cannot find these stretches
(loudnorm lifts room tone; silencedetect at -45 / -32 dB found nothing), so
this reads speech activity from channel_vad's floor-relative detector on the
RAW (aligned) WAV, before the mix and loudnorm blur it:

  * a recording with almost no speech is skipped entirely (logged, and an
    explicit no-speech outcome is saved instead of a transcript);
  * no-speech stretches of at least MIN_GAP_SEC are cut out of the MP3 sent
    to Gemini, keeping PAD_SEC on each side;
  * every timestamp Gemini returns is mapped back to the ORIGINAL recording
    (`TrimPlan.to_original`), so system-track attribution, voice matching and
    live fill keep reading the original WAV with consistent times.

Measured on 2026-11-25 (raw WAV): 2026-11-24_16-00-15, 11.5 min with 0.4 min
of speech -> skipped; 2026-09-14_15-59-56, 69.4 min with a 14-min empty
stretch -> about 16 min cut; normal meetings (41-60 min) -> nothing cut.
"""
from __future__ import annotations

import bisect
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"

MIN_GAP_SEC = 150.0        # cut only no-speech stretches at least this long
PAD_SEC = 10.0             # keep this much audio on each side of a cut
MIN_TOTAL_CUT_SEC = 60.0   # below this, re-encoding isn't worth it
# Skip only when BOTH hold. Losing a real meeting costs far more than the few
# cents a near-empty call costs, so this stays deliberately conservative.
SKIP_MAX_SPEECH_SEC = 30.0
SKIP_MAX_SPEECH_SHARE = 0.05
# ...and the recording must also be QUIET: every channel's 95th-percentile
# speech-band level below this. The floor-relative detector reads "no speech"
# whenever nothing stands 10 dB above the floor, which is also what wall-to-
# wall talk or a uniformly noisy room looks like. Measured 2026-11-25 (raw
# WAV, p95): near-empty 2026-11-24_16-00-15 -62 dBFS; real meetings -16 to
# -20; in-room meeting with silent system channels -40.
SKIP_MAX_P95_DB = -50.0

_TS_RE = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2})\]")


@dataclass
class TrimPlan:
    """What to send to Gemini. `keep` lists (start, end) in original seconds."""

    duration_sec: float
    speech_sec: float
    keep: list[tuple[float, float]] = field(default_factory=list)
    skip: bool = False
    reason: str = ""

    @property
    def trimmed(self) -> bool:
        return not self.skip and self.kept_sec < self.duration_sec - 1e-6

    @property
    def kept_sec(self) -> float:
        return sum(e - s for s, e in self.keep)

    @property
    def cut_sec(self) -> float:
        return self.duration_sec - self.kept_sec

    @property
    def effective_end(self) -> float:
        """Original time where the last kept audio ends: the validator's
        coverage target, since a trailing empty stretch has no transcript."""
        return self.keep[-1][1] if self.keep else 0.0

    def _starts(self) -> list[float]:
        out, acc = [], 0.0
        for s, e in self.keep:
            out.append(acc)
            acc += e - s
        return out

    def to_original(self, t: float) -> float:
        """Map a time in the trimmed audio to the original recording."""
        if not self.trimmed:
            return t
        starts = self._starts()
        i = max(0, bisect.bisect_right(starts, t) - 1)
        s, e = self.keep[i]
        return min(s + (t - starts[i]), e) if i < len(self.keep) - 1 else s + (t - starts[i])

    def to_trimmed_segments(self, segments: Optional[list]) -> Optional[list]:
        """Map original-time segments [(t0, t1, ...rest)] or dicts with
        start/end into trimmed time, clipping away the cut stretches."""
        if not segments or not self.trimmed:
            return segments
        starts = self._starts()
        out = []
        for seg in segments:
            is_dict = isinstance(seg, dict)
            t0, t1 = (seg["start"], seg["end"]) if is_dict else (seg[0], seg[1])
            for (s, e), ts in zip(self.keep, starts):
                a, b = max(t0, s), min(t1, e)
                if b <= a:
                    continue
                n0, n1 = ts + (a - s), ts + (b - s)
                if is_dict:
                    out.append(dict(seg, start=n0, end=n1))
                else:
                    out.append((n0, n1, *seg[2:]))
        return out

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 1),
            "speech_sec": round(self.speech_sec, 1),
            "kept_sec": round(self.kept_sec, 1),
            "cut_sec": round(self.cut_sec, 1),
            "skip": self.skip,
            "reason": self.reason,
            "keep": [[round(s, 2), round(e, 2)] for s, e in self.keep],
            "params": {"min_gap_sec": MIN_GAP_SEC, "pad_sec": PAD_SEC,
                       "skip_max_speech_sec": SKIP_MAX_SPEECH_SEC,
                       "skip_max_speech_share": SKIP_MAX_SPEECH_SHARE,
                       "skip_max_p95_db": SKIP_MAX_P95_DB},
        }


def plan_trim(active, window_sec: float, duration_sec: Optional[float] = None,
              p95_db: Optional[float] = None) -> TrimPlan:
    """Build a TrimPlan from a per-window boolean speech array.

    `p95_db` (loudest channel's 95th-percentile level) must confirm a skip;
    when it is unknown or loud, "no speech detected" is inconclusive and the
    whole recording is sent untouched."""
    n = len(active)
    duration = float(duration_sec if duration_sec is not None else n * window_sec)
    speech = float(sum(bool(x) for x in active)) * window_sec
    if speech < SKIP_MAX_SPEECH_SEC and speech < SKIP_MAX_SPEECH_SHARE * duration:
        if p95_db is not None and p95_db < SKIP_MAX_P95_DB:
            return TrimPlan(duration, speech, keep=[], skip=True,
                            reason=f"almost no speech: {speech:.0f}s in "
                                   f"{duration / 60:.1f} min, p95 level {p95_db:.0f} dBFS")
        level = "unknown" if p95_db is None else f"{p95_db:.0f} dBFS"
        return TrimPlan(duration, speech, keep=[(0.0, duration)],
                        reason=f"detector inconclusive: {speech:.0f}s speech detected "
                               f"but level {level}; sending everything")

    # No-speech runs long enough to cut, shrunk by PAD_SEC at each side. The
    # recording's own start/end need no pad on their outer side.
    cuts: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if active[i]:
            i += 1
            continue
        j = i
        while j < n and not active[j]:
            j += 1
        s, e = i * window_sec, min(j * window_sec, duration)
        if j >= n:
            e = duration
        if e - s >= MIN_GAP_SEC:
            cs = s if i == 0 else s + PAD_SEC
            ce = e if j >= n else e - PAD_SEC
            if ce - cs > 0:
                cuts.append((cs, ce))
        i = j

    if sum(e - s for s, e in cuts) < MIN_TOTAL_CUT_SEC:
        return TrimPlan(duration, speech, keep=[(0.0, duration)], reason="nothing to cut")

    keep, cursor = [], 0.0
    for cs, ce in cuts:
        if cs > cursor:
            keep.append((cursor, cs))
        cursor = ce
    if cursor < duration:
        keep.append((cursor, duration))
    return TrimPlan(duration, speech, keep=keep,
                    reason=f"cut {len(cuts)} no-speech stretch(es) >= {MIN_GAP_SEC:.0f}s")


def plan_from_activity(activity) -> Optional[TrimPlan]:
    """TrimPlan from a channel_vad.SpeechActivity (None stays None)."""
    if activity is None:
        return None
    return plan_trim(activity.active, activity.window_sec, activity.duration_sec,
                     p95_db=getattr(activity, "p95_db", None))


def render_trimmed_mp3(mp3_path: Path, plan: TrimPlan, out_path: Path) -> Path:
    """Concatenate the kept spans of the (already loudness-normalised) MP3.
    Same encoder settings as audio_converter._run_loudnorm_mp3, no second
    loudnorm pass."""
    parts, labels = [], []
    for k, (s, e) in enumerate(plan.keep):
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{k}]")
        labels.append(f"[a{k}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"
    cmd = [FFMPEG_PATH, "-v", "error", "-y", "-i", str(mp3_path),
           "-filter_complex", graph, "-map", "[out]",
           "-codec:a", "libmp3lame", "-qscale:a", "2", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {r.stderr[:300]}")
    return out_path


def _fmt(sec: float) -> str:
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, ss = divmod(rem, 60)
    return f"[{h:02d}:{m:02d}:{ss:02d}]" if h else f"[{m:02d}:{ss:02d}]"


def _parse(ts: str) -> int:
    return sum(int(v) * 60 ** i for i, v in enumerate(reversed(ts.split(":"))))


def remap_text(text: str, plan: TrimPlan) -> str:
    if not text or not plan.trimmed:
        return text
    return _TS_RE.sub(lambda m: _fmt(plan.to_original(_parse(m.group(1)))), text)


def _remap_events(events, plan: TrimPlan):
    out = []
    for ev in events or []:
        if isinstance(ev, dict) and isinstance(ev.get("time"), str):
            ev = dict(ev, time=remap_text(ev["time"], plan))
        out.append(ev)
    return out


def remap_result(result, plan: TrimPlan) -> None:
    """In place: move every timestamp in a GeminiResult from trimmed time
    back to the original recording and record the trim in the result."""
    if result is None or not plan.trimmed:
        return
    result.transcript = remap_text(result.transcript, plan)
    for entry in result.speaker_emotions or []:
        if isinstance(entry, dict) and "arc" in entry:
            entry["arc"] = _remap_events(entry["arc"], plan)
    result.interruptions = _remap_events(result.interruptions, plan)
    for data in (result.energy_levels or {}).values():
        if isinstance(data, dict) and "arc" in data:
            data["arc"] = _remap_events(data["arc"], plan)
    result.missing_time_ranges = [
        (plan.to_original(s), plan.to_original(e)) for s, e in result.missing_time_ranges]
    result.audio_duration_seconds = plan.duration_sec


class TimeMappedProcessor:
    """Wraps a GeminiAudioProcessor for one trimmed recording.

    Every call goes to the trimmed MP3 with segments moved into trimmed time,
    and every result comes back in ORIGINAL time. The watcher's escalation
    ladder can use it exactly like the real processor."""

    def __init__(self, inner, plan: TrimPlan, trimmed_mp3: Path):
        self._inner = inner
        self._plan = plan
        self._mp3 = Path(trimmed_mp3)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _kw(self, kw: dict) -> dict:
        kw = dict(kw)
        for key in ("channel_segments", "diarization_segments"):
            if kw.get(key):
                kw[key] = self._plan.to_trimmed_segments(kw[key])
        return kw

    def _done(self, result):
        remap_result(result, self._plan)
        return result

    def process_audio(self, audio_path, custom_prompt=None, **kw):
        return self._done(self._inner.process_audio(self._mp3, custom_prompt, **self._kw(kw)))

    def _process_single_shot(self, audio_path, custom_prompt, total_duration, **kw):
        return self._done(self._inner._process_single_shot(
            self._mp3, custom_prompt, self._plan.kept_sec, **self._kw(kw)))

    def _process_chunked(self, audio_path, custom_prompt, total_duration, **kw):
        return self._done(self._inner._process_chunked(
            self._mp3, custom_prompt, self._plan.kept_sec, **self._kw(kw)))
