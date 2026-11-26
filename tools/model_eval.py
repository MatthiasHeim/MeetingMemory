#!/usr/bin/env python3
"""Transcription model eval on real meetings, through the production path.

Runs each candidate (model + thinking level) over existing recordings with
exactly what the watcher sends: the pipeline's own converted MP3
(Transcripts/<stem>.mp3), the calendar attendees (calendar_resolve, read
only), the channel map when the WAV still exists and is admissible, the same
single-shot / chunked routing, retries, and the watcher's validator.

Nothing is written to InsightBase, Telegram or Transcripts/. Full candidate
transcripts stay in --out (default ~/.local/share/meetingmemory-model-eval/),
outside Git: meeting content is confidential. Only aggregate numbers belong in
docs/model-eval-2026-11.md.

    venv/bin/python tools/model_eval.py run \\
        --stems 2026-11-05_17-01-23 2026-11-18_16-31-24 \\
        --candidates gemini-3.6-flash:minimal gemini-3.8-flash:low --jobs 4
    venv/bin/python tools/model_eval.py judge --run-dir <dir> --windows 5
    venv/bin/python tools/model_eval.py report --run-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

try:
    from dotenv import load_dotenv
    load_dotenv(TOOLS.parent / ".env")
except ImportError:
    pass

TRANSCRIPTS = Path.home() / "Documents/MeetingRecorder/Transcripts"
RECORDINGS = Path.home() / "Documents/MeetingRecorder/Recordings"
DEFAULT_OUT = Path.home() / ".local/share/meetingmemory-model-eval"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# USD per 1M tokens, Standard tier: (audio/text input, output incl. thinking).
# Verified on aistudio.google.com/docs/pricing 2026-11-25. Input is priced at
# the audio rate throughout (the prompt text is a few k tokens; conservative).
PRICES_NOW = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-4-neon": (3.00, 15.00),
    "gemini-4-carbon": (2.00, 10.00),
    "gemini-3-flash-preview": (1.00, 3.00),
    "gemini-2.5-pro": (1.25, 10.00),   # last published price; no longer listed
}
# Announced changes: Carbon/Argon double 2026-12-01, 3.6/3.8 Flash 2027-01-01.
PRICES_2027 = dict(PRICES_NOW, **{
    "gemini-3.8-flash": (1.50, 7.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-4-carbon": (4.00, 20.00),
})

TS_RE = re.compile(r"^\[(\d{1,2}(?::\d{2}){1,2})\]\s*([^:]{1,60}):\s*(.*)$")
GENERIC_RE = re.compile(r"^(speaker|sprecher|person|unknown|unbekannt)\b", re.I)
WORD_RE = re.compile(r"[\wäöüÄÖÜéèàß]+", re.U)

log = logging.getLogger("model_eval")


def ts_to_sec(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def parse_lines(transcript: str) -> list[tuple[int, str, str]]:
    out = []
    for line in (transcript or "").splitlines():
        m = TS_RE.match(line.strip())
        if m:
            out.append((ts_to_sec(m.group(1)), m.group(2).strip(), m.group(3)))
    return out


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def cost(model: str, inp: int, out_and_thinking: int, table: dict) -> Optional[float]:
    if model not in table:
        return None
    pi, po = table[model]
    return inp / 1e6 * pi + out_and_thinking / 1e6 * po


def window_f1(a: list[tuple[int, str, str]], b: list[tuple[int, str, str]],
              duration: float, win: int = 300) -> Optional[float]:
    """Mean bag-of-words F1 per 5-min window between two transcripts.

    Reference-free agreement: where two independent models agree, both are
    probably right; a candidate far from every other one is the outlier."""
    if not a or not b:
        return None
    scores = []
    for start in range(0, int(duration) + 1, win):
        wa = Counter(w for t, _, x in a if start <= t < start + win for w in words(x))
        wb = Counter(w for t, _, x in b if start <= t < start + win for w in words(x))
        if not wa and not wb:
            continue
        inter = sum((wa & wb).values())
        p = inter / max(1, sum(wa.values()))
        r = inter / max(1, sum(wb.values()))
        scores.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return round(sum(scores) / len(scores), 3) if scores else None


def transcript_metrics(transcript: str, duration: float, attendees: list[str]) -> dict:
    lines = parse_lines(transcript)
    n_words = sum(len(words(x)) for _, _, x in lines)
    labels = Counter(s for _, s, _ in lines)
    last = lines[-1][0] if lines else 0
    texts = [x.strip().lower() for _, _, x in lines if len(x.split()) >= 6]
    dup = sum(c - 1 for c in Counter(texts).values() if c > 1)
    att = {a.lower().split()[0] for a in attendees if a}
    named = sum(c for s, c in labels.items()
                if s.lower().split()[0] in att) if att else None
    return {
        "lines": len(lines),
        "words": n_words,
        "wpm": round(n_words / max(duration / 60, 0.01), 1),
        "last_ts_sec": last,
        "end_coverage_pct": round(100 * last / duration, 1) if duration else None,
        "speaker_labels": len(labels),
        "generic_label_share": round(sum(c for s, c in labels.items()
                                         if GENERIC_RE.match(s)) / max(1, len(lines)), 3),
        "attendee_named_share": (round(named / max(1, len(lines)), 3)
                                 if named is not None else None),
        "duplicate_line_rate": round(dup / max(1, len(texts)), 3),
    }


# --------------------------------------------------------------------- run
def _attendees(stem: str) -> list[dict]:
    try:
        import calendar_resolve
        match = calendar_resolve.resolve(TRANSCRIPTS / f"{stem}.json") or {}
        return match.get("participant_details") or []
    except Exception as e:  # calendar is optional context, as in production
        log.warning(f"{stem}: calendar_resolve failed: {e}")
        return []


def _channel_segments(stem: str) -> Optional[list]:
    wav = RECORDINGS / f"{stem}.wav"
    if not wav.exists():
        return None
    try:
        from channel_vad import compute_channel_vad
        vad = compute_channel_vad(wav)
        if vad and vad.separation_report().get("admissible"):
            return vad.segments
    except Exception as e:
        log.warning(f"{stem}: channel_vad failed: {e}")
    return None


def run_one(stem: str, candidate: str, out_dir: Path, context: dict) -> dict:
    from gemini_processor import GeminiAudioProcessor
    from transcribe_watcher import _validate_gemini_result

    model, _, level = candidate.partition(":")
    level = None if level in ("", "default") else level
    dest = out_dir / "runs" / f"{stem}__{model}__{level or 'default'}.json"
    if dest.exists():
        return json.loads(dest.read_text())["summary"]

    mp3 = TRANSCRIPTS / f"{stem}.mp3"
    proc = GeminiAudioProcessor(model=model, thinking_level=level,
                                timeout_seconds=900, allow_deprecated=True)
    duration = proc._get_duration(mp3)
    summary = {"stem": stem, "model": model, "thinking_level": level,
               "duration_sec": duration}
    t0 = time.time()
    result = None
    try:
        result = proc.process_audio(
            mp3, known_attendees=context["attendees"] or None,
            channel_segments=context["channel_segments"])
        validation = _validate_gemini_result(result, duration)
        summary["first_pass_valid"] = validation.passed
        summary["first_pass_reasons"] = validation.reasons
        # Production escalation ladder, minus the alerts: one fresh single
        # call, then forced chunking. Its cost counts against the candidate.
        if not validation.passed:
            for step in ("single", "chunked"):
                if step == "single":
                    retry = proc._process_single_shot(
                        mp3, None, duration,
                        known_attendees=context["attendees"] or None,
                        channel_segments=context["channel_segments"])
                else:
                    retry = proc._process_chunked(
                        mp3, None, duration,
                        known_attendees=context["attendees"] or None,
                        channel_segments=context["channel_segments"],
                        force_chunk=True)
                v = _validate_gemini_result(retry, duration)
                if v.passed or v.coverage_pct > validation.coverage_pct:
                    result, validation = retry, v
                if v.passed:
                    break
        summary["final_valid"] = validation.passed
        summary["final_coverage_pct"] = validation.coverage_pct
        summary["final_reasons"] = validation.reasons
        summary["chunked"] = result.chunked
        summary["missing_time_ranges"] = [list(r) for r in result.missing_time_ranges]
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"[:400]
    summary["latency_sec"] = round(time.time() - t0, 1)

    usage = proc.usage_log
    inp = sum(u["input_tokens"] for u in usage)
    outp = sum(u["output_tokens"] for u in usage)
    think = sum(u["thinking_tokens"] for u in usage)
    summary.update({
        "calls": len(usage),
        "input_tokens": inp, "output_tokens": outp, "thinking_tokens": think,
        "cost_usd": cost(model, inp, outp + think, PRICES_NOW),
        "cost_usd_2027": cost(model, inp, outp + think, PRICES_2027),
    })
    hours = duration / 3600
    for k in ("cost_usd", "cost_usd_2027"):
        if summary[k] is not None:
            summary[k + "_per_hour"] = round(summary[k] / hours, 4)
            summary[k] = round(summary[k], 4)
    if result is not None:
        names = [a.get("name", "") for a in context["attendees"]]
        summary.update(transcript_metrics(result.transcript, duration, names))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "summary": summary,
        "usage_log": usage,
        "result": result.parsed_response if result is not None else None,
    }, ensure_ascii=False, indent=1))
    log.info(f"done {stem} {candidate}: valid={summary.get('final_valid')} "
             f"wpm={summary.get('wpm')} cost=${summary.get('cost_usd')} "
             f"{summary['latency_sec']}s")
    return summary


def cmd_run(args) -> None:
    out_dir = Path(args.run_dir or DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d_%H%M"))
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts = {}
    for stem in args.stems:
        contexts[stem] = {"attendees": _attendees(stem),
                          "channel_segments": _channel_segments(stem)}
        log.info(f"{stem}: {len(contexts[stem]['attendees'])} attendees, "
                 f"channel map={'yes' if contexts[stem]['channel_segments'] else 'no'}")
    (out_dir / "contexts.json").write_text(json.dumps(
        {s: {"attendees": [a.get("name") for a in c["attendees"]],
             "channel_map": bool(c["channel_segments"])} for s, c in contexts.items()},
        indent=1))
    jobs = [(s, c) for s in args.stems for c in args.candidates]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, s, c, out_dir, contexts[s]): (s, c) for s, c in jobs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                log.error(f"{futs[f]} crashed: {e}")
    print(out_dir)


# -------------------------------------------------------------- transcribe
TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
TRANSCRIBE_CHUNK_SEC = 30 * 60      # diarisation caps a request at 30 min
TRANSCRIBE_PRICE = (2.00, 12.00)    # USD per 1M audio-in / text-out tokens


def _offset(v) -> float:
    return float(str(v).rstrip("s") or 0)


def _turns_from_words(text: str, annotations: list, chunk_offset: float,
                      chunk_idx: int) -> list[str]:
    """word_info annotations -> '[MM:SS] Speaker N: ...' turns (meeting time).
    Speaker ids are per request, so they are prefixed with the chunk number:
    the dedicated STT model has no cross-chunk identity."""
    turns, cur_spk, cur_words, cur_start = [], None, [], 0.0
    raw = text.encode("utf-8")  # indices are UTF-8 byte offsets, not chars
    for a in annotations:
        if a.get("type") != "word_info":
            continue
        w = raw[a["start_index"]:a["end_index"]].decode("utf-8", "replace")
        spk = a.get("speaker") or "spk:?"
        if spk != cur_spk and cur_words:
            turns.append((cur_start, cur_spk, " ".join(cur_words)))
            cur_words = []
        if not cur_words:
            cur_start = chunk_offset + _offset(a.get("start_offset", 0))
        cur_spk = spk
        cur_words.append(w)
    if cur_words:
        turns.append((cur_start, cur_spk, " ".join(cur_words)))
    out = []
    for t, spk, txt in turns:
        s = int(t)
        ts = f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"
        out.append(f"[{ts}] Speaker {chunk_idx}.{spk.split(':')[-1]}: {txt}")
    return out


def cmd_transcribe(args) -> None:
    """gemini-3.5-transcribe needs google-genai >= 2.0 (Interactions API);
    run this subcommand from a venv that has it."""
    from google import genai
    run_dir = Path(args.run_dir)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options={"api_version": "v1beta"})
    cfg = {"transcription_config": {
        "language_codes": ["de-DE"],
        "mode": {"type": "verbatim", "diarization_mode": "speaker",
                 "timestamp_granularities": ["word"]}}}
    for stem in args.stems:
        dest = run_dir / "runs" / f"{stem}__{TRANSCRIBE_MODEL}__none.json"
        if dest.exists():
            continue
        mp3 = TRANSCRIPTS / f"{stem}.mp3"
        dur = float(subprocess.run(
            ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(mp3)], capture_output=True, text=True).stdout.strip())
        lines, inp, outp, t0, err = [], 0, 0, time.time(), None
        with tempfile.TemporaryDirectory() as td:
            for i, off in enumerate(range(0, int(dur) + 1, TRANSCRIBE_CHUNK_SEC)):
                if off >= dur - 1:
                    break
                part = Path(td) / f"c{i}.mp3"
                subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", str(off), "-t",
                                str(TRANSCRIBE_CHUNK_SEC), "-i", str(mp3), "-c", "copy",
                                str(part)], check=True)
                f = client.files.upload(file=str(part))
                while f.state.name == "PROCESSING":
                    time.sleep(2)
                    f = client.files.get(name=f.name)
                try:
                    r = client.interactions.create(
                        model=TRANSCRIBE_MODEL, generation_config=cfg,
                        input=[{"type": "audio", "uri": f.uri, "mime_type": "audio/mpeg"}])
                    d = r.model_dump()
                    content = d["steps"][-1]["content"][0]
                    lines += _turns_from_words(content.get("text", ""),
                                               content.get("annotations") or [], off, i + 1)
                    inp += d["usage"]["total_input_tokens"]
                    outp += d["usage"]["total_output_tokens"]
                except Exception as e:
                    err = f"chunk {i + 1}: {e}"[:300]
                    log.warning(f"{stem}: {err}")
                finally:
                    client.files.delete(name=f.name)
        transcript = "\n".join(lines)
        c = inp / 1e6 * TRANSCRIBE_PRICE[0] + outp / 1e6 * TRANSCRIBE_PRICE[1]
        summary = {"stem": stem, "model": TRANSCRIBE_MODEL, "thinking_level": "none",
                   "duration_sec": dur, "first_pass_valid": err is None,
                   "final_valid": err is None, "error": err,
                   "latency_sec": round(time.time() - t0, 1),
                   "input_tokens": inp, "output_tokens": outp, "thinking_tokens": 0,
                   "cost_usd": round(c, 4), "cost_usd_2027": round(c, 4),
                   "cost_usd_per_hour": round(c / (dur / 3600), 4),
                   "cost_usd_2027_per_hour": round(c / (dur / 3600), 4),
                   **transcript_metrics(transcript, dur, [])}
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"summary": summary, "result": {
            "transcript": transcript, "_meta": {"audio_duration_seconds": dur,
                                                "model": TRANSCRIBE_MODEL}}},
            ensure_ascii=False, indent=1))
        log.info(f"done {stem} transcribe: wpm={summary['wpm']} cost=${summary['cost_usd']}")


# ------------------------------------------------------------------- judge
JUDGE_PROMPT = """You are grading meeting transcripts against the audio excerpt attached.
The meeting is in Swiss German and/or German (sometimes English). Transcripts render
Swiss German speech as standard German text; that is expected, not an error.

The excerpt covers {start} to {end} of the meeting. Below are {n} anonymous candidate
transcripts of that same span (timestamps are meeting time). Listen to the audio and
score EACH candidate from 1 (unusable) to 10 (perfect) on:
- completeness: is everything said present, nothing invented, nothing cut off?
- accuracy: are the words and meaning right (names, numbers, technical terms)?
- speakers: are turns attributed to the right speaker consistently?
Speaker names may differ between candidates; judge consistency with the audio.

{candidates}

Reply with JSON only: {{"scores": {{"A": {{"completeness": n, "accuracy": n, "speakers": n}}, ...}},
"best": "<letter>", "notes": "<one sentence, no quotes from the meeting>"}}"""


def _cut(mp3: Path, start: int, length: int, dest: Path) -> None:
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", str(start), "-t", str(length),
                    "-i", str(mp3), "-ac", "1", "-b:a", "64k", str(dest)], check=True)


def cmd_judge(args) -> None:
    from google import genai
    from google.genai import types
    run_dir = Path(args.run_dir)
    runs: dict[str, dict[str, dict]] = {}
    for f in sorted((run_dir / "runs").glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("result"):
            continue
        s = d["summary"]
        runs.setdefault(s["stem"], {})[f"{s['model']}:{s['thinking_level'] or 'default'}"] = d
    for stem in runs:
        ref = TRANSCRIPTS / f"{stem}.json"
        if ref.exists():
            runs[stem]["production(3.8-flash:medium)"] = {"result": json.loads(ref.read_text())}

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rng = random.Random(args.seed)
    verdicts = []
    for stem, cands in runs.items():
        any_res = next(iter(cands.values()))["result"]
        duration = any_res["_meta"]["audio_duration_seconds"]
        starts = sorted(rng.sample(range(0, max(1, int(duration) - args.length)),
                                   k=min(args.windows, max(1, int(duration) // args.length))))
        for start in starts:
            end = start + args.length
            names = list(cands)
            rng.shuffle(names)
            letters = {chr(65 + i): n for i, n in enumerate(names)}
            blocks = []
            for letter, name in letters.items():
                lines = [l for l in (cands[name]["result"].get("transcript") or "").splitlines()
                         if (m := TS_RE.match(l.strip())) and start - 5 <= ts_to_sec(m.group(1)) < end]
                blocks.append(f"### Candidate {letter}\n" + ("\n".join(lines) or "(nothing)"))
            with tempfile.TemporaryDirectory() as td:
                clip = Path(td) / "clip.mp3"
                _cut(TRANSCRIPTS / f"{stem}.mp3", start, args.length, clip)
                prompt = JUDGE_PROMPT.format(start=f"{start // 60:02d}:{start % 60:02d}",
                                             end=f"{end // 60:02d}:{end % 60:02d}",
                                             n=len(letters), candidates="\n\n".join(blocks))
                for attempt in range(3):
                    try:
                        resp = client.models.generate_content(
                            model=args.judge_model,
                            contents=[prompt, types.Part.from_bytes(
                                data=clip.read_bytes(), mime_type="audio/mpeg")],
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                thinking_config=types.ThinkingConfig(thinking_level="high")))
                        text = re.sub(r"^```(json)?|```$", "", resp.text.strip(), flags=re.M)
                        v = json.loads(text)
                        break
                    except Exception as e:
                        log.warning(f"judge {stem}@{start}: {e}")
                        v = None
            if not v:
                continue
            for letter, sc in (v.get("scores") or {}).items():
                if letter in letters:
                    verdicts.append({"stem": stem, "start": start, "candidate": letters[letter],
                                     **sc, "best": v.get("best") == letter})
            log.info(f"judged {stem}@{start}: best={letters.get(v.get('best'))}")
    (run_dir / "judge.json").write_text(json.dumps(verdicts, indent=1))
    print(run_dir / "judge.json")


# ------------------------------------------------------------------ report
def cmd_report(args) -> None:
    run_dir = Path(args.run_dir)
    rows = []
    texts: dict[str, dict[str, list]] = {}
    for f in sorted((run_dir / "runs").glob("*.json")):
        d = json.loads(f.read_text())
        s = d["summary"]
        key = f"{s['model']}:{s['thinking_level'] or 'default'}"
        rows.append((key, s))
        if d.get("result"):
            texts.setdefault(s["stem"], {})[key] = parse_lines(d["result"]["transcript"])
    for stem in texts:
        ref = TRANSCRIPTS / f"{stem}.json"
        if ref.exists():
            texts[stem]["production"] = parse_lines(json.loads(ref.read_text())["transcript"])

    judge = json.loads((run_dir / "judge.json").read_text()) if (run_dir / "judge.json").exists() else []
    agg: dict[str, dict] = {}
    for key, s in rows:
        a = agg.setdefault(key, {"n": 0, "valid_first": 0, "valid_final": 0, "hours": 0.0,
                                 "cost": 0.0, "cost27": 0.0, "think": 0, "out": 0,
                                 "latency": 0.0, "wpm_ratio": [], "agree": [], "errors": 0,
                                 "dup": [], "endcov": []})
        a["n"] += 1
        a["valid_first"] += bool(s.get("first_pass_valid"))
        a["valid_final"] += bool(s.get("final_valid"))
        a["errors"] += bool(s.get("error"))
        a["hours"] += s["duration_sec"] / 3600
        a["cost"] += s.get("cost_usd") or 0
        a["cost27"] += s.get("cost_usd_2027") or 0
        a["think"] += s.get("thinking_tokens", 0)
        a["out"] += s.get("output_tokens", 0)
        a["latency"] += s.get("latency_sec", 0)
        if s.get("end_coverage_pct") is not None:
            a["endcov"].append(s["end_coverage_pct"])
        if s.get("duplicate_line_rate") is not None:
            a["dup"].append(s["duplicate_line_rate"])
        others = [k for k in texts.get(s["stem"], {}) if k != key]
        mine = texts.get(s["stem"], {}).get(key)
        if mine and others:
            f1 = [window_f1(mine, texts[s["stem"]][o], s["duration_sec"]) for o in others]
            f1 = [x for x in f1 if x is not None]
            if f1:
                a["agree"].append(sum(f1) / len(f1))
            prod = texts[s["stem"]].get("production")
            if prod:
                pw = sum(len(words(x)) for _, _, x in prod)
                a["wpm_ratio"].append(s.get("words", 0) / max(1, pw))

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    jscore: dict[str, list] = {}
    for v in judge:
        jscore.setdefault(v["candidate"], []).append(v)

    print("| candidate | meetings | valid 1st pass | valid final | end coverage | words vs prod | "
          "agreement F1 | dup lines | judge compl/acc/spk | judge best | thinking share | "
          "$/meeting-h now | $/meeting-h 2027 | latency min/h |")
    print("|" + "---|" * 14)
    for key, a in sorted(agg.items(), key=lambda kv: kv[1]["cost"] / max(kv[1]["hours"], 1e-9)):
        js = jscore.get(key, [])
        jtxt = (f"{mean([v['completeness'] for v in js])}/{mean([v['accuracy'] for v in js])}/"
                f"{mean([v['speakers'] for v in js])}") if js else "-"
        best = f"{sum(v['best'] for v in js)}/{len(js)}" if js else "-"
        think_share = a["think"] / max(1, a["think"] + a["out"])
        print(f"| {key} | {a['n']} | {a['valid_first']}/{a['n']} | {a['valid_final']}/{a['n']} | "
              f"{mean(a['endcov'])}% | {mean(a['wpm_ratio'])} | {mean(a['agree'])} | {mean(a['dup'])} | "
              f"{jtxt} | {best} | {think_share:.0%} | "
              f"{a['cost'] / a['hours']:.3f} | {a['cost27'] / a['hours']:.3f} | "
              f"{a['latency'] / 60 / a['hours']:.1f} |")
    prod = jscore.get("production(3.8-flash:medium)", [])
    if prod:
        print(f"\nProduction transcripts (3.8-flash, default=medium thinking) judge "
              f"compl/acc/spk: {mean([v['completeness'] for v in prod])}/"
              f"{mean([v['accuracy'] for v in prod])}/{mean([v['speakers'] for v in prod])}, "
              f"best {sum(v['best'] for v in prod)}/{len(prod)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stems", nargs="+", required=True)
    r.add_argument("--candidates", nargs="+", required=True,
                   help="model[:thinking_level], e.g. gemini-3.6-flash:minimal")
    r.add_argument("--run-dir")
    r.add_argument("--jobs", type=int, default=4)
    j = sub.add_parser("judge")
    j.add_argument("--run-dir", required=True)
    j.add_argument("--windows", type=int, default=5)
    j.add_argument("--length", type=int, default=90)
    j.add_argument("--seed", type=int, default=20261125)
    j.add_argument("--judge-model", default="gemini-4-neon")
    t = sub.add_parser("transcribe", help="gemini-3.5-transcribe baseline (google-genai>=2)")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--stems", nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "judge": cmd_judge, "report": cmd_report,
     "transcribe": cmd_transcribe}[args.cmd](args)


if __name__ == "__main__":
    main()
