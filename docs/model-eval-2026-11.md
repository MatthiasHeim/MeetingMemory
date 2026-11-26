# Transcription model eval, November 2026

**Decision (2026-11-25):** `gemini-3.8-flash` with `thinking_level: low` for short and long meetings. It is the default in `tools/gemini_processor.py` (`DEFAULT_MODEL`, `DEFAULT_THINKING_LEVEL`), and the watcher refuses `gemini-2.5-pro` / `gemini-2.5-flash`. Status: **confirmed on completeness, judge pending**. The clean re-run on 2026-11-26 did not change the ranking (see Results). Still open: the audio-judge pass and a 3.8 Flash low run on the 121-min meeting. Run them at the start of a billing month or after raising the spending cap, because the key is shared with the live watcher.

## Why this came up

Google auto-routes `gemini-2.5-pro` to `gemini-3.8-flash` and `gemini-2.5-flash` to `gemini-3.5-flash-lite` from **2026-12-01** (aistudio.google.com/docs/deprecations, checked 2026-11-25).

What the code and logs actually showed:

- Production (`~/Documents/MeetingRecorder/config.yaml`) has run `gemini-3.8-flash` for both routes since **2026-09-17**, when 2.5 Pro hung on a 33-min recording. All November transcripts are 3.8 Flash. The "existing JSON = 2.5 Pro reference" assumption was wrong.
- The 2.5 exposure was in the **code defaults**. They were used whenever a config key was missing: `model_long` defaulted to `gemini-2.5-pro` and `model` to `gemini-2.5-flash`. The same defaults sat in `speaker_trial.py` and `verify_reliability_fix.py`. The escalation log line "retrying fresh single-call on pro" was a stale label; it used the current model.
- No thinking level was set, so 3.8 Flash ran at its default **medium**. Thinking tokens are billed as output and count against `max_output_tokens` (32,768 in the config). The logged token counts left them out.

## Candidates (prices verified on aistudio.google.com/docs/pricing, 2026-11-25)

| Model | Released | Audio in / output per 1M (USD) | Change | Thinking levels (default) |
|---|---|---|---|---|
| gemini-4-hydrogen | 2026-11-12 | 1.50 / 7.50 | none announced | low..xhigh (high) |
| gemini-4-neon | 2026-11-18 | 3.00 / 15.00 | none | low..xhigh (high) |
| gemini-4-carbon / argon | 2026-11 / 10 | 2.00 / 10.00 | **doubles 2026-12-01** | low..high (medium) |
| gemini-3.8-flash | 2026-09-02 | 0.75 / 3.75 | **doubles 2027-01-01** | low..high, no minimal (medium) |
| gemini-3.6-flash | 2026-07-21 | 0.75 / 3.75 | **doubles 2027-01-01** | minimal..high (medium) |
| gemini-3.5-flash-lite | 2026-07-21 | 0.30 / 2.50 | none | minimal..high (minimal) |
| gemini-3.5-transcribe | 2026-08 | 2.00 audio / 12.00 text (~$0.30/h) | none | none; 30 min/request with diarisation |

From 2027-01-01, 3.8 Flash costs the same as Gemini 4 Hydrogen. Neon and Carbon were not run: they cost 2-4× Hydrogen with no sign that transcription needs them.

## Method

`tools/model_eval.py` runs each candidate through the production path: the pipeline's own converted MP3, calendar attendees from `calendar_resolve`, the channel map when the WAV still exists and is admissible, the same single-shot or chunked routing, retries, the watcher's validator, and the escalation ladder (without alerts). Every billed call is counted, including retries and thinking tokens. Full outputs stay in `~/.local/share/meetingmemory-model-eval/` (meeting content is confidential and never leaves the machine or enters Git).

Meetings: 2026-11-24 15:29 (18 min, short path), 2026-11-18 16:31 (60 min), 2026-11-04 17:01 (66 min), 2026-11-05 17:01 (121 min).

The main metric is **words relative to production** (3.8 Flash, medium). The validator only checks timestamp coverage, so a model that condenses speech passes it. Audio-grounded quality judging (`model_eval.py judge`: Gemini 4 Neon at high thinking scores blind 90-s windows against the audio) is built but **not yet run**.

## Results

Words relative to production. "ok" means final-valid after the ladder. 429 means the run was killed by the spending cap.

| Candidate | 18 min | 60 min | 66 min | 121 min | Verdict |
|---|---|---|---|---|---|
| **3.8 Flash, low** | 1.01 ok | 1.00 ok | 1.00 ok | incomplete (API) | complete everywhere; 0-600 thinking tokens per meeting |
| 3.8 Flash, medium (production) | 1.02 ok | 0.89 ok | not re-run | production = 1.00 | 4-8k thinking tokens per call |
| 3.6 Flash, low | **0.86** ok | 0.99 ok | 0.98 ok | 1.01 ok | close, but dropped 14% on the short meeting |
| 3.6 Flash, minimal | **0.41** ok | 0.99, duplicate span | 1.01 ok | - | condenses silently and passes the validator; rejected |
| Gemini 4 Hydrogen, low | 0.96 ok | 0.98 ok (re-run) | 0.95 ok (re-run) | not run | complete but slightly shorter; $0.64-1.19 per meeting-hour now (15-23k thinking tokens per long meeting even at low), 2-4× 3.8 Flash |
| 3.5 Flash-Lite, minimal | **0.00** | **0.00** | 0.98 ok | - | repetition loops: 86-148k output tokens of junk, paid in full; rejected |
| 3.5 Flash-Lite, low | **0.12** | 429 | 0.87, chunk lost | - | rejected |
| 3.5 transcribe | 0.74 | **0.00** (2 chunks disconnected) | **0.04** | - | 9 of 20 thirty-min requests "Server disconnected" (two runs), one silently near-empty with `status: completed`; text only (no analysis JSON, no names); needs google-genai ≥ 2 (Interactions API, production venv is 1.60). Rejected. |
| 2.5 Pro (reference) | 1.00 ok | not run (collision) | not run | - | deprecated |

**Re-run 2026-11-26** (one process per recording, no chunk collisions): Hydrogen on the 60- and 66-min meetings, and 3.6 Flash low plus 3.8 Flash low on the 121-min meeting. Hydrogen and 3.6 low completed. The 3.8 Flash low run on the 121-min meeting did not: the API was unstable that morning (broken pipe, "Server disconnected", one 39-min read timeout, truncated JSON on chunk 4). The spending cap then throttled the project again after about $3.30 of re-run spend. The Hydrogen run on the 121-min meeting and the audio-judge pass were not run, to keep headroom for the live watcher.

What the re-run changes: nothing in the decision. Hydrogen stays complete but costs 2-4× more now and is not more complete. 3.6 Flash low was complete on all three long meetings and is the runner-up at the same price, but it dropped 14% on the short meeting. 3.8 Flash has been the production model on the 121-min class since September (production transcript = 1.00), so the missing low run there is a gap in the evidence, not a known failure.

### Caveats

1. **Contaminated reliability numbers.** Six parallel jobs ran per recording, and the chunk folder was named after the recording (`gemini_chunks_<stem>`), so concurrent runs deleted each other's chunks: 96 "No such file" errors. Most first-pass failures on the long meetings came from this, not from the models. Fixed: the folder is now unique per call (`tempfile.mkdtemp`). This could also hit production when an escalation retry and a rescan overlap.
2. **Spending cap.** Around $8 of eval spend pushed the "Local automations" project into its **monthly spending cap** (429 "rate-limited to prevent exceeding its monthly spending cap"). The watcher shares this key. The eval was stopped immediately. A 429 for the spending cap is now treated as fatal, so the watcher fails fast with the reason in the alert instead of burning the ladder.
3. Production JSONs carry names applied by later attribution stages, so speaker labels can't be compared directly; that is the judge's job.

## Cost per meeting-hour

Production, November: 80 meetings, 53.5 h; about 100k input and 21k visible output tokens per hour.

| | now | from 2027-01-01 |
|---|---|---|
| Before: 3.8 Flash medium (visible tokens + est. 4-17k thinking/h + paid JSON retries) | ≈ $0.17-0.22 | ≈ $0.34-0.43 |
| After: 3.8 Flash low (thinking ≈ 0-1k/h) | ≈ $0.155 | ≈ $0.31 |

MeetingMemory costs roughly $4-6 a month. The model choice saves the thinking share, and likely some truncated-JSON retries, which were frequent (12 "Unterminated string" / "Expecting ','" retries on 18.-19.11. alone, each re-paying the full audio). The bigger lever is the 2027-01-01 doubling. The only cheaper model, 3.5 Flash-Lite, is not usable.

## Open: audio judge and the 121-min low run

```bash
R=~/.local/share/meetingmemory-model-eval/2026-11-25_eval
# one process per recording; stop at the first 429 (key is shared with the watcher)
venv/bin/python tools/model_eval.py run --run-dir $R --jobs 1 \
  --stems 2026-11-05_17-01-23 --candidates gemini-3.8-flash:low   # ~$0.40
venv/bin/python tools/model_eval.py judge --run-dir $R --windows 5    # ~$1
venv/bin/python tools/model_eval.py report --run-dir $R
```

Runs that were contaminated or killed by the cap were moved to `$R/runs_removed_2026-11-26/`; the run script skips a candidate whose output file already exists.

Re-check Hydrogen in December. From 2027-01-01 it costs the same as 3.8 Flash and is newer. It only wins if the judge rates it better, because its thinking tokens at low still add about 15-20% output cost.

## Speech gate (`tools/speech_trim.py`)

Recordings left running are billed in full. 11.10.: 87.8 min at 6 words/min. 15.10.: 56.6 min at 15 wpm, and 126.2 min at 9 wpm. Normal meetings run 140-200 wpm.

- **The detector must read the raw/aligned WAV.** On the loudnormed MP3, channel_vad's floor-relative detector marks 25-41% of these empty recordings as "speech" and finds no long gaps. On the raw 3-channel WAV the same detector separates cleanly, because system audio goes digitally silent and the mic threshold is anchored to the host's speech level.

  | Recording (raw WAV) | Length | Speech found | Action |
  |---|---|---|---|
  | 2026-11-24 16:00, near-empty | 11.5 min | 0.4 min, p95 −62 dBFS | skipped, no Gemini call |
  | 2026-09-14 15:59, forgot to stop | 69.4 min | 38.6 min | ~16 min cut (14-min empty stretch + 2.7-min lead-in) |
  | Normal meetings (18.11., three files) | 41-60 min | 65-79% | untouched |

  The three October cases survive only as MP3, so they could not be re-checked on the WAV.
- **Rules.** Cut no-speech stretches of at least 150 s, keeping 10 s on each side, and only if at least 60 s gets cut. Skip only when speech is under 30 s, under 5% of the length, **and** every channel's 95th-percentile level is below −50 dBFS. The level check exists because the floor-relative detector also reads "no speech" for wall-to-wall talk or a uniformly noisy room. When the level is loud or unknown, the result is "inconclusive" and the whole recording is sent.
- **Timestamps.** Gemini hears the shortened MP3. `TimeMappedProcessor` maps channel and diarisation segments into trimmed time and maps every timestamp in the result (transcript, emotion and energy arcs, interruptions, missing ranges) back to the original recording before validation. System-track attribution, voice matching and live fill keep reading the original WAV. Validation measures coverage against the end of the last kept audio. The full MP3 stays in Transcripts/ as evidence; the trimmed copy is deleted after the run. `_meta.speech_trim` records the plan.
- **Savings.** At today's prices the three October recordings cost $0.17 together, so the money is small. The skip also stops a fake "meeting" from reaching InsightBase, a headless Claude follow-up session and a Telegram ping.

## Duplicate transcriptions

From `~/Documents/MeetingRecorder/logs/watcher.log`; `~/Library/Logs/MeetingRecorder/watcher_stdout.log` is empty.

| Case | Cause | Paid twice? | Status |
|---|---|---|---|
| 09.08., 08.10., 14.10., 19.10., 28.10. ×2 | Rescan and the live file handler both queued the WAV; after the first run saved, the second queued copy ran again | **yes** | fixed 2026-10-28 (86cbe279: queue dedup + skip when a transcript exists) |
| 09.11. 21:29 (67.5 min ×2) | Socket hung 12 h with no response; restart re-ran it | first call likely never billed | fixed 2026-11-10 (62dea1b8: request timeout) |
| 17.11. 09:59 (31.6 min ×4) | **Invalid API key** (key rotated in .env); 3 full ladders + 2 rescans until a manual restart | no (400s) | fixed here: non-retryable errors (bad key, 401/403, billing, unknown model, spending cap) raise `GeminiFatalError` through every retry layer; the watcher re-reads `.env` once and retries if the key changed |
| 03.11. 17:00 | Watcher restarted mid-run | possibly | intended (no output existed yet) |

Remaining paid re-runs are intended: JSON-validation retries, the escalation ladder, and rescans of files without output. `_meta.billed_usage` now records every billed call per recording, so they are visible.
