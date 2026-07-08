# Transcription Pipeline Reliability Plan (July 2026)

**Goal:** No lost meetings, no silently missing minutes, no wrong speaker allocations. Written 2026-07-08 after auditing source 429 (BlueCare PO-Enablement 1:1 with Lukas, 59 min) where a chunked fallback produced duplicated segments with inverted speaker labels, a fully lost chunk (~min 29–43), and a runaway repetition loop.

## Evidence base (what actually goes wrong)

| Failure | Evidence | Root cause |
|---|---|---|
| Whole meeting lost | 2026-07-06_15-00-03 (28.5 min): all Gemini calls 0-byte, all chunks failed, only Telegram alert | `gemini-3-flash-preview` 0-byte disconnects on multi-source audio **below** the 30-min pro-routing threshold |
| ~15 min silently missing | 7 of 58 sources since May contain `[CHUNK N FAILED]`; meeting still reported as success | `_process_chunked` swallows per-chunk failure (`gemini_processor.py:823-826`), no alert, no coverage check |
| Duplicated text with conflicting speakers | Source 429: [06:49]–[08:51] transcribed twice, first copy labels inverted | 30 s chunk overlap concatenated with **no dedup** (`:842-846`); each chunk diarized independently by Gemini |
| Wrong speaker on short turns | 429 [01:06] question mis-attributed (confirmed by Matthias) | `speaker_verify` only flips turns ≥ 8 s; single-call diarization errors on short turns pass through |
| Repetition loop | 429 [52:07]: "s'heisst" repeated ~400× | No repetition detector anywhere in the pipeline |
| Stale credentials | Watcher held pre-rotation `INSIGHTBASE_DATABASE_URL` for 4 days (process older than `.env` fix) | `.env` loaded once at startup; no restart after rotation; seed failure is best-effort |

**Microphone/channel status (user asked to confirm):** capture is fine. The Aggregate Device records 3 channels (host mic separate from system audio), `channel_vad.py` computes host/remote ground truth from them, and it IS wired into both the Gemini prompt ("AUDIO CHANNEL MAP") and post-hoc `speaker_verify` flips. The remaining weaknesses are: (a) Gemini itself only ever hears a **mono mix** (`audio_converter.py:264-280`) — channel separation is reduced to a text hint; (b) verify thresholds are so conservative (8 s min turn) that most real misattributions survive; (c) the stronger `diarization.channel_fusion` path exists but is **off by default**.

## Decision: chunking is NOT essential — retire it as a primary path

Meetings ≤ 60 min (nearly all of them) already go single-call, which has **zero** cross-chunk drift by construction. Chunking exists only as (1) the > 60 min path and (2) the fallback when a single call fails. Both justifications weaken once the model is reliable (pro, not flash). Chunking is demoted to a last-resort emergency mode with a redesigned, drift-proof implementation (Phase 3), and the fallback order becomes *retry single-call on pro* first.

---

## Phase 0 — Ops fixes (DONE 2026-07-08, no code)

- ✅ `config.yaml`: `model: gemini-2.5-pro` for ALL meetings (flash 0-byte-fails even short multi-source recordings; ~$0.25/meeting extra is irrelevant vs. losing a client meeting).
- ✅ Watcher restarted → picks up rotated `INSIGHTBASE_DATABASE_URL`, requeues + reprocesses the lost 2026-07-06_15-00-03 recording.
- ✅ Source 429 insights flagged for re-extraction (`flag_reextraction.py`, 14 insights).
- ☐ Optional: re-transcribe 2026-06-30_15-31-11.wav (source 429's audio) single-call on pro and UPDATE `sources.content_text` in place (do NOT re-seed — would duplicate the source row). Recovers the missing min 29–43 and fixes the flipped labels.

## Model choice: gemini-3.5-flash evaluated and REJECTED (2026-07-08)

3.5 Flash (stable, released 2026-05-19) replaces the deprecated `gemini-3-flash-preview` we were running. Tested empirically because specs can't reveal the disconnect issue. Two findings, both against a switch:

- **Reliability: still broken.** On the 28.5-min multi-source recording that killed the old flash, 3.5-flash failed **2 of 3 runs** ("Server disconnected without sending a response", 0 bytes, all single-call retries + all chunk-fallback retries exhausted); each failure burned ~440 s. It also failed the 59-min recording. gemini-2.5-pro transcribed the same 28.5-min file cleanly in 130 s. The disconnect is a streaming-endpoint problem for long/multi-source audio on the Flash tier, not a model-intelligence issue — 3.5-flash inherits it.
- **Cost: not actually cheaper.** Audio transcription is input-token-dominated (~1920 tok/audio-min). 3.5-flash is $1.50/M in ÷ $9/M out; 2.5-pro is $1.00/M in ÷ $10/M out. A 28.5-min meeting (~61k in / 8.6k out) ≈ $0.17 on flash vs ≈ $0.15 on pro. Flash is ~15% *more* expensive here. ("3x the model it replaced"; the cheap-Flash era is over.)

**Decision: stay on gemini-2.5-pro for all meetings** (Phase 0). Re-evaluate only a *pro*-tier successor (e.g. gemini-3.5-pro) via the Phase 4 eval harness; do not put any Flash-tier model back on the live path until a batch reliability test shows 0 disconnects across ≥10 long/multi-source runs.

### Second finding from the same test: pro silently TRUNCATES long single calls

The A/B test surfaced a failure mode independent of the model question. On a 50.5-min single call, gemini-2.5-pro returned **valid JSON that ended at [08:17]** — 16.4% coverage, 3,349 output tokens (nowhere near the 32k cap), no disconnect, no error. The current pipeline would insert this as a complete 50-min meeting. This is the same silent-partial-data class as `[CHUNK N FAILED]` but with no marker at all. Two consequences:

1. **The Phase 1 coverage gate (last-timestamp ≥ 90% of audio duration) is the single highest-priority fix** — it is the only thing that catches this.
2. **Lower `CHUNK_THRESHOLD_SEC` from 60 min to ~35 min.** The 60-min single-call window was chosen (doc `transcription-single-call-investigation.md`) to avoid cross-chunk drift, but it trades drift for truncation/disconnect risk on 40–60 min calls. With Phase 3 drift-proof chunking (silence-aligned, zero-overlap, continuity context), chunking no longer causes drift, so the reason to push single-call so long is gone. Route > ~35 min to drift-proof chunks; keep single-call for the short majority.

## Phase 1 — Validation gate + no silent loss (highest ROI, ~1 day)

New module `tools/transcript_validator.py`, called in `_process_with_gemini` between the Gemini result and the JSON write. A transcript must pass ALL checks or the meeting is retried/escalated — never silently inserted:

1. **Coverage check:** max transcript timestamp ≥ 90% of `audio_duration_seconds` AND no gap > 3 min between consecutive timestamps during VAD-active spans. Catches failed chunks, early stops, truncation-inside-valid-JSON.
2. **Chunk-failure = hard failure:** any `[CHUNK N FAILED]` marker → validation fails. Retry the failed chunk(s) on pro; if still failing, Telegram alert names the exact missing time range and the meeting is inserted with a `partial: true` metadata flag (visible to `/meeting-actions` and wiki compile).
3. **Repetition detector:** any token/phrase (1–4 words) repeated > 15× consecutively → strip the loop, log, and count as a validation warning; > 1 loop or a loop > 200 tokens → fail and retry.
4. **Duplicate-span detector:** shingle (8-gram) match between transcript regions > 30 words long → overlap dedup failed → fail (should be impossible once Phase 3 removes overlap; keep as invariant).
5. **VAD agreement score:** % of transcript turns whose speaker label contradicts channel-VAD host/remote ground truth. Score stored in `_meta`; > 20% disagreement → Telegram warning + auto-flag source for review.

**Escalation ladder on validation failure:** retry single-call pro (fresh call) → retry with re-encoded audio (occasionally the MP3 encode is the trigger) → drift-proof chunked mode (Phase 3) → insert best result with `partial: true` + Telegram alert. Never drop a recording; never insert silently-bad data.

**Reconciliation sweep (backstop):** small script + daily launchd job: every `Recordings/*.wav` ≥ MIN_RECORDING_BYTES and > 3 h old must have (a) a Transcripts JSON and (b) a `sources` row (match on filename in `_meta`). Any orphan → Telegram + requeue. This single check would have caught every data-loss incident to date, including failure modes we haven't met yet.

## Phase 2 — Channel-first speaker attribution (~2 days)

Make the deterministic signal authoritative instead of advisory:

1. **Enable and harden `diarization.channel_fusion`** for `multi_source_genuine` topology (it exists, default-off). pyannote diarizes the remote mix → remote speakers separated acoustically; channel VAD separates host vs remote with hardware certainty.
2. **Deterministic post-pass replaces conservative verify:** for every transcript turn, compute host_share from VAD. In 1:1 meetings (the majority), assignment becomes near-deterministic: host segments = Matthias, remote = counterpart; Gemini's labels are only consulted inside `both`-overlap spans. Lower flip threshold from 8 s to 2.5 s when VAD confidence is high (host_share < 5% or > 95%) — this fixes the short-question misattribution class (429 [01:06]).
3. **Multi-party:** flips target the pyannote remote cluster active at that time, removing the "exactly one remote speaker" restriction in `speaker_verify.py:263`. Guard against the known multiparty-collapse failure (memory: `speaker_resolution_multiparty_collapse`) with a rule: never collapse two remote clusters that pyannote separates.
4. **Mic-channel sanity probe at recording start:** meeting_recorder checks within the first 60 s that the mic channel carries signal (RMS above noise floor) and that channel order matches config; menu-bar warning if not. Confirms "same microphone" regressions immediately instead of post-hoc.

## Phase 3 — Drift-proof chunking for > 60 min only (~1 day)

Keep chunking solely for audio the single call cannot cover, redesigned so drift is structurally impossible:

1. **Silence-aligned seams, zero overlap:** cut at the longest VAD-silent point within ±60 s of the nominal 15-min boundary. No overlap → no duplicate text → no conflicting-label duplicates (429's core defect).
2. **Continuity context:** pass each chunk the final ~15 transcript lines of the previous chunk (with resolved names) plus the attendee roster, with an instruction to continue the same speaker mapping.
3. **Per-chunk validation:** each chunk passes the Phase 1 coverage check for its own span before merge; a failed chunk retries independently on pro.
4. Global VAD/pyannote label reconciliation after merge (Phase 2 post-pass runs on the merged transcript, so even residual per-chunk drift is corrected against ground truth).

## Phase 4 — Regression eval harness (~half day, then continuous)

- Golden set: 5–8 past meetings with human-confirmed labels — start with source 429 (Matthias confirmed the correct attribution on 2026-07-08) and the itesys/Sascha call from the speaker-attribution memory.
- `tools/eval_attribution.py`: reprocesses golden audio, scores (a) turn-level speaker accuracy vs. confirmed labels, (b) coverage %, (c) validation-gate pass. Run before merging any pipeline change; also the place to trial new models (gemini-3-pro etc.) with data instead of vibes.

## Order & effort

| Phase | Effort | Impact |
|---|---|---|
| 0 Ops fixes | done | Stops active bleeding (model, credential, lost meeting) |
| 1 Validation gate + sweep | ~1 day | Eliminates silent loss — biggest reliability win |
| 2 Channel-first attribution | ~2 days | Eliminates the speaker-allocation error class |
| 3 Drift-proof chunking | ~1 day | Fixes the rare > 60 min path |
| 4 Eval harness | ~0.5 day | Keeps it fixed |

Implementation happens in this repo (MeetingMemory) via a dedicated Claude Code session; Brain only records the plan reference.
