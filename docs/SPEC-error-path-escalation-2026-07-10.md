# SPEC: Error-Path Escalation + Degenerate-Chunk Fallback Fix

**Date:** 2026-07-10
**Status:** proposed
**Trigger incident:** `2026-07-10_13-59-12.wav` (24.0 min) produced no transcript — Telegram failure alert, no JSON, no InsightBase row.
**Relates to:** `RELIABILITY_PLAN_2026-07.md` (Phase 1 validation gate + escalation ladder, merged 2026-07-09 in PR #5)

---

## 1. Incident timeline (from `watcher_stderr.log`)

1. **14:23** — 24.0-min recording picked up, converted to MP3 (17.1 MB), channel VAD computed (555 segments), topology `multi_source_genuine`.
2. **Single-shot** (24 min < 35-min chunk threshold): 3 Gemini attempts, all `Server disconnected without sending a response` after 0 chars.
3. **Fallback to "chunked" processing** — but 24 min is below `CHUNK_THRESHOLD_SEC` and the fallback does not pass `force_chunk=True`, so `_chunk_audio` returned the **whole file as one degenerate chunk**. The "fallback" was the same single-shot call again:
   - Chunk attempt 1/2: one more disconnect, then output that failed per-chunk validation (duplicate span).
   - Chunk attempt 2/2: output failed validation (repetition loop + duplicate span).
4. **`All chunks failed`** → `GeminiResult(error="All chunks failed")`.
5. **Watcher hard-stops on `result.error`** (`transcribe_watcher.py:969-972`): log + Telegram alert + `return` — *before* reaching `_validate_and_escalate` at line 979. The escalation ladder (fresh single-call → force-chunked mode) never ran.
6. No JSON written, no source seeded. The WAV/MP3 stay on disk, but nothing retries them until a manual watcher restart.

**Context:** the previous day (`2026-07-09_13-58-39.wav`, 39.1 min) hit the same Gemini disconnect storm at ~14:00, lost chunk 2, **failed validation — and was rescued by the escalation ladder's fresh single-call**. The only reason today's meeting died and yesterday's didn't is which code path the failure surfaced on. Both incidents also show Gemini API instability concentrated around 14:00 HKT.

## 2. Root causes

### RC1 (primary): `result.error` short-circuit bypasses the escalation ladder

`transcribe_watcher.py:969-972`:

```python
if result.error:
    self.logger.error(f"Gemini processing error: {result.error}")
    self._notify_telegram_failure(audio_file, f"Gemini error: {result.error}")
    return
```

The escalation ladder (`_validate_and_escalate`, line 979) was built for results that *look* successful but fail validation. A result that is *worse* — a hard error with no usable transcript — gets *less* recovery: none. The ladder's step 2 (fresh single-call) and step 3 (force-chunked) are exactly the moves that would have rescued this meeting once the disconnect storm passed or on genuinely smaller sub-problems.

### RC2: single-shot→chunked fallback degenerates for 15–35 min recordings

`gemini_processor.py:582-591` (`process_audio` exception handler) calls `_process_chunked(...)` **without** `force_chunk=True`. For audio ≤ 35 min, `_chunk_audio` then returns `[(audio_path, 0.0)]` — the whole file, unsplit. The "chunked fallback" is therefore 2 more attempts × 3 retries of the *identical* single-shot request that just failed three times.

This exact degenerate-disguise failure is already documented and fixed in `_chunk_audio`'s docstring **for the escalation ladder** (source 427, 2026-07-08) — the same reasoning was never applied to the `process_audio` fallback path.

### RC3: no retry after terminal failure within a running watcher

`_process_existing_files` (startup scan: WAV without JSON → queue) runs **only at watcher start**. After a terminal failure, the meeting sits unprocessed until someone manually restarts the watcher. A 30-minute Gemini outage therefore permanently loses any meeting that ends inside it, even though the audio is safely on disk and a retry an hour later would almost certainly succeed.

### RC4 (latent, observed 2026-07-09): chunked-mode timestamp drift — coverage 143.2%

Yesterday's validation failure reported `coverage 143.2%` — the last transcript timestamp (~56 min) exceeded the 39.1-min audio. Mechanism: the drift-proof chunking continuity prefix shows the previous chunk's tail lines *with their chunk-local timestamps* and says "continue". Gemini sometimes obeys too literally and **continues the timestamp sequence** instead of restarting at `[00:00]` for the new chunk's audio; `_shift_timestamps` then adds the chunk offset on top → double-counted time. Consequences:

- Timestamps in accepted chunked transcripts can be wrong by up to a full chunk length (breaks speaker verification, temporal-neighbor retrieval, and any `[MM:SS]` citation downstream).
- The validator's coverage gate only checks `>= 90%` — a drifted transcript at 143% passes the coverage check trivially, so drift is invisible unless something else fails.

## 3. Fixes

### F1: Route error results into the escalation ladder (fixes RC1)

In `_process_with_gemini`, replace the `if result.error: ... return` block with entry into `_validate_and_escalate`. Concretely:

- `_validate_gemini_result` on an error-result already yields `passed=False` (empty/garbage transcript → no timestamps → coverage 0), so the ladder's step 1 naturally falls through to step 2 (fresh single-call) and step 3 (force-chunked).
- Guard the "accept best-available as partial" terminal step: if the best candidate has **no usable transcript at all** (e.g. `coverage_pct == 0` and empty transcript), do **not** store a junk JSON — keep today's behavior (Telegram failure alert, no JSON) so RC3's rescan can retry it later. A junk JSON would mark the file "processed" forever.
- The escalation ladder's existing per-step validation gates stay unchanged — a rescue result is only accepted clean if it fully passes.

### F2: `force_chunk=True` in the single-shot fallback (fixes RC2)

`gemini_processor.py` `process_audio` exception handler: call `_process_chunked(..., force_chunk=True)`. Real 15-min chunks give Gemini genuinely smaller inputs (empirically far more reliable against both disconnects and repetition-loop garbage) instead of replaying the identical failed request.

### F3: Periodic rescan with attempt cap (fixes RC3)

Run the existing `_process_existing_files` logic on a timer (e.g. every 30 min) inside the watcher loop, not only at startup, with:

- **Attempt tracking** per WAV (in-memory dict + a JSON sidecar/state file so restarts don't reset it): max ~4 automatic attempts spaced ≥ 30 min, then stop and alert "giving up — manual reprocess needed".
- **Alert de-duplication**: only the first failure and the final give-up fire Telegram; intermediate retries log only. (Today's behavior would otherwise alert every 30 min.)
- The existing `MIN_RECORDING_BYTES` corrupt-file guard already prevents re-queueing junk captures.

This turns "meeting lost during a Gemini outage" into "meeting delayed until the outage ends".

### F4: Timestamp-drift hardening (fixes RC4)

1. **Prompt**: in `_build_continuity_prefix`, add an explicit instruction: *"Timestamps in this segment MUST restart at [00:00] for the beginning of THIS audio file — do not continue the timestamps shown above."*
2. **Validator**: add an upper coverage bound — `coverage_pct > 110%` fails validation with reason "last timestamp exceeds audio duration — timestamp drift". (110% allows normal rounding/dictation slop; genuine drift lands at ~130–200%.)
3. **Per-chunk validation** already runs against `chunk_duration`, so the upper bound also catches the drift at the chunk level, where a retry with the corrected prompt can fix it cheaply.

### F5 (open decision — NOT in scope until decided): last-resort model fallback

When pro fails every ladder step, a final attempt on `gemini-2.5-flash` (fully validation-gated) would trade the "never Flash" quality rule against total loss of the meeting. Since F1+F3 already convert most total losses into delayed successes, recommendation is to **defer F5** and revisit only if post-fix telemetry still shows terminal failures.

## 4. Test plan

Unit tests (follow the existing `tests/test_watcher_validation_escalation.py` pattern, mocked Gemini):

1. **RC1 regression**: processor returns `error="All chunks failed"` → assert `_validate_and_escalate` is entered and a passing rescue result is stored clean.
2. **F1 junk guard**: all ladder steps fail with empty transcripts → assert no JSON is written and failure alert fires once.
3. **RC2 regression**: 24-min audio, single-shot raises → assert `_process_chunked` receives `force_chunk=True` and processes ≥ 2 chunks.
4. **F3**: simulate failed WAV + timer tick → re-queued; after max attempts → give-up alert, no further queueing; restart mid-sequence → attempt count survives via state file.
5. **F4 validator**: transcript with last timestamp at 143% of duration → `passed=False`, reason mentions drift; 105% → passes.

Integration smoke test: replay `2026-07-10_13-59-12.mp3` through the patched pipeline (it is the perfect real-world fixture — keep a copy before cleanup).

## 5. Rollout

1. Implement F1–F4 in MeetingMemory (branch, PR, tests green).
2. Restart watcher (`launchctl kickstart -k gui/501/com.user.transcribewatcher`) — required; the stale-code check warns but does not hot-reload.
3. Verify next 2–3 live meetings pass validation clean; check `_meta.validation` in their JSONs.
4. Backfill: re-run the known-truncated meetings list from `RELIABILITY_PLAN_2026-07.md` (ids 410, 427, 142, 419) now that the coverage gate + drift fix make re-transcription safe.
