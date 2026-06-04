# Investigation: single-call vs chunked transcription & the speaker-attribution bug

Date: 2026-06-04
Question (Matthias): we chunk meeting audio and keep getting speakers wrong. Should we do
a single whole-transcript call with Gemini (like the `video-analyzer` skill) and drop chunking?
Would that fix the speaker problem?

## How transcription works today (verified in code, not docs)

- **Engine:** `tools/gemini_processor.py`, model **`gemini-2.5-flash`**, via the **Gemini
  developer API** (`GEMINI_API_KEY`, `from google import genai`). **Not Vertex.** The local
  noScribe/Whisper/pyannote pipeline is legacy/unused.
- **Chunking:** audio **> 15 min** is split into **15-min chunks, 30 s overlap**
  (`CHUNK_THRESHOLD_SEC = CHUNK_DURATION_SEC = 15*60`, `CHUNK_OVERLAP_SEC = 30`). Each chunk is
  transcribed independently, then a **text-only reduce-pass** merges per-chunk analyses.
- Stated reason for chunking (code comment): *"Flash models hallucinate on long single-shot
  audio; Pro is slow+flaky. 15 min stays well under model thinking-loop thresholds."* And:
  *"Gemini's server disconnects with RemoteProtocolError on ~30% of long-audio requests.
  Streaming + retry makes the pipeline reliable."*
- **Speaker scaffolding built to fight chunking:**
  - `known_attendees` (calendar) prefix forwarded to **every** chunk.
  - `tools/speaker_reconcile.py` canonicalises Gemini's guessed names against the calendar
    attendee list, including a **`singleton_collapse`** pass that exists *only* to undo chunk
    drift (one physical voice split into "Speaker 1"/"Speaker 2"/"Vivienne" across chunks).

## Data: how often does chunking fire?

InsightBase `sources` (source_type='meeting', n=125 with duration):
- **105 / 125 (84%) exceed 15 min** → chunked.
- median **31 min**, mean **35.7 min**, max **136 min**; 36 meetings (29%) > 45 min.

So the chunk-drift path is exercised on the large majority of meetings.

## Two distinct speaker failure modes (do not conflate)

1. **Cross-chunk inconsistency** — a chunk is diarized inconsistently with the others. Two
   flavors: *drift* (one voice gets different labels across chunks → `singleton_collapse` fights
   it) and *swap* (within one chunk both speakers' labels are flipped, e.g. the itesys chunk-2
   Matthias↔Sascha flip). The dominant class. **A single call eliminates both by construction**
   (verified: the single-call run nailed the itesys chunk-2 region the chunked run swapped).
2. **Within-call name hallucination** — Gemini invents a wrong real name (docstring example:
   "Nadine Maricic" for "Ladina Walicki-Kasper"). Flagged as the *dangerous* case; it happens in
   **1:1s under 15 min that never chunk**. **A single call does nothing for this.** → calendar
   reconciliation (`known_attendees` + `speaker_reconcile`) stays load-bearing either way.

## A second chunking failure mode found in the wild

`2026-06-04_16-03-16.json` transcript begins with
`[CHUNK 1 FAILED: All 3 attempts failed: Server disconnected without sending a response.]` —
chunking didn't just drift speakers, it **silently dropped the first 14.5 min** of that meeting.
A per-chunk failure loses a segment; the meeting still "succeeds".

## Empirical test — COMPLETED 2026-06-04 ~22:23 CET

Goal: run the 60-min itesys 1:1 (`2026-06-04_19-59-36.mp3`) single-call and check repeat-loop
hallucination + output tokens + speaker consistency.

**Result (production model `gemini-3-flash-preview`, single-call, cap 32768, roster = Matthias +
Sascha Lioi):**
- ✅ completed in **221 s**, full coverage to `[60:10]`, 60 237 chars of clean Swiss German.
- ✅ **no repeat-loop hallucination** (longest consecutive identical line = 1) — the exact
  2.5-flash failure that motivated chunking did **not** occur on the 3.x model.
- ✅ **speakers perfectly consistent**: only `Matthias:` / `Sascha:` alternating end-to-end,
  **zero drift**, used the roster name "Sascha" (no `Speaker 1/2`, no handle).
- **output = 20 240 tokens** for 60 min → natural ceiling ~20 k; the 65536 cap is ~3× overkill.

→ For the user's question this is the direct evidence: **single-call eliminates the cross-chunk
drift class and produces a clean, fully-covered transcript on a real 60-min Swiss-German meeting
with the production model.**

### Side-quest (a degraded-API detour, mostly resolved — do not over-read)

Before the successful run, 21:40–22:33 produced a confusing run of `0-char` stream disconnects.
Bisecting (sandbox on/off, streaming vs non-streaming, duration, prompt weight, output cap, model)
landed here:
- `gemini-3-flash-preview` (**production model**): worked at cap 32768 (60-min) **and** 65536
  (short) → **production path is fine.**
- `gemini-2.5-flash` and `gemini-3.5-flash`: repeatedly failed at cap 65536, worked at 8192.
- Files API upload always succeeded; only `generate_content_stream` dropped.

Not cleanly isolated (models were tested at different times during what looks like a partial API
wobble), so treat as **observation, not verdict**: the *older* flash models appear unhappy with a
65536 output cap right now; the production model is not. My earlier "65536 is the trigger /
production at risk tonight" note was an **overreach** — retracted. Lowering `max_output_tokens` to
~24–32 k is still a cheap, sensible robustness+cost win (60-min needs ~20 k; chunked 15-min needs
~5 k), just not urgent.

## What the itesys doc being hand-fixed actually shows (failure-mode tie-in)

**Correction to an earlier mis-read in this doc.** A first pass concluded the itesys 1:1 had "no
drift" because a label-count showed only two stable labels (`Matthias`, `sascha.lioi`). That was
wrong: a label-count is **blind to a speaker SWAP** (both labels stay present, only the assignment
flips). Per Matthias's own 21:58 fix, the real defect is exactly that — **chunk 2 (~14:30–29:00)
flipped Matthias↔Sascha**: "ich lieb's zum baue" / "Biologie studiert" / "Müllheim" / "mir sind zu
dritt" were tagged Sascha but are Matthias. That is a **chunk artifact** (chunk 2 diarized the two
voices inconsistently with chunks 1/3/4), i.e. the cross-chunk-inconsistency family, NOT mode #2.

**The single-call run got this exact region 100% right** — every one of those lines is correctly
`Matthias:`, and the "Aike / Leistungserfassig / 14 Cases" turns are correctly `Sascha:`. So
**single-call WOULD have prevented the swap Matthias is hand-fixing.** This is direct evidence for
the user's hypothesis, not against it.

(The Teams-*handle* name `sascha.lioi` is a **separate** defect — failure mode #2, from
`calendar_resolve`, fixed below — independent of chunking.)

## "Vertex" is orthogonal

Vertex (`video-analyzer --backend vertex-eu`, batch in `europe-west4`) is about **EU data
residency / auth**, not single-call. Same Gemini model underneath. It's a separate (and real, for
Swiss client audio) decision from the speaker bug. Today's meeting audio runs through the
non-resident developer API.

## Recommendation (minimal, reversible)

Model is already good: production runs **`gemini-3-flash-preview`** (config.yaml), which the test
above proved handles a full 60-min single call cleanly. So the lever is the threshold, not the
model.

1. **Raise `CHUNK_THRESHOLD_SEC`** from 15 min to **~60 min** so the bulk of meetings (median 31,
   ~90th pct ≤ 60) go single-call and skip cross-chunk drift entirely — **keep streaming+retry,
   keep the chunked path as a fallback** for multi-hour recordings and as a retry after a
   single-call failure (so we never lose a whole long meeting to one disconnect, the way
   `2026-06-04_16-03-16` lost chunk 1).
2. **Keep `known_attendees` + `speaker_reconcile`** regardless — single-call does not fix the
   within-call name hallucination (failure mode #2). The roster still drove the correct "Sascha"
   label in the single-call test.
3. (Optional, non-urgent) lower `max_output_tokens` 65536 → ~32 k: 60-min needs ~20 k, chunked
   15-min needs ~5 k. Cheaper, and sidesteps the older-flash-model 65536 wobble seen during
   testing.
4. Decide Vertex/`vertex-eu` separately, driven by client data-residency needs.

Net: single-call removes the *dominant* speaker bug (cross-chunk drift) and the per-chunk
data-loss bug, and on the production model it is reliable on a real 60-min meeting. It does **not**
retire the calendar-reconciliation layer (failure mode #2 lives on, and is what the
`calendar_resolve` fix below addresses).

## Companion fix shipped this session

`calendar_resolve.py`: when a calendar attendee has no `displayName` (typical for external
attendees), the email local-part was stored verbatim as the name (`sascha.lioi`, `phillip.fumolo`,
`alex.oehler`, `philipp.baltensperger`, `cmr.huth` — 5 stored meetings). Added
`_humanize_email_local()` → "Sascha Lioi" etc.; real displayNames and non-name locals
(`info`, `noreply`, anything with digits) untouched. Fixes future meetings; the 5 existing rows
can be backfilled on request. (Uncommitted, on `main`.)
