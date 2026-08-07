# Backfill proposal — speaker attribution (2026-08-07)

**Proposal only. Nothing here has been executed.** Root cause and evidence:
`docs/SPEC-speaker-attribution-2026-08-07.md`.

## How affected sources were identified

Two independent signals, both computable without re-transcribing anything.

**Signal A — the flip signature.** `speaker_verify`'s decision log is stored in
every transcript JSON. Scanning all 221 stored transcripts:

| | count |
|---|---|
| transcripts total | 221 |
| ran channel verification at all | 32 |
| had ≥ 1 flip | **11** |
| **all flips toward the host** | **11 of 11** |
| total flips applied | **81** |

Every one of the 81 flips ever applied moved a turn onto Matthias. Not one went
the other way. A working oracle corrects in both directions; a saturated one
cannot. That 81/81 is the corpus-wide fingerprint of the defect, and it makes
those 11 transcripts the high-confidence affected set.

**Signal B — bleed on the recording.** `channel_vad.host_bleed_rate` can be
recomputed from any retained WAV. All 8 recordings measured were inadmissible
(0.36–0.80 against a 0.35 threshold), so the *prompt-side* damage — Gemini told
a bleeding map is "GROUND TRUTH" and that it "wins" over voice — plausibly
touched every hybrid recording, not just the 11 that also got flipped. That
damage is diffuse and unquantified; it is the reason for tier 3 below.

## Affected sources

### Tier 1 — flipped, with insights already extracted (re-extract)

| source | date | company | meeting | flips | chunks | insights |
|---|---|---|---|---|---|---|
| **414** | 2026-06-22 | Zur Rose | B2C Vision-Sprint Tag 1 — Claude-Setup | **34** | 8 | 10 |
| **427** | 2026-06-23 | Zur Rose | B2C Vision-Sprint — Onboarding-Walkthrough | **18** | 36 | 7 |
| **530** | 2026-07-30 | BBC Group | VR bejaht AI-native und Claude | 6 | 19 | 28 |
| **425** | 2026-06-30 | Brame | BMS-Enablement-Session | 2 | 20 | 17 |
| **429** | 2026-07-01 | BlueCare | PO-Enablement — Lukas | 2 | 32 | 28 |
| **517** | 2026-07-28 | Brame | BMS AI Enablement #3 — Supabase | 2 | 33 | 44 |
| **486** | 2026-07-20 | BlueCare | Zur-Rose-PMO-Prototyp Troubleshooting | 1 | 5 | 10 |
| **523** | 2026-07-29 | BlueCare | Prisco — curaMED-Integration | 1 | 18 | 36 |

Total: **8 sources, 180 insights, 66 of the 81 flips.**

Sources 414 and 427 are the worst by a wide margin — 34 and 18 flips — and both
are Zur Rose sprint sessions, i.e. multi-party. Start there.

### Tier 2 — flipped, nothing extracted yet (repair in place, no re-extraction)

| source | date | transcript | flips | note |
|---|---|---|---|---|
| **767** | 2026-08-07 | `2026-08-07_14-11-24` | 12 | 0 chunks / 0 insights — the fixture |
| **416** | 2026-06-25 | `2026-06-25_10-59-37` | 2 | 1 chunk / 0 insights, personal |

767 is clean: repair `content_text`, then let extraction run once. No
re-extraction needed because none has happened.

### Tier 3 — no flips, but transcribed with a bleeding channel map (audit only)

Every hybrid recording since channel-map injection went live received a
"GROUND TRUTH" map built from a bleeding mic. The 21 remaining transcripts that
ran channel verification without producing flips are in this class, plus any
earlier hybrid recording. There is no per-source evidence of damage here — only
a known-bad input. **Recommendation: do not re-extract tier 3 blindly.** Run the
coherence gate in audit-only mode (see below) and re-extract only what it
actually flags.

## Proposed sweep

Three steps, each independently checkable, executed only on approval.

**Step 1 — repair the transcripts (no DB writes).**
For each tier 1 + tier 2 source: revert `speaker_verify`'s flips using the
`speaker_verification.flips` log already in the JSON (the log records
`from`/`to`/`time_span` per flip, so this is exact and reversible), then run
`tools/speaker_coherence.py` over the reverted transcript. Write the result to
a side file, not over the original, and diff. ~10 min of wall clock per source,
parallelisable.

**Step 2 — review before anything is written.**
Present a per-source diff of label changes for Matthias to sanity-check on the
two he knows best (429 and 767) before the remaining eight are committed. This
is the step that must not be skipped: the gate's measured accuracy on source
767 is good on provable violations and imperfect overall, and re-writing
`content_text` is not something to do unattended across 10 client meetings.

**Step 3 — commit and re-extract.**
For approved sources, update `content_text` in place — do **not** re-seed, that
duplicates the source row (Phase 0 note in the reliability plan). Note there is
no existing helper for this: `neon_insert.insert_source` writes `content_text`
only at seed time and `update_source_with_gemini` does not touch it, so the
sweep needs a small purpose-built updater that also refreshes
`metadata.content_revision_id` (`_compute_content_revision_id`) — otherwise the
revision hash silently stops matching the stored text. Then queue
re-extraction:

```bash
python3 /Users/Matthias/Repos/Brain/.claude/scripts/flag_reextraction.py --source-id 414
# ...427, 530, 425, 429, 517, 486, 523
```

Tier 2 (767, 416) needs the `content_text` update only — extraction has not run,
so it will pick up the repaired text on its next pass.

**Cost:** ~10 sources × ~4 audit windows ≈ 40 headless Claude calls for step 1,
plus re-embedding 180 insights across 8 sources in step 3.

## Open question for Matthias

Tier 1 sources 414 and 427 (Zur Rose, 34 and 18 flips) are **multi-party**
sessions. `speaker_verify` can only ever flip toward "the host" or "the single
dominant remote", so on a multi-party recording its 52 combined flips are
unlikely to be repairable to the right *individual* by any text-only pass — the
gate can tell Matthias from not-Matthias, but distinguishing three Zur Rose
attendees from content alone is much weaker. Options:

1. Repair to host/not-host only and accept generic labels for the remote side.
2. Re-transcribe those two recordings with the fixed pipeline (the WAVs are
   retained) — contradicts "do not re-transcribe as a fix for mislabelling",
   but these two are the corpus's worst cases and were never correct.
3. Leave them and mark the insights low-confidence.

Recommendation: **option 1** for now, revisit if the Zur Rose material is
needed for client-facing work.
