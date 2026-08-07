# Speaker misattribution: root cause and fix (2026-08-07)

**Status:** root-caused, fixed on `reliability/speaker-attribution-bleed-guard-coherence-gate`.
**Fixture:** `2026-08-07_14-11-24` — BlueCare 1:1, Matthias Heim + Philipp
Baltensperger, 35.1 min, InsightBase source **767**. Two parties, remote,
single-call (not chunked).

## Summary

Speaker labels have been wrong for months because the pipeline's "physical
ground truth" is not ground truth. `channel_vad` treats the host's microphone
channel as an oracle for *is the host speaking*. On recordings made without
headphones the mic hears the remote participants through the loudspeakers, so
the oracle answers "yes, the host" almost regardless of who spoke. Two stages
consume that answer as certainty:

1. the Gemini prompt, which is told the channel map is **GROUND TRUTH** and
   that when voice and map disagree, "the map wins";
2. `speaker_verify`, which flips transcript turns whose label contradicts it.

The result on source 767: **12 flips, every single one in the same direction**,
moving 8.4 minutes of speech onto Matthias and turning a 70 %-Matthias
transcript into a **94 %-Matthias** one. In a two-person conversation.

The bleed is **not specific to this recording** — it is present on every one of
the last 8 recordings measured. This defect has been silently degrading
attribution across the corpus.

A second, independent defect: Gemini's own diarization is wrong on some
question/answer adjacencies even before any post-processing. No acoustic signal
can repair that, which is why a semantic gate is also warranted.

## What the evidence shows

### The measurement that was never taken

`channel_vad.py`'s docstring justified trusting the mic channel with a
"measured sample-level correlation ≈ 0.000" between the mic and system
channels. That statistic cannot detect acoustic bleed. Room coupling delays and
filters the loudspeaker signal, so a mic that plainly hears the speakers still
correlates ≈ 0 sample-wise. Re-measured on the fixture:

| measurement | value | meaning |
|---|---|---|
| `corr(mic, ch1)` sample-level | **-0.0002** | the original evidence — reads clean |
| `corr(ch1, ch2)` sample-level | +1.0000 | system audio is duplicated mono |
| envelope `corr(mic, remote)`, best lag | **+0.454** @ 750 ms | the mic tracks the remote's loudness |
| mic level, remote LOUD vs SILENT | −26.9 dB vs −30.7 dB | **remote speech lifts the mic +3.8 dB** |
| remote-active windows the host VAD marks active | **66.9 %** | the oracle fires during two-thirds of the remote's speech |

The last row is the one that matters, and it is now computed in-pipeline as
`channel_vad.host_bleed_rate` = `both / (remote_only + both)`.

### It is chronic, not a one-off

```
2026-07-29_16-06-36   38.0min  bleed_lift= +3.3dB  remote-loud-marked-HOST= 57.6%
2026-07-30_14-34-53    7.8min  bleed_lift= -2.8dB  remote-loud-marked-HOST= 36.5%
2026-07-30_15-00-00   42.8min  bleed_lift=+15.9dB  remote-loud-marked-HOST= 80.1%
2026-08-05_16-00-34   31.0min  bleed_lift= +4.5dB  remote-loud-marked-HOST= 43.2%
2026-08-06_16-00-49   40.5min  bleed_lift=+14.2dB  remote-loud-marked-HOST= 65.5%
2026-08-07_11-30-40    7.2min  bleed_lift= +8.5dB  remote-loud-marked-HOST= 75.6%
2026-08-07_14-11-24   35.1min  bleed_lift= +3.8dB  remote-loud-marked-HOST= 66.9%
```

Every recording is above the 35 % admissibility line. The channel oracle has
been inadmissible on essentially the whole corpus.

### Why the damage is one-directional

`speaker_verify` counts `both` windows toward the host, on the documented
reasoning that this "biases AGAINST flipping in both directions". That holds
only when `both` is rare. On the fixture `both` is **39.8 %** of all windows,
so:

- `host_share` never falls below ~0.43 for any turn longer than the 2.5 s floor.
- The reverse rule `host_label_but_mic_silent` needs `host_share < 0.15`. It is
  **unreachable** — it cannot fire on this recording, or on any bleeding one.
- The forward rule `remote_label_but_mic_dominant` needs `host_share > 0.85`,
  which is satisfied by host turns and remote turns alike. Whatever survives
  the `both_share ≤ 0.40` gate gets flipped to Matthias.

Only 17 of 119 turns were even evaluated, and 12 of those 17 were flipped —
all toward the host. Raw Gemini gave Matthias 1469 s of 2107 s (70 %);
`speaker_verify` raised it to 1973 s (**94 %**).

### The confirmed defects, stage by stage

| line | text (abridged) | truth | raw Gemini | after `speaker_verify` | blame |
|---|---|---|---|---|---|
| `[13:26]` | "…Weisch du eigentlich, dünd ihr monatlich release…" | Matthias | Speaker 2 | Matthias | flip, correct by luck |
| `[13:58]` | "Es wird amene Release zueordnet…26.8…" | **Philipp** | Matthias | Matthias | **Gemini** |
| `[15:07]` | "Ja, genau. Und wenn mer da denn hätted…" | **Matthias** | Speaker 2 | Speaker 2 | **Gemini** |
| `[15:29]` | "Aber das isch jetzt einigermasse unabhängig…" | **Matthias** | Speaker 2 | Speaker 2 | **Gemini** |
| `[15:36]` | "Aber da het ja jetzt au nöd mit de Technologie…" | Matthias | Matthias | Matthias | — |
| `[15:39]` | "Nöd würklich, also eifach dur da…" | **Philipp** | Speaker 2 | Matthias | **flip** |

Two mechanisms, clearly separated:

- `[15:39]` — Gemini had it **right** (Speaker 2). `speaker_verify` flipped it
  onto Matthias with `host_share=0.944`, producing the "question and its answer
  three seconds apart, both labelled Matthias" symptom together with `[15:36]`.
- `[13:58]`, `[15:07]`, `[15:29]` — Gemini's own errors, untouched by the
  acoustic stages.

### Ruled out

Each of the previously documented failure modes was checked against this file
rather than assumed:

| candidate | verdict | evidence |
|---|---|---|
| Chunk-boundary drift | **not implicated** | `_meta.chunked = false`, `chunk_count = 1` |
| Duplicate span, inverted speakers | **not implicated** | `validation.has_duplicate_span = false` |
| `singleton_collapse` | **did not run** | no generic labels beyond a single `Speaker 2` |
| Calendar binding to an absent invitee | **did not run** | `participants: []`, no calendar match |
| `classify_source_topology` single-source | **passed correctly** | topology gate admitted it as `multi_source_genuine`; 3 channels genuinely active |

The `participants: []` is a real contributing defect, though not the cause of
the misattribution: with no participant list the counterpart never got a name,
so the transcript carries `Speaker 2` throughout and
`_recompute_speaking_stats` did nothing (`speaking_stats_recomputed: false`).

## The fix

### 1. The oracle must pass a self-test before anyone trusts it

`channel_vad.host_bleed_rate()` / `channel_separation_report()` /
`is_admissible()` measure `P(mic active | remote speaking)` and refuse the
channel above `MAX_HOST_BLEED_RATE = 0.35`. The verdict is computed once in the
watcher and applied to **every** consumer:

- the Gemini prompt map (`_build_channel_map_prefix` returns `""` — the no-map
  prompt, so Gemini diarizes on voice instead of being told a false map wins);
- pyannote host-cluster fusion (`channel_fusion`);
- `speaker_verify`, which also re-checks independently and records
  `skipped_channel_bleed` plus the measurement it refused on.

`insufficient_remote_speech` (< 60 s of remote) is recorded but stays
admissible: the defect being gated is *measured* bleed, and a recording with
almost no remote speech has almost nothing to misattribute. The
"everyone-on-ch0" case that this reasoning would miss is already caught
upstream by `classify_source_topology`.

The verdict lands in `_meta.channel_separation`, so an audit can tell an
attribution that was channel-verified from one that never was.

### 2. A semantic coherence gate, before any insight is written

`tools/speaker_coherence.py` runs after transcription and **before** the JSON
write — which is also before the InsightBase seed, so the repaired transcript
is what reaches `sources.content_text` and everything built from it. It asks
Claude whether the attribution is internally coherent (who answers a question
didn't ask it; who says "mir/öisi" about the client company works there; who
describes their own process is that person) and rewrites the labels it can
prove.

Guards, each one earned from a prior repair that went wrong:

- **Closed label universe** — a rewrite may only target a label already in the
  transcript, the host, or a calendar attendee. No invented names (source 434's
  `singleton_collapse`).
- **Stale-index rejection** — each proposal names the line's current label; a
  mismatch means the model's index is wrong, and the proposal is dropped rather
  than applied to whichever line sits at that offset.
- **High confidence only** — medium/low proposals are recorded in
  `uncertain_regions` and the text is left alone. Unresolvable regions are
  flagged, not guessed.
- **Runaway cap** — more than 35 % of lines proposed for relabelling is a
  re-diarization, not a repair; the whole set is refused.
- **Degrade loudly** — a failed or partial audit leaves the transcript
  untouched, sets `ok: false` / `failed_windows`, logs at ERROR and fires a
  Telegram alert naming the unaudited line ranges.

**Windowing.** Audit latency scales badly with transcript length: 30 lines
returns in ~3m45s, the full 119-line transcript did not return within 15
minutes. The audit runs over 30-line windows with an 8-line read-only context
lead-in, up to 3 in parallel. A window may only relabel lines in its own range,
which makes overlapping windows conflict-free by construction. Line numbers are
global, so no offset arithmetic can mis-target a line.

**Identity binding** fixes the `participants: []` case: a generic `Speaker N`
is renamed to a real name only when that name is on the calendar — an identity
the model inferred from the audio alone can never introduce a name nobody
verified.

### Operational notes

**Latency.** The gate adds roughly 4 min (one window) to ~12 min (a 60-min
meeting, ~7 windows, 3 at a time) between transcription and the JSON write.
Transcription is already complete and safe at that point, so nothing is at risk
— but the InsightBase seed and the downstream `/meeting-actions` session are
delayed by that much. Meetings are processed asynchronously, so this is
acceptable; it is called out because it is a real change to pipeline timing.

**Config** (`~/Documents/MeetingRecorder/config.yaml`, all optional):

```yaml
speaker_coherence:
  enabled: true          # false disables the gate entirely
  timeout: 900           # seconds per window
  max_parallel: 3        # concurrent headless sessions — also a quota cap
  model: null            # null = CLI default
  config_dir: null       # falls back to claude_trigger.config_dir (quota isolation)
```

`max_parallel` is a quota control as much as a speed knob. Each window is a
headless Claude session, and the 2026-07-10 incident was fire-and-forget
sessions starving each other on a shared pool. Run the gate under its own
`CLAUDE_CONFIG_DIR` (it inherits `claude_trigger.config_dir` by default) and
lower this if audits start dying on session limits.

**Re-running over an existing transcript** (this is what the backfill uses):

```bash
python3 tools/speaker_coherence.py ~/Documents/MeetingRecorder/Transcripts/2026-08-07_14-11-24.json \
  --attendee "Philipp Baltensperger,BlueCare" --revert-flips
# dry run by default; add --write (in place) or --out PATH
```

`--revert-flips` undoes `speaker_verify`'s flips first using its own decision
log, which is exactly what a transcript stored before the admissibility gate
needs.

## Measured result on source 767

Ground truth is the six labels Matthias confirmed by hand. Reported without
rounding up, including where the gate does not help.

| pipeline | correct | notes |
|---|---|---|
| raw Gemini | 2/6 | `[13:26]`+`[13:58]` swapped as a pair; `[15:07]`, `[15:29]` wrong |
| **+ `speaker_verify` — shipped behaviour** | **2/6** | 12 one-way flips; **94 % Matthias**; breaks `[15:39]`, fixes `[13:26]` |
| + bleed guard alone | 2/6 | flips suppressed, Gemini's own errors remain — but the 94 % collapse is gone and speaking time is a plausible 1469 s / 638 s |
| + bleed guard + coherence gate | **see below** | |

**Verified separately: the guard does exactly what it claims.** Re-running the
real recording through the fixed code: `host_bleed_rate = 0.71`, channel map
prefix suppressed to 0 characters, all 12 flips suppressed, transcript
byte-identical. Speaking time returns from 94 %/6 % to 70 %/30 %.

**The coherence gate's result depends on which transcript it audits, and this
matters more than the headline number.**

- Against the *damaged* transcript (the 94 %-Matthias version), it scored
  **4/6**: it repaired `[13:58]` and `[15:39]` and independently undid 3 of
  `speaker_verify`'s 12 flips — corroborating evidence that those flips were
  wrong, from a method that never saw the audio.
- Against *raw Gemini output* — which is what it will actually receive in
  production, since the guard suppresses the flips — it initially applied
  **zero** relabels. The reason is instructive: raw Gemini labels `[13:26]` as
  Speaker 2 and `[13:58]` as Matthias, i.e. the question and its answer are
  swapped *as a pair*. Locally that reads as a perfectly ordinary exchange, so
  the question/answer adjacency rule sees no violation. **A consistent swap is
  locally coherent.**

That blind spot was closed by adding the mirror of the company-voice rule —
whoever addresses a company in the second person ("dünd **ihr** monatlich
release?", "vo **eurer** Siite") is not part of it — and an instruction to
establish each speaker's identity first and check every line against it rather
than relying on turn-taking. This is the general rule symmetric to the existing
one, not a patch fitted to the fixture.

**Known remaining gap:** `[15:07]` / `[15:29]` — Matthias's own argument, which
Gemini attributed to Speaker 2 and the audit read as the counterpart agreeing
and elaborating ("Ja, genau. Und wenn mer da denn hätted…"). The text supports
both readings and the audit did not flag it as uncertain either, which is a
miss rather than a repair. It is a pre-existing Gemini error the gate fails to
catch, not one it introduces. Rule 7 now names this pattern explicitly.

**Honest summary of the fix's value.** The guard is the load-bearing change and
its effect is unambiguous and deterministic: it removes 81 corpus-wide flips
that were all wrong-by-construction, and it is verified on the real audio. The
coherence gate is a genuine improvement on the class of error the audio cannot
touch — question/answer collapse — but it is a probabilistic stage, it does not
reach every error, and its measured accuracy is a function of what the diarizer
handed it. It is not a guarantee, and the `uncertain_regions` output plus the
Telegram alert on incomplete audits exist so that its limits stay visible.

## Consequences for the reliability plan

**Phase 2 "channel-first speaker attribution" must not be implemented as
written.** It proposes making the channel signal *authoritative* — "assignment
becomes near-deterministic: host segments = Matthias, remote = counterpart" —
and lowering the flip threshold to 2.5 s. On a corpus where the mic hears the
remote 36-80 % of the time, that would apply the source-767 failure to every
meeting, with more turns and higher confidence. Phase 2 is now gated on the
recording actually being separated.

**The capture side is where the channel signal gets fixed**, not the software:
headphones restore separation. The bleed rate is now measured and logged every
run, and the watcher warns explicitly when a recording is inadmissible, so the
regression is visible immediately rather than inferred months later from wrong
wiki pages. Phase 2.4's proposed mic sanity probe should check bleed, not just
signal presence.

## Backfill

See `docs/BACKFILL-speaker-attribution-2026-08-07.md`.
