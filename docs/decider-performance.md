# Decider latency: diagnosis and measurement plan

2026-10-11. Static inspection and saved measurements only. No inference requests,
runtime changes, service restarts, downloads, or power-setting changes were made
for this assessment. New benchmarks await an explicit coordination message that
the Qwen historical-email and workflow grading has finished.

## What the existing timings establish

**The 12–16 second figures measure context-rich requests with four concurrent
clients, including queueing. They do not establish isolated interactive latency.**
They also cannot be divided by four to estimate individual response time.

| Saved check | Measurement | Interpretation |
| --- | --- | --- |
| 15 short synthetic cases | 1.331 s median request | Short-input reference; not a normal meeting tick |
| Four concurrent short calls | 2.127 s total | Evidence that batching can help short-call throughput; not a live-latency percentile |
| English relevance, 60 lines | 12.156 s median request; 194.39 s total | Four concurrent callers, six preceding lines per decision |
| Swiss German relevance, 60 lines | 14.405 s median request; 211.10 s total | Same client concurrency; different inputs |
| Swiss German prompt detection, 161 lines | 16.066 s median request; 705.12 s total | Also overlapped recorder/demo smoke tests; a mixed-load result |
| Two-line recorder card check | 3.460 s to one card | Real local decisions, synthetic transcription, identity cleaner; excludes real transcription and rewriting latency |

These checks used different inputs and workloads. They are not an A/B experiment.
The saved Jev reference contains scores, not timings: there is no defensible Jev
versus Decider speed ratio. The 1.9 s startup log times model loading only, before
kernel warmup; it is not full cold-start readiness. No clean cold/warm comparison
or per-request queue breakdown was saved.

## Where time goes

1. **Recorder cadence and upstream work.** `LiveEngine._loop` waits 20 seconds,
   processes a tick synchronously, then waits another 20 seconds. Tick starts are
   therefore separated by 20 seconds plus processing time. Each tick transcribes
   up to 60 seconds of audio, commits/revises lines, scores them, and may clean a
   prompt before publishing. Gemini transcription and cleaning are separate
   latency sources. Changing the cadence can change audio windows and decisions;
   it is outside the first output-preserving experiment. Audio capture is separate.

2. **Context and repeated scoring.** After six existing lines, adding `n` lines
   normally scores `n + 6` lines. The overlapping six receive 0–5 preceding lines
   within the newly sliced subset, potentially different from their earlier
   context. Their scores are merged with the earlier scores using `max`. These
   are not automatically duplicate requests. Revisions can rescore a longer
   suffix. Skipping this work or caching by line ID would change semantics.

3. **Queueing and padding.** One MLX worker runs GPU batches. Current settings are
   maximum batch 4, padded-token budget 2,048, two preparation threads, and a
   20 ms collection wait. The scheduler groups similar lengths (longest at most
   1.25 times shortest) and selects the group containing the oldest pending job.
   There is no live-versus-bulk priority. Two inputs over 1,024 tokens cannot share
   a batch under this budget; an individual input may exceed the budget, up to
   the 8,192-token request limit. Removing a 20 ms wait alone cannot explain away
   multi-second requests. More batching might improve throughput or worsen tail
   latency and memory use; that remains to be measured.

4. **Client overhead and limited caching.** Each line starts a Python helper
   process, serializes JSON, opens a loopback HTTP request, validates the answer,
   and writes metadata. The live judge is not wrapped in `CachedJudge`; finished
   transcript judgments are. Repeated topic clips may therefore repeat identical
   decisions. The saved helper audit had 58 successful requests and 56 distinct
   request hashes at inspection: only two duplicate entries. That small, mixed
   sample does not support a large routine-live cache saving. Direct API callers
   are not covered by this helper audit; its token counts sum across questions.

5. **Memory and warm state.** The separate Qwen experiment recorded a Decider OS
   process footprint of **35.896 GiB** on the 48 GiB laptop, plus substantial
   system compression and swap. This is a point-in-time footprint, not a measured
   peak, resident-RAM count, or permanent requirement. Weight files of about
   15 GB do not describe runtime memory. Qwen generation and Decider grading were
   sequential; those records do not measure simultaneous inference contention.
   Existing swap also predates individual runs. Memory refaults and competing
   applications are plausible contributors, not a quantified causal explanation.
   The LaunchAgent keeps Decider loaded; it does not reload weights for each call.

The custom forward uses noncausal full attention and a dedicated classification
head. Conventional causal-chat prefix/KV caching cannot be assumed correct:
extending a sequence can change earlier hidden states. Preserve the custom loader,
attention masks, pooling, temperature, and head.

Current observability is insufficient to assign percentages to these causes.
Server `gpu_s` is wall time for collation, forward, and probability materialization,
not GPU hardware telemetry. `/stats.queued` excludes the scheduler's private
pending list and active batch. Audit records contain no request durations.

## Improvements worth testing, in order

| Candidate | Expected benefit and limits | Preservation requirements |
| --- | --- | --- |
| Schedule bulk work outside live use; later consider fair live-priority admission | Potentially largest reduction under contention; no benefit when already idle. Priority shifts waiting to bulk work and cannot preempt an active GPU batch. | Same requests; no dropped decisions. Preserve fair progress, ordering, and timeout behavior. |
| Exact-request memoization and in-flight deduplication | Avoids a complete forward on a true hit. Current evidence suggests limited general hit rate; repeated clips are a better target. | Include exact ordered state/question bytes, model revision, tokenizer/template/runtime identity, temperature and schema. Keep recording gates and audits on hits. Bounded local cache; no raw private text in logs. Never line-ID-only caching. |
| Reuse the helper's client in-process, optionally with persistent HTTP connections | Removes process startup/imports and connection setup per line. Likely modest against multi-second model execution; unmeasured. | Preserve loopback pinning, no proxies/redirects, model/probability checks, audits, errors, timeouts, and ordered results. |
| Tune client concurrency and scheduler budget separately | Could reduce padding, queueing or memory pressure. Increasing limits is not inherently faster. | Same full inputs, precision and thresholds; verify numerical equivalence and downstream decisions across batch shapes. |
| Investigate MLX allocation retention or kernel compilation only after profiling | Unproven here; clearing useful caches can make latency worse. | No runtime upgrade, allocator change or compilation activation without an isolated comparison and equivalence checks. |

Reducing history, shortening/rephrasing questions, skipping the six overlap
decisions, changing polling windows, weakening precision/model, or moving to a
cloud model are separate quality/behavior changes, not claimed optimizations here.

## Comparable benchmark, after the coordination signal

**Freeze the baseline:** `pplx-decider-v1.1-27b`, MLX 4-bit revision
`66383f28b37637658b02b8e82b1029598b138642`, temperature
`1.0087417621345625`, exact source context and question/criteria order. Keep prompt
threshold **0.92** and clip thresholds **0.10/0.025** (wide **0.05/0.01**).
Installed packages at inspection: MLX 0.32.3, mlx-vlm 0.7.6, Transformers 5.19.0,
NumPy 2.5.3, FastAPI 0.142.2. Record complete package and source hashes when running.

1. **Unchanged service baseline first.** After explicit clearance, use a frozen,
   locally stored manifest of exact request bodies from the existing reference
   material, with no private text exported. Replay sequential warm requests,
   then the identical manifest at concurrency four. Cover observed short/medium/
   long token lengths and multiple-question requests. Use three paired passes
   of at least 60 decisions per scenario; keep warmup separate and report sample
   counts. No additional cloud transcription/cleaning calls are needed.

2. **Recorder replay.** Preserve existing source lines, revision events and question
   definitions. Measure whole scoring bursts with 1, 3, 6 and 12 new lines plus
   their actual overlap decisions, and repeated topic clips. Use fixed transcriber/
   cleaner outputs. Report both decision latency and card availability; explain
   that real audio-to-card latency additionally includes cadence, transcription
   and cleaning. Do not modify live recorder behavior for this replay.

3. **Decompose before tuning.** A separately reviewed instrumented build should
   record monotonic timestamps for client start, server receipt, preparation,
   queue entry, worker start/end, response, validation/audit and card publication.
   Record request hash, question count, unpadded/padded lengths, batch size,
   pending/active counts and cache hits, without payload text. Existing API timing
   alone cannot supply this breakdown. Sample process footprint, MLX active/cache/
   peak allocation, system compression and swap-in/out deltas, other model
   residency, and thermal/power state without changing power settings.

4. **One variant at a time, A/B/A.** Start with exact caching and client reuse.
   Only then compare concurrency 1/2/4 and scheduler budgets 2,048/4,096; evaluate
   wait 0/20 ms if its contribution warrants it. Report request and burst p50/p95,
   throughput, errors, memory and cache-hit rate separately, with comparable load
   and fixture ordering. A mixed live/bulk scenario is a later explicitly
   coordinated test. No cold-start test or instrumented-server/scheduler change
   may restart the current service without a separately agreed maintenance window;
   do not load a second 27B model alongside it to bypass that constraint.

5. **Acceptance gates.** Client/cache changes must preserve the returned answer
   fields and probabilities exactly, apart from transport IDs/timing. For
   scheduling/kernel variants, predeclare maximum absolute probability drift
   of 1e-5, unchanged option winners and no crossings of any fixed threshold,
   with identical replayed clips/cards. Check baseline repeatability first;
   if it already exceeds this bound, stop and diagnose instead of relaxing it
   after seeing the candidate. These are finite-fixture regression checks, not
   proof of universal equivalence. Require a repeatable latency improvement
   larger than run-to-run variation, no new errors, and no memory-pressure
   regression before recommending rollout. Borderline inputs must remain in the
   comparison; the old machine-labelled calibration set is not held-out accuracy.

The first deliverable from that experiment should be an isolated baseline and a
queue-versus-compute breakdown, not a promised speedup factor.

## Evidence locations

- Recorder source at commit `5d6e824d`: `tools/sidecar/live.py` (`_loop`,
  `process_snapshot`, `_extend_scores`), `live_session.py`, `judges.py`
  (`DeciderJudge`, `CachedJudge`), `decider_policy.py`.
- [Migration and functional validation](local-decider.md).
- Local-only reports under `~/.local/share/agent-skill-state/decider/`:
  `acceptance.json`, `meeting-reference-results.json`, `evaluate_meetings.py`,
  `live-card-check.json`, `audit.jsonl`; upstream `model/server.py` and
  `model/decider_mlx.py`.
- Shared wrapper: `~/Repos/agent-skills/skills/decider/scripts/serve.py`;
  LaunchAgent `~/Library/LaunchAgents/com.lailix.decider.plist`;
  startup log `~/Library/Logs/Decider/stderr.log`.
- Saved resource measurements:
  `~/Repos/Brain/outputs/qwen38-historical-email-2026-10-10/memory-report.json`.
  Decider footprint snapshot saved at 2026-10-10 21:50:34 UTC.

No production behavior changed. Functional suites from the migration were not
rerun because this work only adds documentation.
