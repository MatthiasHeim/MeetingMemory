# Live transcription and prompt-worker repair

2026-10-14. Prepared on `codex/live-prompt-nonblocking`, based on `f54f7a96`.
The running recorder, its checkout, configuration and model service were not changed
or restarted. No audio replay or additional model inference was performed.

## Observed runtime

The running `meeting_recorder.py` process uses `/Users/Matthias/Repos/MeetingMemory`
on `codex/local-decider`. The active recording sidecar and preference both select
Decider. Loopback `/health` returned `ok`; `/memory` reported the model loaded;
`/stats` reported no queued requests at the observation instant.

Six sampled live batches between 20:37 and 20:41 Europe/Zurich requested 2, 4, 6,
8, 9 and 10 decisions, with 39 matching successful helper entries. Their starts
were 34–59 seconds apart despite the nominal 20-second tick. The helper audit's
timestamp is the request **start**, written after success; it is not a completion
timestamp or a direct duration measurement. Neither audit records probabilities.

The user's screenshot at 20:41 shows local and remote transcript lines and one
detected card. Consequently, total detection failure and a missing microphone
stream are not supported by that evidence. Earlier requested examples might
precede this recording's start. The exact recall of the ongoing conversation is
unverified; no threshold or taxonomy change is justified by these observations.

## Confirmed code defects and repair

- The live worker used to transcribe, commit, classify, rewrite, then publish.
  Classification and rewriting blocked both visible transcript updates and the
  next transcription pass. Judge initialization could also block transcription
  startup. Transcript, scoring and rewriting now use separate workers, with
  short synchronized state updates and no provider call under the state lock.
- A failed score batch left the score prefix behind the transcript. The next
  request was anchored to the number of newly added lines, potentially skipping
  older unscored lines; silence did not trigger a retry at all. Requests now
  resume from the processed prefix, including on unchanged tails. Each request
  contains at most eight new lines and six context lines. Successful chunks are
  committed independently. A failing range is retried unchanged up to three
  times, with backoff, then marked explicitly unscored so later lines can proceed.
  Unknown gaps are hard card boundaries, never scored negatives or implicit
  instruction bridges. Exhausted gaps require later context rescoring or an
  authorized replay; they are not retried indefinitely in silence.
- Published cards could retain replaced transcript text. Revisions now invalidate
  affected scores/cards; in-flight responses for superseded text are discarded.
  Clean rewrites are applied only to the exact card text they were generated from.
  Cards with unchanged text retain their rewrite when an earlier merge shifts
  their indices. A card whose actual source wording changes is removed pending
  rescoring and can remain absent if the new wording scores below threshold;
  carrying its old classification forward is a separate UX/semantic decision.

There is at most one scoring worker and one rewriting worker per session.
Pending scoring work is coalesced into the latest full transcript, retaining all
committed lines. Appended lines do not invalidate an already scored prefix;
accepted verbatim cards appear before rewriting. Slow providers cannot block
Stop or the audio callback. Workers drain pending work without joining the UI.

Model, provider gates, routing, concurrency inside Decider, threshold (0.92),
canonical questions and six-line context/rescoring policy are preserved. Bounded
request splitting changes batch boundaries for large backlogs and can change
context rescoring/model batch shapes; identical probabilities are not claimed.
Offline
`replay_wav` explicitly uses synchronous prompt processing for deterministic
reports. The recorder uses asynchronous processing by default.

The cadence still waits 20 seconds between transcription passes; ASR request
latency remains. This change removes downstream delays, not the initial capture
window or transcription latency. Overloaded classification can still lag behind
the transcript, now visibly marked as pending.

## Text-free diagnostics after activation

The recorder's existing stderr log will contain `LIVE_SIDECAR` JSON events:

- `transcript`: request duration, committed/added line counts, revision index,
  microphone sample count, system-stream presence and local-line count.
- `prompts`: backend, duration, processed/scored/card counts, threshold, cumulative
  maximum and positive count, plus the requested size, new line index range, new
  maximum and positive count, pending/failed counts and whether the response was
  stale. Later near-misses are not hidden behind an earlier positive score.
- Failure events include only the stage and exception class, not provider error
  messages, plus request size, range, processed/pending/failed/skipped counts and
  consecutive failures. No audio, transcript, rewritten prompt, speaker names or
  secrets are logged.

These distinguish no microphone/transcript input, pending decisions, successful
negative scores, rejected stale responses and provider failure. They do not by
themselves establish semantic recall; that requires an explicitly authorized,
representative labelled evaluation.

## Screenshot and prompt-span follow-up

`LivePanel._render_cards` gives each card 92 points total, the clean text 52 points
and the complete original text only 28 points. It passes the full joined source
lines to that small label. The screenshot's clipped source is therefore explained
by layout; it is not proof of truncated stored text. The document view's bottom
origin also explains the card sitting low in a mostly empty scroll region.

Current extraction groups above-threshold lines using a 20-second difference
between **line starts**, rather than the pause from one line's end to the next,
and limited low-scored bridges. Long continuous turns can therefore be split.
It has no explicit notion of a complete user request, semantic
topic change, assistant turn or a correction to a previous request. Those are
limitations of the span model, separate from classification latency and display
clipping. This repair does not redesign those rules, apart from preventing a
failed score from being used as an instruction bridge.

A subsequent span design should represent one actionable user request with its
constraints and corrections. Show a provisional card early and extend it while
the speaker continues the same request; finalize on a credible request boundary
or topic change, using a pause only as a hint. Keep assistant responses as context,
not copied instructions, once speaker roles are established. A short clarification
exchange may amend the same request; a new task should create a separate card.
Keep full source ranges inspectable and use expandable, content-sized card rows.
Evaluate segmentation separately from detection, including long requests,
corrections, interruptions, assistant replies and distinct adjacent tasks.

## Validation and activation

The focused recorder/sidecar suite uses synthetic providers and audio. Regression
coverage includes blocked initialization, blocked classification while subsequent
transcriptions continue, non-blocking Stop, coalescing and ordering, revision
invalidation, failure recovery with and without new speech, invalid probabilities,
verbatim-first cards, blocked/stale rewrites, diagnostic privacy and unchanged
provider gates.

Activation requires applying the reviewed commit to the recorder's checkout and
restarting the recorder only after the active recording has ended, with the user's
approval. Then verify one explicitly addressed request, one ordinary discussion,
one continued/corrected request and source integrity against the visible transcript
and metadata. Prompt-span and layout changes remain a separate proposed follow-up.

One independent Opus review completed (967 seconds, 34 turns). It found a blocking
persistent-failure stall in the first implementation; bounded chunk progress,
retry caps/backoff and explicit unknown gaps fix that finding. Per-pass diagnostics
and index-shift retention fix the additional factual findings. Keeping a changed
card visible as provisional remains a design choice for the span/UI follow-up.
The review was not rerun after these fixes; regression tests validate the changes.
Artifacts and adjudication are in
`~/.local/share/agent-skill-state/panel-review/meetingmemory-live-workers-20261014/`.
