# Speaker attribution production trial

The September 2026 trial prevents unsupported channel rewrites, retains capture
and attribution evidence, and tracks verification independently of transcription.
The existing Gemini model, confirmed-bleed ducking and semantic repair remain
unchanged. A new model, authoritative diarizer and echo cancellation must earn
promotion through audio evaluation; the offline tools are experiments only.

## Production behavior

- Less than 60 seconds of system speech is unknown isolation. No channel prompt,
  fusion or speaker rewrite is permitted on that evidence.
- Across the whole recording, two-minute windows with at least 60 seconds of mic
  activity and less than five seconds of system activity are marked uncertain.
  This includes possible phone handovers and legitimate long monologues. The
  prompt map and host-cluster fusion are suppressed; verification cannot rewrite
  turns intersecting these windows. Other well-supported intervals remain usable.
- Unknown isolation does not trigger ducking. Confirmed bleed still uses the
  existing mix. Exact digital silence produces an explicit empty result without
  a model call or downstream extraction.
- Label changes invalidate speaker-specific audio statistics. Transcript gap
  durations are not claimed to be speaking time.
- `_meta.speaker_attribution` separates completed text checks, partial channel
  checks and unresolved acoustic identity. Raw transcripts and neutral metadata
  are stored immediately. The legacy workflow combines extraction with actions;
  held meetings therefore wait in the stage queue before insights/actions/drafts
  run. This deliberate delay must be reported in the trial's usefulness metrics.
  Held enrichment omits model-inferred people and per-speaker signals; verified
  calendar attendees remain attendance metadata. The zero-insight reconciler
  excludes held sources, so it cannot bypass this restriction.
  Capture notifications explicitly report the hold rather than claiming the
  action workflow is running. Transcript-derived counterpart hints cannot
  populate held attendance/company updates.
- Capture pads the shorter stream when merging instead of truncating the longer
  one. Original mic/system streams are moved to CaptureArchive with mic ADC timing,
  device details and discontinuities. This is still independent-clock capture:
  the current tap binary does not expose system sample timestamps. The code does
  not claim sample synchronization or apply an estimated delay to production.

## Evidence and controls

State: `~/.local/share/meeting-pipeline-trial/2026-09-05/` (private, outside Git).
`manifest.json` freezes baseline commit, original config hash, eight historical
meetings, dates and decision rules. `config.before.yaml` is private mode 0600.
The baseline worktree stays detached; each run checks HEAD, tracked/untracked
cleanliness and the config hash before import and again before certification.
Candidates fingerprint the original WAV before processing and verify it again
before archiving. Baseline replays require that candidate fingerprint before and
after processing; collection rejects certificates for a different audio hash.
Collected baselines must also match the code/config certificate and output hash.
Only trial configuration is added to the
live config; original credentials and model settings remain intact.

Each meeting archives immutable revisions before attribution, before channel
verification, before semantic repair, and at candidate publication. A pending
verification queue is keyed to these stages, even after content extraction.
`speaker_trial.py collect` discovers missing transcripts and current queue state.
It uses the archived candidate revision so later human corrections cannot inflate
candidate quality. `shadow-next` replays the frozen baseline against the same
recording and attendee list, with all DB writes, extraction and notifications
disabled. It retains baseline outputs and failures; at most two attempts per file.
The normal usage guard still applies to baseline Claude checks.

`retry-audits` runs a bounded advisory text audit through the already configured
Gemini API independently of Claude's interactive quota. It never applies those
proposals or claims that a text audit established acoustic identity. Failed audits
retry at most three times. Acoustic review and missing-content recovery stay queued.

`audio_reference_probe.py` compares delay and a static, first-half-trained echo
filter on derived excerpts. Its held-out energy reduction is not speaker accuracy.
Use `--confirmed-remote-only` only on independently established remote-only audio.
Use the existing local `diarize.py` worker as a shadow anonymous-clustering challenger.
The earlier model runner and evidence remain in the separate September 5 assessment
directory; changing a model string is not a dedicated-ASR integration.

## Frozen decision rules

Review on Saturday 12 September 2026, 09:00 Asia/Hong_Kong, after next week's calls.
Aim for 20+ minutes from at least five meetings; sample at fixed beginning/middle/end
positions as well as every changed or uncertain region. Include dialect, phone
handover, overlap and three-plus-speaker cases if they actually occur. Preserve
the six historically confirmed labels as a regression fixture; do not replace
them with model-generated labels. Report missing strata and insufficient volume.

Primary metric: paired confirmed speaker-confusion seconds / reviewed speech
seconds. Improvement target: at least 50% relative reduction. Roll back if:

- confirmed speaker error or WER worsens by more than 2 percentage points;
- overlapping-host recall falls by more than 5 percentage points;
- a newly introduced, confirmed critical owner error, omitted utterance or duplicate
  echo utterance is found;
- paired partial-transcript rate increases by more than 10 percentage points with
  at least five comparable meetings;
- a trial-caused capture loss, repeated processing failure, unsafe speaker rewrite,
  stale post-relabel metadata or silence hallucination is confirmed.

Also report end-to-end latency, failed checks, queue age, timestamp validity,
missing-chunk duration, downstream owner/draft holds and free disk. Establish
causality for operational failures instead of attributing existing quota/disk
problems or old malformed output to the patch. Baseline cohort rates are descriptive,
not a substitute for paired recordings. Human/channel-confirmed evidence can score
accuracy; model-only adjudications are explicitly unconfirmed and cannot count.

Otherwise keep production. With insufficient labels or volume, the verdict is
`keep_inconclusive`, following Matthias's requested policy; do not claim improvement.
The threshold for a measured improvement does not become lower at the deadline.

Commands (use MeetingMemory's venv Python):

```bash
python tools/speaker_trial.py collect
python tools/speaker_trial.py shadow-next --limit 2
python tools/speaker_trial.py retry-audits --limit 2
python tools/speaker_trial.py evaluate
python tools/compress_capture.py --target-free-gb 15
```

Append confirmed review rows to `adjudications.jsonl`. Required keys: `meeting`
(recording stem), `start`, `end` (seconds), `truth_source` (`human_confirmed` or
`channel_confirmed`), `audio_evidence`, exact `baseline_sha256` and
`candidate_sha256` from collected metrics, and boolean `baseline_speaker_wrong` /
`candidate_speaker_wrong`. Optional paired flags: `critical_owner_error`,
`omitted_speech`, `duplicate_echo_text` (prefix each with `baseline_` / `candidate_`).
For WER include `reference_words`, `baseline_word_errors`, `candidate_word_errors`.
For overlap include `overlapping_host_seconds`, `baseline_host_recalled_seconds`,
`candidate_host_recalled_seconds`. Segment rows must not overlap. Include both
correct and incorrect samples. Pure model review belongs in a separate proxy file.
Confirmed operational regressions use `kind: confirmed_operational_regression`,
`introduced_by_trial: true` and specific `evidence`.

## Deployment, review and authorized rollback

Publish through a MeetingMemory feature branch and PR; never push that repo's main
directly. Deploy by fast-forwarding the clean main checkout and restarting its
watcher/recorder LaunchAgents only when idle. Record commit, PR, PIDs and runtime
config in the manifest. Do not stop an active capture or processing job.
`python tools/recorder_status.py` queries the recorder's own CoreAudio process:
exit 0 means idle, 1 active, 2 unknown. Do not treat another application's mic
usage as recorder activity, or a missing system tap as proof that its mic is idle.

The daily Codex heartbeat collects evidence, runs bounded shadows/advisory retries,
checks storage and missing stages, and stays quiet unless action is required. The
Saturday standalone automation opens the requested new review session. It finishes
backlog shadows, reviews paired audio, runs evaluation, then keeps or rolls back
without asking again: Matthias authorized both outcomes on September 5.

Rollback: save the evidence and verdict first. On a new `codex/` branch off current
origin/main, revert ONLY the recorded MeetingMemory trial merge (use `-m 1` for a
merge commit), resolve later changes conservatively, test, publish a PR and merge.
Then fast-forward the production checkout and restart idle services. Never reset
main to an old commit or overwrite unrelated work. Restore only the trial config
keys from the private backup; preserve later credentials and unrelated settings.
Revert the recorded Brain gate/document integration commit through Brain's normal
main workflow, also preserving later edits. Never revert manual transcript fixes,
delete recordings or undo transparent byte-identical compression. If current work
conflicts, isolate the trial hunks and keep the evidence, rather than discarding
other changes. Verify capture/service health after rollback and record the result.

After either decision, record `review_complete`, result and timestamp in the
manifest; pause both trial automations using their saved IDs. Keep raw evidence.
Historical transcript/insight repair remains gated on confirmed audio evidence;
use Brain's existing revision/re-extraction workflow, not direct insight edits.
