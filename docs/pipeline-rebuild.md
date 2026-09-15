# Source-preserving pipeline rollout

Implemented in an isolated worktree. Keep the live recorder until native capture passes acceptance; no model winner is assumed.

## Configuration (opt-in)

Merge these keys into a reviewed configuration; do not replace credentials or paths:

```yaml
audio:
  capture_backend: screencapturekit
  native_bundle: ~/Documents/MeetingRecorder/bin/MeetingNativeCapture.app
  native_start_timeout_seconds: 30
processing:
  mode: gemini
  transcription_pipeline: durable_sources
durable_transcription:
  chunk_seconds: 180
  request_timeout_seconds: 120
  job_timeout_seconds: 600
  state_dir: ~/Documents/MeetingRecorder/Transcripts/.source-jobs
```

Default backend/pipeline remains `legacy` for an explicit migration boundary. Selecting an unknown native backend fails rather than silently switching to unsynchronized capture. See `native-capture.md` for build, signing and permission setup. A running Python recorder cannot load changes without a controlled restart while idle.

Native completed segments and `capture-manifest.json` live under CaptureArchive. A finalized compatibility WAV is published atomically with a `.capture.json` binding. The new transcription route reads the original source manifest; it does not infer source identity from the compatibility mix. Derived per-track WAVs preserve the common timeline; zero padding is accompanied by capture-gap metadata and is never evidence of silence. Mixed segment formats or overlaps fail visibly and leave originals intact.

The new route publishes anonymous draft words before source storage, naming or analysis. It bypasses calendar-name guessing, channel flips and semantic relabeling. Source-local speaker labels do not establish cross-chunk identity. Failed source chunks and failed database publication are retryable; uncertainty requiring human review is not automatically retried. Existing historical partial outputs are not automatically backfilled on enabling the option. Durable database publication uses a stable capture identity and a transaction lock to update one source while preserving transcript revisions locally.

The durable route does not initialize noScribe or the legacy Gemini analysis processor. A missing API key remains a visible, retryable ASR failure; it does not silently fall back to another engine. Unknown pipeline names fail during startup. Provider requests use a validated structured response schema, and its request contract is part of the cache identity.

## Recovery without publishing

Run the following from this checkout with the existing Python runtime, setting GEMINI_API_KEY securely in the calling environment. No database, messaging or canonical transcript changes occur:

```sh
python tools/recover_source_transcript.py /absolute/recording.wav \
  --output-dir /absolute/private-recovery-dir --prepare-only
python tools/recover_source_transcript.py /absolute/recording.wav \
  --output-dir /absolute/private-recovery-dir --model gemini-2.5-pro
```

Repeat the second command to resume failed jobs; successful jobs are cached by input/model/prompt identity. A run has a finite budget and may need another invocation. Do not treat a draft as a verified transcript.

Legacy files have unverified timing. `--offsets-json` accepts a reviewed `{ "system": 46.28 }` mapping for a specific diagnosis, not a universal fix. Native timestamps cannot be overridden with this option. Never reuse the late-call estimate on another recording.

After a native process crash, use `--manifest /path/capture-manifest.json --recover-stopped-native --prepare-only` with the recovery command. This refuses a still-live recorder PID and leaves the original manifest and partial CAF untouched. It recovers only closed, checkpointed segments and explicitly marks the final tail duration unknown. Remove `--prepare-only` to transcribe that recovery as a held partial draft.

## Sentence-level identity review

Export a small review batch with `python tools/source_speaker_review.py /path/transcript.draft.json --output /path/examples.json --candidates "Matthias Heim" "Another participant"`. Each item contains the verbatim sentence, times, source, choices and no default selection. Selection balances sources and prefers complete sentences. The existing clip-review UI can use these records. Candidate names are choices for the human; they are never ASR prompt context.

To apply explicit answers, pass `--confirmations /path/answers.json --output /path/confirmed.draft.json`. Answers are a list containing `segment_id`, `sentence`, `name`, and `basis: "human_confirmed"`. The hash binds the sentence to its durable source job, preventing a response from another recording from matching. Confirmation applies only to that segment; it does not name the whole source or release downstream actions.

## Preservation and acceptance

The revised pruner preserves sources unless a separately certified lossless replacement exists. An MP3 plus a JSON file is insufficient. `source-inputs.json` is not such a certificate. Storage must be monitored; indefinite retention is not a substitute for verified lossless compression.

Before switching capture: native compile/self-test; synthetic delayed-track/gap tests; real cold-start capture with both permissions; delayed remote start; route change; sleep/wake; forced crash and write failure; a 60-minute timing check. Target <=50ms alignment error and no unreported gaps. Retain original samples and logs from each test. OS permission prompts and human identity labels cannot be simulated as passed.

Before switching ASR: replay representative source clips and a failed meeting; interrupt/resume to prove completed chunks are not repeated; malformed responses and silence must stay explicit; source publication must remain idempotent. Compare words and voices on human-labelled examples before changing the selected model. Ten shadow meetings are the subsequent reliability acceptance stage, not a completed claim.

The current production worker processes one bounded request at a time. The five-minute draft target is **not yet met or certified**. Separate diagnostic recovery jobs may run concurrently, but that does not establish production throughput. Freeze a runtime snapshot before real comparisons: Python spawned workers import the module from disk, so editing that module during a run can otherwise change the request contract mid-job.
