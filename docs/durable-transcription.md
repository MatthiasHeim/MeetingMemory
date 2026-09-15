# Durable source transcription

`tools/transcription_jobs.py` is a standalone ASR boundary. It has no watcher,
calendar, database, webhook, attendee list, or semantic-analysis dependency.
It accepts immutable source tracks and produces a source-separated draft with
anonymous speaker labels. The original tracks remain authoritative and are
never deleted or replaced.

## Input contract

Call `SourceTranscriptionPipeline.run()` with source records in this shape:

```json
[
  {
    "source_id": "mic",
    "path": "/absolute/path/to/mic-full-timeline.wav",
    "start_seconds": 0.0,
    "timing_basis": "host_clock"
  },
  {
    "source_id": "system",
    "path": "/absolute/path/to/system-full-timeline.flac",
    "start_seconds": 0.0,
    "timing_basis": "unverified_sample_zero"
  }
]
```

`source_id` is an opaque lower-case source token, never a person's name.
`start_seconds` is a declared source offset; it is not evidence that tracks
share a clock. The worker marks only explicit `host_clock`, `common_clock`, or
equivalent verified timing bases as verified. It leaves legacy sample-zero and
manual offsets unverified and never interleaves sources into a synthetic shared
timeline.

Pass capture evidence separately when available:

```python
payload = pipeline.run(
    sources,
    session_id="opaque-session-id",
    capture_gaps=[
        {"source_id": "system", "start_seconds": 0, "end_seconds": 46.28,
         "reason": "capture_gap"}
    ],
)
```

If `system` is absent altogether but the capture manifest declares a gap for
it, the worker still transcribes `mic`. The missing source is retained in
`_meta.missing_ranges` and `_meta.source_provenance`; it is not replaced by
silence or rejected as malformed input.

## Durable model

The default chunk is 180 seconds. The constructor permits only 120–300-second
chunks, so each network request has a bounded audio problem. A job identity is
a SHA-256 digest over all input file hashes, source timing records, capture-gap
records, model, prompt version/hash, schema hash, Gemini response-request
contract version/hash, and chunk size. Changing a source byte, model, prompt,
schema, response contract, gap declaration, or chunk plan creates a different
durable job rather than reusing stale output. Existing job directories remain
intact; a changed request contract starts a new job and only reuses results
whose complete input and request identity matches.

State is written under:

```text
<state-dir>/jobs/<job-hash>/
  manifest.json
  clips/<source-id>/chunk-0000.wav
  attempts/<source-id>/<chunk-id>/attempt-0001.json
  segments/<source-id>/<chunk-id>.json
```

Clips are source-local uncompressed WAV files and retain the source channel
layout and native sample subtype where supported (including FLOAT timeline
WAVs); the worker never mixes microphone and system tracks. JSON attempts and successful
segment results are written through a temporary file followed by an atomic
rename. The per-job advisory lock prevents a watcher and a manual recovery run
from processing the same hash concurrently.

Every planned region has one durable outcome:

- `transcribed`: valid, source-local text segments were retained.
- `no_speech`: the model said no speech **and** the derived PCM clip was exact
  digital zero.
- `uncertain`: usable text may be retained, but the region needs review.
- `failed`: no durable terminal response; it remains retryable.

On another call with the same identity, `retry_failed=True` processes only
failed/pending regions. It does not repeat chunks already marked transcribed,
no-speech, or uncertain. Useful uncertain text stays in its result file;
uncertainty is not silently replaced by a retry loop.

## Completeness and speech evidence

The worker does not use the final timestamp of a transcript as coverage. It
reports `missing_ranges`, `retryable_ranges`, `review_flags`, and a separate
`structural_completeness` object. `accuracy_verified` is always false: durable
accounting says that every audio region has an outcome, not that every word is
correct.

The default detector uses WebRTC VAD if installed, but a VAD-negative nonzero
clip remains non-conclusive. A model-only `no_speech` answer becomes
`uncertain` unless the clip is exact digital zero. A VAD-positive/model-silent
disagreement is flagged as `unexplained_speech`; known capture gaps are flagged
as `known_capture_gap`; unavailable detector evidence is flagged as unknown.
This prevents a model response from erasing quiet speech or a known recording
failure.

Malformed segment timestamps are not clamped. Their raw response is retained,
valid segments (if any) are kept, and the clip becomes `uncertain` with a
`malformed_segment_bounds` flag. A model-declared `uncertain_ranges` interval
is retained as an exact missing range, so a successful-looking final timestamp
cannot hide it.

## Speaker and timing boundaries

The transcription prompt requests only source-local start/end times, words,
and anonymous speaker labels. It forbids names, attendee information, calendar
information, emotions, sentiment, summaries, actions, and analysis. Labels are
namespaced as `source-id:chunkNNNN:speaker_01`; identical local labels in two
chunks never claim to describe the same human. `_meta.speaker_attribution`
always remains `hold` with an anonymous basis, including
`speaker_dependent_actions: "hold"`.

Segment `start_seconds` and `end_seconds` are source-local positions plus the
source's declared offset. Each carries provenance in the source metadata. A
consumer must not compare values across unverified sources as if they were a
common clock.

## Time limits and Gemini

`request_timeout_seconds` is a per-adapter-call limit (120 seconds by default).
For Gemini, it is passed to `google.genai.types.HttpOptions(timeout=...)` in
milliseconds; installed SDK 1.60.0 documents that field in milliseconds. The
request also runs in an isolated worker process. A timed-out worker receives a
process-group termination and is reaped, so it cannot block subsequent jobs.

The Gemini request sets both `response_mime_type="application/json"` and
`response_schema=TRANSCRIPTION_SCHEMA` in `GenerateContentConfig`. The local
SDK accepts that structured-output field (its underlying SDK name is
`responseSchema`). This makes JSON output a request contract rather than only
a formatting preference. The contract is versioned and hashed into the durable
job key, so a structured-output run does not treat an older JSON-only cache as
equivalent.

`job_timeout_seconds` is a wall-clock budget for one whole `run()` call (600
seconds by default), including source hashing, duration probing, job-lock wait,
clips, and retries. When it expires after a durable job can be identified,
remaining regions are marked retryable and a later invocation resumes them. If
it expires during input preparation, no unsafe partial identity is invented and
the caller can retry preparation. The built-in
Gemini adapter uses `spawn`, rather than forking a watcher process that may
already hold CoreAudio or Torch state. Explicit local/test adapters may use
`fork` on POSIX for simple isolated fakes.

No Gemini request occurs on import or pipeline construction. The optional
adapter reads only `GEMINI_API_KEY` when a pending clip is actually processed.

## CLI

The standalone CLI reads a JSON manifest and writes one payload. It does not
load MeetingRecorder configuration or trigger a service.

```bash
python3 tools/transcribe_sources.py /path/to/source-inputs.json \
  --state-dir /path/to/durable-state \
  --output /path/to/durable-transcript.json \
  --model gemini-2.5-pro \
  --request-timeout-seconds 120 \
  --job-timeout-seconds 600 \
  --chunk-seconds 180
```

The manifest can be either a source list or an object with `sources`, optional
`session_id`, and optional `_meta.capture_gaps`. Exit status `2` means durable
state contains retryable failures; a review-needed result is still written and
exits successfully because it is a usable draft rather than an execution
failure.
