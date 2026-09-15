# Pipeline rebuild implementation — 15 September 2026

Implementation lives on `codex/meeting-pipeline-rebuild` in the isolated MeetingMemory rebuild checkout. The existing recorder, watcher, production configuration and canonical transcripts have not been replaced. Four retention holds protect the affected 14 September recordings during recovery.

## What changed

- **Capture:** a signed foreground ScreenCaptureKit app records microphone and system audio separately, retains timestamps on a shared clock, checkpoints CAF segments and records gaps, readiness, route changes and failures. The menu app has an opt-in adapter and keeps a Stop action while startup or shutdown needs attention. Capture files are bound to their native manifest by hashes; existing publications cannot be overwritten accidentally.
- **Transcription:** immutable sources feed bounded jobs with atomic chunk results, request deadlines, resumable failures and validated structured responses. Words are saved before database or analysis work. The source route bypasses noScribe and the legacy analysis processor. It never silently falls back to another engine.
- **Evidence:** all sources survive pruning unless a separately verified lossless replacement exists. First-minute channel selection no longer determines whether a source is retained. Capture gaps, uncertain speech, failed requests and unknown names remain explicit.
- **Identity:** labels remain anonymous and local to a source/chunk. Review exports show a sentence and choices with no selection made. Human confirmation is bound to the actual source job and sentence, and applies only to that segment.
- **Publication:** retries use one durable source identity and a database transaction lock. Previous transcript versions remain available. Downstream speaker-dependent actions remain held until identity evidence exists.

The recorded source files remain authoritative. A successful API response or accounted-for interval is not a word-accuracy or speaker-accuracy score.

## Validation and limitations

The first real native capture attempt was blocked by macOS screen/system-audio permission. That failed attempt also exposed a shutdown issue, which was fixed. No successful real capture, 60-minute alignment test, route-change test or sleep/wake acceptance is claimed. The tested bundle must receive macOS permission before those checks can proceed.

Synthetic and unit checks cover delayed sources, gap metadata, physical channel preservation, binding mismatch, readiness/stop failure handling, corrupt cache recovery, request deadlines, exact silence, uncertain ranges, anonymous labels and idempotent publication. The existing runtime produces a Pyannote/TorchCodec warning; the source-separated ASR path does not require Pyannote decoding.

Final full-suite result: **406 passed**. Native Swift typecheck, signed build, production-path synthetic gap/owner-exit tests, and 50 direct bridge lifecycle stress runs passed. The integrated synthetic capture produced two sources, three verified timing sidecars and a three-channel WAV with the initial and internal gaps preserved. The final fractional-frame validation was covered by the focused native source tests.

A real structured-response probe succeeded after correcting a provider-rejected schema. Diagnostic runs made before the runtime was frozen remain preserved and are not a controlled model comparison. New recovery jobs use a frozen runtime and retain the original hashes, source ranges, raw attempts and any unresolved intervals.

The five-minute draft target is not yet certified. The production worker remains serial. Parallel diagnostic recovery jobs do not establish its future live throughput. Model selection still requires a representative human-labelled benchmark, including Swiss German and overlapping speech.

## Recovered afternoon section

The previously omitted **30:00–69:25** portion now has **153 draft segments (2,539 words)** across microphone and system sources. This includes background speech recorded after the meeting; it is not all meeting content. Two three-minute system requests repeatedly disconnected after roughly 73–76 seconds. Six one-minute, source-specific rescue jobs recovered those request intervals without repeating successful chunks. **No failed-request intervals remain; 15 coverage-review intervals remain.** These numbers are accounting results, not an accuracy score.

The combined artifact is `afternoon-omitted-recovery/recovered-omitted-section.draft.md` under the private evidence directory. JSON includes original hashes, global recording ranges, individual source-job provenance, unresolved intervals and anonymous labels. No canonical transcript or database row was changed by recovery. All six rescue jobs were replayed without an API key; their six attempt files remained byte-identical, proving cached jobs made no new provider calls.

Shorter rescue segmentation was a controlled recovery step, not an automatic production fallback. The dense-audio disconnect and suitable chunk size remain benchmark inputs before live rollout.

## Before live rollout

1. Grant MeetingNativeCapture microphone and screen/system-audio permissions, then pass the capture acceptance checks in `pipeline-rebuild.md`.
2. Review recovered text and a small set of identity examples. Confirm useful words, missed speech and speaker evidence separately.
3. Authenticate the actual downstream automation profile if analysis is to run: `CLAUDE_CONFIG_DIR=/Users/Matthias/.claude-automation claude auth login`. Login does not clear speaker holds.
4. Run representative shadow meetings, measure draft latency and failures, then approve the concrete production switch. Both completed speaker-trial automations remain paused.

Configuration, recovery commands and acceptance criteria: [pipeline-rebuild.md](pipeline-rebuild.md).

Private evidence directory: `/Users/Matthias/.local/share/meeting-pipeline-trial/rebuild-2026-09-15`.
