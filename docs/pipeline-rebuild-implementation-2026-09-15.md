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

The first real native capture attempt was blocked by macOS permissions. A missing hardened-runtime audio-input entitlement and a stale ad-hoc signing identity were corrected. Both permissions then matched the tested bundle, and a 90-second session saved separate microphone and system CAF segments. No 60-minute alignment, route-change, sleep/wake or speech-accuracy acceptance is claimed.

Synthetic and unit checks cover delayed sources, gap metadata, physical channel preservation, binding mismatch, readiness/stop failure handling, corrupt cache recovery, request deadlines, exact silence, uncertain ranges, anonymous labels and idempotent publication. The existing runtime produces a Pyannote/TorchCodec warning; the source-separated ASR path does not require Pyannote decoding.

Final full-suite result: **406 passed**. Native Swift typecheck, signed build, production-path synthetic gap/owner-exit tests, and 50 direct bridge lifecycle stress runs passed. The integrated synthetic capture produced two sources, three verified timing sidecars and a three-channel WAV with the initial and internal gaps preserved. The final fractional-frame validation was covered by the focused native source tests.

A real structured-response probe succeeded after correcting a provider-rejected schema. Diagnostic runs made before the runtime was frozen remain preserved and are not a controlled model comparison. New recovery jobs use a frozen runtime and retain the original hashes, source ranges, raw attempts and any unresolved intervals.

The five-minute draft target is not yet certified. The production worker remains serial. Parallel diagnostic recovery jobs do not establish its future live throughput. Model selection still requires a representative human-labelled benchmark, including Swiss German and overlapping speech.

## Recovered afternoon section

The previously omitted **30:00–69:25** portion now has **153 draft segments (2,539 words)** across microphone and system sources. This includes background speech recorded after the meeting; it is not all meeting content. Two three-minute system requests repeatedly disconnected after roughly 73–76 seconds. Six one-minute, source-specific rescue jobs recovered those request intervals without repeating successful chunks. **No failed-request intervals remain; 15 coverage-review intervals remain.** These numbers are accounting results, not an accuracy score.

The combined artifact is `afternoon-omitted-recovery/recovered-omitted-section.draft.md` under the private evidence directory. JSON includes original hashes, global recording ranges, individual source-job provenance, unresolved intervals and anonymous labels. No canonical transcript or database row was changed by recovery. All six rescue jobs were replayed without an API key; their six attempt files remained byte-identical, proving cached jobs made no new provider calls.

Shorter rescue segmentation was a controlled recovery step, not an automatic production fallback. The dense-audio disconnect and suitable chunk size remain benchmark inputs before live rollout.

## Before live rollout

1. Complete a live spoken capture and the acceptance checks in `pipeline-rebuild.md`. The current bundle has both permissions; replacing its ad-hoc signed binary changes its identity and may require granting them again.
2. Review recovered text and a small set of identity examples. Confirm useful words, missed speech and speaker evidence separately.
3. Authenticate the actual downstream automation profile if analysis is to run: `CLAUDE_CONFIG_DIR=/Users/Matthias/.claude-automation claude auth login`. Login does not clear speaker holds.
4. Run representative shadow meetings, measure draft latency and failures, then approve the concrete production switch. Both completed speaker-trial automations remain paused.

Configuration, recovery commands and acceptance criteria: [pipeline-rebuild.md](pipeline-rebuild.md).

Private evidence directory: `/Users/Matthias/.local/share/meeting-pipeline-trial/rebuild-2026-09-15`.

## First permitted live test — 11:04 Hong Kong time

The 90-second test finalized with no native errors and preserved ten raw CAF segments plus timing sidecars. The selected microphone was `BuiltInMicrophoneDevice`; the system source was digital silence. The microphone contained signal, but the full-source structured transcription returned `no_speech`. Voice activity detection disagreed, so the worker correctly left the interval uncertain rather than marking it verified silence. An independent PCM16 check of the loudest final eight seconds described keyboard typing and no intelligible speech. Whether the user spoke during this window, and which microphone was intended, remains to be clarified. This is a successful file-capture test, not a successful Swiss German transcription test.

Actual capture exposed a timestamp-origin bug: ScreenCaptureKit returned zero mapping anchors while sample timestamps were around 220,792 seconds after boot. The code selected zero as the session origin, which would create about 61 hours of leading silence. The source fix now uses the controller's recorded host-clock start and preserves the raw mapping anchors as evidence. A Python wall-duration guard rejects implausible timelines before allocating derived WAVs. A separate temporary Swift executable passed synthetic zero-anchor and owner-exit checks, and 19 focused Python tests passed (including recovery from an unfinalized checkpoint). The permitted app bundle was deliberately left byte-identical during the user test; these timing corrections are source changes awaiting the next built-bundle test.

For analysis only, an explicitly labelled derivative applied one common 220792.444913416-second shift using the original recorded host-clock start, retaining all original audio and raw PTS. The resulting timeline is 90.258 seconds. The original and normalized manifests, input bindings, microphone check and empty/uncertain transcript result remain under `swiss-german-live-110426` in the private evidence directory. No canonical transcript, database publication or production recorder setting was changed.
