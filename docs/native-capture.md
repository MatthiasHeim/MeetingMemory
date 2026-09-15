# Native timestamped capture boundary

`tools/native_capture.swift` is the macOS 15+ capture boundary for MeetingMemory.
It receives separate ScreenCaptureKit `.audio` (system) and `.microphone`
`CMSampleBuffer`s from one `SCStream`, writes immutable lossless source segments,
and records their timing before any merge or transcription work happens.

This exists because a successful process launch is not evidence that system audio
has started. The manifest remains `starting` until buffers arrive and retains the
actual delay as a source coordinate rather than placing every source at zero.

Apple documents the microphone output as a macOS 15 ScreenCaptureKit feature and
documents `SCStream.synchronizationClock` as the stream media clock. See
[ScreenCaptureKit capture sample](https://developer.apple.com/documentation/screencapturekit/capturing-screen-content-in-macos),
[SCStreamOutputType.microphone](https://developer.apple.com/documentation/screencapturekit/scstreamoutputtype/microphone),
and [WWDC24: Capture HDR content with ScreenCaptureKit](https://developer.apple.com/videos/play/wwdc2024/10088/).

## Build

```zsh
tools/build_native_capture.sh
```

The result is `build/MeetingNativeCapture.app`. The script creates a regular,
foreground-capable app bundle and adds microphone, screen-capture, and audio
capture usage descriptions. It signs locally with an ad-hoc signature by
default. For a distributable build, set `NATIVE_CAPTURE_SIGN_IDENTITY` to the
Developer ID identity before building.

The build never installs the app, changes TCC permissions, starts capture, or
changes any audio route. It validates the app signature. A writer/timeline
self-test that does not ask for permissions or touch live audio is available:

```zsh
tools/build_native_capture.sh --output /tmp/MeetingNativeCapture.app --self-test
```

The self-test does not request capture permissions. It routes synthetic packets
through the production writer path: a `-0.1` second mic packet, a `+3.25`
second system packet, and a later system timestamp gap. It verifies that the
late packet is written into a second CAF segment instead of being appended into
the earlier CAF and collapsing the declared gap. It also runs a separate local
owner-exit monitor probe; that probe starts no capture stream or audio device.

## Python integration

Use the bridge, not `subprocess.Popen` from a recorder UI:

```python
from pathlib import Path
from native_capture_bridge import (
    NativeCaptureReadinessTimeout,
    NativeCaptureSession,
)

session = NativeCaptureSession(
    bundle=Path("build/MeetingNativeCapture.app"),
    session_dir=Path("/safe/session-directory"),
)
try:
    startup = session.start()  # full manifest/status dictionary
except NativeCaptureReadinessTimeout:
    # Do not destroy the session: inspect session.status() and retain it.
    startup = session.status()

# Cheap, file-only UI polling; no process probe or state change.
current = session.status()

# SIGINT only the native PID observed for this session, then wait for both an
# atomic finalized manifest and process exit.
finished = session.stop()
manifest_path = session.manifest_path
```

`NativeCaptureSession` defaults to `launchservices`, effectively running:

```zsh
open -n MeetingNativeCapture.app --args --output-dir SESSION_DIR ...
```

That is intentional. Existing evidence shows that direct helper execution and
LaunchServices app execution do not necessarily have the same TCC attribution.
The bridge offers `launch_mode="direct"` only for non-capture tests. It does
not claim that this mode can receive production screen-recording or microphone
permissions.

The bridge gives every process a private random session token. It will only
signal a PID after it has seen a matching token, stable session ID, stable OS
process-start signature, and native executable command. It never uses a global
`pkill` pattern. `status()` only reads `capture-manifest.json`; it is safe for a
frequent menu-bar health check.

At launch the bridge also passes its own PID as `--owner-pid`. The native app
registers that exact process with macOS kqueue using `EVFILT_PROC` and
`NOTE_EXIT`, before it begins permission or stream setup. This follows the
specific process generation rather than polling a numeric PID, so a later PID
reuse cannot keep a LaunchServices capture alive. If the bridge process exits,
the native app records `owner_process_exited`, stops the stream, flushes queued
packets, finalizes closed segments, and exits. The manifest's `owner_process`
object records monitor state and whether that exit was observed.

`start()` returns a full manifest dictionary once recording is ready, or once a
source-ready degraded capture is explicit. A failed native start raises
`NativeCaptureStartError`. A bridge readiness timeout raises
`NativeCaptureReadinessTimeout` and deliberately leaves the recorder alive so
completed microphone or system segments are not discarded.

## Manifest contract

The native process writes `capture-manifest.json` atomically in the session
directory. Segment and timing paths are relative to that directory. Consumers
must resolve them relative to the manifest, never the current working directory.

```json
{
  "schema_version": 2,
  "capture_backend": "screencapturekit",
  "status": "complete",
  "finalized": true,
  "session_id": "...",
  "started_wall_time": "2026-09-15T...Z",
  "ended_wall_time": "2026-09-15T...Z",
  "timeline": "host_clock",
  "timeline_metadata": {
    "epoch_raw_pts": {"value": 0, "timescale": 1},
    "host_clock_epoch_raw": {"value": 0, "timescale": 1},
    "start_seconds_definition": "CMSampleBuffer.presentationTimeStamp converted from SCStream.synchronizationClock to host clock, minus the one shared host_clock_epoch_raw; individual tracks are never independently zeroed"
  },
  "readiness": {"mic": true, "system": true},
  "tracks": [
    {
      "source_id": "mic",
      "first_buffer_ready": true,
      "segments": [
        {
          "path": "segments/mic-0001.caf",
          "timing_path": "segments/mic-0001.timing.json",
          "start_seconds": -0.1,
          "duration_seconds": 20.0,
          "sample_rate": 48000,
          "channels": 1,
          "frames": 960000,
          "format": "caf/pcm_f32le",
          "raw_pts_start": {"value": 0, "timescale": 1}
        }
      ],
      "gaps": []
    },
    {
      "source_id": "system",
      "first_buffer_ready": true,
      "segments": [{"path": "segments/system-0001.caf", "start_seconds": 46.28}],
      "gaps": [{"begin_seconds": 0.0, "end_seconds": 46.28, "duration_seconds": 46.28, "reason": "initial_no_samples"}]
    }
  ],
  "errors": [],
  "final_duration_seconds": 66.28
}
```

`start_seconds` is always measured from one shared host-clock epoch. The native
recorder preserves the original stream-clock PTS and maps it with Core Media's
clock synchronization API; it is never a per-track sample-zero coordinate. If
a valid packet predates that epoch, the native recorder retains its negative
coordinate and records a
`pre_epoch_first_packet` event. A downstream assembler may shift every source
by one shared amount to obtain nonnegative ASR times, provided it records that
shift in its own provenance. It must never independently shift mic and system
tracks.

Each completed segment is a float32 PCM CAF file readable by libsndfile. The
sidecar timing JSON retains every packet's raw PTS, sample duration, original
frame count, written frame count, and shared relative coordinate. The manifest
also retains raw PTS at segment and track boundaries. A segment is written under
a `.partial.caf` name and only renamed after the CAF header and timing sidecar
are finalized; its manifest entry is atomically checkpointed immediately after
that close. A process crash can therefore lose at most the active 20-second
checkpoint, not already-completed segments.

`gaps` use `begin_seconds` and `end_seconds` in that same shared coordinate.
The recorder marks an initial late source and any timestamp discontinuity as a
gap. It preserves no fabricated silence in the authoritative source CAF. A
gap larger than the 20 ms jitter allowance finalizes the active CAF before the
late packet is written, so each source segment contains contiguous samples and
the segment start continues to match the shared manifest coordinate.

Consumers may process a manifest only when `finalized` is `true`, including a
terminal `status: "degraded"`. An active degraded capture has
`finalized: false` and must remain untouched. `errors` represent real failure
or degradation evidence; ordinary start/stop lifecycle notes live in `events`.

## States and failure behavior

| Status | `finalized` | Meaning |
|---|---:|---|
| `starting` | false | Stream launch has begun; no readiness claim yet. |
| `recording` | false | Required sources delivered buffers. |
| `degraded` | false | A source was late/unavailable or a discontinuity was observed; capture may still retain the healthy source. |
| `complete` | true | All queued writes and segment finalization completed cleanly. |
| `degraded` | true | Capture finalized with explicit recoverable defects. |
| `failed` | true | A fatal writer, queue, stream, or startup failure finalized what was recoverable. |

The callback only snapshots an audio buffer into a bounded queue. It never
writes files or manifests synchronously. The serial writer owns CAF conversion,
segment close, timing sidecars, and atomic manifest updates. Queue overflow,
sample snapshot failure, conversion/write failure, nonmonotonic timing, and
stream stop errors are surfaced in the manifest; they never silently turn into
an apparently healthy recording. A source format change is recorded and that
source is stopped rather than concatenating incompatible sample rates or layouts.

The recorder polls the selected microphone route for evidence and records route
events. A default-input change is evidence for review, not a claim that an
already configured ScreenCaptureKit stream switched devices. A stream stop or
format change is recorded as a capture failure/degradation event.

## Verification

```zsh
swiftc -swift-version 5 -typecheck -target arm64-apple-macosx15.0 \
  -framework AppKit -framework AVFoundation -framework AudioToolbox \
  -framework CoreMedia -framework CoreVideo -framework ScreenCaptureKit \
  tools/native_capture.swift

/Users/Matthias/Repos/MeetingMemory/venv/bin/python -m pytest -q \
  tests/test_native_capture_bridge.py
```

The Python tests exercise an atomic synthetic late-source manifest, active
versus finalized degraded state, token/PID ownership refusal, and a fake direct
child's graceful finalization. They do not exercise live ScreenCaptureKit,
permissions, hardware routes, or actual meeting audio.
