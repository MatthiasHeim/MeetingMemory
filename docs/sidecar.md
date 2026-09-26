# Meeting sidecar: Slice 1 clerk

Status on 2026-09-26: Slice 1 has the finished-transcript clerk plus its
recorder menu-bar entry points. `gemini-3.8-flash` is the only Gemini judge.
The recorder always starts capture before optional sidecar/calendar work. The
watcher change is one additive final-JSON metadata field for an alignment lag;
WAV capture, transcription decisions, and all other output stay unchanged.

## Use

The CLI reads a completed recorder JSON from
`~/Documents/MeetingRecorder/Transcripts/<stem>.json`; it never edits that
file. The menu-bar app loads the repository `.env` exactly like
`transcribe_watcher.py` and reads the configured `gemini.api_key_env` (default
`GEMINI_API_KEY`) without logging it. Direct CLI use still needs that key in
its invoking environment.

```bash
python -m tools.sidecar clip \
  --stem 2026-09-08_14-31-16 \
  --topic "User roles and permissions in the dashboard" \
  --judge gemini

python -m tools.sidecar prompts --stem 2026-09-08_14-31-16
```

`clip` produces the fixed header plus verbatim transcript lines. Its approved
default hysteresis is seed `0.60`, grow `0.25`, bridge at most two lines,
require two seeds, and remove pure fillers. `--widen` deliberately uses the
one-step lower pair `0.50` / `0.15`.

The rumps menu bar app adds these actions:

- **Start Recording** starts capture immediately with a fail-closed local
  sidecar (`jev: false`, `external_attendees: null`). Calendar resolution then
  runs on a background worker with a three-second timeout; it never delays mic
  or system capture. The later **“Jev verwenden?”** checkbox is off by default.
  It remains switchable for external **and unresolved** meetings, but displays
  **“nicht durch Kunden-DPA gedeckt”**. If Matthias enables it in either case,
  the amended sidecar records `jev: true` and
  `jev_external_acknowledged: true`. A timeout stays `external_attendees: null`
  and receives the same warning. A sidecar write failure is logged/notified,
  forces Jev off, and never stops recording.
- **Prompt markieren** is enabled while recording and appends a monotonic
  elapsed offset to `Recordings/<stem>.sidecar.json`. The global shortcut is
  **Ctrl–Option–Command–P** (`⌃⌥⌘P`). It uses AppKit global and local key-event
  monitors; macOS requires Accessibility permission for global key monitoring.
  Grant it to the interpreter/app running MeetingRecorder under **System
  Settings → Privacy & Security → Accessibility**. The menu item still works
  without that permission. After the first mic callback, a background writer
  also persists `mic_first_sample_offset_seconds` relative to this mark-zero
  boundary; the callback itself performs no file I/O.
- **Clip…** offers the newest ten valid finished transcripts (metadata/calendar
  title when available, otherwise the stem), asks for a topic, then evaluates
  and copies the clip on a background thread. Its completion notification gives
  the copied line count.
- **Prompts…** similarly selects a finished transcript, evaluates marked and
  suggested prompt cards on a background thread, then asks which card to copy.

The score cache is local-only at `~/.local/share/meeting-sidecar/cache/`. Its
entries contain probability vectors and hashes, never transcript text. The
calibration corpus and aggregate report remain under
`~/.local/share/meeting-sidecar/` and are never copied into this repository.
Each Jev use also appends a text-free audit record (`stem`, attendee state,
external acknowledgement, UTC timestamp, and request count) to
`~/.local/share/meeting-sidecar/jev-audit.jsonl`.

## Provider gate

Gemini is the default. `--judge jev` is accepted only when the exact recording
has a regular, non-symlinked `Recordings/<stem>.sidecar.json` whose
`recording_stem` matches the transcript stem, and it records:

```
jev is true AND (
  external_attendees is explicitly false OR
  jev_external_acknowledged is explicitly true
)
```

`external_attendees` is tri-state: `false` means a uniquely matched timed event
with an explicit roster and verified `matthias@lailix.com` self identity; `true`
means a verified other attendee; `null` means unknown. An empty roster, a
resolver-synthesised self row, an all-day/no timed match, a malformed roster,
or a display-name-only self match is unknown and receives the external warning.
Missing, malformed, opt-out, or unacknowledged external/unknown sidecars fail
closed to Gemini. The CLI rejects Jev if its recordings root comes from
`MEETING_SIDECAR_RECORDINGS_DIR`.

The factory decides before constructing `JevJudge`, and `JevJudge` checks again
at direct construction and immediately before each helper request. The helper
is always invoked with its internal/approved contract flags; the local audit
record preserves the actual attendee/acknowledgement facts. Gemini construction
explicitly pins `https://generativelanguage.googleapis.com/`, so inherited
`GOOGLE_GEMINI_BASE_URL` and SDK endpoint overrides cannot redirect transcript
text to another provider.

## Gemini calibration and owner override

All baseline A/B/C measurements used the same single-question, 20-line,
six-prior-line request layouts as runtime `clip` and `prompts` calls. The
historical measurement table is aggregate-only:

| Gemini model | A EN relevance AUC | B Swiss German relevance AUC | C Swiss German prompt recall / FP | Prompt threshold | Input / output tokens | p50 batch request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemini-3.8-flash` | 0.908 | 0.815 | 1.00 / 0 | 0.900 | 1,704,866 / 175,505 | 4.83 s |
| `gemini-3.1-flash-lite` (historical; not routed or used) | 0.819 | 0.904 | 0.50 / 0 | 0.800 | 1,704,866 / 175,905 | 2.60 s |

**Owner override — Matthias, 2026-09-26.** The B Swiss-German relevance labels
are noisy and repeated `gemini-3.8-flash` B AUCs ranged from **0.788 to
0.884**. Criterion 2 is therefore relaxed only for that Swiss-German relevance
cell. The approved Slice 1 configuration is the single
`gemini-3.8-flash` judge, prompt threshold **0.90**, and the default clip
hysteresis **0.60 / 0.25**. There is no Flash/Lite routing.

`~/.local/share/meeting-sidecar/calibration/latest.json` records this as
`status: "owner_approved_override"` with the date, model, thresholds, and B
range. `prompts` accepts that exact dated override; a missing or altered
override does not activate prompt extraction. The waiver applies only to B:
English relevance A must still pass, and C prompt recall/false-positive checks
must still pass at the approved `0.90` threshold. A failed A or C remains a
failed calibration with prompts disabled. To amend an existing
aggregate-only report without a provider call:

```bash
python -m tools.sidecar calibrate --apply-owner-override
```

### Source-955 clip diagnostic

The default `gemini-3.8-flash` clip request for the specified roles topic
selected 20 lines from exactly `43:05` through `45:42`, with zero selected
lines outside that interval at seed/grow `0.60` / `0.25`. This satisfies
criterion 1.

### Known Swiss German limitation

Swiss-German relevance is accepted for this slice under the owner override,
not because it meets the original `0.88` AUC bar. Its observed B range is
`0.788–0.884`; it must be re-measured against the human-checked reference set
defined in spec §11 before tightening or generalising these thresholds.

## Mark-to-transcript mapping

The sidecar format stores numeric `marks` as seconds since the app's monotonic
recording-start boundary, captured immediately before `AudioRecorder.start()`.
It later stores `mic_first_sample_offset_seconds`: the monotonic instant of the
first mic sample relative to that same mark-zero boundary. When the watcher
actually aligns channels, it adds
`_meta.channel_alignment.lag_seconds` to the final transcript JSON. Prompt
extraction applies both through:

```
max(0, mark - mic_first_sample_offset_seconds - channel_alignment.lag_seconds)
```

It then searches five seconds before that mapped point and stops at `m + 3 min`
or the line-level prompt end. A sentence-level yes/no judge strips only a
leading non-prompt prefix; it never rewrites prompt text.

Prompt lines merge only when consecutive transcript timestamps are at most
12 seconds apart. A marked card is bounded to the exact mapped `m - 5 s`
through `m + 3 min` window, even when a detected prompt run is longer. A
low-scored bridging acknowledgement may connect a detected run but is never
copied into the prompt card.

This nominal map is not an exact shared-clock map in the current recorder:

- `AudioRecorder` launches the system tap before the mic, and the streams have
  independent sample-zero origins.
- `capture_provenance.merge_filter` merges/pads them at file sample zero; the
  current tap binary does not retain system host timestamps.
- The watcher later runs `channel_align` only on a derived analysis WAV. A
  positive lag means the mic was late and is advanced, while system zero stays
  unchanged.

When either persisted value is missing, the mapper searches backward through
`channel_align.MAX_LAG_SEC` (currently 60 seconds) rather than pretending the
nominal mark is exact. Any resulting marked card is labelled
**“Zuordnung unsicher”** in the menu and CLI. This is intentionally conservative
for historical recordings and for runs where no correctable alignment was
applied.

## Headless verification and attended checklist

The automated suite covers the tri-state/acknowledged Jev gate, endpoint pin,
regular-sidecar authority, text-free Jev audit, capture-first sidecar failure,
monotonic mark offsets, alignment mapping, and a scripted concurrent
mark-plus-clip run without opening an audio device. The AppKit dialogs and
global hotkey need an attended macOS session. Before merge:

1. Start an internal calendar event, leave Jev off, mark once from the menu and
   once with `⌃⌥⌘P`, then confirm both numeric marks and the first-mic offset in
   its local sidecar.
2. Start an external event and an unresolved/no-attendee event. Confirm capture
   starts immediately, **“nicht durch Kunden-DPA gedeckt”** appears in each
   later dialog, the checkbox is switchable but off by default, and enabling it
   writes `jev_external_acknowledged: true` alongside `jev: true`.
3. Temporarily make the sidecar directory unwritable; confirm recording still
   starts, a failure notification appears, and Jev remains off.
4. With a prior completed transcript, run **Clip…** during a short recording;
   confirm the clip notification shows a line count and the WAV duration still
   matches wall clock.
5. Run **Prompts…** on one aligned recording and one historical/mapping-missing
   recording; confirm the latter card displays **“Zuordnung unsicher”**.

## Known offline limitation

The loader judges only complete `[mm:ss] Speaker: text` turns. It deliberately
does not invent timestamps or speakers for malformed/untimestamped continuation
text, so such text is not sent to a judge; nonblank skipped lines are retained
only in local debug logs. Historic transcripts lacking either mapping datum are
still usable, but their marked prompt cards are explicitly uncertain.
