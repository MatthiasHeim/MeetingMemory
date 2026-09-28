# Meeting sidecar

Status on 2026-09-28: Slice 1 is the finished-transcript clerk. Slice 2 adds a
live side window and removes the start dialog. `gemini-3.8-flash` is the only
Gemini model for judging, live transcription, and clean prompts. The recorder
starts capture before optional sidecar or calendar work. The audio callback is
unchanged. The watcher change from slice 1 is one additive final-JSON metadata
field for an alignment lag; WAV capture, transcription decisions, and all other
output stay unchanged.

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

`clip` produces the fixed header plus verbatim transcript lines. Gap markers
appear only where substantive lines between selected clusters were excluded;
omitted acknowledgements inside a kept stretch do not create a marker. Its
approved default hysteresis is seed `0.60`, grow `0.25`, bridge at most two lines,
require two seeds, and remove pure fillers. An isolated non-filler seed at
`0.90` or above is retained for sparse topics. `--widen` deliberately uses the
one-step lower pair `0.50` / `0.15`. If no line qualifies, `clip` prints
**“Nichts Passendes gefunden.”** followed by the three highest-scored verbatim
candidate lines and their scores, rather than returning only its header.

The rumps menu bar app adds these actions:

- **Start Recording** starts capture immediately. There is no dialog and no
  modal. The sidecar JSON records the **Jev verwenden** preference at that
  moment (`external_attendees` is still null until calendar metadata arrives).
  Calendar lookup then runs on a background worker with a three-second timeout
  and only fills metadata. It never delays mic or system capture and never
  changes `jev`. A sidecar write failure is logged/notified, forces Jev off,
  and never stops recording. The live side window opens at the same time.
- **Jev verwenden** is a Preferences switch in the menu, default off. It is
  stored in `sidecar-preferences.json` next to the recorder config, not by
  rewriting `config.yaml`. Turn it on only once the direct TypeSafe account
  and DPA are active. The next recording then starts with `jev: true`.
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
  The clipboard receives the clean prompt. The verbatim lines stay on the card.

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
jev is true
```

`external_attendees` is still stored, and it must be an explicit boolean or
null, but it does not authorise or block Jev. The old external-attendee
acknowledgement is no longer required (owner decision, 2026-09-28).
`jev_external_acknowledged` remains in the file for older sidecars and is
ignored by the gate.

`external_attendees` is tri-state metadata: `false` means a uniquely matched
timed event with an explicit roster and verified `matthias@lailix.com` self
identity; `true` means a verified other attendee; `null` means unknown. An
empty roster, a resolver-synthesised self row, an all-day/no timed match, a
malformed roster, or a display-name-only self match is unknown. Missing,
malformed, or `jev: false` sidecars fail closed to Gemini. A stored `jev: true`
is enough even when attendance is external or unknown. The CLI rejects Jev if
its recordings root comes from `MEETING_SIDECAR_RECORDINGS_DIR`.

The factory decides before constructing `JevJudge`, and `JevJudge` checks again
at direct construction and immediately before each helper request. The helper
is always invoked with its internal/approved contract flags; the local audit
record preserves the actual attendee/acknowledgement facts. Gemini construction
explicitly pins `https://generativelanguage.googleapis.com/`, so inherited
`GOOGLE_GEMINI_BASE_URL` and SDK endpoint overrides cannot redirect transcript
text to another provider.

Gemini clip and prompt scoring keeps the same 20-line request contract, retry
behaviour, and score-only cache, while sending independent batches through a
bounded pool of up to eight workers. Results are placed back in transcript-line
order before deterministic selection or prompt assembly.

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
or the line-level prompt end. A sentence-level yes/no judge strips leading and
trailing non-prompt sentences; it never rewrites prompt text. A high-confidence
(`>= 0.90`) line-level prompt decision remains verbatim in a card if the
sentence pass has a recall miss, so calibrated prompt-scored lines remain
available for copying.

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

## Live side window

A small floating panel opens when recording starts. It is an `NSPanel` with the
non-activating style mask, floating level, and `orderFrontRegardless`, so it
stays on top, does not become key on open, and does not run a modal loop. The
menu stays usable. Closing the panel does not stop recording. Stopping the
recording leaves the panel open until it is closed.

The panel shows:

- the live Hochdeutsch transcript
- prompt cards as they appear, each with the clean prompt, the verbatim lines
  underneath, and a **Kopieren** button that copies the clean prompt
- a topic field and **Clip kopieren**, which runs the existing clip selection
  (including the sparse-topic rule) on the live lines and copies verbatim text

The live worker is a daemon thread. About every 20 seconds it copies references
to the in-memory mic blocks and reads the tail of the system-audio WAV (the
last 60 seconds of each, taken at the same moment). It sends those tails to
`gemini-3.8-flash` on the pinned Gemini endpoint. Lines already committed are
dropped. A transcription or prompt failure shows a status line in the panel and
does not stop or delay capture. The audio callback is not on this path.

Prompt detection uses the broadened judge question: Swiss German and Hochdeutsch
indirect instructions (“ich würd em Claude säge, er söll …”, “mir müessted em
Agent säge …”) and English dictation inside dialect. Gemini then writes the
clean prompt (intent only; English for AI tools unless the speaker clearly
wants German). Clips stay verbatim.

Headless replay, no window:

```bash
python -m tools.sidecar replay --wav /path/to/recording.wav
```

It reveals the WAV in 20-second steps, prints each new line with its lag, and
prints each prompt card again when a later tail completes it. Lag is the
simulated tick time plus the real transcription duration, minus the line's
audio time. A line that is still cut off at the end of a tick is held for the
next tail; a finished sentence is committed immediately, and a later tail can
replace a fragment with the complete hearing. Optional `--from-seconds`
and `--to-seconds` limit the span. The command reads `GEMINI_API_KEY` from the
environment or from the repository `.env` (the main checkout's `.env` when
this directory is a worktree).

## Headless verification and attended checklist

The automated suite covers the preference gate (jev true authorises Jev without
an external acknowledgement), the endpoint pin, regular-sidecar authority,
text-free Jev audit, capture-first sidecar failure, no modal on start, an
unchanged audio callback, a crashing or slow live worker, monotonic mark
offsets, alignment mapping, and a scripted concurrent mark-plus-clip run
without opening an audio device. The global hotkey and the real panel still
need an attended macOS session. Before merge:

1. Start recording from the menu. Confirm there is no dialog, Stop is available
   immediately, and the side window appears without taking focus. Type in the
   other app and confirm keystrokes do not land in the topic field until you
   click it.
2. Leave **Jev verwenden** off. Confirm the sidecar JSON has `jev: false`.
   Turn the switch on, start another recording, and confirm `jev: true` even
   if the calendar event has external attendees. Turn the switch off again.
3. Speak for about a minute. Confirm Hochdeutsch lines show up within about
   half a minute, and that a failed network status line does not stop the
   recording. Stop, confirm the panel stays open, then close it yourself.
4. During or after the recording, type a topic and press **Clip kopieren**.
   Confirm the clipboard is verbatim transcript text with the fixed header.
5. Dictate an indirect instruction (“ich würde Claude sagen, er soll …”).
   Confirm a card appears, **Kopieren** pastes a clean English prompt, and the
   verbatim lines are visible under it. Run **Prompts…** on the finished
   transcript and confirm that path also copies the clean prompt.
6. Mark once from the menu and once with `⌃⌥⌘P`. Confirm both numeric marks
   and the first-mic offset in the sidecar.
7. Temporarily make the sidecar directory unwritable; confirm recording still
   starts, a failure notification appears, and Jev remains off.
8. With a prior completed transcript, run **Clip…** during a short recording;
   confirm the clip notification shows a line count and the WAV duration still
   matches wall clock.
9. Run **Prompts…** on one aligned recording and one historical/mapping-missing
   recording; confirm the latter card displays **“Zuordnung unsicher”**.

## Known offline limitation

The loader judges only complete `[mm:ss] Speaker: text` turns. It deliberately
does not invent timestamps or speakers for malformed/untimestamped continuation
text, so such text is not sent to a judge; nonblank skipped lines are retained
only in local debug logs. Historic transcripts lacking either mapping datum are
still usable, but their marked prompt cards are explicitly uncertain.

## Known prompt recall limitation

The line at `01:28:16` in the July 8 recording that asks for a status-line
prompt scored `0.15` for `dictating_prompt`. This is a Gemini judge recall miss;
it is documented here only and does not change the calibrated threshold.
