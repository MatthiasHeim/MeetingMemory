# Meeting sidecar

Status on 2026-09-28: Slice 1 is the finished-transcript clerk. Slice 2 adds a
live side window and removes the start dialog. `gemini-3.8-flash` is the only
Gemini model for judging, live transcription, and clean prompts. The recorder
starts capture before optional sidecar or calendar work. The microphone
callback copies each block into a bounded queue and a two-minute ring; a
writer thread checkpoints the mic WAV. It does not write the file itself.
The watcher still adds one alignment-lag field, and a partial transcript can
gain `[live]` lines from the saved live session. A complete transcript is
left as Gemini wrote it.

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
  the copied line count. While a recording is running this item is disabled.
  The live panel's topic field is the clip control during capture. Outside a
  recording, the app is activated before the dialog so it cannot sit behind
  other windows.
- **Prompts…** similarly selects a finished transcript, evaluates marked and
  suggested prompt cards on a background thread, then asks which card to copy.
  Cards saved in `Recordings/<stem>.live.json` are offered after those, labelled
  **Live bei …**, without another Gemini clean pass. The clipboard receives the
  clean prompt. The verbatim lines stay on the card. This item is also disabled
  while recording; prompt cards are on the live panel. The same activation rule
  applies when the dialog is allowed.
- **List Audio Devices** is disabled while recording. Outside a recording it
  activates the app and then shows the device list. It never uses a modal
  during capture.

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

The live worker is a daemon thread named `meeting-sidecar-live`. About every
20 seconds it copies references to the in-memory mic ring and reads the tail
of the system-audio WAV (the last 60 seconds of each, taken at the same
moment). The mic tail is walked from the newest block and stops once those
60 seconds are in hand; the recorder's frame count supplies the clock, so
dropping older blocks from the ring does not move the timestamps. Gemini
clients and the calibration-file read are created on that worker's first
tick. Start only opens the panel. If the session cannot start (for example a
missing Gemini key), a non-modal notification says so and recording
continues. A new Start closes the previous panel.

Each tick, after the transcription attempt, the worker atomically replaces
`Recordings/<stem>.live.json` (write a temp file, then `os.replace`). It
does this at most once per tick, and once more when Stop asks the worker to
finish, including the lines and cards committed so far. The file holds
committed lines (`start`, `end`, `speaker`, `text`) and prompt cards (clean
text, verbatim lines, time range). Closing the side window does not clear
that file and does not stop the worker. A persist failure sets a status line
and leaves recording running.

The microphone is not held in RAM for the whole meeting. The PortAudio
callback copies the block and `put_nowait`s it onto a queue of about thirty
seconds (`audio.mic_queue_seconds`, default 30). A `meeting-mic-writer` thread
appends PCM and checkpoints the WAV header about once a second (`flush` and
`fsync`). A full queue does not drop the timeline: those blocks, and every
block after them, stay in an in-memory list, and one merged `mic_queue_full`
range is recorded. The callback never waits on disk. If a checkpoint raises
(disk full, for example), the writer stops touching the file and the rest of
the meeting stays in that list. `stop()` rebuilds the mic track as the flushed
prefix plus that remainder, and a writer error does not skip stopping the
system tap, the merge, the replace into Recordings, or the archive.
`_finish_mic_writer` closes the WAV only after the writer thread has exited.
The callback also keeps a ring of about the last two minutes
(`audio.mic_ring_seconds`, default 120) for the live tail. `stop()` still
writes the merged 3-channel WAV the watcher expects (mic, system left, system
right). A crash keeps every checkpointed second and loses at most the open
checkpoint plus the queued tail. On the next launch the menu app scans
`~/Documents/MeetingRecorder/.tmp` for orphaned `<stem>.mic.wav` and
`<stem>.sys.wav` files, recovers a playable WAV into Recordings, and posts a
non-modal notification. An orphan that cannot be read is left in place. If
the writer cannot open the file, `stop()` still falls back to the in-memory
blocks.

It sends those tails to `gemini-3.8-flash` on the pinned Gemini endpoint.
An incoming line that overlaps a committed line of the same speaker by more
than half of the shorter interval replaces that line when the new text is
longer and still covers its start; otherwise the re-hear is dropped. A tail
that has slid forward keeps the committed words and appends only the new
aligned suffix. A short fragment is absorbed when a longer same-speaker
hearing starts within two seconds and its text begins with that fragment.
Any other line that starts more than half a second before the
frontier is dropped. Offsets within the tail are clip-relative. An offset
past the length of the tail is read as a meeting time and kept only when it
falls inside the window; anything else is rejected, not pinned to the end of
the tail. When that answer leaves at least half a second of detected speech
uncovered, each gap is sent again as a 15 second slice with a short prompt
that asks for every word. A remark that is still missing is asked once more
on a slice that starts where that answer stopped and, at the end of the clip,
runs through the last sample. Only lines that land in those gaps are kept.
An unfinished
line at the edge of an open tail is held for one tick and then committed, so
a long unpunctuated monologue is not left behind the frontier. A later tick
can still upgrade that fragment. After Stop, the final tick is skipped when
the system WAV has already been archived. A transcription or prompt failure
shows a status line in the panel and does not stop or delay capture. A clean
prompt is reused only when the cleaned text is non-empty; an empty result is
retried up to three times per card. The audio callback is not on this path.

When the watcher marks a transcript partial (coverage below the minimum) and
`Recordings/<stem>.live.json` exists, it inserts live lines whose start falls
in the missing ranges, including the tail after the last timestamp. Before
that, spans between final lines that sit within 30 seconds of each other are
subtracted, so a `missing_time_ranges` entry that overlaps speech already in
the transcript does not duplicate it. A line that only repeats an existing
timestamp is skipped. A line that starts within a fraction of a second of that
timestamp and continues into the gap is inserted. When channel alignment
actually shifted the mic, live timestamps are moved by the same
`channel_alignment.lag_seconds` (positive lag means the mic was late, so the
live clock moves earlier). Each inserted line is tagged `[live]`, and
`partial` stays true. The filled ranges and line
count are stored on `_meta.live_fill`. The same fill runs again just before
the JSON is written, so a later speaker-reconcile pass cannot drop the tags;
a second pass is idempotent. A transcript that already meets the coverage
minimum is not modified. Escalation (one fresh single-call retry, then
chunked retry when the audio is long enough) and the Telegram partial alert
are unchanged. When lines were inserted, the alert adds: "Gaps were filled
from the live transcript." The live file is written with mode `0600`.

Pyannote diarization now also runs for single-source recordings. That is a
behaviour change: the acoustic prior can change speaker labels compared with
voice-only Gemini. The watcher logs when the prior is used. Set
`diarization.enabled: false` to skip it. The worker timeout defaults to 900
seconds (`diarization.timeout_seconds`), not an hour. Pyannote is started
from `tools/diarize.py`. launchd runs
`python tools/transcribe_watcher.py`, so `sys.path[0]` is `tools/` and a
direct `import pyannote_mp_worker` fails (`No module named 'pyannote_mp_worker'`).
The spawn target is `_pyannote_child` in `tools/diarize.py`. That child puts
the repository root on `sys.path` and then imports `pyannote_mp_worker.py`.

Prompt detection uses the broadened judge question: Swiss German and Hochdeutsch
indirect instructions (“ich würd em Claude säge, er söll …”, “mir müessted em
Agent säge …”) and English dictation inside dialect. A line that once scored
at or above the threshold keeps that score, so a later window cannot split
the same dictation onto a second card. An unfinished line that tells Claude
or an agent what to do stays on that card when the same speaker continues
it within 20 seconds, even if one judge call scored the opening under the
threshold. Gemini then writes the
clean prompt (intent only; in the language spoken: German or Swiss German becomes Hochdeutsch, English stays English, unless the speaker clearly
asks for another language). Clips stay verbatim.

Headless replay, no window:

```bash
python -m tools.sidecar replay --wav /path/to/recording.wav
```

It reveals the WAV in 20-second steps, prints each new line with its lag, and
prints each prompt card again when a later tail completes it. Lag is the
simulated tick time plus the real transcription duration, minus the line's
audio time. A line that is still cut off at the end of a tick is held for that
one tick and committed on the next; a finished sentence is committed
immediately, and a later tail can replace a fragment with the complete
hearing. Optional `--from-seconds`
and `--to-seconds` limit the span. The command reads `GEMINI_API_KEY` from the
environment or from the repository `.env` (the main checkout's `.env` when
this directory is a worktree).

## Headless verification and attended checklist

The automated suite covers the preference gate (jev true authorises Jev without
an external acknowledgement), the endpoint pin, regular-sidecar authority,
text-free Jev audit, capture-first sidecar failure, no modal on start, a
non-blocking mic callback with a crash-recoverable WAV, a crashing or slow
live worker, monotonic mark offsets, alignment mapping, persisted live
sessions, partial-transcript live fill, and a scripted concurrent
mark-plus-clip run without opening an audio device. The global hotkey and the
real panel still need an attended macOS session. Before merge:

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
   `Recordings/<stem>.live.json` is still there after the window is closed.
4. During or after the recording, type a topic and press **Clip kopieren**.
   Confirm the clipboard is verbatim transcript text with the fixed header.
5. Dictate an indirect instruction (“ich würde Claude sagen, er soll …”).
   Confirm a card appears, **Kopieren** pastes a clean prompt in the spoken language, and the
   verbatim lines are visible under it. Run **Prompts…** on the finished
   transcript and confirm that path copies the clean prompt and also offers
   any card saved in the live file.
6. Mark once from the menu and once with `⌃⌥⌘P`. Confirm both numeric marks
   and the first-mic offset in the sidecar.
7. Temporarily make the sidecar directory unwritable; confirm recording still
   starts, a failure notification appears, and Jev remains off.
8. During a short recording, confirm **Clip…**, **Prompts…**, and **List Audio
   Devices** are disabled and Stop stays clickable. After stop, run **Clip…**
   on a prior transcript and confirm the dialog comes to the front and the
   notification shows a line count. The WAV duration still matches wall clock.
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
