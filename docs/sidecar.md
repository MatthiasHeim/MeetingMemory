# Meeting sidecar: Slice 1 clerk

Status on 2026-09-26: Slice 1 has the finished-transcript clerk plus its
recorder menu-bar entry points. `gemini-3.8-flash` is the only Gemini judge.
The recorder changes are menu/UI and local sidecar metadata only: capture,
WAV writing, the watcher, and transcription are unchanged.

## Use

The CLI reads a completed recorder JSON from
`~/Documents/MeetingRecorder/Transcripts/<stem>.json`; it never edits that
file. Set the recorder's existing `GEMINI_API_KEY` in the invoking environment.

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

- **Start Recording** asks **“Jev verwenden?”** with its checkbox off by
  default. The current event is resolved through the existing calendar resolver
  before recording begins. A uniquely resolved external event displays
  **“nicht durch Kunden-DPA gedeckt”** and leaves Jev disabled. An ambiguous or
  failed lookup is also fail-closed: its local sidecar records
  `external_attendees: true` and Jev is disabled.
- **Prompt markieren** is enabled while recording and appends a monotonic
  elapsed offset to `Recordings/<stem>.sidecar.json`. The global shortcut is
  **Ctrl–Option–Command–P** (`⌃⌥⌘P`). It uses AppKit global and local key-event
  monitors; macOS requires Accessibility permission for global key monitoring.
  Grant it to the interpreter/app running MeetingRecorder under **System
  Settings → Privacy & Security → Accessibility**. The menu item still works
  without that permission.
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

## Provider gate

Gemini is the default. `--judge jev` is rejected unless the exact recording has
`Recordings/<stem>.sidecar.json` containing both `"jev": true` and an explicit
boolean `"external_attendees": false`. Missing, malformed, opt-out, unresolved,
or external sidecars fail closed to Gemini. The factory decides before
constructing `JevJudge`, and `JevJudge` checks again at direct construction and
immediately before its helper request, so a denied path cannot create or use an
OpenRouter-capable client.

The menu writes `jev`, `external_attendees`, `marks`, and small calendar
metadata to the local sidecar before capture starts. The strict offline gate is
unchanged: a checkbox is never legal authority for client material.

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
override does not activate prompt extraction. To amend an existing
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
Prompt extraction nominally maps a mark `m` to transcript time `m`, searches
from `m - 5 s`, and stops at `m + 3 min` or the line-level prompt end. A
sentence-level yes/no judge strips only a leading non-prompt prefix; it never
rewrites prompt text.

Prompt lines merge only when consecutive transcript timestamps are at most
12 seconds apart. A marked card is bounded to the exact `m - 5 s` through
`m + 3 min` window, even when a detected prompt run is longer.

This nominal map is not an exact shared-clock map in the current recorder:

- `AudioRecorder` launches the system tap before the mic, and the streams have
  independent sample-zero origins.
- `capture_provenance.merge_filter` merges/pads them at file sample zero; the
  current tap binary does not retain system host timestamps.
- The watcher later runs `channel_align` only on a derived analysis WAV. A
  positive lag means the mic was late and is advanced, while system zero stays
  unchanged.

If those offsets were persisted, a mic-derived timestamp would be
`max(0, m - mic_origin_delay - channel_lag)`. They are not persisted today, so
the menu uses the documented nominal map plus its five-second look-back
tolerance.

## Headless verification and attended checklist

The automated suite covers the strict Jev gate, monotonic mark offsets, and a
scripted concurrent mark-plus-clip run without opening an audio device; the
test proves all marks survive while a fake recording remains active. The
AppKit dialogs and global hotkey need an attended macOS session. Before merge:

1. Start an internal calendar event, leave Jev off, mark once from the menu and
   once with `⌃⌥⌘P`, then confirm both numeric marks in its local sidecar.
2. Start an external event and confirm the DPA note appears and Jev cannot be
   selected.
3. With a prior completed transcript, run **Clip…** during a short recording;
   confirm the clip notification shows a line count and the WAV duration still
   matches wall clock.
4. Run **Prompts…** and confirm a marked or suggested prompt can be selected
   and copied.

## Known offline limitation

The loader judges only complete `[mm:ss] Speaker: text` turns. It deliberately
does not invent timestamps or speakers for malformed/untimestamped continuation
text, so such text is not sent to a judge.
