# Meeting sidecar: first-slice offline clerk

Status on 2026-09-26: the offline package, provider gate, cache, clip selector,
prompt selector, and calibration harness are present. The release gate did **not**
pass Gemini criterion 2, so the recorder menu-bar work is intentionally absent
from this branch. `tools/meeting_recorder.py`, the WAV writer, watcher, and
transcription path have no diff.

## Usage

The package reads a completed recorder JSON from
`~/Documents/MeetingRecorder/Transcripts/<stem>.json`. It never edits that file.
Set the recorder's existing `GEMINI_API_KEY` in the invoking environment.

```bash
python -m tools.sidecar clip \
  --stem 2026-09-08_14-31-16 \
  --topic "User roles and permissions in the dashboard" \
  --judge gemini

python -m tools.sidecar clip \
  --stem 2026-09-08_14-31-16 \
  --topic "User roles and permissions in the dashboard" \
  --without-timestamps

python -m tools.sidecar prompts --stem 2026-09-08_14-31-16

python -m tools.sidecar calibrate \
  --eval-dir ~/.local/share/meeting-sidecar/eval-2026-09-26
```

`clip` produces only the fixed header plus verbatim transcript lines. Default
hysteresis is seed `0.60`, grow `0.25`, bridge at most two lines, require two
seeds, and remove pure fillers. `--widen` uses the explicit one-step lower pair
`0.50` / `0.15`. The retained `0.60` / `0.25` pair passed the source-955
range diagnostic below, but was not independently optimized for Gemini; no
release follows from that diagnostic because the Gemini calibration gate failed.

`prompts` refuses Gemini extraction unless `calibration/latest.json` names the
same Gemini model as a passing selected model and supplies its chosen threshold.
It will therefore currently report that prompt extraction is disabled, rather
than use the historical `0.45` candidate after the failed gate.

The score cache is local-only at `~/.local/share/meeting-sidecar/cache/`. Each
entry contains probability vectors and hashes, never transcript text. Its key
includes the normalized question criteria, full ordered question layout, model,
context, a human prompt version, and a deterministic hash of the rendered batch
template/schema plus batch size. A Gemini prompt threshold additionally binds
that hash to the canonical `dictating_prompt` definition, so a prompt change
cannot reuse stale scores. The
calibration corpus at `~/.local/share/meeting-sidecar/eval-2026-09-26/` is read
in place and is never copied into this repository. Calibration deliberately
bypasses this cache so its reported requests, token volume, and latency are
from a fresh provider pass.

## Provider gate

Gemini is always the default. `--judge jev` is rejected unless the exact
recording has `Recordings/<stem>.sidecar.json` containing `"jev": true` **and**
an explicit boolean `"external_attendees": false`; a missing, malformed,
opt-out, unresolved, or external sidecar fails closed to Gemini. The factory
runs this decision before it constructs `JevJudge`, so a denied request does
not create an OpenRouter-capable client. `JevJudge` repeats the gate at direct
construction and immediately before its helper request, so a later caller
cannot bypass the decision. The adapter uses the existing audited
`jev_decide.py` helper and therefore writes its helper audit metadata when it
is permitted.

For client meetings, do not set `jev: true`: the TypeSafe/OpenRouter route is
not covered by the client DPA described in the governing spec. The external
attendee gate is deliberately a necessary second check rather than treating the
human checkbox as legal approval. `external_attendees: false` is an explicit
recording classification, not an independently verified DPA determination; a
future enabled recorder path must obtain it from resolved calendar/client data.
This branch makes no automatic or inferred Jev choice and does not ship the UI
that could create new Jev sidecars.

## Gemini calibration

All A/B/C language versions are evaluated with the same single-question,
20-line contextual request layouts used by `clip` and `prompts`. Each line
carries its prior six lines. The fresh v3 pass made 246 requests per model; AI
Studio returned token counts but no bill amount, so observed token volume and
p50 request time are retained as reproducible cost/latency telemetry.

| Gemini model | A EN relevance AUC | B Swiss German relevance AUC | C Swiss German prompt recall / FP | Prompt threshold | Input / output tokens | p50 batch request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemini-3.8-flash` | 0.908 | 0.815 | 1.00 / 0 | 0.900 | 1,704,866 / 175,505 | 4.83 s |
| `gemini-3.1-flash-lite` | 0.819 | 0.904 | 0.50 / 0 | 0.800 | 1,704,866 / 175,905 | 2.60 s |

| Model | A EN | A DE | B Swiss German | B DE | B EN | C Swiss German | C DE | C EN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-3.8-flash` relevance AUC | 0.908 | 0.916 | 0.815 | 0.844 | 0.794 | 0.488 | 0.484 | 0.500 |
| `gemini-3.8-flash` prompt recall / FP | 0.50 / 0 | 0.50 / 0 | 0.00 / 0 | 0.00 / 0 | 0.00 / 0 | 1.00 / 0 | 0.83 / 0 | 0.33 / 0 |
| `gemini-3.1-flash-lite` relevance AUC | 0.819 | 0.830 | 0.904 | 0.902 | 0.862 | 0.511 | 0.522 | 0.492 |
| `gemini-3.1-flash-lite` prompt recall / FP | 0.50 / 0 | 0.00 / 0 | 0.00 / 0 | 0.00 / 0 | 0.00 / 0 | 0.50 / 0 | 0.67 / 0 | 0.17 / 0 |

Neither model satisfies all release conditions: 3.8 misses B relevance by
`0.065`, and 3.1-lite misses A relevance by `0.061` plus C prompt recall by
`0.333`. Accordingly, no model is selected and no recorder UI is included.
The v3 aggregate-only report is local at
`~/.local/share/meeting-sidecar/calibration/latest.json`.
The failed report predates the final deterministic protocol/definition-fingerprint
metadata added after measurement; that is safe because it has no selected model
and cannot enable Gemini prompts. Any future passing calibration must be run
again with the fingerprinted report format.

### Source-955 clip diagnostic

The default `gemini-3.8-flash` clip request for the specified roles topic
selected 20 lines from exactly `43:05` through `45:42`, with zero selected
lines outside that interval, at seed/grow `0.60` / `0.25`. It therefore meets
acceptance criterion 1 as a range-only diagnostic. It does not override the
criterion-2 fail-stop or justify recorder UI work.

The stored Jev reference remains materially stronger on relevance (A EN AUC
`0.919`, B Swiss German AUC `0.923`), but must not be used for client meeting
text under the DPA gate.

### Evidence disagreement: Jev prompt threshold

The build brief says Jev achieved 5/6 C-Swiss-German prompt recall at `0.45`.
The checked-in `evalj.py` instead reports 5/6 at `0.50`; directly applying its
per-line metric to the stored scores at `0.45` yields 6/6 with one false
positive. This document follows the stored script/data rather than silently
repeating the conflicting `0.45` wording.

## Mark-to-transcript mapping

The sidecar format reserves `marks` for numeric seconds since the recorder's
monotonic start clock. Prompt extraction nominally maps a mark `m` to transcript
time `m`, searches from `m - 5 s`, and stops at `m + 3 min` or the
line-level prompt end. A sentence-level yes/no judge removes complete lead-in
sentences only; it never rewrites prompt text.

Lead-in filtering splits a timestamped transcript turn into deterministic,
verbatim sentence substrings before asking its yes/no question. It can therefore
remove a lead-in while retaining a prompt in the same turn; it never asks a
model to rewrite either sentence. It removes only an initial non-prompt prefix;
once prompt content begins, every later sentence is retained verbatim so an
ambiguous interior constraint is never silently deleted.

That nominal map is **not an exact shared-clock map in the current recorder**:

- `AudioRecorder` launches the system tap before the mic, and the two WAVs have
  independent sample-zero origins.
- `capture_provenance.merge_filter` merges/pads those streams at file sample
  zero; the current tap binary does not retain system host timestamps.
- The watcher later runs `channel_align` only on a derived analysis WAV. A
  positive lag means the mic was late and is advanced, while the system
  channel's zero remains unchanged.

If the missing offsets were persisted, a mic-derived timestamp would be
`max(0, m - mic_origin_delay - channel_lag)`. They are not persisted today, so
the implemented nominal mapping is deliberately limited and uses the five
second look-back tolerance. The menu shortcut/mark UI remains gated behind a
passing Gemini calibration and, later, the spec's shared-timebase live-capture
work; it has not been enabled by this failed slice.

## Known offline limitation

The loader judges only complete `[mm:ss] Speaker: text` turns. It deliberately
does not invent timestamps or speakers for malformed/untimestamped continuation
text, so such text is not sent to a judge. A future enabled UI must surface a
partial-transcript state rather than imply that a clip covers those omitted
lines; this failed offline slice has no recorder UI in which to do so.
