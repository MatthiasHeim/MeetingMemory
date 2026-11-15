# Privacy scan (shadow)

Goal: keep private passages (a speaker's family life, an identifiable person's health) away from
Gemini and Claude while the rest of a meeting is still transcribed. Scope agreed by Matthias on
2026-11-15: block health and patient data; cut private or family talk at segment level.

`tools/privacy_scan.py` is the detector only. It changes nothing in the pipeline and writes
timestamps and probabilities, never text.

1. Local Whisper (`mlx-community/whisper-large-v3-turbo`, MLX) transcribes the audio roughly.
2. ~30 s windows go to the local Decider with two `noul` questions: `private_family`, `personal_health`.
3. Windows at p >= 0.5 become spans, padded 15 s each side and merged.

## First measurement (2026-11-15, one 67-min client meeting with known private passages)

| Path | ASR | Decider | Spans | Cut |
|---|---|---|---|---|
| Existing Gemini transcript | n/a | 428 s | 3 | 11.1 min |
| Local Whisper on the audio | 124 s | 534 s | 4 | 11.2 min |

The audio-only path covered 93% of the seconds the transcript path flagged; 8% of its cut was extra.
So a local first pass is good enough to decide what to cut before audio leaves the Mac. Spans not yet
checked by a person against the recording.

## Business meetings, false-alarm check (2026-11-15, transcript path)

12 meetings from 03.–14.11. (6 h 40 min), including five with a health-industry client about
patient-data systems: the health-industry meetings had **no** flagged second. Flags elsewhere:

| Meeting (stem) | Length | Flagged | Kind |
|---|---|---|---|
| 2026-11-09_15-59-47 | 35 min | 31:56–32:52 (p 0.65) | health |
| 2026-11-11_15-22-28 | 3 min | 0:58–1:53 (p 0.90) | family |
| 2026-11-14_11-05-24 | 25 min | five spans, 6.7 min (p 0.81–0.99) | family |
| the other 9 | | none | |

Not yet checked by a person whether these are real private talk or false alarms.

## Not built yet

- Silencing flagged spans in the audio before the Gemini call in `transcribe_watcher.py`
  (a hold that removes content: needs Matthias's OK before it is installed).
- The live sidecar sends ~24 s windows to Gemini during the meeting; a check after the meeting
  cannot protect those. Options: a "Privat" button that pauses live sending, live off for
  personal meetings, or a local check per window.
- Threshold calibration on more meetings, including health-industry client meetings
  (business talk about patient-data systems must not be cut).
