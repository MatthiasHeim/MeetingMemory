# MeetingMemory

> Capture audio from any meeting on macOS, transcribe and analyse it with Gemini, and feed the result into a Brain-driven LLM pipeline that extracts insights, drafts follow-ups, updates client context, and writes to the CRM.

MeetingMemory is the personal recording-to-knowledge stack used at [Lailix](https://lailix.com). It started as a fork of [noScribe](https://github.com/kaixxx/noScribe) by Kai Dröge, but the local-transcription path is no longer used — audio goes straight to Gemini 2.5 Flash, and the watcher hands off to a headless Claude session in a separate `Brain` repo that owns all downstream automation.

## Pipeline

```
┌─────────────────────────┐
│ MeetingRecorder (menu   │   user clicks Start/Stop
│ bar app, macOS)         │   ───────────────────────►  WAV in ~/Documents/MeetingRecorder/Recordings/
└────────────┬────────────┘
             │ watchdog file event
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ transcribe_watcher.py  (LaunchAgent, KeepAlive)                             │
│                                                                             │
│   1. Pre-mix multi-channel WAV → mono                                       │
│   2. ffmpeg WAV → MP3                                                       │
│   3. Gemini 2.5 Flash:  transcript + audio-derived signals                  │
│        (sentiment, speaker_emotions, pacing, interruptions, energy)         │
│   4. Calendar resolve  (gws CLI → ±15min match against Matthias's primary)  │
│   5. Speaker reconciliation  (canonicalise Gemini-guessed names against     │
│        calendar attendees BEFORE persisting — see tools/speaker_reconcile)  │
│   6. Persist:  sources INSERT + meeting_metadata INSERT + sources UPDATE    │
│                (calendar_event_id, company, participant_details)            │
│   7. Telegram ping  ("Meeting captured #N — title — sentiment …")           │
│   8. Trigger headless Claude  /meeting-actions  in the Brain repo           │
└─────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Brain repo  /meeting-actions  (Claude Code, headless)                       │
│                                                                             │
│   • Extracts insights to InsightBase (topics + insights tables)             │
│   • Drafts a follow-up email as a Gmail draft                               │
│   • Updates ClientContext, CRM contacts and opportunity notes               │
│   • Files action items as Linear tasks                                      │
│   • Compiles to the Brain wiki                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The transcript JSON, the seeded `sources` row, and the Telegram ping all exist within ~30 seconds of the recording stopping, before the Claude session has finished its LLM work.

## Components

| Path | Purpose |
|------|---------|
| `tools/meeting_recorder.py` | Menu-bar app for start/stop recording |
| `tools/transcribe_watcher.py` | Watchdog service: WAV → Gemini → InsightBase → Claude trigger |
| `tools/gemini_processor.py` | Audio-only Gemini prompt + chunking + reduce-pass for long meetings |
| `tools/audio_converter.py` | WAV pre-mix and MP3 encode for Gemini |
| `tools/calendar_resolve.py` | Pulls attendees from Google Calendar via the Brain `gws` wrapper |
| `tools/speaker_reconcile.py` | Canonicalises Gemini-guessed speaker names against the calendar |
| `tools/neon_insert.py` | InsightBase writes (sources + meeting_metadata) |
| `tools/launchagents/com.user.transcribewatcher.plist` | macOS LaunchAgent, KeepAlive |

## Setup (macOS, Apple Silicon)

```bash
git clone https://github.com/MatthiasHeim/MeetingMemory.git
cd MeetingMemory
python3 -m venv venv
source venv/bin/activate
pip install -r tools/requirements.txt
```

### `.env` (project root)

```
GEMINI_API_KEY=...                      # Google AI Studio
INSIGHTBASE_DATABASE_URL=postgres://... # Neon, project bitter-waterfall-54453207
```

### `~/Documents/MeetingRecorder/config.yaml`

```yaml
paths:
  recordings:  ~/Documents/MeetingRecorder/Recordings
  transcripts: ~/Documents/MeetingRecorder/Transcripts
  logs:        ~/Documents/MeetingRecorder/logs

processing:
  mode: gemini   # whisper | gemini | both — gemini is the supported path

gemini:
  model: gemini-2.5-flash
  api_key_env: GEMINI_API_KEY

claude_trigger:
  enabled: true
  brain_repo: ~/Desktop/Repos/Brain
  command: meeting-actions
```

### Audio routing (system audio + mic)

1. `brew install --cask background-music`
2. Open **Audio MIDI Setup**, create an Aggregate Device with Background Music (2ch) + your mic, sample rate 48000 Hz.
3. Set Background Music's output to your speakers/headphones from its menu bar icon.

### Install the LaunchAgents

```bash
cd tools
./install.sh    # copies plists, bootstraps the watcher and the recorder
```

Verify:

```bash
launchctl list | grep -E 'transcribewatcher|meetingrecorder'
tail -f ~/Documents/MeetingRecorder/logs/watcher.log
```

## Operations

| | |
|---|---|
| Watcher logs | `~/Documents/MeetingRecorder/logs/watcher.log` |
| Per-Claude-run logs | `~/Documents/MeetingRecorder/logs/claude-YYYY-MM-DD_HH-MM-SS.log` |
| Recordings | `~/Documents/MeetingRecorder/Recordings/` |
| Transcripts (JSON) | `~/Documents/MeetingRecorder/Transcripts/` |
| Watcher config | `~/Documents/MeetingRecorder/config.yaml` |
| Restart watcher | `launchctl bootout gui/$(id -u)/com.user.transcribewatcher && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.transcribewatcher.plist` |

A healthy run, on the next recording, logs:

```
Saved result to: 2026-04-30_10-12-34.json
Speaker reconciliation rewrote 1 name(s) using calendar
Seeded InsightBase source id=…
Enriched source id=…
Calendar-resolved source id=…
Telegram ping sent for source_id=…
Claude trigger fired: PID=…
```

## Tests

```bash
pytest tests/
```

Includes `tests/test_speaker_reconcile.py` (rules: self / confident / singleton / fuzzy / none) and `tests/test_utils.py`.

## Heritage

This repo started life as a fork of [kaixxx/noScribe](https://github.com/kaixxx/noScribe) — the original `noScribe.py` GUI, `whisper_mp_worker.py`, and `pyannote_mp_worker.py` are still in the tree as a fallback transcription path, but Gemini is the supported route and the upstream remote has been dropped.

Acknowledgements:
- **[Kai Dröge](https://github.com/kaixxx)** — original noScribe author
- **[Background Music](https://github.com/kyleneideck/BackgroundMusic)** — system audio capture on macOS
- **[Neon](https://neon.tech)** — serverless Postgres for InsightBase

## License

GPL-3.0 (inherited from noScribe). See `LICENSE`.
