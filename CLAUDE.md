# noScribe - Claude Code Context

## Project Overview

noScribe is an AI-powered audio transcription tool using Whisper and pyannote for speaker diarization.

**Main app:** `noScribe.py`

## Key Files

| File | Purpose |
|------|---------|
| `noScribe.py` | Main GUI application and transcription logic |
| `pyannote_mp_worker.py` | Speaker diarization subprocess worker |
| `whisper_mp_worker.py` | Whisper transcription subprocess worker |
| `noScribeEdit/` | Transcript editor application |

## MeetingRecorder (macOS companion tool)

Located in `tools/`:

| File | Purpose |
|------|---------|
| `tools/meeting_recorder.py` | Menu bar app for recording control |
| `tools/transcribe_watcher.py` | Background service that auto-transcribes new recordings |
| `tools/install.sh` | Installation script |
| `tools/uninstall.sh` | Uninstallation script |
| `tools/launchagents/` | LaunchAgent plists for auto-start |

**Config location:** `~/Documents/MeetingRecorder/config.yaml`

## Important Technical Notes

### PyTorch 2.6+ Compatibility

PyTorch 2.6 changed `torch.load()` default to `weights_only=True`, breaking pyannote model loading. The fix in `pyannote_mp_worker.py` adds safe globals before loading:

```python
from pyannote.audio.core.task import Specifications, Problem, Resolution
from omegaconf import ListConfig, DictConfig
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, ListConfig, DictConfig])
```

### launchd PATH Issues

When running under launchd (LaunchAgents), PATH is minimal. Use absolute paths for system tools:
- `/opt/homebrew/bin/ffmpeg`
- `/opt/homebrew/bin/ffprobe`

### Multi-Channel Audio Mixing

noScribe's ffmpeg uses `-ac 1` which only takes the first 2 channels. For 3+ channel recordings (e.g., stereo system audio + mono mic), pre-mix with:

```python
weight = 1.0 / channels
mix_filter = f"pan=mono|c0={'+'.join([f'{weight}*c{i}' for i in range(channels)])}"
```

See `_premix_audio()` in `tools/transcribe_watcher.py`.

### Audio Setup for MeetingRecorder

Uses Background Music app + Aggregate Device:
- Background Music: Captures system audio with volume control
- Aggregate Device: Combines Background Music (2ch) + Microphone (1ch) = 3 channels
- Sample rate: 48000 Hz

## Requirements Files

Located in `environments/`:
- `requirements_macOS_arm64.txt` - Apple Silicon Macs
- `requirements_linux.txt` - Linux
- `requirements_win_cpu.txt` - Windows (CPU)
- `requirements_win_cuda.txt` - Windows (NVIDIA GPU)

**Key dependencies:**
- `pyannote.audio>=4` - Speaker diarization
- `faster-whisper` - Transcription
- `torch==2.8` / `torchaudio==2.8` - PyTorch (pyannote 4 not compatible with torch 2.9)
- `omegaconf` - Required for PyTorch 2.6+ safe globals fix

## Git Remotes

This fork setup:
```
origin   → https://github.com/MatthiasHeim/noScribe (this fork)
upstream → https://github.com/kaixxx/noScribe (original)
```

To sync with upstream:
```bash
git fetch upstream
git merge upstream/main
```

## CLI Usage

```bash
# Basic transcription
python noScribe.py input.wav output.html --no-gui --language auto

# With speaker detection
python noScribe.py input.wav output.html --no-gui --speaker-detection auto --timestamps

# Models: tiny, small, medium, precise (default)
python noScribe.py input.wav output.html --no-gui --model precise
```

## Logs

- MeetingRecorder watcher: `~/Documents/MeetingRecorder/logs/watcher.log`
- Recordings: `~/Documents/MeetingRecorder/Recordings/`
- Transcripts: `~/Documents/MeetingRecorder/Transcripts/`
