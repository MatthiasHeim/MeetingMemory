#!/usr/bin/env python3
"""
TranscribeWatcher - Automatic transcription service for MeetingRecorder

Monitors a folder for new audio files and automatically transcribes them
using either noScribe (Whisper) or Gemini Flash, then sends results to n8n webhook.

Processing modes:
- "whisper": Traditional noScribe transcription → n8n analysis
- "gemini": Direct Gemini 2.5 Flash transcription + analysis
- "both": Run both pipelines in parallel for comparison

Usage:
    python transcribe_watcher.py [--config PATH]
"""

import os
import sys
import time
import json
import queue
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import yaml
import requests
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError as e:
    Observer = None
    FileSystemEventHandler = object
    FileCreatedEvent = object
    WATCHDOG_AVAILABLE = False
    _WATCHDOG_IMPORT_ERROR = str(e)

# Marker used by the stale-code check below. Captured at module-import time
# so we can warn when tools/*.py changes on disk without a process restart —
# the operational failure mode behind the Stefan-mislabel incident where the
# fixes were on disk but the long-running watcher was still serving the
# pre-fix bytecode in memory.
_PROCESS_START_MONOTONIC = time.time()
_PROCESS_START_WALL = time.time()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Add tools directory to path for sibling module imports
_tools_dir = Path(__file__).parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

# Import Gemini processing modules (optional - gracefully handle if not available)
try:
    from audio_converter import (
        TOPOLOGY_MULTI_SOURCE_GENUINE,
        TOPOLOGY_SINGLE_SOURCE,
        classify_source_topology,
        convert_for_gemini,
        detect_active_channels,
        get_audio_duration,
    )
    from gemini_processor import GeminiAudioProcessor, GeminiResult
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    _GEMINI_IMPORT_ERROR = str(e)

# Import neon_insert for programmatic InsightBase source creation +
# enrichment. Graceful degrade: if the module or psycopg2 is missing, the
# watcher still processes transcripts and triggers Claude (which has its
# own insert fallback in /meeting-actions Step 1).
try:
    from neon_insert import (
        insert_source as _insightbase_insert_source,
        update_source_with_gemini as _insightbase_update_with_gemini,
        update_source_calendar_match as _insightbase_update_calendar,
    )
    NEON_INSERT_AVAILABLE = True
except ImportError as e:
    NEON_INSERT_AVAILABLE = False
    _NEON_INSERT_IMPORT_ERROR = str(e)

# Import calendar_resolve for participant resolution. Optional — if the
# import fails we skip calendar resolution and let Claude handle it.
try:
    import calendar_resolve as _calendar_resolve
    CALENDAR_RESOLVE_AVAILABLE = True
except ImportError as e:
    CALENDAR_RESOLVE_AVAILABLE = False
    _CALENDAR_RESOLVE_IMPORT_ERROR = str(e)

# Import speaker_reconcile to canonicalize Gemini-guessed speaker names
# against the calendar attendee list before persisting the transcript.
try:
    from speaker_reconcile import reconcile as _reconcile_speakers
    SPEAKER_RECONCILE_AVAILABLE = True
except ImportError as e:
    SPEAKER_RECONCILE_AVAILABLE = False
    _SPEAKER_RECONCILE_IMPORT_ERROR = str(e)

# Import channel_vad + speaker_verify for channel-based speaker attribution:
# the 3-channel hybrid WAV carries the host mic on its own channel, which is
# physical ground truth for who speaks when. channel_vad feeds (a) a prompt
# map into the Gemini call and (b) the post-transcription verification pass
# that flips confidently-misattributed turns.
try:
    from channel_vad import compute_channel_vad as _compute_channel_vad
    CHANNEL_VAD_AVAILABLE = True
except ImportError as e:
    CHANNEL_VAD_AVAILABLE = False
    _CHANNEL_VAD_IMPORT_ERROR = str(e)

try:
    from speaker_verify import verify as _verify_speakers
    SPEAKER_VERIFY_AVAILABLE = True
except ImportError as e:
    SPEAKER_VERIFY_AVAILABLE = False
    _SPEAKER_VERIFY_IMPORT_ERROR = str(e)

try:
    from diarize import (
        PYANNOTE_AVAILABLE,
        PYANNOTE_IMPORT_ERROR,
        fuse_host_cluster_with_channel_vad,
        run_pyannote_diarization,
    )
    DIARIZATION_AVAILABLE = True
except ImportError as e:
    DIARIZATION_AVAILABLE = False
    PYANNOTE_AVAILABLE = False
    PYANNOTE_IMPORT_ERROR = str(e)
    _DIARIZATION_IMPORT_ERROR = str(e)

# Import speaker_hints for counterpart inference when the calendar lookup
# returned no external attendees (e.g. BlueCare meetings live on their
# Teams/M365 calendar, not Matthias's Google Calendar). High-precision
# transcript signals (real-name speaker labels, direct address) validated
# against the Brain ClientContext directories.
try:
    from speaker_hints import detect_counterpart as _detect_counterpart
    SPEAKER_HINTS_AVAILABLE = True
except ImportError as e:
    SPEAKER_HINTS_AVAILABLE = False
    _SPEAKER_HINTS_IMPORT_ERROR = str(e)


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"


def _git_sha() -> Optional[str]:
    """Return the short git SHA of this repo, or None if unavailable."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def _tool_file_mtimes() -> dict[str, float]:
    """Return mtime (seconds since epoch) for each .py in tools/."""
    out: dict[str, float] = {}
    for p in Path(__file__).resolve().parent.glob("*.py"):
        try:
            out[p.name] = p.stat().st_mtime
        except OSError:
            continue
    return out


def _log_code_version(logger: logging.Logger) -> None:
    """Emit a one-shot banner with git SHA + tool-file mtimes.

    Anchors every log file to a specific code revision. If the long-running
    daemon is ever stale-deployed (fixes on disk, old bytecode in memory),
    the operator can grep for this banner to see which revision is actually
    running — that ambiguity is what hid the Stefan-mislabel root cause for
    days.
    """
    sha = _git_sha() or "unknown"
    mtimes = _tool_file_mtimes()
    newest = max(mtimes.values()) if mtimes else 0
    newest_iso = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S") if newest else "n/a"
    logger.info(f"Code version: git={sha}, newest tools/*.py mtime={newest_iso}")


def _warn_if_code_is_stale(logger: logging.Logger) -> None:
    """Warn if any tools/*.py has been modified after this process started.

    Fires once per detected file (suppressed by a module-level set so the log
    doesn't get spammed). The next clean restart picks up the new code; until
    then, the message gives the operator a single clear signal that a new
    commit has landed and the watcher needs to be cycled.
    """
    stale: list[str] = []
    for name, mtime in _tool_file_mtimes().items():
        if mtime > _PROCESS_START_WALL and name not in _STALE_FILES_REPORTED:
            stale.append(name)
            _STALE_FILES_REPORTED.add(name)
    if stale:
        sha = _git_sha() or "unknown"
        logger.warning(
            f"Stale watcher: tools/*.py modified after process start — "
            f"{', '.join(stale)}. Restart the watcher to load new code "
            f"(current loaded SHA approx: {sha} but disk may be ahead)."
        )


_STALE_FILES_REPORTED: set[str] = set()


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "watcher.log"

    logger = logging.getLogger("TranscribeWatcher")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def html_to_text(html_content: str) -> str:
    """Extract plain text from HTML transcript."""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_content)
    # Decode HTML entities
    import html
    text = html.unescape(text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TranscriptionQueue:
    """Thread-safe queue for managing transcription jobs."""

    def __init__(self, logger: logging.Logger):
        self.queue = queue.Queue()
        self.processing = False
        self.current_file: Optional[Path] = None
        self.logger = logger

    def add(self, audio_file: Path):
        """Add a file to the transcription queue."""
        self.queue.put(audio_file)
        self.logger.info(f"Queued for transcription: {audio_file.name}")

    def get(self) -> Optional[Path]:
        """Get next file from queue, non-blocking."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def is_empty(self) -> bool:
        return self.queue.empty()


class AudioFileHandler(FileSystemEventHandler):
    """Handles new audio file events."""

    def __init__(self, transcription_queue: TranscriptionQueue,
                 debounce_seconds: float, logger: logging.Logger):
        self.queue = transcription_queue
        self.debounce_seconds = debounce_seconds
        self.logger = logger
        self.pending_files = {}  # file_path -> timer

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process WAV files
        if file_path.suffix.lower() != '.wav':
            return

        self.logger.debug(f"File detected: {file_path.name}")

        # Cancel existing timer for this file
        if file_path in self.pending_files:
            self.pending_files[file_path].cancel()

        # Set debounce timer
        timer = threading.Timer(
            self.debounce_seconds,
            self._add_to_queue,
            args=[file_path]
        )
        self.pending_files[file_path] = timer
        timer.start()

    def _add_to_queue(self, file_path: Path):
        """Add file to queue after debounce period."""
        if file_path in self.pending_files:
            del self.pending_files[file_path]

        # Verify file still exists and has content
        if file_path.exists() and file_path.stat().st_size > 0:
            self.queue.add(file_path)
        else:
            self.logger.warning(f"File no longer exists or is empty: {file_path.name}")


class TranscribeWatcher:
    """Main watcher service that monitors folder and processes transcriptions."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Expand paths
        self.recordings_dir = expand_path(config['paths']['recordings'])
        self.transcripts_dir = expand_path(config['paths']['transcripts'])
        self.noscribe_path = Path(config['noscribe']['path'])

        # Ensure directories exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Processing mode: "whisper", "gemini", or "both"
        self.processing_mode = config.get('processing', {}).get('mode', 'whisper')
        self.logger.info(f"Processing mode: {self.processing_mode}")

        # Validate Gemini availability if needed
        if self.processing_mode in ('gemini', 'both') and not GEMINI_AVAILABLE:
            self.logger.warning(
                "Gemini processing requested but modules not available. "
                "Falling back to whisper mode. Install: pip install google-genai"
            )
            self.processing_mode = 'whisper'

        # Initialize Gemini processor if needed
        self.gemini_processor = None
        if self.processing_mode in ('gemini', 'both') and GEMINI_AVAILABLE:
            gemini_config = config.get('gemini', {})
            api_key = os.environ.get(gemini_config.get('api_key_env', 'GEMINI_API_KEY'))
            if api_key:
                self.gemini_processor = GeminiAudioProcessor(
                    api_key=api_key,
                    model=gemini_config.get('model', 'gemini-2.5-flash'),
                    max_output_tokens=gemini_config.get('max_output_tokens', 65536),
                    temperature=gemini_config.get('temperature', 0.1),
                    timeout_seconds=gemini_config.get('timeout', 600)
                )
                self.logger.info(f"Gemini processor initialized with model: {gemini_config.get('model', 'gemini-2.5-flash')}")
            else:
                self.logger.error(f"GEMINI_API_KEY not found in environment. Gemini processing disabled.")
                if self.processing_mode == 'gemini':
                    self.processing_mode = 'whisper'

        # Initialize queue
        self.queue = TranscriptionQueue(logger)

        # Set up file watcher
        if not WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog package not installed; TranscribeWatcher cannot "
                f"watch files ({_WATCHDOG_IMPORT_ERROR})"
            )
        debounce = config.get('watcher', {}).get('debounce_seconds', 2)
        self.handler = AudioFileHandler(self.queue, debounce, logger)
        self.observer = Observer()

        self.running = False

    def start(self):
        """Start watching for new files."""
        self.logger.info(f"Starting TranscribeWatcher...")
        _log_code_version(self.logger)
        self.logger.info(f"Watching: {self.recordings_dir}")
        self.logger.info(f"Transcripts: {self.transcripts_dir}")

        # Check for existing unprocessed files
        self._process_existing_files()

        # Start file watcher
        self.observer.schedule(self.handler, str(self.recordings_dir), recursive=False)
        self.observer.start()

        self.running = True
        self.logger.info("Watcher started. Press Ctrl+C to stop.")

        # Process queue in main loop
        try:
            while self.running:
                self._process_queue()
                time.sleep(self.config.get('watcher', {}).get('poll_interval', 1))
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the watcher."""
        self.logger.info("Stopping watcher...")
        self.running = False

        # Terminate any running Claude sessions
        for proc in getattr(self, '_claude_pids', []):
            if proc.poll() is None:
                self.logger.info(f"Terminating Claude session PID={proc.pid}")
                proc.terminate()

        self.observer.stop()
        self.observer.join()
        self.logger.info("Watcher stopped.")

    def _process_existing_files(self):
        """Check for WAV files that don't have corresponding outputs."""
        for wav_file in self.recordings_dir.glob("*.wav"):
            needs_processing = False

            if self.processing_mode in ('whisper', 'both'):
                # Check for Whisper HTML transcript
                transcript_file = self.transcripts_dir / f"{wav_file.stem}.html"
                if not transcript_file.exists():
                    needs_processing = True

            if self.processing_mode in ('gemini', 'both'):
                # Check for Gemini JSON output
                gemini_json = self.transcripts_dir / f"{wav_file.stem}.json"
                if not gemini_json.exists():
                    needs_processing = True

            if needs_processing:
                self.logger.info(f"Found unprocessed file: {wav_file.name}")
                self.queue.add(wav_file)

    def _process_queue(self):
        """Process the next file in the queue based on processing mode."""
        audio_file = self.queue.get()
        if audio_file is None:
            return

        self.queue.current_file = audio_file
        self.queue.processing = True

        try:
            if self.processing_mode == 'whisper':
                self._process_with_whisper(audio_file)
            elif self.processing_mode == 'gemini':
                self._process_with_gemini(audio_file)
            elif self.processing_mode == 'both':
                # Run both pipelines
                self.logger.info(f"Running both pipelines for: {audio_file.name}")
                self._process_with_whisper(audio_file)
                self._process_with_gemini(audio_file)
            else:
                self.logger.error(f"Unknown processing mode: {self.processing_mode}")
        except Exception as e:
            self.logger.error(f"Error processing {audio_file.name}: {e}")
        finally:
            self.queue.processing = False
            self.queue.current_file = None

    def _premix_audio(self, audio_file: Path) -> Path:
        """Pre-mix multi-channel audio to mono, dropping silent channels.

        Background: the macOS capture pipeline sometimes writes a 3-channel WAV
        where only one channel contains audio (observed with BlackHole routing:
        audio lands in channel 2 (LFE); channels 0/1 are digital silence). A
        naive equal-weight mix across all channels attenuates the signal by
        ~10dB and pushes Swiss German ASR below its accuracy threshold.

        Strategy: per-channel volumedetect → keep channels with mean_volume
        above SILENCE_THRESHOLD_DB → mix only those channels.
        """
        import tempfile

        ffprobe_path = '/opt/homebrew/bin/ffprobe'
        ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
        SILENCE_THRESHOLD_DB = -60.0  # channels below this are treated as silent

        # Check channel count
        probe_cmd = [ffprobe_path, '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=channels', '-of', 'csv=p=0',
                     str(audio_file)]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            channels = int(result.stdout.strip())
            self.logger.debug(f"Detected {channels} channels in {audio_file.name}")
        except Exception as e:
            self.logger.warning(f"ffprobe failed: {e}, assuming 2 channels")
            channels = 2

        if channels <= 1:
            return audio_file

        # Measure per-channel mean volume (probe first 60s for speed).
        try:
            active_channels = detect_active_channels(
                audio_file, probe_seconds=60,
                silence_db=SILENCE_THRESHOLD_DB,
            )
        except Exception as e:
            self.logger.warning(
                f"Active-channel detection failed ({e}); "
                f"falling back to equal-weight mix of all {channels}"
            )
            active_channels = list(range(channels))

        if not active_channels:
            self.logger.warning(f"No active channels detected in {audio_file.name}, "
                                f"falling back to equal-weight mix of all {channels}")
            active_channels = list(range(channels))

        if channels == 2 and active_channels == [0, 1]:
            # Standard stereo with audio on both channels — noScribe's -ac 1 handles this fine
            return audio_file

        self.logger.info(
            f"Pre-mixing to mono: {channels} channels detected, "
            f"using active channels {active_channels}"
        )

        temp_dir = Path(tempfile.gettempdir())
        mixed_file = temp_dir / f"{audio_file.stem}_mixed.wav"

        weight = 1.0 / len(active_channels)
        mix_filter = (
            f"pan=mono|c0=" + '+'.join([f'{weight}*c{i}' for i in active_channels])
        )

        mix_cmd = [
            ffmpeg_path, '-y', '-i', str(audio_file),
            '-af', mix_filter,
            '-ar', '48000',
            '-c:a', 'pcm_s16le',
            str(mixed_file),
        ]

        try:
            result = subprocess.run(mix_cmd, capture_output=True, text=True)
            if result.returncode == 0 and mixed_file.exists():
                self.logger.debug(f"Pre-mixed audio saved to: {mixed_file}")
                return mixed_file
            else:
                self.logger.warning(f"Pre-mix failed, using original: {result.stderr}")
                return audio_file
        except Exception as e:
            self.logger.warning(f"Pre-mix error, using original: {e}")
            return audio_file

    def _process_with_whisper(self, audio_file: Path):
        """Transcribe a single audio file using noScribe (Whisper)."""
        transcript_file = self.transcripts_dir / f"{audio_file.stem}.html"

        self.logger.info(f"Starting transcription: {audio_file.name}")
        start_time = time.time()

        # Pre-mix multi-channel audio to ensure all channels are transcribed
        processed_audio = self._premix_audio(audio_file)

        # Build noScribe command
        cmd = [
            sys.executable,
            str(self.noscribe_path),
            str(processed_audio),
            str(transcript_file),
            "--no-gui",
            "--language", self.config['noscribe'].get('language', 'auto'),
            "--speaker-detection", str(self.config['noscribe'].get('speaker_detection', 'auto')),
        ]

        # Add optional flags
        if self.config['noscribe'].get('timestamps', True):
            cmd.append("--timestamps")

        pause = self.config['noscribe'].get('pause', 'none')
        if pause and pause != 'none':
            cmd.extend(["--pause", pause])

        model = self.config['noscribe'].get('model')
        if model:
            cmd.extend(["--model", model])

        self.logger.debug(f"Command: {' '.join(cmd)}")

        # Run noScribe
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.noscribe_path.parent
            )

            if result.returncode != 0:
                self.logger.error(f"noScribe failed with exit code {result.returncode}")
                self.logger.error(f"stdout: {result.stdout}")
                self.logger.error(f"stderr: {result.stderr}")
                return

        except Exception as e:
            self.logger.error(f"Failed to run noScribe: {e}")
            return

        processing_time = time.time() - start_time
        self.logger.info(f"Transcription complete: {transcript_file.name} ({processing_time:.1f}s)")

        # Cleanup temp mixed audio file if created
        if processed_audio != audio_file and processed_audio.exists():
            try:
                processed_audio.unlink()
                self.logger.debug(f"Cleaned up temp file: {processed_audio}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file: {e}")

        # Send webhook notification
        if self.config.get('webhook', {}).get('enabled', False):
            self._send_webhook(audio_file, transcript_file, processing_time)

        # Seed InsightBase with a sources row BEFORE triggering Claude.
        # If this fails, we still trigger Claude (graceful degrade).
        source_id = self._seed_insightbase_source(transcript_file)

        # Trigger Claude for immediate processing
        self._trigger_claude(transcript_file, source_id=source_id)

    def _send_webhook(self, audio_file: Path, transcript_file: Path, processing_time: float):
        """Send transcription result to n8n webhook.

        Sends payload matching n8n workflow expectations:
        {
            "transcript_path": "/path/to/transcript.html",
            "transcript_html": "<html>...</html>",
            "started_at": "2025-12-05T15:56:47Z",
            "audio_duration_seconds": 2279
        }
        """
        webhook_url = self.config.get('webhook', {}).get('url', '')

        if not webhook_url:
            self.logger.debug("Webhook URL not configured, skipping notification")
            return

        try:
            # Read transcript HTML content
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_html = f.read()

            # Get audio duration (approximate from file size, assuming 16kHz mono 16-bit)
            audio_size = audio_file.stat().st_size
            # WAV header is ~44 bytes, 16kHz * 2 bytes = 32000 bytes/second
            duration_seconds = max(0, (audio_size - 44) / 32000)

            # Parse start time from filename (format: YYYY-MM-DD_HH-MM-SS.wav)
            # Example: 2025-12-05_15-56-47.wav
            # Note: Filename timestamp is in LOCAL time, need to convert to UTC
            filename_without_ext = audio_file.stem
            try:
                date_part, time_part = filename_without_ext.split('_')
                year, month, day = date_part.split('-')
                hour, minute, second = time_part.split('-')
                # Parse as local time (no timezone = naive datetime interpreted as local)
                local_dt = datetime(int(year), int(month), int(day),
                                   int(hour), int(minute), int(second))
                # Convert local time to UTC
                # astimezone() treats naive datetime as local time and converts to UTC
                utc_dt = local_dt.astimezone(timezone.utc)
                started_at = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                self.logger.debug(f"Parsed timestamp: local={local_dt}, utc={started_at}")
            except (ValueError, IndexError) as e:
                # Fallback: use current time if filename doesn't match expected format
                self.logger.warning(f"Could not parse timestamp from filename: {filename_without_ext} ({e})")
                started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Build payload matching n8n workflow expectations
            payload = {
                "transcript_path": str(transcript_file),
                "transcript_html": transcript_html,
                "started_at": started_at,
                "audio_duration_seconds": round(duration_seconds)
            }

            timeout = self.config.get('webhook', {}).get('timeout', 30)
            response = requests.post(webhook_url, json=payload, timeout=timeout)
            response.raise_for_status()

            self.logger.info(f"✅ Webhook sent successfully to n8n: {response.status_code}")
            self.logger.debug(f"Payload sent (HTML content: {len(transcript_html)} chars)")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Webhook request failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error preparing webhook: {e}")

    def _process_with_gemini(self, audio_file: Path):
        """Process audio file using Gemini 2.5 Flash for transcription + analysis."""
        if not self.gemini_processor:
            self.logger.error("Gemini processor not initialized, skipping Gemini processing")
            return

        self.logger.info(f"Starting Gemini processing: {audio_file.name}")
        _warn_if_code_is_stale(self.logger)
        start_time = time.time()

        try:
            # Step 1: Convert WAV to MP3 (extracts mic channel)
            self.logger.info("Converting WAV to MP3...")
            mp3_path = convert_for_gemini(audio_file, output_dir=self.transcripts_dir)
            self.logger.info(f"Converted to: {mp3_path.name} ({mp3_path.stat().st_size / 1024 / 1024:.1f} MB)")

            # Step 2: Get audio duration
            audio_duration = get_audio_duration(mp3_path)
            self.logger.info(f"Audio duration: {audio_duration / 60:.1f} minutes")

            # Step 2b: JSON output path used downstream (and as the source
            # of truth for calendar timestamp parsing — filename-derived).
            json_path = mp3_path.with_suffix('.json')

            # Step 2c: Resolve calendar attendees BEFORE the Gemini call.
            # Two reasons:
            #   1. Pass attendees into Gemini so it doesn't guess names —
            #      this prevents the cross-chunk drift that produced bogus
            #      "Vivienne" / "Speaker 1" / "Speaker 2" labels in long
            #      meetings (each chunk re-guessed in isolation).
            #   2. The result also feeds speaker_reconcile as a safety net
            #      for any remaining generic "Speaker A/B" labels.
            # Filename encodes the timestamp, so resolve() works even though
            # the JSON doesn't exist yet.
            cal_match = self._resolve_calendar(json_path)
            known_attendees = (cal_match or {}).get("participant_details") or []
            known_attendees = self._merge_configured_diarization_roster(
                audio_file, known_attendees
            )
            if known_attendees:
                self.logger.info(
                    f"Calendar attendees ({len(known_attendees)}): "
                    f"{', '.join(a.get('name', '?') for a in known_attendees)}"
                )

            # Step 2d: Classify source topology from the ORIGINAL WAV. Channel
            # count alone is unsafe: in-room recordings can be 3-channel files
            # with only ch0 active. In that topology, channel_vad MUST stay
            # None for both prompt injection and speaker verification.
            topology = self._classify_source_topology_safe(audio_file)

            channel_vad = None
            if topology and topology.topology == TOPOLOGY_SINGLE_SOURCE:
                self.logger.info(
                    "Source topology is single_source "
                    f"(active_channels={topology.active_channels}); "
                    "disabling channel VAD and speaker verification"
                )
            else:
                # Channel VAD from the ORIGINAL 3-channel WAV (the MP3 is
                # already mixed mono). Gives Gemini a ground-truth host/remote
                # speaking map and feeds post-transcription verification.
                # None for mono/stereo/non-genuine recordings.
                channel_vad = self._compute_channel_vad_safe(audio_file)

            diarization_segments = None
            diarization_cfg = self.config.get("diarization", {}) or {}
            diarization_enabled = diarization_cfg.get("enabled", True)
            channel_fusion = diarization_cfg.get("channel_fusion", False)
            if diarization_enabled:
                should_run_diarization = self._should_run_diarization_prior(
                    topology, channel_fusion
                )
                if should_run_diarization:
                    num_speakers = self._diarization_num_speakers(
                        audio_file, known_attendees
                    )
                    diarization_segments = self._run_diarization_safe(
                        mp3_path, num_speakers=num_speakers
                    )
                    if (
                        diarization_segments
                        and channel_fusion
                        and topology
                        and topology.topology == TOPOLOGY_MULTI_SOURCE_GENUINE
                    ):
                        diarization_segments = fuse_host_cluster_with_channel_vad(
                            diarization_segments, channel_vad
                        )
                elif topology and topology.topology == TOPOLOGY_MULTI_SOURCE_GENUINE:
                    self.logger.info(
                        "Source topology is multi_source_genuine; "
                        "diarization.channel_fusion is off, preserving "
                        "legacy channel_vad + speaker_verify behavior"
                    )

            # Step 3: Process with Gemini (attendees + channel map injected)
            self.logger.info("Sending to Gemini API...")
            result = self.gemini_processor.process_audio(
                mp3_path, known_attendees=known_attendees or None,
                channel_segments=channel_vad.segments if channel_vad else None,
                diarization_segments=diarization_segments,
            )

            processing_time = time.time() - start_time
            self.logger.info(f"Gemini processing complete in {processing_time:.1f}s")

            if result.error:
                self.logger.error(f"Gemini processing error: {result.error}")
                self._notify_telegram_failure(audio_file, f"Gemini error: {result.error}")
                return

            # Log some stats
            if result.input_tokens and result.output_tokens:
                self.logger.info(f"Tokens - Input: {result.input_tokens}, Output: {result.output_tokens}")

            # Step 4b: Speaker reconciliation safety net. Even with attendees
            # in the prompt, Gemini may still emit "Speaker A/B" for unclear
            # voices — map those to canonicals. Mutates `result` in place so
            # downstream JSON write, DB seed, and `update_source_with_gemini`
            # all see canonical names from the calendar.
            # Step 4a2: Counterpart hint when calendar gave no externals.
            # Mutates cal_match (or creates one) so the reconcile below has
            # a canonical name to rewrite "Speaker B" to.
            cal_match = self._infer_counterpart_if_unknown(result, cal_match)

            self._reconcile_speakers_inplace(result, cal_match)

            # Step 4b2: Channel-based attribution verification. Runs AFTER
            # reconcile so flips target canonical names. Flips turns whose
            # transcript label confidently contradicts the mic-channel
            # ground truth (see speaker_verify module docstring).
            self._verify_speakers_inplace(result, channel_vad, cal_match)

            # Step 4c: Save JSON result alongside MP3 (now canonical).
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.parsed_response, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved result to: {json_path.name}")

            # Step 5: Send webhook if configured (legacy path; disabled by default)
            if self.config.get('webhook_gemini', {}).get('enabled', False):
                self._send_gemini_webhook(audio_file, mp3_path, result, audio_duration, processing_time)

            # Step 6a: Seed sources row in InsightBase.
            source_id = self._seed_insightbase_source(json_path)

            # Step 6b: Deterministic ingest — populate audio-derived fields and
            # write meeting_metadata. We do this BEFORE Claude so /meeting-actions
            # can skip its sources-UPDATE block. Each step is best-effort.
            self._enrich_with_gemini(source_id, result, audio_duration)

            # Step 6c: Persist the calendar resolution we computed in Step 4a.
            self._persist_calendar(source_id, cal_match)

            # Step 6d: Telegram ping — fires as soon as the row is in the DB
            # and ready for downstream LLM work. The Claude session below
            # finishes minutes later and posts no second notification.
            self._notify_telegram_meeting_captured(
                source_id=source_id, result=result, cal_match=cal_match,
                audio_duration=audio_duration,
            )

            # Step 7: Trigger Claude for the LLM-only steps (insights, email
            # draft, ClientContext, CRM, wiki). It reads the seeded fields
            # and skips its old Step 1b/Step 2-UPDATE/Step 8.
            self._trigger_claude(json_path, source_id=source_id)

        except Exception as e:
            self.logger.error(f"Gemini processing failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self._notify_telegram_failure(audio_file, f"{type(e).__name__}: {e}")

    def _send_gemini_webhook(self, audio_file: Path, mp3_path: Path,
                             result: 'GeminiResult', audio_duration: float,
                             processing_time: float):
        """Send Gemini transcription result to n8n webhook.

        Sends payload optimized for the conversations_gemini table:
        {
            "source_audio_path": "/path/to/original.wav",
            "source_mp3_path": "/path/to/converted.mp3",
            "started_at": "2025-01-01T00:00:00Z",
            "duration_seconds": 1800,
            "processing_time_seconds": 45.2,
            "gemini_model": "gemini-2.5-flash",
            "gemini_input_tokens": 57600,
            "gemini_output_tokens": 8500,
            "transcript_text": "Full transcript...",
            "transcript_language": "de",
            "title": "Meeting Title",
            "summary": "Meeting summary...",
            "key_points": ["point1", "point2"],
            "tags": ["tag1", "tag2"],
            "participants": {...},
            "sentiment": "positive",
            "meeting_type": "client_call",
            "lailix_communication_score": 7,
            "lailix_communication_feedback": "...",
            "lailix_sales_score": 6,
            "lailix_sales_feedback": "...",
            "lailix_strategic_alignment": "...",
            "lailix_improvement_areas": ["area1", "area2"],
            "lailix_strengths": ["strength1", "strength2"]
        }
        """
        webhook_url = self.config.get('webhook_gemini', {}).get('url', '')

        if not webhook_url:
            self.logger.debug("Gemini webhook URL not configured, skipping notification")
            return

        try:
            # Parse start time from filename
            filename_without_ext = audio_file.stem
            try:
                date_part, time_part = filename_without_ext.split('_')
                year, month, day = date_part.split('-')
                hour, minute, second = time_part.split('-')
                local_dt = datetime(int(year), int(month), int(day),
                                   int(hour), int(minute), int(second))
                utc_dt = local_dt.astimezone(timezone.utc)
                started_at = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, IndexError):
                started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Extract data from Gemini result
            data = result.parsed_response or {}
            lailix_feedback = data.get('lailix_feedback', {})

            # Build payload for conversations_gemini table
            payload = {
                # Source info
                "source_audio_path": str(audio_file),
                "source_mp3_path": str(mp3_path),
                "started_at": started_at,
                "duration_seconds": round(audio_duration),

                # Processing metadata
                "processing_time_seconds": round(processing_time, 2),
                "gemini_model": result.model,
                "gemini_input_tokens": result.input_tokens,
                "gemini_output_tokens": result.output_tokens,

                # Transcript
                "transcript_text": data.get('transcript', ''),
                "transcript_language": data.get('language', ''),

                # Standard metadata
                "title": data.get('title', ''),
                "summary": data.get('summary', ''),
                "key_points": data.get('key_points', []),
                "tags": data.get('tags', []),
                "participants": data.get('participants', {}),
                "sentiment": data.get('sentiment', ''),
                "meeting_type": data.get('meeting_type', ''),

                # Lailix-specific feedback
                "lailix_communication_score": lailix_feedback.get('communication_score'),
                "lailix_communication_feedback": lailix_feedback.get('communication_feedback', ''),
                "lailix_sales_score": lailix_feedback.get('sales_score'),
                "lailix_sales_feedback": lailix_feedback.get('sales_feedback', ''),
                "lailix_strategic_alignment": lailix_feedback.get('strategic_alignment', ''),
                "lailix_improvement_areas": lailix_feedback.get('improvement_areas', []),
                "lailix_strengths": lailix_feedback.get('strengths', []),

                # Raw response for debugging
                "gemini_raw_response": data
            }

            timeout = self.config.get('webhook_gemini', {}).get('timeout', 60)
            response = requests.post(webhook_url, json=payload, timeout=timeout)
            response.raise_for_status()

            self.logger.info(f"✅ Gemini webhook sent successfully: {response.status_code}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Gemini webhook request failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error preparing Gemini webhook: {e}")

    def _seed_insightbase_source(self, transcript_path: Path) -> Optional[int]:
        """Insert a minimal sources row in InsightBase and return its id.

        Called BEFORE launching Claude so that a record of the meeting
        persists even if /meeting-actions fails to run. /meeting-actions
        Step 1 receives the id via --source-id and UPDATEs the row with
        enriched metadata (summary, participant_details, calendar_event_id,
        company) that requires Google Calendar resolution.

        Returns the source_id on success, or None if the insert fails or
        the neon_insert module isn't available. Failure here is NEVER
        fatal — we always fall through to the Claude trigger.
        """
        if not NEON_INSERT_AVAILABLE:
            self.logger.warning(
                f"neon_insert unavailable ({_NEON_INSERT_IMPORT_ERROR}); "
                f"skipping InsightBase seed. /meeting-actions will insert inline."
            )
            return None

        # Derive started_at from filename (YYYY-MM-DD_HH-MM-SS) if possible.
        started_at = None
        try:
            date_part, time_part = transcript_path.stem.split('_')
            y, mo, d = date_part.split('-')
            h, mi, s = time_part.split('-')
            local_dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
            started_at = local_dt.astimezone(timezone.utc)
        except (ValueError, IndexError):
            self.logger.debug(f"Could not parse started_at from {transcript_path.stem}")

        try:
            source_id = _insightbase_insert_source(
                transcript_path=str(transcript_path),
                title=transcript_path.stem,  # /meeting-actions will overwrite with real title
                started_at=started_at,
                origin='noscribe',
                sensitivity_level='internal',
            )
            self.logger.info(
                f"Seeded InsightBase source id={source_id} for {transcript_path.name}"
            )
            return source_id
        except Exception as e:
            # Graceful degrade — never block the pipeline on DB issues.
            self.logger.warning(
                f"InsightBase seed failed for {transcript_path.name}: {e}. "
                f"/meeting-actions will insert inline as a fallback."
            )
            return None

    def _enrich_with_gemini(self, source_id: Optional[int], result,
                              audio_duration: float) -> None:
        """Populate sources audio-derived fields + INSERT meeting_metadata.

        Best-effort. Failures here do not block calendar resolution, Telegram,
        or the Claude trigger.
        """
        if source_id is None:
            self.logger.debug("No source_id; skipping Gemini enrichment")
            return
        if not NEON_INSERT_AVAILABLE:
            self.logger.warning(
                f"neon_insert unavailable; skipping Gemini enrichment "
                f"({_NEON_INSERT_IMPORT_ERROR})"
            )
            return
        try:
            _insightbase_update_with_gemini(
                source_id=source_id,
                gemini_result=result.parsed_response,
                duration_seconds=audio_duration,
            )
        except Exception as e:
            self.logger.warning(
                f"Gemini enrichment failed for source_id={source_id}: {e}"
            )

    def _resolve_calendar(self, transcript_path: Path) -> Optional[dict]:
        """Look up the calendar event matching the transcript timestamp.

        Returns the resolution dict (calendar_event_id, company,
        participant_details, participant_resolution_log) or None if the
        lookup didn't run / failed. No DB writes here — `_persist_calendar`
        does that AFTER the source row exists.
        """
        if not CALENDAR_RESOLVE_AVAILABLE:
            self.logger.warning(
                f"calendar_resolve unavailable; Claude will resolve "
                f"({_CALENDAR_RESOLVE_IMPORT_ERROR})"
            )
            return None
        try:
            return _calendar_resolve.resolve(transcript_path)
        except Exception as e:
            self.logger.warning(f"calendar_resolve failed: {e}")
            return None

    def _reconcile_speakers_inplace(self, result, cal_match: Optional[dict]) -> None:
        """Canonicalize Gemini-guessed speaker names against calendar attendees.

        Mutates the GeminiResult fields (transcript, participants,
        speaker_emotions, speaker_pacing, interruptions, energy_levels) so the
        JSON write, the source content_text, the meeting_metadata row, and
        Claude all see the same canonical names. Stashes a forensic log on
        cal_match so it gets persisted via `participant_resolution_log`.
        """
        if not SPEAKER_RECONCILE_AVAILABLE:
            self.logger.warning(
                f"speaker_reconcile unavailable; speaker names not canonicalized "
                f"({_SPEAKER_RECONCILE_IMPORT_ERROR})"
            )
            return
        if not cal_match:
            return
        try:
            d = result.parsed_response
            recon_log = _reconcile_speakers(d, cal_match)
            # Push reconciled values back into the dataclass.
            result.transcript = d.get("transcript", result.transcript)
            result.participants = d.get("participants", result.participants)
            result.speaker_emotions = d.get("speaker_emotions", result.speaker_emotions)
            result.speaker_pacing = d.get("speaker_pacing", result.speaker_pacing)
            result.interruptions = d.get("interruptions", result.interruptions)
            result.energy_levels = d.get("energy_levels", result.energy_levels)
            # Attach to participant_resolution_log so it lands in the DB.
            prl = cal_match.setdefault("participant_resolution_log", {})
            prl["speaker_reconciliation"] = recon_log
            if recon_log.get("rewrote_speakers"):
                self.logger.info(
                    f"Speaker reconciliation rewrote "
                    f"{recon_log['rewrote_speakers']} name(s) using calendar"
                )
        except Exception as e:
            self.logger.warning(f"Speaker reconciliation failed: {e}")

    def _infer_counterpart_if_unknown(self, result, cal_match: Optional[dict]):
        """Infer the counterpart from the transcript when calendar gave none.

        Only runs when the calendar resolution produced no external
        attendee (the ~37%-of-meetings case). On a confident hint, injects
        the person into cal_match.participant_details so the downstream
        speaker_reconcile rewrites the generic labels, and records the
        decision in participant_resolution_log. Best-effort: returns
        cal_match unchanged on any failure or abstention.
        """
        try:
            externals = [
                p for p in (cal_match or {}).get("participant_details") or []
                if isinstance(p, dict) and (p.get("role") or "").lower() != "self"
            ]
            if externals:
                return cal_match
            if not SPEAKER_HINTS_AVAILABLE:
                self.logger.debug(
                    f"speaker_hints unavailable ({_SPEAKER_HINTS_IMPORT_ERROR})"
                )
                return cal_match
            hint = _detect_counterpart(result.parsed_response)
            if not hint:
                self.logger.info(
                    "No calendar attendees and no confident transcript "
                    "hint — counterpart stays unresolved (Brain Step 1c "
                    "may still infer it)"
                )
                return cal_match
            self.logger.info(
                f"Counterpart inferred from transcript: {hint['name']} "
                f"({hint['company']}) via {hint['method']} — {hint['evidence']}"
            )
            if cal_match is None:
                cal_match = {
                    "participant_details": [{
                        "name": "Matthias Heim",
                        "email": "matthias@lailix.com",
                        "company": "Lailix", "role": "self",
                    }],
                    "participant_resolution_log": {},
                    "calendar_event_id": None,
                    "company": None,
                }
            cal_match.setdefault("participant_details", []).append({
                "name": hint["name"],
                "email": hint.get("email"),
                "company": hint.get("company"),
                "role": "participant",
                "confidence": hint.get("confidence", "high"),
                "resolution_method": hint["method"],
            })
            if not cal_match.get("company") and hint.get("company"):
                cal_match["company"] = hint["company"]
            prl = cal_match.setdefault("participant_resolution_log", {})
            prl.setdefault("resolutions", []).append({
                "name": hint["name"],
                "method": hint["method"],
                "confidence": hint.get("confidence", "high"),
                "evidence": hint.get("evidence", ""),
            })
            return cal_match
        except Exception as e:
            self.logger.warning(f"Counterpart inference failed: {e}")
            return cal_match

    def _classify_source_topology_safe(self, audio_file: Path):
        """Classify source topology. Never raises."""
        try:
            topology = classify_source_topology(audio_file)
            self.logger.info(
                f"Source topology: {topology.topology} "
                f"(channels={topology.total_channels}, "
                f"active={topology.active_channels})"
            )
            return topology
        except Exception as e:
            self.logger.warning(
                f"Source topology detection failed for {audio_file.name}: {e}; "
                "falling back to legacy channel VAD behavior"
            )
            return None

    def _merge_configured_diarization_roster(
        self, audio_file: Path, known_attendees: list[dict]
    ) -> list[dict]:
        """Merge optional explicit roster entries into calendar attendees.

        Supported low-friction inputs:
          - config.yaml: diarization.roster: ["Name", {name, company, role}]
          - per-recording sidecar: <recording>.diarization.json with
            {"roster": [...]} or {"attendees": [...]}
        """
        roster_items = []
        cfg = self.config.get("diarization", {}) or {}
        roster_items.extend(cfg.get("roster") or [])

        sidecar = audio_file.with_suffix(".diarization.json")
        if sidecar.exists():
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    roster_items.extend(
                        data.get("roster") or data.get("attendees") or []
                    )
            except Exception as e:
                self.logger.warning(
                    f"Diarization sidecar roster unreadable "
                    f"({sidecar.name}): {e}"
                )

        if not roster_items:
            return known_attendees

        merged = list(known_attendees or [])
        existing = {
            (p.get("name") or "").strip().lower()
            for p in merged
            if isinstance(p, dict)
        }
        for item in roster_items:
            if isinstance(item, str):
                entry = {"name": item.strip(), "role": "participant"}
            elif isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("role", "participant")
            else:
                continue
            name = (entry.get("name") or "").strip()
            if not name or name.lower() in existing:
                continue
            entry["name"] = name
            merged.append(entry)
            existing.add(name.lower())
        return merged

    def _diarization_num_speakers(
        self, audio_file: Path, known_attendees: list[dict]
    ) -> Optional[int]:
        """Return explicit pyannote speaker count, or None for auto mode."""
        candidates = []
        sidecar = audio_file.with_suffix(".diarization.json")
        if sidecar.exists():
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    candidates.append(data.get("num_speakers"))
            except Exception as e:
                self.logger.warning(
                    f"Diarization sidecar unreadable ({sidecar.name}): {e}"
                )
        candidates.append(os.environ.get("MEETINGMEMORY_NUM_SPEAKERS"))
        candidates.append(
            (self.config.get("diarization", {}) or {}).get("num_speakers")
        )

        for value in candidates:
            if value in (None, "", "auto"):
                continue
            try:
                n = int(value)
            except (TypeError, ValueError):
                self.logger.warning(
                    f"Ignoring invalid diarization num_speakers={value!r}"
                )
                continue
            if n > 0:
                self.logger.info(f"Using explicit pyannote num_speakers={n}")
                return n

        attendee_count = len([
            p for p in known_attendees or []
            if isinstance(p, dict) and (p.get("name") or "").strip()
        ])
        if attendee_count:
            self.logger.info(
                f"Calendar/roster attendee count is {attendee_count}; "
                "leaving pyannote num_speakers on auto (soft hint only)"
            )
        return None

    def _run_diarization_safe(
        self, audio_file: Path, num_speakers: Optional[int] = None
    ) -> Optional[list[dict]]:
        """Run pyannote prior generation. Never raises."""
        if not DIARIZATION_AVAILABLE:
            self.logger.warning(
                f"diarization module unavailable; no pyannote prior "
                f"({_DIARIZATION_IMPORT_ERROR})"
            )
            return None
        if not PYANNOTE_AVAILABLE:
            self.logger.warning(
                f"pyannote unavailable; no diarization prior "
                f"({PYANNOTE_IMPORT_ERROR})"
            )
            return None
        cfg = self.config.get("diarization", {}) or {}
        timeout = int(cfg.get("timeout_seconds", 3600))
        device = str(cfg.get("device", ""))
        try:
            return run_pyannote_diarization(
                audio_file,
                num_speakers=num_speakers,
                timeout_seconds=timeout,
                device=device,
            )
        except Exception as e:
            self.logger.warning(f"pyannote prior failed: {e}")
            return None

    @staticmethod
    def _should_run_diarization_prior(topology, channel_fusion: bool) -> bool:
        """Default-on only for single-source; remote fusion is opt-in."""
        if topology is None:
            return False
        topo = getattr(topology, "topology", None)
        return (
            topo == TOPOLOGY_SINGLE_SOURCE
            or (channel_fusion and topo == TOPOLOGY_MULTI_SOURCE_GENUINE)
        )

    def _compute_channel_vad_safe(self, audio_file: Path):
        """Compute channel VAD from the original WAV. Never raises.

        Returns a ChannelVAD or None (module unavailable, non-hybrid
        recording, decode failure) — callers treat None as "no channel
        ground truth available" and proceed voice-only as before.
        """
        if not CHANNEL_VAD_AVAILABLE:
            self.logger.warning(
                f"channel_vad unavailable; voice-only attribution "
                f"({_CHANNEL_VAD_IMPORT_ERROR})"
            )
            return None
        try:
            vad = _compute_channel_vad(audio_file)
        except Exception as e:
            self.logger.warning(f"channel_vad failed for {audio_file.name}: {e}")
            return None
        if vad is not None:
            self.logger.info(
                f"Channel VAD: {len(vad.segments)} segments over "
                f"{vad.duration_sec / 60:.1f} min"
            )
        return vad

    def _verify_speakers_inplace(self, result, channel_vad,
                                 cal_match: Optional[dict]) -> None:
        """Flip confidently-misattributed turns using channel ground truth.

        Mutates the GeminiResult (transcript, participants) so the JSON
        write, DB enrichment, and Claude all see verified attribution.
        Attaches the forensic log to participant_resolution_log (next to
        speaker_reconciliation) when a calendar match exists.
        """
        if channel_vad is None:
            return
        if not SPEAKER_VERIFY_AVAILABLE:
            self.logger.warning(
                f"speaker_verify unavailable; attribution not verified "
                f"({_SPEAKER_VERIFY_IMPORT_ERROR})"
            )
            return
        # Namesake guard: verification anchors every "Matthias ..." label to
        # the host. If the calendar shows a REMOTE participant also named
        # Matthias, that anchor is wrong for half the labels — skip entirely.
        for att in (cal_match or {}).get("participant_details") or []:
            name = (att.get("name") or "").strip().lower() if isinstance(att, dict) else ""
            if (name.split() and name.split()[0] == "matthias"
                    and (att.get("role") or "").lower() != "self"):
                self.logger.warning(
                    f"Speaker verification skipped: remote attendee "
                    f"{att.get('name')!r} shares the host's first name"
                )
                return
        try:
            d = result.parsed_response
            verify_log = _verify_speakers(d, channel_vad)
            result.transcript = d.get("transcript") or result.transcript
            result.participants = d.get("participants") or result.participants
            # Persist the decision log in the on-disk JSON in all cases...
            result.speaker_verification_log = verify_log
            # ...and in the DB's participant_resolution_log when a calendar
            # match exists to carry it.
            if cal_match is not None:
                prl = cal_match.setdefault("participant_resolution_log", {})
                prl["speaker_verification"] = verify_log
            n_flips = len(verify_log.get("flips") or [])
            if n_flips:
                self.logger.info(
                    f"Speaker verification flipped {n_flips} turn(s) "
                    f"using channel ground truth"
                )
            else:
                self.logger.info(
                    f"Speaker verification: no flips "
                    f"({verify_log.get('turns_checked', 0)} turns checked)"
                )
        except Exception as e:
            self.logger.warning(f"Speaker verification failed: {e}")

    def _persist_calendar(self, source_id: Optional[int],
                            cal_match: Optional[dict]) -> None:
        """Write the already-resolved calendar match to the sources row."""
        if source_id is None or not cal_match:
            return
        if not NEON_INSERT_AVAILABLE:
            return
        try:
            _insightbase_update_calendar(
                source_id=source_id,
                participant_details=cal_match.get("participant_details") or [],
                participant_resolution_log=cal_match.get("participant_resolution_log") or {},
                calendar_event_id=cal_match.get("calendar_event_id"),
                company=cal_match.get("company"),
            )
        except Exception as e:
            self.logger.warning(
                f"Calendar UPDATE failed for source_id={source_id}: {e}"
            )

    def _notify_telegram_meeting_captured(self, source_id: Optional[int],
                                            result, cal_match: Optional[dict],
                                            audio_duration: float) -> None:
        """Send a one-line Telegram ping that the meeting is in InsightBase.

        Fires BEFORE the Claude session — user knows the row exists and the
        transcript is ready to be pulled into a session if needed. Best-effort.
        """
        notify_path = os.path.expanduser(
            "~/.claude/scripts/telegram_notify.py"
        )
        if not os.path.exists(notify_path):
            self.logger.debug("telegram_notify.py not found; skipping ping")
            return

        title = (cal_match or {}).get("calendar_event_id") and (
            (cal_match or {}).get("participant_resolution_log", {})
            .get("calendar_search", {}).get("chosen_event_title")
        ) or "Untitled meeting"
        company = (cal_match or {}).get("company") or "—"
        duration_min = int(audio_duration // 60) if audio_duration else 0
        sentiment = result.overall_sentiment if result else "neutral"
        intensity = result.sentiment_intensity if result else "moderate"
        chunked = bool(result and result.chunked)
        n_emotions = (
            sum(len(e.get("arc", []) or []) for e in (result.speaker_emotions or []))
            if result else 0
        )
        msg = (
            f"Meeting captured (#{source_id})\n"
            f"{title}\n"
            f"Client: {company} • {duration_min}min • "
            f"sentiment {sentiment}/{intensity}\n"
            f"{n_emotions} emotion arc events"
            + (" • chunked" if chunked else "")
            + "\nClaude /meeting-actions running…"
        )
        try:
            subprocess.run(
                [sys.executable, notify_path, "--category", "Meeting", msg],
                capture_output=True, text=True, timeout=15,
            )
            self.logger.info(f"Telegram ping sent for source_id={source_id}")
        except Exception as e:
            self.logger.warning(f"Telegram ping failed: {e}")

    def _notify_telegram_failure(self, audio_file: Path, reason: str) -> None:
        """Alert that a recording FAILED transcription and was NOT captured.

        Without this the watcher dropped failed recordings silently — a
        malformed Gemini JSON response looked identical to "no meeting today",
        so a real meeting could vanish unnoticed (happened 2026-06-22 10:17 and
        an 888 MB 2026-06-12 recording). The audio is retained, so the file can
        be re-triggered after a fix. Best-effort.
        """
        notify_path = os.path.expanduser(
            "~/.claude/scripts/telegram_notify.py"
        )
        if not os.path.exists(notify_path):
            self.logger.debug("telegram_notify.py not found; skipping failure alert")
            return
        msg = (
            f"⚠️ Transcription FAILED — recording not captured\n"
            f"{audio_file.name}\n"
            f"{reason}\n"
            f"Audio retained; re-trigger after a fix."
        )
        try:
            subprocess.run(
                [sys.executable, notify_path, "--category", "Meeting", msg],
                capture_output=True, text=True, timeout=15,
            )
            self.logger.info(f"Telegram failure alert sent for {audio_file.name}")
        except Exception as e:
            self.logger.warning(f"Telegram failure alert failed: {e}")

    def _trigger_claude(self, transcript_path: Path, source_id: Optional[int] = None):
        """Fire-and-forget headless Claude session to process transcript.

        If source_id is provided, it is passed to /meeting-actions as a
        --source-id flag so Step 1 UPDATEs that row instead of INSERTing
        a fresh one. The watcher seeds the row (see
        _seed_insightbase_source) so a record persists even if this
        Claude session never runs.
        """
        claude_config = self.config.get('claude_trigger', {})
        if not claude_config.get('enabled', False):
            self.logger.debug("Claude trigger disabled, skipping")
            return

        claude_path = claude_config.get('claude_path', '/Users/Matthias/.local/bin/claude')
        brain_repo = claude_config.get('brain_repo', '/Users/Matthias/Repos/Brain')
        command = claude_config.get('command', 'meeting-actions')

        source_id_arg = f" --source-id {source_id}" if source_id else ""
        prompt = (
            f"Read .claude/commands/{command}.md and process transcript: "
            f"{transcript_path}{source_id_arg}"
        )

        # Log file for this Claude session
        log_dir = expand_path(self.config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"claude-{timestamp}.log"

        env = os.environ.copy()
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/Users/Matthias/.local/bin"

        try:
            with open(log_file, 'w') as lf:
                proc = subprocess.Popen(
                    [claude_path, "-p", prompt, "--dangerously-skip-permissions"],
                    cwd=brain_repo,
                    env=env,
                    stdout=lf,
                    stderr=lf,
                )
            self.logger.info(f"Claude trigger fired: PID={proc.pid}, log={log_file.name}")
            if not hasattr(self, '_claude_pids'):
                self._claude_pids = []
            self._claude_pids.append(proc)
        except Exception as e:
            self.logger.error(f"Failed to trigger Claude: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Watch for audio files and automatically transcribe them"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up logging
    log_dir = expand_path(config['paths']['logs'])
    logger = setup_logging(log_dir)

    logger.info(f"Config loaded from: {args.config}")

    # Create and start watcher
    watcher = TranscribeWatcher(config, logger)
    watcher.start()


if __name__ == "__main__":
    main()
