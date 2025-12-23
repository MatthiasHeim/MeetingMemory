#!/usr/bin/env python3
"""
TranscribeWatcher - Automatic transcription service for MeetingRecorder

Monitors a folder for new audio files and automatically transcribes them
using noScribe, then sends results to an n8n webhook.

Usage:
    python transcribe_watcher.py [--config PATH]
"""

import os
import sys
import time
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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"


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

        # Initialize queue
        self.queue = TranscriptionQueue(logger)

        # Set up file watcher
        debounce = config.get('watcher', {}).get('debounce_seconds', 2)
        self.handler = AudioFileHandler(self.queue, debounce, logger)
        self.observer = Observer()

        self.running = False

    def start(self):
        """Start watching for new files."""
        self.logger.info(f"Starting TranscribeWatcher...")
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
        self.observer.stop()
        self.observer.join()
        self.logger.info("Watcher stopped.")

    def _process_existing_files(self):
        """Check for WAV files that don't have corresponding transcripts."""
        for wav_file in self.recordings_dir.glob("*.wav"):
            transcript_file = self.transcripts_dir / f"{wav_file.stem}.html"
            if not transcript_file.exists():
                self.logger.info(f"Found unprocessed file: {wav_file.name}")
                self.queue.add(wav_file)

    def _process_queue(self):
        """Process the next file in the queue."""
        audio_file = self.queue.get()
        if audio_file is None:
            return

        self.queue.current_file = audio_file
        self.queue.processing = True

        try:
            self._transcribe_file(audio_file)
        except Exception as e:
            self.logger.error(f"Error transcribing {audio_file.name}: {e}")
        finally:
            self.queue.processing = False
            self.queue.current_file = None

    def _premix_audio(self, audio_file: Path) -> Path:
        """Pre-mix multi-channel audio to mono, ensuring all channels are included.

        noScribe's ffmpeg uses -ac 1 which ignores channel 3 in 3-channel files.
        This function properly mixes all channels with equal weight.
        """
        import tempfile

        # Check channel count (use full path for ffprobe in case PATH isn't set in launchd)
        ffprobe_path = '/opt/homebrew/bin/ffprobe'
        probe_cmd = [ffprobe_path, '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=channels', '-of', 'csv=p=0',
                     str(audio_file)]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            channels = int(result.stdout.strip())
            self.logger.debug(f"Detected {channels} channels in {audio_file.name}")
        except Exception as e:
            self.logger.warning(f"ffprobe failed: {e}, assuming 2 channels")
            channels = 2  # Assume stereo if probe fails

        if channels <= 2:
            # Standard stereo or mono, no pre-processing needed
            return audio_file

        self.logger.info(f"Pre-mixing {channels}-channel audio to mono")

        # Create temp file for mixed audio
        temp_dir = Path(tempfile.gettempdir())
        mixed_file = temp_dir / f"{audio_file.stem}_mixed.wav"

        # Mix all channels equally: for 3 channels, each gets 1/3 weight
        # pan filter: mono output, mix all input channels
        weight = 1.0 / channels
        mix_filter = f"pan=mono|c0={'+'.join([f'{weight}*c{i}' for i in range(channels)])}"

        ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
        mix_cmd = [
            ffmpeg_path, '-y', '-i', str(audio_file),
            '-af', mix_filter,
            '-ar', '48000',  # Keep original sample rate
            '-c:a', 'pcm_s16le',
            str(mixed_file)
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

    def _transcribe_file(self, audio_file: Path):
        """Transcribe a single audio file using noScribe."""
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
