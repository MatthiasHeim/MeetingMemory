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
import copy
import queue
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from speaker_integrity import atomic_json, digital_silence, finalize_attribution, save_trial_stage

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
        convert_for_gemini_ducked,
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
    from speaker_verify import verify as _verify_speakers, SELF_NAME as _SELF_FULL_NAME
    SPEAKER_VERIFY_AVAILABLE = True
except ImportError as e:
    SPEAKER_VERIFY_AVAILABLE = False
    _SPEAKER_VERIFY_IMPORT_ERROR = str(e)

# Semantic speaker-coherence gate: the repair path for the (common) case where
# the mic channel bleeds and channel attribution is inadmissible. Runs after
# transcription and BEFORE the JSON write, so no insight is ever extracted
# from an unaudited attribution.
try:
    from speaker_coherence import (
        check_and_repair as _coherence_repair,
        claude_cli_runner as _coherence_cli_runner,
    )
    SPEAKER_COHERENCE_AVAILABLE = True
except ImportError as e:
    SPEAKER_COHERENCE_AVAILABLE = False
    _SPEAKER_COHERENCE_IMPORT_ERROR = str(e)

# Import transcript_validator for the coverage/completeness gate (see
# docs/RELIABILITY_PLAN_2026-07.md Phase 1) that runs on every Gemini result
# before it's written to disk. Unavailable is treated as "cannot validate" —
# _validate_gemini_result degrades to a permissive pass rather than blocking
# the pipeline, matching every other optional module in this file.
try:
    from transcript_validator import (
        validate_transcript as _validate_transcript,
        sanitize_repetition_loops as _sanitize_repetition_loops,
    )
    TRANSCRIPT_VALIDATOR_AVAILABLE = True
except ImportError as e:
    TRANSCRIPT_VALIDATOR_AVAILABLE = False
    _TRANSCRIPT_VALIDATOR_IMPORT_ERROR = str(e)

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

# Recordings below this size are treated as corrupt/empty and skipped rather
# than queued: at 48kHz/3ch/16-bit a real recording is ~288 KB per second, so
# anything under ~0.1 MB has no usable audio. Without this guard a truncated
# capture (e.g. a 102-byte file) is re-queued on every startup, fails ffprobe
# ("Invalid data found"), and fires a Telegram failure alert each time.
MIN_RECORDING_BYTES = 100_000

# F3 periodic rescan (docs/SPEC-error-path-escalation-2026-07-10.md RC3):
# _process_existing_files used to run only at watcher startup, so a meeting
# that ended inside a Gemini outage sat unprocessed until someone manually
# restarted the watcher. Re-running it on a timer turns that into "delayed",
# bounded by a per-WAV attempt cap + min spacing so a flapping watcher
# (crash-restart loop) can't burn through retries in seconds, and an
# alert-dedup so an ongoing outage doesn't re-alert every tick.
RESCAN_INTERVAL_SEC = 30 * 60
RESCAN_MAX_ATTEMPTS = 4
RESCAN_MIN_SPACING_SEC = 30 * 60
RESCAN_STATE_FILENAME = ".rescan_state.json"

# P1 (docs/../meeting-pipeline-investigation-2026-08-10.md §6): the Telegram
# ping in _notify_telegram_meeting_captured fires when the fire-and-forget
# Claude trigger STARTS, not when it succeeds — a session that dies on its
# first line (expired OAuth, hit the org spend limit, no session quota,
# untrusted workspace) produced a ~400-byte log and no further signal,
# indistinguishable from a healthy run in progress. Three meetings sat with
# zero insights for days in 07.08 because of exactly this. _trigger_claude
# now watches the child for CLAUDE_STARTUP_FAILFAST_SEC and alerts if its
# log tail matches one of these known fatal-startup signatures.
CLAUDE_STARTUP_FAILFAST_SEC = 90
# Kept in sync by hand with FATAL_SIGNATURES in
# /Users/Matthias/Repos/Brain/.claude/scripts/meeting_extraction_reconciler.py
# — same fatal-startup class, two separate repos, deliberately not a shared
# import. Update both lists together.
FATAL_STARTUP_SIGNATURES = (
    "OAuth session expired",
    "spend limit",
    "session limit",
    "not been trusted",
    "Failed to authenticate",
    "Execution error",
)


def _load_rescan_state(path: Path) -> dict:
    """Load the per-WAV rescan attempt-tracking state.

    Tolerates a missing or corrupt file (treated as "no attempts recorded
    yet") so a bad state file can never block processing.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_rescan_state(path: Path, state: dict) -> None:
    """Persist rescan attempt-tracking state. Best-effort — a failed write
    just means the next watcher restart re-attempts from attempt 0, which is
    the safe direction to fail in (extra retries, not lost meetings)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


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


def _fmt_mmss(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class _PermissiveValidation:
    """Stand-in for transcript_validator.ValidationResult when that module
    is unavailable — always passes so a missing optional module never
    blocks the pipeline, matching the rest of this file's degrade style."""

    def __init__(self, reasons: list[str]):
        self.passed = True
        self.coverage_pct = 100.0
        self.last_timestamp_sec = 0.0
        self.reasons = reasons
        self.has_chunk_failure_marker = False
        self.has_repetition_loop = False
        self.has_duplicate_span = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "coverage_pct": self.coverage_pct,
            "last_timestamp_sec": self.last_timestamp_sec,
            "reasons": self.reasons,
        }


def _apply_missing_ranges(validation, missing: list) -> None:
    if missing:
        validation.passed = False
        ranges_str = ", ".join(
            f"{_fmt_mmss(s)}-{_fmt_mmss(e)}" for s, e in missing
        )
        validation.reasons.append(
            f"Missing time range(s) from failed chunk(s): {ranges_str}"
        )


def _validate_gemini_result(result, audio_duration_seconds: float):
    """transcript_validator.validate_transcript, plus a GeminiResult-specific
    check the text-only validator can't see: a chunk that failed after
    retries leaves a hole in the middle of an otherwise full-coverage
    transcript (e.g. chunks 1 and 3 succeed, chunk 2 doesn't) — the trailing
    timestamp still reaches near the end, so the coverage percentage alone
    would pass it. `GeminiResult.missing_time_ranges` (see gemini_processor)
    forces a failure in that case.

    F6 (docs/SPEC-error-path-escalation-2026-07-10.md): when the failure is
    a runaway repetition loop, a tiny glitch (observed: "de," x595 covering
    ~5s of a 24-min recording, source 463) shouldn't cost the whole
    transcript. Sanitize it (collapse the loop to one instance + a glitch
    marker) and re-validate; if the sanitized transcript now passes, mutate
    `result.transcript` in place and accept it as a clean success with
    `sanitized: true` recorded, instead of entering the escalation ladder.
    The duplicate-span check runs as part of that re-validation — i.e.
    AFTER sanitization — because the loop's own shingles trivially recur
    >30 words apart and would otherwise cascade into a second false
    "duplicate span" failure reason for the same root cause.

    Degrades to a permissive pass if transcript_validator is unavailable —
    never blocks the pipeline on a missing optional module.
    """
    if not TRANSCRIPT_VALIDATOR_AVAILABLE:
        return _PermissiveValidation(
            [f"transcript_validator unavailable: {_TRANSCRIPT_VALIDATOR_IMPORT_ERROR}"]
        )
    validation = _validate_transcript(result.transcript, audio_duration_seconds)
    missing = getattr(result, "missing_time_ranges", None) or []
    _apply_missing_ranges(validation, missing)

    if not validation.passed and validation.has_repetition_loop:
        sanitized_text, locations = _sanitize_repetition_loops(result.transcript)
        if locations:
            resanitized = _validate_transcript(sanitized_text, audio_duration_seconds)
            _apply_missing_ranges(resanitized, missing)
            if resanitized.passed:
                result.transcript = sanitized_text
                resanitized.sanitized = True
                resanitized.sanitized_locations = locations
                return resanitized

    return validation


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

        # Verify file still exists and holds a usable amount of audio.
        if not file_path.exists() or file_path.stat().st_size == 0:
            self.logger.warning(f"File no longer exists or is empty: {file_path.name}")
        elif file_path.stat().st_size < MIN_RECORDING_BYTES:
            self.logger.warning(
                f"Skipping corrupt/too-small recording "
                f"({file_path.stat().st_size} bytes < {MIN_RECORDING_BYTES}): {file_path.name}"
            )
        else:
            self.queue.add(file_path)


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
                # Length-based model routing. The flash models intermittently
                # fail long multi-source recordings with "Server disconnected
                # without sending a response" (0 bytes back, all retries +
                # chunk fallback exhausted -> the recording is silently
                # dropped). gemini-2.5-pro is reliable on that same audio, so
                # meetings longer than `long_duration_min` are routed to it;
                # shorter meetings stay on the cheap/fast `model`. Applied
                # per-file in _process_with_gemini.
                self._gemini_model_short = gemini_config.get('model', 'gemini-2.5-flash')
                self._gemini_model_long = gemini_config.get('model_long', 'gemini-2.5-pro')
                self._gemini_long_min = float(gemini_config.get('long_duration_min', 30))
                if self._gemini_model_long != self._gemini_model_short:
                    self.logger.info(
                        f"Model routing enabled: >{self._gemini_long_min:.0f}min -> "
                        f"{self._gemini_model_long}, else {self._gemini_model_short}"
                    )
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

        # F3 periodic rescan state (see RESCAN_* constants above).
        self._rescan_state_path = self.transcripts_dir / RESCAN_STATE_FILENAME
        self._rescan_state = _load_rescan_state(self._rescan_state_path)
        self._last_rescan_monotonic = time.time()

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
                now = time.time()
                if now - self._last_rescan_monotonic >= RESCAN_INTERVAL_SEC:
                    self._last_rescan_monotonic = now
                    try:
                        self._rescan_unprocessed_wavs(now)
                    except Exception as e:
                        self.logger.error(f"Periodic rescan failed: {e}")
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

    def _wav_needs_processing(self, wav_file: Path) -> bool:
        """True if wav_file lacks the output(s) required by processing_mode."""
        if self.processing_mode in ('whisper', 'both'):
            if not (self.transcripts_dir / f"{wav_file.stem}.html").exists():
                return True
        if self.processing_mode in ('gemini', 'both'):
            if not (self.transcripts_dir / f"{wav_file.stem}.json").exists():
                return True
        return False

    def _process_existing_files(self):
        """Check for WAV files that don't have corresponding outputs.

        Runs once at watcher startup (see _rescan_unprocessed_wavs for the
        F3 periodic retry that covers files that fail *after* startup)."""
        for wav_file in self.recordings_dir.glob("*.wav"):
            # Corrupt/empty captures never produce a transcript, so without
            # this they'd be re-queued (and fail-alert) on every startup.
            if wav_file.stat().st_size < MIN_RECORDING_BYTES:
                self.logger.warning(
                    f"Skipping corrupt/too-small recording "
                    f"({wav_file.stat().st_size} bytes): {wav_file.name}"
                )
                continue

            if self._wav_needs_processing(wav_file):
                self.logger.info(f"Found unprocessed file: {wav_file.name}")
                self.queue.add(wav_file)

    def _rescan_attempt_count(self, audio_file: Path) -> int:
        """Number of F3 rescan-triggered (re)queues recorded for this WAV.

        0 means this file has never needed a rescan retry — either it's
        being processed for the first time via the live file-watcher, or it
        hasn't been seen by the rescan mechanism at all. Deliberately
        defensive (getattr) so code paths / tests that construct a
        TranscribeWatcher without going through __init__ (bare stubs) don't
        need to know about rescan state to keep working exactly as before.
        """
        return getattr(self, "_rescan_state", {}).get(audio_file.stem, {}).get("attempts", 0)

    def _rescan_unprocessed_wavs(self, now: float) -> None:
        """Periodic retry for WAVs still lacking output after their first
        attempt (F3, docs/SPEC-error-path-escalation-2026-07-10.md RC3).

        _process_existing_files used to run only at startup, so a meeting
        that failed (and, per F1's junk guard, left no JSON) during a
        Gemini outage sat unprocessed until someone manually restarted the
        watcher. Called every RESCAN_INTERVAL_SEC from the main loop.

        Per-WAV bookkeeping (persisted to a state-file sidecar so a watcher
        restart doesn't reset it):
          - Requeue at most RESCAN_MAX_ATTEMPTS times, spaced at least
            RESCAN_MIN_SPACING_SEC apart (protects against a flapping
            watcher burning through retries via repeated startup scans).
          - Once attempts reach the cap and the file *still* has no output,
            give up permanently (until the state entry is cleared or a JSON
            appears) and fire exactly one "giving up" alert.

        The per-attempt failure/partial Telegram alerts fired from inside
        normal processing (_notify_telegram_failure / _notify_telegram_partial)
        are suppressed for every rescan-triggered attempt (attempts >= 1) —
        only the original live attempt's alert and this method's own
        give-up alert reach Telegram, so an ongoing outage doesn't re-alert
        every 30 minutes.
        """
        state = self._rescan_state
        changed = False
        for wav_file in self.recordings_dir.glob("*.wav"):
            if wav_file.stat().st_size < MIN_RECORDING_BYTES:
                continue
            if not self._wav_needs_processing(wav_file):
                continue

            stem = wav_file.stem
            entry = state.get(stem, {"attempts": 0, "last_attempt_ts": 0.0, "gave_up": False})

            if entry.get("gave_up"):
                continue

            if entry["attempts"] >= RESCAN_MAX_ATTEMPTS:
                entry["gave_up"] = True
                state[stem] = entry
                changed = True
                self.logger.error(
                    f"Rescan: giving up on {wav_file.name} after "
                    f"{entry['attempts']} automatic attempts — manual "
                    f"reprocess needed"
                )
                self._notify_telegram_giveup(wav_file, entry["attempts"])
                continue

            if entry["attempts"] > 0 and (now - entry["last_attempt_ts"]) < RESCAN_MIN_SPACING_SEC:
                continue  # too soon since the last attempt

            entry["attempts"] += 1
            entry["last_attempt_ts"] = now
            state[stem] = entry
            changed = True
            self.logger.info(
                f"Rescan: re-queueing {wav_file.name} "
                f"(attempt {entry['attempts']}/{RESCAN_MAX_ATTEMPTS})"
            )
            self.queue.add(wav_file)

        if changed:
            _save_rescan_state(self._rescan_state_path, state)

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
            # Step 1: JSON output path used downstream (and as the source of
            # truth for calendar timestamp parsing — filename-derived).
            # Built from audio_file's stem directly rather than from the
            # eventual mp3_path: conversion (Step 3 below) now runs AFTER
            # channel VAD because it needs the VAD to pick ducked vs. legacy
            # pre-mix, but nothing before that point needs the converted
            # bytes, only this path.
            json_path = self.transcripts_dir / f"{audio_file.stem}.json"
            if digital_silence(audio_file):
                # No generative model is allowed to invent a meeting from
                # an all-zero capture. Persist the explicit skipped outcome.
                empty = GeminiResult(transcript="", language="unknown",
                                     audio_duration_seconds=get_audio_duration(audio_file))
                empty.speaker_attribution = {
                    "status": "no_speech", "identity_basis": "none",
                    "speaker_dependent_actions": "hold", "missing_stages": [],
                    "accuracy_measured": False,
                }
                data = empty.parsed_response
                data["_meta"]["speech_gate"] = "digital_silence"
                atomic_json(json_path, data)
                save_trial_stage(self.config, audio_file, "candidate", data,
                                 elapsed_seconds=time.time() - start_time)
                self.logger.info("Digital silence: saved no-speech outcome, no extraction")
                return

            # Step 1b: Resolve calendar attendees BEFORE the Gemini call.
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

            # Step 2: Classify source topology from the ORIGINAL WAV. Channel
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
                # Channel VAD from the ORIGINAL 3-channel WAV — must run
                # BEFORE conversion now (Step 3), which needs it to choose
                # the pre-mix. Also gives Gemini a ground-truth host/remote
                # speaking map and feeds post-transcription verification.
                # None for mono/stereo/non-genuine recordings.
                channel_vad = self._compute_channel_vad_safe(audio_file)

            # One admissibility verdict, applied to every consumer of the mic
            # channel: the pre-mix choice below, the Gemini prompt map,
            # pyannote host-cluster fusion and speaker_verify. A bleeding mic
            # must not reach any of them at full, undecked weight.
            channel_separation = (
                channel_vad.separation_report() if channel_vad else None
            )
            channel_admissible = bool(
                channel_separation and channel_separation.get("admissible")
            )

            # Step 2b: Convert WAV to MP3. Ducked pre-mix (attenuates ch0
            # while ch1/ch2 are active) when the mic channel is confirmed
            # bleeding; legacy equal-weight mix otherwise — byte-for-byte
            # unchanged for clean/headphone recordings and non-hybrid
            # layouts. See _convert_for_gemini_routed / convert_for_gemini_ducked.
            self.logger.info("Converting WAV to MP3...")
            mp3_path = self._convert_for_gemini_routed(
                audio_file, channel_vad, channel_admissible
            )
            self.logger.info(f"Converted to: {mp3_path.name} ({mp3_path.stat().st_size / 1024 / 1024:.1f} MB)")

            # Step 2c: Get audio duration
            audio_duration = get_audio_duration(mp3_path)
            self.logger.info(f"Audio duration: {audio_duration / 60:.1f} minutes")

            # Route long recordings to the reliable model (see __init__).
            _routed_model = (
                self._gemini_model_long
                if audio_duration / 60 > self._gemini_long_min
                else self._gemini_model_short
            )
            if _routed_model != self.gemini_processor.model:
                self.logger.info(
                    f"Model routing: {audio_duration / 60:.1f}min -> {_routed_model} "
                    f"(was {self.gemini_processor.model})"
                )
                self.gemini_processor.model = _routed_model

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
                        and channel_admissible
                        and not channel_separation.get("reference_uncertain_intervals")
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
            channel_segments = (
                channel_vad.segments if (channel_vad and channel_admissible)
                else None
            )
            result = self.gemini_processor.process_audio(
                mp3_path, known_attendees=known_attendees or None,
                channel_segments=channel_segments,
                diarization_segments=diarization_segments,
            )

            processing_time = time.time() - start_time
            self.logger.info(f"Gemini processing complete in {processing_time:.1f}s")

            if result.error:
                # F1 (docs/SPEC-error-path-escalation-2026-07-10.md RC1): a
                # hard error result (e.g. "All chunks failed" after a
                # disconnect storm) used to short-circuit here BEFORE the
                # escalation ladder ever ran — the ladder's fresh single-call
                # and force-chunked steps are exactly the moves that would
                # rescue this once the disconnect storm passes or on a
                # genuinely smaller sub-problem. _validate_gemini_result on
                # an error-result already yields passed=False (empty
                # transcript -> no timestamps -> 0% coverage), so it falls
                # straight into the same ladder a validation failure does.
                self.logger.warning(
                    f"Gemini processing error: {result.error} — routing "
                    f"into the validation/escalation ladder instead of "
                    f"giving up immediately"
                )

            # Step 3b: Coverage/completeness gate + escalation ladder (see
            # docs/RELIABILITY_PLAN_2026-07.md Phase 1). Never store a
            # failed-validation transcript as a clean success — this is the
            # fix for the silent-truncation and undeduped-overlap failure
            # modes, neither of which raises `result.error` above.
            result, validation, partial = self._validate_and_escalate(
                audio_file, mp3_path, audio_duration, result,
                known_attendees=known_attendees or None,
                channel_segments=channel_segments,
                diarization_segments=diarization_segments,
            )
            if result is None:
                # F1 junk guard: every ladder step produced no usable
                # transcript at all (coverage 0%, empty) — do NOT write a
                # JSON (that would mark the file "processed" forever).
                # _validate_and_escalate already fired the failure alert;
                # the file stays eligible for F3's periodic rescan.
                return
            result.validation_report = validation.to_dict()
            result.partial = partial
            pre_attribution = copy.deepcopy(result.parsed_response)
            save_trial_stage(self.config, audio_file, "before_attribution",
                             pre_attribution, known_attendees=known_attendees)

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
            save_trial_stage(self.config, audio_file, "before_channel_verify",
                             result.parsed_response, known_attendees=known_attendees)

            # Step 4b2: Channel-based attribution verification. Runs AFTER
            # reconcile so flips target canonical names. Flips turns whose
            # transcript label confidently contradicts the mic-channel
            # ground truth (see speaker_verify module docstring).
            self._verify_speakers_inplace(result, channel_vad, cal_match)
            save_trial_stage(self.config, audio_file, "before_coherence",
                             result.parsed_response, known_attendees=known_attendees)

            # Step 4b3: Semantic speaker-coherence gate. The acoustic stages
            # above can only ask whose microphone was hot — and on a bleeding
            # recording (the common case, see the separation report) they are
            # skipped entirely. This asks whether the labels make SENSE: who
            # answers a question didn't ask it, who says "we" about the client
            # company works there. Runs BEFORE the JSON write, so the repaired
            # transcript is what reaches sources.content_text and every
            # insight, wiki page and prep doc built from it.
            self._coherence_repair_inplace(
                result, audio_file, known_attendees or None, cal_match
            )

            # Provenance: whether attribution was channel-verified at all.
            result.channel_separation = channel_separation
            finalize_attribution(result, pre_attribution)
            save_trial_stage(self.config, audio_file, "candidate",
                             result.parsed_response, known_attendees=known_attendees,
                             elapsed_seconds=time.time() - start_time)

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
            report = vad.separation_report()
            self.logger.info(
                f"Channel VAD: {len(vad.segments)} segments over "
                f"{vad.duration_sec / 60:.1f} min "
                f"(host_bleed_rate={report.get('host_bleed_rate')}, "
                f"admissible={report.get('admissible')})"
            )
            if not report.get("admissible"):
                # 2026-08-07: the mic hears the loudspeakers on most of our
                # recordings, so its host/remote spans are noise dressed as
                # certainty. Everything downstream that would treat them as
                # ground truth is disabled for this recording; semantic
                # coherence repair is the attribution path instead.
                self.logger.warning(
                    f"Channel attribution DISABLED for {audio_file.name}: "
                    f"{report.get('reason')} "
                    f"(host_bleed_rate={report.get('host_bleed_rate')} > "
                    f"{report.get('max_host_bleed_rate')}). The host mic is "
                    f"not a reliable identity signal in this recording. "
                    f"See docs/SPEC-speaker-attribution-2026-08-07.md."
                )
        return vad

    def _convert_for_gemini_routed(
        self, audio_file: Path, channel_vad, channel_admissible: bool
    ) -> Path:
        """Pick the ducked or legacy pre-mix for the Gemini upload.

        Ducked pre-mix (`convert_for_gemini_ducked`) only when a channel VAD
        exists AND it found the mic bleeding (`channel_admissible=False`) —
        that is the one case where the equal-weight mix sends every remote
        utterance to Gemini twice (see that function's docstring). Every
        other case — clean/headphone recordings, mono/stereo fallbacks,
        channel_vad unavailable — keeps today's `convert_for_gemini`
        behaviour byte-for-byte.
        """
        # Unknown isolation is not measured echo. In particular, do not duck
        # the only usable speech channel after a phone handover.
        bleed_rate = channel_vad.host_bleed_rate() if channel_vad else None
        if (channel_vad is not None and not channel_admissible
                and bleed_rate is not None and bleed_rate > 0.35):
            duck_cfg = self.config.get("audio_ducking", {}) or {}
            duck_db = float(duck_cfg.get("duck_db", 18.0))
            self.logger.info(
                f"Bleeding mic detected (host_bleed_rate="
                f"{channel_vad.host_bleed_rate()}) — using ducked pre-mix "
                f"(-{duck_db:.0f}dB on ch0 while remote speaks) instead of "
                f"the equal-weight mix"
            )
            return convert_for_gemini_ducked(
                audio_file, channel_vad,
                output_dir=self.transcripts_dir, duck_db=duck_db,
            )
        self.logger.debug(
            "Using legacy equal-weight pre-mix "
            "(channel admissible or no channel VAD)"
        )
        return convert_for_gemini(audio_file, output_dir=self.transcripts_dir)

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
        # Exact full-name duplicates of the host himself (calendar listing
        # him twice — 2026-08-10: "Matthias Heim, Matthias Heim, Stefan
        # Sieber") are NOT this case: calendar_resolve dedupes those before
        # they reach participant_details, but this stays defensive against
        # any other path (roster merge, counterpart inference) that could
        # still hand us a duplicate.
        for att in (cal_match or {}).get("participant_details") or []:
            name = (att.get("name") or "").strip() if isinstance(att, dict) else ""
            name_lower = name.lower()
            if not name_lower.split() or name_lower.split()[0] != "matthias":
                continue
            if (att.get("role") or "").lower() == "self":
                continue
            if name_lower == _SELF_FULL_NAME.lower():
                continue
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

    def _notify_telegram_coherence_unverified(self, audio_file: Path,
                                              reason: str) -> None:
        """Alert that a transcript was stored with UNVERIFIED speaker labels.

        Not a transcription failure — the meeting is captured — but the
        attribution behind every downstream insight was never checked, so it
        must not pass silently (the whole point of this stage).
        """
        notify_path = self._telegram_notify_script()
        if not notify_path:
            return
        msg = (
            f"⚠️ Speaker attribution UNVERIFIED\n"
            f"{audio_file.name}\n"
            f"Coherence audit did not run: {reason}\n"
            f"Transcript stored; labels may be wrong. Re-run "
            f"tools/speaker_coherence.py over the JSON to repair."
        )
        try:
            subprocess.run(
                [sys.executable, notify_path, "--category", "Meeting", msg],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as e:
            self.logger.warning(f"Telegram coherence alert failed: {e}")

    def _coherence_repair_inplace(self, result, audio_file: Path,
                                  known_attendees: Optional[list],
                                  cal_match: Optional[dict]) -> None:
        """Semantic speaker-attribution audit + repair before anything reads it.

        Mutates the GeminiResult so the JSON write, the InsightBase seed
        (sources.content_text) and every downstream insight see the repaired
        labels. Best-effort by construction: a failed audit leaves the
        transcript exactly as it was and says so in the log — but it says so
        LOUDLY, because an unaudited transcript on a bleeding recording has no
        attribution guarantee at all.
        """
        cfg = self.config.get("speaker_coherence", {}) or {}
        if not cfg.get("enabled", True):
            self.logger.info("speaker_coherence disabled in config; skipping")
            return
        if not SPEAKER_COHERENCE_AVAILABLE:
            self.logger.warning(
                f"speaker_coherence unavailable; speaker attribution NOT "
                f"semantically verified ({_SPEAKER_COHERENCE_IMPORT_ERROR})"
            )
            return
        try:
            claude_cfg = self.config.get("claude_trigger", {}) or {}
            runner = _coherence_cli_runner(
                claude_path=cfg.get("claude_path")
                or claude_cfg.get("claude_path",
                                  "/Users/Matthias/.local/bin/claude"),
                config_dir=(str(expand_path(cfg.get("config_dir")))
                            if cfg.get("config_dir")
                            else (str(expand_path(claude_cfg["config_dir"]))
                                  if claude_cfg.get("config_dir") else None)),
                timeout=int(cfg.get("timeout", 900)),
                model=cfg.get("model"),
            )
            d = result.parsed_response
            log = _coherence_repair(
                d, known_attendees=known_attendees, runner=runner,
                max_parallel=int(cfg.get("max_parallel", 3)),
            )
            result.transcript = d.get("transcript") or result.transcript
            result.participants = d.get("participants") or result.participants
            result.speaker_coherence_log = log
            if cal_match is not None:
                prl = cal_match.setdefault("participant_resolution_log", {})
                prl["speaker_coherence"] = log

            if not log.get("ok") or log.get("failed_windows"):
                # Either nothing was audited, or some windows were not — both
                # leave regions of the transcript with unchecked attribution,
                # and both must surface rather than read as success.
                self.logger.error(
                    f"Speaker coherence incomplete ({log.get('error')}) "
                    f"— attribution for this meeting is unverified"
                )
                self._notify_telegram_coherence_unverified(
                    audio_file, str(log.get("error"))
                )
            elif log.get("refused_runaway"):
                self.logger.error(
                    "Speaker coherence REFUSED a runaway relabel set — "
                    "transcript unchanged, attribution unverified"
                )
            elif log.get("changed"):
                self.logger.info(
                    "Speaker coherence repaired %d line(s), bound %d "
                    "identity/identities; %d region(s) left low-confidence",
                    len(log.get("relabels_applied") or []),
                    len(log.get("identity_bindings") or []),
                    len(log.get("uncertain_regions") or []),
                )
            else:
                self.logger.info(
                    "Speaker coherence: attribution coherent over %d lines",
                    log.get("lines_total", 0),
                )
        except Exception as e:
            self.logger.warning(f"Speaker coherence repair failed: {e}")

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

    @staticmethod
    def _telegram_notify_script() -> Optional[str]:
        """Resolve telegram_notify.py across known locations.

        The canonical copy lives in the Brain repo; an older deploy expected
        ~/.claude/scripts. Hardcoding the latter silently disabled ALL pings
        (captured + failure alerts) once the script moved — checked both now so
        a repo move can't break alerting unnoticed. Optional env override first.
        """
        candidates = [
            os.environ.get("TELEGRAM_NOTIFY_SCRIPT", ""),
            os.path.expanduser("~/Repos/Brain/.claude/scripts/telegram_notify.py"),
            os.path.expanduser("~/.claude/scripts/telegram_notify.py"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def _notify_telegram_meeting_captured(self, source_id: Optional[int],
                                            result, cal_match: Optional[dict],
                                            audio_duration: float) -> None:
        """Send a one-line Telegram ping that the meeting is in InsightBase.

        Fires BEFORE the Claude session — user knows the row exists and the
        transcript is ready to be pulled into a session if needed. Best-effort.
        """
        notify_path = self._telegram_notify_script()
        if not notify_path:
            self.logger.warning(
                "telegram_notify.py not found in any known location; skipping ping"
            )
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
        partial = bool(result and getattr(result, "partial", False))
        msg = (
            f"Meeting captured (#{source_id})\n"
            f"{title}\n"
            f"Client: {company} • {duration_min}min • "
            f"sentiment {sentiment}/{intensity}\n"
            f"{n_emotions} emotion arc events"
            + (" • chunked" if chunked else "")
            + (" • ⚠️ PARTIAL" if partial else "")
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

        F3 alert dedup: suppressed for any rescan-triggered retry (attempt
        count >= 1) — the original live attempt already alerted once, and
        _rescan_unprocessed_wavs fires its own distinct "giving up" alert
        when the attempt cap is reached. Without this an ongoing outage
        would re-alert every ~30min for as long as it lasted.
        """
        if self._rescan_attempt_count(audio_file) >= 1:
            self.logger.info(
                f"Suppressing failure alert for {audio_file.name} "
                f"(rescan retry in progress: {reason})"
            )
            return
        notify_path = self._telegram_notify_script()
        if not notify_path:
            self.logger.warning(
                "telegram_notify.py not found in any known location; "
                "skipping failure alert"
            )
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

    def _validate_and_escalate(self, audio_file: Path, mp3_path: Path,
                                audio_duration: float, result,
                                known_attendees: Optional[list[dict]],
                                channel_segments: Optional[list],
                                diarization_segments: Optional[list]):
        """Coverage/completeness gate + escalation ladder
        (docs/RELIABILITY_PLAN_2026-07.md Phase 1). Never store a
        failed-validation transcript as a clean success:

          1. Validate the Gemini result already in hand (this also runs the
             F6 sanitize-and-revalidate step — see _validate_gemini_result).
          2. On failure, retry a FRESH single-call on pro.
          3. On failure, try drift-proof chunked mode (skipped for
             recordings too short to meaningfully chunk).
          4. If every step still fails validation, accept the best-available
             candidate (highest coverage_pct) with partial=True and fire a
             Telegram alert naming what's missing — never silently. EXCEPT
             (F1 junk guard): if that best candidate has no usable
             transcript at all (0% coverage AND empty), don't store a junk
             JSON that would mark the file "processed" forever — fire the
             plain failure alert instead and return `result=None`, which
             tells the caller to skip the JSON write so F3's periodic
             rescan can retry the file later.

        Returns (result, validation, partial). `result` is the result to
        persist, or None if the junk guard fired (nothing should be
        persisted). `partial` is True iff `result` did not pass validation.
        """
        validation = _validate_gemini_result(result, audio_duration)
        if validation.passed:
            self.logger.info(
                f"Validation passed: coverage {validation.coverage_pct:.1f}%"
            )
            return result, validation, False

        self.logger.warning(
            f"Validation FAILED (coverage {validation.coverage_pct:.1f}%): "
            f"{'; '.join(validation.reasons)}"
        )
        best_result, best_validation = result, validation

        self.logger.info("Escalation: retrying fresh single-call on pro...")
        try:
            retry_result = self.gemini_processor._process_single_shot(
                mp3_path, None, audio_duration,
                known_attendees=known_attendees,
                channel_segments=channel_segments,
                diarization_segments=diarization_segments,
            )
            retry_validation = _validate_gemini_result(retry_result, audio_duration)
            if retry_validation.passed:
                self.logger.info("Escalation: fresh single-call retry passed validation")
                return retry_result, retry_validation, False
            self.logger.warning(
                f"Escalation: fresh single-call retry still failed "
                f"(coverage {retry_validation.coverage_pct:.1f}%): "
                f"{'; '.join(retry_validation.reasons)}"
            )
            if retry_validation.coverage_pct > best_validation.coverage_pct:
                best_result, best_validation = retry_result, retry_validation
        except Exception as e:
            self.logger.warning(f"Escalation: fresh single-call retry raised: {e}")

        if audio_duration > self.gemini_processor.CHUNK_DURATION_SEC:
            self.logger.info("Escalation: trying drift-proof chunked mode...")
            try:
                chunked_result = self.gemini_processor._process_chunked(
                    mp3_path, None, audio_duration,
                    known_attendees=known_attendees,
                    channel_segments=channel_segments,
                    diarization_segments=diarization_segments,
                    force_chunk=True,
                )
                chunked_validation = _validate_gemini_result(chunked_result, audio_duration)
                if chunked_validation.passed:
                    self.logger.info("Escalation: chunked mode passed validation")
                    return chunked_result, chunked_validation, False
                self.logger.warning(
                    f"Escalation: chunked mode still failed "
                    f"(coverage {chunked_validation.coverage_pct:.1f}%): "
                    f"{'; '.join(chunked_validation.reasons)}"
                )
                if chunked_validation.coverage_pct > best_validation.coverage_pct:
                    best_result, best_validation = chunked_result, chunked_validation
            except Exception as e:
                self.logger.warning(f"Escalation: chunked mode raised: {e}")
        else:
            self.logger.info(
                f"Escalation: skipping chunked mode — {audio_duration:.0f}s "
                f"is too short to meaningfully chunk"
            )

        if best_validation.coverage_pct == 0.0 and not (best_result.transcript or "").strip():
            # F1 junk guard: nothing rescued anything -- every step returned
            # an effectively empty transcript (e.g. the original RC1
            # "All chunks failed" error result, with no valid retry either).
            # Storing this as a "partial" JSON would mark the file processed
            # forever, permanently defeating F3's periodic rescan.
            self.logger.error(
                f"Escalation exhausted with NO usable transcript at all — "
                f"not writing a JSON so the periodic rescan can retry: "
                f"{'; '.join(best_validation.reasons)}"
            )
            self._notify_telegram_failure(
                audio_file,
                f"All escalation steps failed with no usable transcript: "
                f"{'; '.join(best_validation.reasons)}",
            )
            return None, best_validation, True

        self.logger.error(
            f"Escalation exhausted — accepting best-available result as "
            f"PARTIAL (coverage {best_validation.coverage_pct:.1f}%): "
            f"{'; '.join(best_validation.reasons)}"
        )
        self._notify_telegram_partial(audio_file, best_validation)
        return best_result, best_validation, True

    def _notify_telegram_partial(self, audio_file: Path, validation) -> None:
        """Alert that a meeting was captured but FAILED the completeness
        gate after every escalation step — stored anyway (partial=true in
        _meta) rather than lost, but needs human review. Best-effort.
        """
        notify_path = self._telegram_notify_script()
        if not notify_path:
            self.logger.warning(
                "telegram_notify.py not found in any known location; "
                "skipping partial-transcript alert"
            )
            return
        msg = (
            f"⚠️ Meeting captured but INCOMPLETE (coverage "
            f"{validation.coverage_pct:.1f}%)\n"
            f"{audio_file.name}\n"
            f"{'; '.join(validation.reasons)}\n"
            f"Stored as partial — needs review/reprocessing."
        )
        try:
            subprocess.run(
                [sys.executable, notify_path, "--category", "Meeting", msg],
                capture_output=True, text=True, timeout=15,
            )
            self.logger.info(f"Telegram partial-transcript alert sent for {audio_file.name}")
        except Exception as e:
            self.logger.warning(f"Telegram partial-transcript alert failed: {e}")

    def _notify_telegram_giveup(self, audio_file: Path, attempts: int) -> None:
        """F3 final alert: the periodic rescan exhausted its attempt cap and
        the file still has no output. Distinct from _notify_telegram_failure
        (suppressed for the quiet intermediate retries in between) so the
        user gets exactly two signals across a whole outage: the first
        failure, then this. Best-effort.
        """
        notify_path = self._telegram_notify_script()
        if not notify_path:
            self.logger.warning(
                "telegram_notify.py not found in any known location; "
                "skipping give-up alert"
            )
            return
        msg = (
            f"🛑 Giving up on transcription after {attempts} automatic attempts\n"
            f"{audio_file.name}\n"
            f"Manual reprocess needed — audio retained."
        )
        try:
            subprocess.run(
                [sys.executable, notify_path, "--category", "Meeting", msg],
                capture_output=True, text=True, timeout=15,
            )
            self.logger.info(f"Telegram give-up alert sent for {audio_file.name}")
        except Exception as e:
            self.logger.warning(f"Telegram give-up alert failed: {e}")

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

        # Quota isolation (2026-07-10 incident). Claude Code rate-limit/quota is
        # keyed by CLAUDE_CONFIG_DIR (Brain memory reference_claude_multiaccount_auth,
        # Option A: each config dir = its own credential + login + quota pool).
        # Without isolation these fire-and-forget headless meeting-actions
        # sessions compete with interactive usage for one shared pool — and on
        # 2026-07-10 that competition starved every session into a
        # "You've hit your session limit" death on its first line, silently
        # dropping all downstream actions (Linear tasks, follow-up emails, …).
        #
        # Give the automation its OWN config dir. Resolution order:
        #   1. claude_trigger.config_dir in config.yaml (preferred)
        #   2. an inherited CLAUDE_CONFIG_DIR (e.g. set in the watcher's plist)
        #   3. none → share the default pool (legacy behaviour), with a warning
        # ONE-TIME setup for the isolated dir (it needs its own login):
        #   CLAUDE_CONFIG_DIR=<dir> claude    # then run /login in that session
        # See tools/CLAUDE_QUOTA_ISOLATION.md.
        config_dir = claude_config.get('config_dir') or env.get('CLAUDE_CONFIG_DIR')
        if config_dir:
            config_dir = str(expand_path(config_dir))
            env["CLAUDE_CONFIG_DIR"] = config_dir
            self.logger.info(
                f"Claude session quota-isolated via CLAUDE_CONFIG_DIR={config_dir}"
            )
        else:
            self.logger.warning(
                "claude_trigger.config_dir not set and no CLAUDE_CONFIG_DIR in env "
                "— headless session shares the default Claude quota pool (no "
                "isolation). A burst of interactive usage can starve it into a "
                "session-limit death. See tools/CLAUDE_QUOTA_ISOLATION.md."
            )

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
            threading.Thread(
                target=self._monitor_claude_startup,
                args=(proc, log_file, source_id),
                daemon=True,
            ).start()
        except Exception as e:
            self.logger.error(f"Failed to trigger Claude: {e}")

    def _monitor_claude_startup(self, proc: subprocess.Popen,
                                 log_file: Path,
                                 source_id: Optional[int]) -> None:
        """Fail-fast feedback for a fire-and-forget headless Claude session.

        Waits up to CLAUDE_STARTUP_FAILFAST_SEC for `proc` to exit. If it
        exits within that window AND its log matches a known
        fatal-startup signature (see FATAL_STARTUP_SIGNATURES), alerts
        with the cause instead of leaving the earlier "running…" Telegram
        ping standing as the only (misleading) signal. A session still
        alive after the window is assumed to have started healthily —
        real /meeting-actions runs take minutes, so this only catches
        immediate deaths, never flags a slow-but-working session.

        Runs in a daemon thread so it can never block watcher shutdown.
        Every step is wrapped: a bug in this monitor must never affect
        transcription itself.
        """
        try:
            try:
                proc.wait(timeout=CLAUDE_STARTUP_FAILFAST_SEC)
            except subprocess.TimeoutExpired:
                return  # still running after the window — assume healthy
            try:
                log_tail = log_file.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                self.logger.warning(
                    f"Claude startup monitor: could not read {log_file} "
                    f"after early exit (PID={proc.pid}): {e}"
                )
                return
            matched = next(
                (sig for sig in FATAL_STARTUP_SIGNATURES if sig in log_tail),
                None,
            )
            if not matched:
                # Died early but not on a signature we recognize as a
                # fatal-startup failure — leave it to the regular logs.
                return
            self.logger.error(
                f"Claude session died within {CLAUDE_STARTUP_FAILFAST_SEC}s "
                f"of starting (PID={proc.pid}, source_id={source_id}, "
                f"exit={proc.returncode}): {matched!r} — {log_file}"
            )
            self._notify_telegram_claude_startup_failure(source_id, matched, log_file)
        except Exception as e:
            # Monitoring must never break transcription.
            self.logger.warning(f"Claude startup monitor failed: {e}")

    def _notify_telegram_claude_startup_failure(self, source_id: Optional[int],
                                                 cause: str, log_file: Path) -> None:
        """Alert that a headless Claude session died at startup.

        Distinct from _notify_telegram_meeting_captured, whose "Claude
        /meeting-actions running…" ping fires when the trigger STARTS —
        this fires when the child actually DIES within the fail-fast
        window, naming the cause (OAuth vs quota vs trust) so the fix is
        obvious from the phone instead of a silent multi-day extraction
        gap. Best-effort.
        """
        notify_path = self._telegram_notify_script()
        if not notify_path:
            self.logger.warning(
                "telegram_notify.py not found in any known location; "
                "skipping Claude startup-failure alert"
            )
            return
        msg = (
            f"🛑 Claude /meeting-actions died at startup (#{source_id})\n"
            f"Cause: {cause}\n"
            f"Log: {log_file}\n"
            f"Insights NOT extracted — re-trigger after fixing auth/quota."
        )
        try:
            subprocess.run(
                [sys.executable, notify_path, "--category", "Meeting", msg],
                capture_output=True, text=True, timeout=15,
            )
            self.logger.info(
                f"Telegram Claude-startup-failure alert sent for source_id={source_id}"
            )
        except Exception as e:
            self.logger.warning(f"Telegram Claude-startup-failure alert failed: {e}")


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
