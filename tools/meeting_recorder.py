#!/usr/bin/env python3
"""
MeetingRecorder - macOS menu bar app for recording meetings

A simple menu bar app that records audio from your microphone (or combined
mic + system audio via BlackHole) and saves it for automatic transcription.

Usage:
    python meeting_recorder.py [--config PATH]
"""

import os
import queue
import sys
import time
import shutil
import threading
import subprocess
from collections import deque
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import yaml
import numpy as np
import sounddevice as sd
import soundfile as sf
import rumps

from capture_provenance import archive_capture, merge_filter
from mic_writer import (
    MIC_CHECKPOINT_SECONDS,
    MIC_QUEUE_SECONDS,
    MIC_RING_SECONDS,
    DiskWriteSuppressed,
    IncrementalWavWriter,
    SampleQueue,
    recover_wav,
)
from sidecar.gate import amend_recording_sidecar, initialise_recording_sidecar
from sidecar.recorder_support import (
    CalendarRecordingContext,
    append_prompt_mark,
    calendar_context_from_resolution,
    jev_preference_enabled,
    recent_transcript_title,
    sidecar_metadata_from_calendar,
    write_jev_preference,
)
from sidecar.prompts import prompt_choice_label
from sidecar.service import clip_for_stem, prompts_for_stem
from sidecar.transcript import format_timestamp, recent_transcripts

# launchd does not pass GEMINI_API_KEY to the menu-bar application.  Mirror the
# watcher: load the repository .env if python-dotenv is available, without ever
# logging the key or its value.
try:
    from dotenv import load_dotenv

    _recorder_env_path = Path(__file__).parent.parent / ".env"
    if _recorder_env_path.exists():
        load_dotenv(_recorder_env_path)
except ImportError:
    pass


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"

# Menu bar icons (using emoji as fallback)
ICON_IDLE = None  # Will use title instead
ICON_RECORDING = None
TITLE_IDLE = "🎙️"
TITLE_RECORDING = "🔴"

# This is observed globally through AppKit, plus locally while this menu-bar
# app is active. The Ctrl modifier keeps it distinct from ordinary Cmd-P.
PROMPT_MARK_SHORTCUT = "⌃⌥⌘P"


def _appkit_constant(appkit, modern_name: str, legacy_name: str, fallback: int) -> int:
    """Read a PyObjC constant across its old/new spelling variants."""
    return int(getattr(appkit, modern_name, getattr(appkit, legacy_name, fallback)))


def is_prompt_mark_shortcut(event, appkit) -> bool:
    """Return whether an AppKit key event is the global prompt-mark shortcut."""
    try:
        is_repeat = event.isARepeat() if callable(event.isARepeat) else event.isARepeat
        if is_repeat:
            return False
        characters = event.charactersIgnoringModifiers()
        flags = int(event.modifierFlags())
    except (AttributeError, TypeError, ValueError):
        return False
    if not isinstance(characters, str) or characters.casefold() != "p":
        return False
    required = (
        _appkit_constant(appkit, "NSEventModifierFlagControl", "NSControlKeyMask", 1 << 18)
        | _appkit_constant(appkit, "NSEventModifierFlagOption", "NSAlternateKeyMask", 1 << 19)
        | _appkit_constant(appkit, "NSEventModifierFlagCommand", "NSCommandKeyMask", 1 << 20)
    )
    return flags & required == required


def activate_app_for_modal(appkit=None) -> None:
    """Bring MeetingRecorder forward so a dialog is not stuck behind other apps.

    A modal that opens behind the frontmost window blocks the menu, including
    Stop, until it is found. Tests pass ``appkit``; under pytest the real
    activation is skipped unless a double is injected.
    """
    if appkit is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        if appkit is None:
            import AppKit

            appkit = AppKit
        app = appkit.NSApp
        if app is None:
            app = appkit.NSApplication.sharedApplication()
        options = int(getattr(appkit, "NSApplicationActivateIgnoringOtherApps", 2))
        if hasattr(app, "activateWithOptions_"):
            app.activateWithOptions_(options)
        elif hasattr(app, "activateIgnoringOtherApps_"):
            app.activateIgnoringOtherApps_(True)
    except Exception as exc:
        print(f"Could not activate MeetingRecorder before a dialog: {exc}", file=sys.stderr)


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        # Return defaults if config doesn't exist
        return {
            'audio': {
                'device': 'default',
                'sample_rate': 16000,
                'channels': 1
            },
            'paths': {
                'recordings': '~/Documents/MeetingRecorder/Recordings',
                'transcripts': '~/Documents/MeetingRecorder/Transcripts',
                'logs': '~/Documents/MeetingRecorder/logs'
            }
        }

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_audio_device_index(device_name: str) -> Optional[int]:
    """Get device index by name, or None for default."""
    if device_name.lower() == 'default':
        return None

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if device_name.lower() in dev['name'].lower():
            return i

    return None


def _mono_int16(block) -> np.ndarray:
    """One channel of int16 samples from a callback block or a recovered array."""
    array = np.asarray(block)
    if array.size == 0:
        return np.zeros(0, dtype=np.int16)
    if array.ndim == 2:
        array = array[:, 0]
    return np.ascontiguousarray(array, dtype=np.int16).reshape(-1)


_ORPHAN_MIC_SUFFIX = ".mic.wav"
_ORPHAN_SYS_SUFFIX = ".sys.wav"


def recover_orphaned_captures(
    tmp_dir: Path,
    recordings_dir: Path,
    *,
    notify=None,
) -> list[Path]:
    """Move crashed mic/system WAVs from ``.tmp`` into ``Recordings``.

    An orphan is deleted only after a playable file has been written. A file
    that cannot be read stays where it is. ``notify`` is a non-modal callback
    ``(path) -> None``; it must not open a dialog.
    """
    tmp_dir = Path(tmp_dir)
    recordings_dir = Path(recordings_dir)
    if not tmp_dir.is_dir():
        return []
    recordings_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(tmp_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(_ORPHAN_MIC_SUFFIX):
            stem = name[: -len(_ORPHAN_MIC_SUFFIX)]
            kind = "mic"
        elif name.endswith(_ORPHAN_SYS_SUFFIX):
            stem = name[: -len(_ORPHAN_SYS_SUFFIX)]
            kind = "sys"
        else:
            continue
        if not stem or stem in {".", ".."} or "/" in stem or stem.startswith("."):
            continue
        grouped.setdefault(stem, {})[kind] = path
    recovered: list[Path] = []
    for stem, files in sorted(grouped.items()):
        dest = _recover_one_orphan(stem, files, tmp_dir, recordings_dir)
        if dest is None:
            continue
        recovered.append(dest)
        if notify is not None:
            try:
                notify(dest)
            except Exception as exc:
                print(f"Recovery notification failed: {exc}", file=sys.stderr)
    return recovered


def _recover_one_orphan(
    stem: str,
    files: dict[str, Path],
    tmp_dir: Path,
    recordings_dir: Path,
) -> Optional[Path]:
    loaded: dict[str, tuple[np.ndarray, int]] = {}
    for kind, path in files.items():
        try:
            pcm, rate = recover_wav(path)
        except (OSError, ValueError):
            continue
        if rate <= 0 or pcm.size == 0:
            continue
        loaded[kind] = (pcm, int(rate))
    if not loaded:
        return None
    dest = _recovered_destination(recordings_dir, stem)
    partial = dest.with_name(f".{dest.stem}.partial.wav")
    produced = False
    consumed: list[Path] = []
    try:
        if "mic" in loaded and "sys" in loaded:
            produced = _merge_recovered_pair(
                stem, loaded["mic"], loaded["sys"], tmp_dir, partial
            )
            if produced:
                consumed = [files["mic"], files["sys"]]
        if not produced and "mic" in loaded:
            pcm, rate = loaded["mic"]
            sf.write(str(partial), pcm, rate, subtype="PCM_16", format="WAV")
            produced = partial.exists() and partial.stat().st_size > 44
            if produced:
                consumed = [files["mic"]]
        if not produced and "sys" in loaded:
            pcm, rate = loaded["sys"]
            sf.write(str(partial), pcm, rate, subtype="PCM_16", format="WAV")
            produced = partial.exists() and partial.stat().st_size > 44
            if produced:
                consumed = [files["sys"]]
        if not produced:
            return None
        os.replace(partial, dest)
    except Exception as exc:
        print(f"Could not recover {stem}: {exc}", file=sys.stderr)
        return None
    finally:
        if partial.exists():
            partial.unlink()
    if not dest.exists() or dest.stat().st_size <= 44:
        return None
    for path in consumed:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            print(f"Recovered {dest.name} but could not remove {path.name}: {exc}", file=sys.stderr)
    return dest


def _recovered_destination(recordings_dir: Path, stem: str) -> Path:
    plain = recordings_dir / f"{stem}.wav"
    if not plain.exists():
        return plain
    recovered = recordings_dir / f"{stem}-recovered.wav"
    if not recovered.exists():
        return recovered
    return recordings_dir / f"{stem}-recovered-{time.time_ns()}.wav"


def _merge_recovered_pair(
    stem: str,
    mic: tuple[np.ndarray, int],
    system: tuple[np.ndarray, int],
    tmp_dir: Path,
    dest: Path,
) -> bool:
    """Merge recovered mic and system PCM. False leaves the caller on the mic fallback."""
    mic_pcm, mic_rate = mic
    sys_pcm, sys_rate = system
    if mic_rate != sys_rate or mic_rate <= 0:
        return False
    mic_path = tmp_dir / f".{stem}.recover-mic.wav"
    sys_path = tmp_dir / f".{stem}.recover-sys.wav"
    try:
        sf.write(str(mic_path), mic_pcm, mic_rate, subtype="PCM_16", format="WAV")
        sf.write(str(sys_path), sys_pcm, sys_rate, subtype="PCM_16", format="WAV")
        duration = max(mic_pcm.shape[0], sys_pcm.shape[0]) / float(mic_rate)
        result = subprocess.run(
            [AudioRecorder._FFMPEG, "-y",
             "-i", str(mic_path), "-i", str(sys_path),
             "-filter_complex", merge_filter(duration),
             "-map", "[a]", "-c:a", "pcm_s16le", str(dest)],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and dest.exists() and dest.stat().st_size > 44
    except Exception:
        return False
    finally:
        for path in (mic_path, sys_path):
            if path.exists():
                path.unlink()


class AudioRecorder:
    """Handles audio recording in a background thread."""

    # Hybrid capture: the microphone is recorded in-process via sounddevice
    # (this Python process holds the macOS Microphone permission), while system
    # audio is captured by the signed Core Audio tap bundle (which holds the
    # System-Audio-Recording permission). The two are merged at stop into one
    # 3-channel WAV: ch0=mic (host), ch1/ch2=system (remote participants).
    #
    # Why hybrid: a single tap+mic aggregate would need BOTH TCC grants on the
    # tap bundle, but a background/ad-hoc bundle can't surface the microphone
    # prompt. Splitting capture across the two processes that already hold each
    # permission sidesteps that entirely, and IS the Phase-5 per-channel layout.
    _PROC_PATTERN = "AudioTapRecorder.app/Contents/MacOS/audio_tap_recorder"
    _FFMPEG = "/opt/homebrew/bin/ffmpeg"

    def __init__(self, config: dict):
        self.config = config
        audio_cfg = config.get('audio', {})
        # Mic: default input device (the microphone), mono, 48 kHz. NOT the old
        # Aggregate Device (whose BlackHole channels were always silent).
        self.sample_rate = int(audio_cfg.get('mic_sample_rate', 48000))
        self.mic_device = None  # None => system default input = the mic
        # Signed system-audio tap bundle (system-only mode).
        self.tap_bundle = expand_path(audio_cfg.get(
            'tap_bundle', '~/Documents/MeetingRecorder/bin/AudioTapRecorder.app'))
        # Intermediates live OUTSIDE the watched Recordings dir so the watcher
        # never picks up a half-written or duplicate file.
        self.tmp_dir = expand_path('~/Documents/MeetingRecorder/.tmp')
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.recording = False
        self.audio_data = []
        self.stream: Optional[sd.InputStream] = None
        self.output_file: Optional[Path] = None
        self._sys_wav: Optional[Path] = None
        self._mic_wav: Optional[Path] = None
        self._sys_active = False
        self._capture_meta = {}
        self._mic_frames = 0
        self._previous_adc_end = None
        self._next_timing_sample = 0
        self._mic_first_sample_monotonic: float | None = None
        self._mic_first_sample_event = threading.Event()
        # The callback keeps only this much mic audio for the live tail.
        # The full track is the incremental WAV written off the callback.
        self._mic_ring_seconds = float(audio_cfg.get("mic_ring_seconds", MIC_RING_SECONDS))
        self._mic_checkpoint_seconds = float(
            audio_cfg.get("mic_checkpoint_seconds", MIC_CHECKPOINT_SECONDS)
        )
        self._mic_queue_seconds = float(audio_cfg.get("mic_queue_seconds", MIC_QUEUE_SECONDS))
        self._mic_queue = None
        self._mic_writer = None
        self._mic_writer_thread = None
        self._mic_writer_stop = threading.Event()
        self._mic_writer_abandon = threading.Event()
        self._mic_disk_capture = False
        self._mic_dropped_blocks = 0
        self._ring_samples = 0
        self._mic_io_lock = threading.Lock()
        self._mic_ram_fallback = False
        self._mic_writer_failed = False
        self._mic_ram_remainder: list = []
        self._mic_inflight = None
        self._mic_inflight_mark = 0
        self._mic_block_kept = False
        self._mic_writer_close_when_done = False
        self._mic_writer_join_timeout = 30.0
        self._mic_writer_inflight_join_timeout = 5.0

    def _sys_proc_running(self) -> bool:
        return subprocess.run(
            ["pgrep", "-f", self._PROC_PATTERN], capture_output=True
        ).returncode == 0

    def start(self, output_file: Path):
        """Start mic (sounddevice) + system-audio tap capture simultaneously."""
        if self.recording:
            return False

        self.output_file = output_file
        self.audio_data = []
        self._ring_samples = 0
        self._mic_dropped_blocks = 0
        self._mic_frames = 0
        self._previous_adc_end = None
        self._next_timing_sample = 0
        self._mic_first_sample_monotonic = None
        self._mic_first_sample_event.clear()
        self._mic_writer_stop.clear()
        self._mic_writer_abandon.clear()
        self._mic_disk_capture = False
        self._mic_ram_fallback = False
        self._mic_writer_failed = False
        self._mic_ram_remainder = []
        self._mic_inflight = None
        self._mic_inflight_mark = 0
        self._mic_block_kept = False
        self._mic_writer_close_when_done = False
        self._mic_queue = None
        self._mic_writer = None
        self._capture_meta = {"schema_version": 1, "started_wall_time": time.time(),
                              "started_monotonic": time.monotonic(),
                              "mic_sample_rate": self.sample_rate,
                              "mic_timing_samples": [], "discontinuities": []}
        try:
            self._capture_meta["mic_device"] = dict(sd.query_devices(kind="input"))
            self._capture_meta["output_device"] = dict(sd.query_devices(kind="output"))
        except Exception as e:
            self._capture_meta["device_query_error"] = str(e)
        stem = output_file.stem
        self._sys_wav = self.tmp_dir / f"{stem}.sys.wav"
        self._mic_wav = self.tmp_dir / f"{stem}.mic.wav"
        for p in (self._sys_wav, self._mic_wav):
            if p.exists():
                p.unlink()

        # 1) System-audio tap first (it has launch latency). Best-effort: if it
        #    fails we still record the mic, never losing the meeting.
        self._sys_active = False
        if self.tap_bundle.exists():
            try:
                subprocess.run(
                    ["open", str(self.tap_bundle), "--args",
                     str(self._sys_wav), "--system-only"],
                    check=True,
                )
                self._sys_active = True
            except Exception as e:
                print(f"System-audio tap launch failed: {e}", file=sys.stderr)
        else:
            print(f"Tap bundle not found ({self.tap_bundle}); mic-only.", file=sys.stderr)

        # 2) Microphone via sounddevice (this process holds the mic permission).
        # The writer is open before the stream so the first callback can enqueue.
        self._start_mic_writer()
        self.recording = True
        try:
            self.stream = sd.InputStream(
                device=self.mic_device,
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.int16,
                callback=self._audio_callback,
            )
            self.stream.start()
            return True
        except Exception as e:
            self.recording = False
            self._mic_writer_abandon.set()
            self._finish_mic_writer()
            if self._sys_active:
                subprocess.run(["pkill", "-INT", "-f", self._PROC_PATTERN])
            raise RuntimeError(f"Failed to start mic recording: {e}")

    def stop(self) -> Optional[Path]:
        """Stop both captures, merge into one 3-channel WAV (mic + system)."""
        if not self.recording:
            return None
        self.recording = False

        # Stop mic. A writer failure must not skip the tap, the merge, or the archive.
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as exc:
            print(f"Mic stream stop failed: {exc}", file=sys.stderr)
        try:
            self._finish_mic_writer()
        except Exception as exc:
            print(f"Mic writer finish failed: {exc}", file=sys.stderr)
        try:
            self._materialize_mic_track()
        except Exception as exc:
            print(f"Mic track rebuild failed: {exc}", file=sys.stderr)
        mic_ok = False
        if (
            self._mic_disk_capture
            and self._mic_wav is not None
            and self._mic_wav.exists()
            and self._mic_wav.stat().st_size > 1000
        ):
            mic_ok = True
        elif self.audio_data:
            # Writer never started (tests, or a disk-open failure). The ring
            # still holds the capture, which is the previous stop() behaviour.
            arr = np.concatenate(self.audio_data, axis=0)
            sf.write(str(self._mic_wav), arr, self.sample_rate, subtype='PCM_16')
            mic_ok = self._mic_wav.exists() and self._mic_wav.stat().st_size > 1000

        # Stop system tap via SIGINT (graceful teardown => valid WAV + tap freed).
        if self._sys_active:
            try:
                subprocess.run(["pkill", "-INT", "-f", self._PROC_PATTERN])
                for _ in range(60):  # up to ~6s for clean teardown
                    if not self._sys_proc_running():
                        break
                    time.sleep(0.1)
                time.sleep(0.3)
            except Exception as exc:
                print(f"System-audio tap stop failed: {exc}", file=sys.stderr)
        sys_ok = (self._sys_wav.exists() and self._sys_wav.stat().st_size > 1000)

        # Merge into a temp file, then atomically move into the watched dir so
        # the watcher never sees a partial file. Fallbacks guarantee we keep
        # whatever audio we captured.
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_tmp = self.tmp_dir / f"{self.output_file.stem}.final.wav"
        if merged_tmp.exists():
            merged_tmp.unlink()

        produced = None
        if mic_ok and sys_ok:
            try:
                merge_duration = max(sf.info(str(self._mic_wav)).duration,
                                     sf.info(str(self._sys_wav)).duration)
            except (OSError, RuntimeError) as e:
                # A stale/crashed tap can leave an unfinished WAV header.
                # Keep the mic fallback reachable instead of losing stop().
                print(f"Cannot read system duration ({e}); falling back to mic.", file=sys.stderr)
                sys_ok = False
        if mic_ok and sys_ok:
            # Merge mic (1ch) + system (2ch) into a 3-channel WAV with a FIXED
            # physical channel order: ch0=mic(host), ch1=sysL, ch2=sysR.
            # NOTE: bare `amerge` of a mono+stereo pair reorders by channel
            # label (mono FC sorts last → mic ends up on ch2), which silently
            # swaps host/remote downstream. amerge deterministically yields
            # [sysL, sysR, mic]; the pan then puts them back as [mic, sysL, sysR].
            # Verified with synthetic tones (440=mic, 200=sysL, 800=sysR).
            r = subprocess.run(
                [self._FFMPEG, "-y",
                 "-i", str(self._mic_wav), "-i", str(self._sys_wav),
                 "-filter_complex",
                 merge_filter(merge_duration),
                 "-map", "[a]", "-c:a", "pcm_s16le", str(merged_tmp)],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and merged_tmp.exists():
                produced = merged_tmp
            else:
                print(f"Merge failed ({r.stderr[:200]}); falling back to mic.", file=sys.stderr)
        if produced is None:
            # Fallback order: mic (host voice is most important) > system > none.
            src = self._mic_wav if mic_ok else (self._sys_wav if sys_ok else None)
            if src is not None:
                shutil.copy(str(src), str(merged_tmp))
                produced = merged_tmp

        result = None
        if produced is not None:
            os.replace(str(produced), str(self.output_file))  # atomic move into Recordings/
            result = self.output_file

        # Original streams are the only way to evaluate offset/drift and
        # missing reference after capture. A failed archive leaves .tmp files
        # intact; it must never delete the only unmerged evidence.
        self._capture_meta.update(stopped_wall_time=time.time(),
                                  mic_frames=self._mic_frames,
                                  mic_ok=mic_ok, system_ok=sys_ok,
                                  output_file=str(result) if result else None)
        try:
            archive_capture(expand_path(self.config.get("audio", {}).get(
                "archive_dir", "~/Documents/MeetingRecorder/CaptureArchive")),
                self.output_file.stem, [self._mic_wav, self._sys_wav],
                self._capture_meta)
        except Exception as e:
            print(f"Capture archive failed; originals retained in .tmp or CaptureArchive: {e}", file=sys.stderr)

        # Closed older captures can be compressed transparently off the GUI
        # thread. Keep this after the trial too: raw-track retention must not
        # depend on a one-week monitor to keep disk usage under control.
        if self.config.get("audio", {}).get("compress_archives", True):
            try:
                log_dir = expand_path(self.config.get("paths", {}).get(
                    "logs", "~/Documents/MeetingRecorder/logs"))
                log_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / "capture-housekeeping.log").open("a") as log:
                    subprocess.Popen([sys.executable, str(Path(__file__).with_name("compress_capture.py")),
                                      "--max-files", "10"], stdout=log, stderr=log,
                                     start_new_session=True)
                    # Raw WAVs whose MP3 + transcript already exist are pruned
                    # after audio.wav_retention_days (default 7). The daily
                    # launchd job com.user.prunerecordings runs the same script.
                    retention_days = self.config.get("audio", {}).get("wav_retention_days", 7)
                    subprocess.Popen([sys.executable, str(Path(__file__).with_name("prune_recordings.py")),
                                      "--days", str(retention_days)], stdout=log, stderr=log,
                                     start_new_session=True)
            except Exception as e:
                print(f"Capture housekeeping could not start: {e}", file=sys.stderr)

        return result

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for the microphone input stream.

        Copies the block into a bounded queue and a short ring. Disk I/O
        stays on the writer thread. A full queue keeps the block in RAM.
        """
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        if self.recording:
            # The callback must remain I/O-free.  The menu app's sidecar worker
            # waits on this event and persists the offset away from audio.
            if self._mic_first_sample_monotonic is None:
                self._mic_first_sample_monotonic = time.monotonic()
                self._mic_first_sample_event.set()
            block = indata.copy()
            self._enqueue_mic_block(block)
            self._keep_recent_block(block)
            adc = float(time_info.inputBufferAdcTime)
            gap = None if self._previous_adc_end is None else adc - self._previous_adc_end
            if status or (gap is not None and abs(gap) > 0.005):
                self._capture_meta["discontinuities"].append({
                    "frame": self._mic_frames, "adc_gap_seconds": gap, "status": str(status)})
            if self._mic_frames >= self._next_timing_sample:
                self._capture_meta["mic_timing_samples"].append({
                    "frame": self._mic_frames, "adc_time": adc,
                    "current_time": float(time_info.currentTime),
                    "callback_monotonic": time.monotonic()})
                self._next_timing_sample = self._mic_frames + self.sample_rate
            self._previous_adc_end = adc + frames / self.sample_rate
            self._mic_frames += frames

    def _keep_recent_block(self, block) -> None:
        """Ring for the live tail. In memory only; the WAV writer owns the file."""
        self.audio_data.append(block)
        if not getattr(self, "_mic_disk_capture", False):
            return
        count = int(np.asarray(block).shape[0])
        self._ring_samples += count
        limit = int(self.sample_rate * self._mic_ring_seconds)
        while len(self.audio_data) > 1 and self._ring_samples > limit:
            removed = self.audio_data.popleft()
            self._ring_samples -= int(np.asarray(removed).shape[0])

    def _enqueue_mic_block(self, block) -> None:
        """Queue one callback block, or keep it in RAM when the disk path cannot."""
        mic_queue = getattr(self, "_mic_queue", None)
        if mic_queue is None:
            return
        with self._mic_io_lock:
            if self._mic_ram_fallback or self._mic_writer_failed:
                if self._mic_ram_fallback and not self._mic_writer_failed:
                    self._note_queue_full(block)
                self._mic_ram_remainder.append(block)
                return
            try:
                mic_queue.put_nowait(block)
            except queue.Full:
                self._mic_ram_fallback = True
                self._note_queue_full(block)
                self._mic_ram_remainder.append(block)

    def _note_queue_full(self, block) -> None:
        """Record one merged range for a run of queue overflows. The samples are kept."""
        samples = int(np.asarray(block).shape[0])
        self._mic_dropped_blocks = getattr(self, "_mic_dropped_blocks", 0) + 1
        meta = getattr(self, "_capture_meta", None)
        discontinuities = meta.get("discontinuities") if isinstance(meta, dict) else None
        if not isinstance(discontinuities, list):
            return
        frame = int(getattr(self, "_mic_frames", 0) or 0)
        if discontinuities:
            last = discontinuities[-1]
            if (
                isinstance(last, dict)
                and last.get("reason") == "mic_queue_full"
                and int(last.get("end_frame", -1)) == frame
            ):
                last["end_frame"] = frame + samples
                last["samples"] = int(last.get("samples", 0)) + samples
                return
        discontinuities.append({
            "start_frame": frame,
            "end_frame": frame + samples,
            "samples": samples,
            "reason": "mic_queue_full",
            "kept": "ram",
        })

    def _start_mic_writer(self) -> None:
        """Open the incremental mic WAV and start the thread that fills it."""
        try:
            mic_queue = SampleQueue(max(1, int(self.sample_rate * self._mic_queue_seconds)))
            writer = IncrementalWavWriter(
                self._mic_wav,
                self.sample_rate,
                channels=1,
                checkpoint_seconds=self._mic_checkpoint_seconds,
            )
        except Exception as exc:
            print(
                f"Durable mic writer unavailable; mic stays in memory until stop: {exc}",
                file=sys.stderr,
            )
            self._mic_queue = None
            self._mic_writer = None
            self._mic_disk_capture = False
            return
        self._mic_queue = mic_queue
        self._mic_writer = writer
        self._mic_disk_capture = True
        # popleft stays off the callback's critical path. The live snapshot
        # only needs to copy the references.
        self.audio_data = deque()
        self._mic_writer_thread = threading.Thread(
            target=self._mic_writer_loop, name="meeting-mic-writer", daemon=True
        )
        self._mic_writer_thread.start()

    def _mic_writer_loop(self) -> None:
        writer = self._mic_writer
        try:
            self._mic_writer_body(writer)
        finally:
            if writer is None:
                return
            try:
                if self._mic_writer_abandon.is_set():
                    writer.abort()
                elif self._mic_writer_close_when_done:
                    writer.close()
            except Exception as exc:
                print(f"Mic writer cleanup failed: {exc}", file=sys.stderr)

    def _mic_writer_body(self, writer) -> None:
        mic_queue = self._mic_queue
        if writer is None or mic_queue is None:
            return
        while True:
            if self._mic_writer_abandon.is_set():
                return
            try:
                block = mic_queue.get(0.05)
            except queue.Empty:
                if self._mic_writer_stop.is_set() or self._mic_ram_fallback:
                    break
                continue
            if self._mic_writer_abandon.is_set():
                self._spill_block_locked(block)
                return
            if not self._write_queued_block(writer, block):
                return
        if self._mic_writer_abandon.is_set() or self._mic_writer_failed:
            return
        while True:
            try:
                block = mic_queue.get(0)
            except queue.Empty:
                break
            if self._mic_writer_abandon.is_set():
                self._spill_block_locked(block)
                return
            if not self._write_queued_block(writer, block):
                return
        try:
            writer.flush()
        except DiskWriteSuppressed:
            return
        except Exception as exc:
            print(f"Mic checkpoint failed; keeping the rest in memory: {exc}", file=sys.stderr)
            self._fail_disk_writer(writer, None)

    def _write_queued_block(self, writer, block) -> bool:
        """Write one queued block. False means the disk path has stopped."""
        with self._mic_io_lock:
            if self._mic_writer_failed:
                self._mic_ram_remainder.append(block)
                return True
            self._mic_inflight = block
            self._mic_inflight_mark = int(writer.samples_written)
            self._mic_block_kept = False
        try:
            writer.write(block)
            return True
        except DiskWriteSuppressed as exc:
            self._keep_suppressed_block(writer, block, exc.consumed)
            with self._mic_io_lock:
                self._mic_ram_fallback = True
                self._append_queued_blocks_locked()
            return False
        except Exception as exc:
            print(f"Mic writer failed; keeping the rest in memory: {exc}", file=sys.stderr)
            self._fail_disk_writer(writer, block)
            return False
        finally:
            with self._mic_io_lock:
                self._mic_inflight = None

    def _keep_suppressed_block(self, writer, block, consumed: bool) -> None:
        with self._mic_io_lock:
            pending = writer.take_pending()
            inflight_n = int(np.asarray(block).shape[0]) if block is not None else 0
            if pending is not None and len(pending):
                if consumed and self._mic_block_kept and len(pending) >= inflight_n:
                    older = pending[:-inflight_n]
                    if len(older):
                        self._mic_ram_remainder.insert(0, older)
                elif not (consumed and self._mic_block_kept):
                    self._mic_ram_remainder.insert(0, pending)
            elif not self._mic_block_kept and block is not None:
                self._mic_ram_remainder.insert(0, block)
            self._mic_block_kept = False

    def _spill_block_locked(self, block) -> None:
        with self._mic_io_lock:
            self._mic_ram_remainder.append(block)

    def _fail_disk_writer(self, writer, block) -> None:
        """Keep every sample the file does not already contain, then stop writing."""
        with self._mic_io_lock:
            self._mic_ram_fallback = True
            self._mic_writer_failed = True
            older: list = []
            if writer is not None:
                writer.suppress_disk = True
                pending = writer.take_pending()
                if pending is not None and len(pending):
                    older.append(pending)
                elif block is not None and int(writer.samples_written) == int(self._mic_inflight_mark):
                    older.append(block)
            elif block is not None:
                older.append(block)
            queued = self._take_queued_blocks_locked()
            self._mic_ram_remainder = older + queued + self._mic_ram_remainder

    def _take_queued_blocks_locked(self) -> list:
        mic_queue = self._mic_queue
        if mic_queue is None:
            return []
        drained = []
        while True:
            try:
                drained.append(mic_queue.get(0))
            except queue.Empty:
                break
        return drained

    def _append_queued_blocks_locked(self) -> None:
        queued = self._take_queued_blocks_locked()
        if queued:
            self._mic_ram_remainder.extend(queued)

    def _handoff_mic_queue_to_ram(self) -> None:
        """Move blocks the writer has not started onto the RAM remainder."""
        with self._mic_io_lock:
            self._mic_ram_fallback = True
            drained = []
            mic_queue = self._mic_queue
            if mic_queue is not None:
                while True:
                    try:
                        drained.append(mic_queue.get(0))
                    except queue.Empty:
                        break
            if drained:
                self._mic_ram_remainder = drained + self._mic_ram_remainder

    def _steal_uncommitted_inflight(self) -> None:
        """Copy the block stuck inside write() when it is not in the file yet."""
        with self._mic_io_lock:
            writer = self._mic_writer
            block = self._mic_inflight
            if writer is None or block is None:
                return
            if int(writer.samples_written) != int(self._mic_inflight_mark):
                return
            self._mic_ram_remainder.insert(0, block)
            self._mic_block_kept = True

    def _finish_mic_writer(self) -> None:
        """Drain the queue and close the WAV. Never raises."""
        try:
            self._finish_mic_writer_impl()
        except Exception as exc:
            print(f"Mic writer finish failed: {exc}", file=sys.stderr)

    def _finish_mic_writer_impl(self) -> None:
        """Drain the queue and close the WAV.

        The file handle is closed only after the writer thread has exited.
        A thread that is still inside write() keeps the handle; its unwritten
        blocks are handed to the RAM remainder instead.
        """
        writer = self._mic_writer
        thread = self._mic_writer_thread
        if self._mic_writer_abandon.is_set():
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=2)
            if thread is not None and thread.is_alive():
                if writer is not None:
                    writer.suppress_disk = True
            return
        if not self._mic_disk_capture and writer is None:
            return
        self._mic_writer_stop.set()
        if thread is None or thread is threading.current_thread():
            if writer is not None:
                writer.close()
            return
        if not thread.is_alive():
            if writer is not None:
                writer.close()
            return
        thread.join(timeout=self._mic_writer_join_timeout)
        if not thread.is_alive():
            if writer is not None:
                writer.close()
            return
        self._handoff_mic_queue_to_ram()
        thread.join(timeout=self._mic_writer_inflight_join_timeout)
        if not thread.is_alive():
            if writer is not None:
                writer.close()
            return
        self._steal_uncommitted_inflight()
        if writer is not None:
            writer.suppress_disk = True
        self._mic_writer_close_when_done = True

    def _materialize_mic_track(self) -> None:
        """Rewrite the mic WAV as the flushed prefix plus the RAM remainder."""
        with self._mic_io_lock:
            remainder = list(self._mic_ram_remainder)
        if not remainder:
            return
        prefix = np.zeros((0, 1), dtype=np.int16)
        rate = int(self.sample_rate)
        path = self._mic_wav
        if path is not None and path.exists() and path.stat().st_size > 44:
            try:
                loaded, loaded_rate = recover_wav(path)
                if loaded_rate > 0:
                    rate = int(loaded_rate)
                prefix = loaded
            except (OSError, ValueError) as exc:
                print(f"Mic prefix unreadable; keeping the RAM remainder only: {exc}", file=sys.stderr)
        parts = []
        if prefix.size:
            parts.append(prefix[:, 0] if prefix.ndim == 2 else np.reshape(prefix, -1))
        for block in remainder:
            mono = _mono_int16(block)
            if mono.size:
                parts.append(mono)
        if not parts:
            return
        pcm = np.concatenate(parts)
        if path is None:
            return
        temporary = path.with_name(f".{path.stem}.rebuild.wav")
        try:
            sf.write(str(temporary), pcm.reshape(-1, 1), rate, subtype="PCM_16", format="WAV")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def abandon_mic_writer(self) -> None:
        """Simulate a crash: keep flushed checkpoints, drop the queued tail."""
        self._mic_writer_abandon.set()
        self._finish_mic_writer()

    @property
    def is_recording(self) -> bool:
        return self.recording

    def wait_for_mic_first_sample(self, timeout_seconds: float = 15.0) -> float | None:
        """Wait outside the audio callback for the first mic sample's clock."""
        self._mic_first_sample_event.wait(timeout=max(0.0, timeout_seconds))
        return self._mic_first_sample_monotonic


class MeetingRecorderApp(rumps.App):
    """macOS menu bar application for recording meetings."""

    def __init__(self, config: dict, config_path: Path):
        super().__init__(
            name="MeetingRecorder",
            title=TITLE_IDLE,
            icon=ICON_IDLE,
            quit_button=None  # We'll add our own
        )

        self.config = config
        self.config_path = config_path
        self.recordings_dir = expand_path(config['paths']['recordings'])
        self.transcripts_dir = expand_path(config['paths']['transcripts'])

        # Ensure directories exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize recorder
        self.recorder = AudioRecorder(config)
        self.recording_start_time: Optional[datetime] = None
        self._recording_stem: Optional[str] = None
        self._recording_started_monotonic: Optional[float] = None
        self._prompt_mark_item = None
        self._appkit = None
        self._event_monitor_api = None
        self._global_hotkey_monitor = None
        self._local_hotkey_monitor = None
        self._sidecar_threads: set[threading.Thread] = set()
        self._sidecar_write_failed_stems: set[str] = set()

        # Build menu
        self._build_menu()
        self._install_prompt_mark_shortcut()
        self._recover_orphaned_recordings()

    def _recover_orphaned_recordings(self) -> None:
        """Play back crashed captures into Recordings without a modal dialog."""
        try:
            recover_orphaned_captures(
                self.recorder.tmp_dir,
                self.recordings_dir,
                notify=self._notify_recovered_recording,
            )
        except Exception as exc:
            print(f"Orphaned recording recovery failed: {exc}", file=sys.stderr)

    def _notify_recovered_recording(self, path: Path) -> None:
        rumps.notification(
            title="MeetingRecorder",
            subtitle="Aufnahme wiederhergestellt",
            message=f"Unterbrochene Aufnahme gespeichert: {path.name}",
        )

    def _build_menu(self):
        """Build the menu bar menu."""
        self._start_stop_item = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self._prompt_mark_item = rumps.MenuItem("Prompt markieren", callback=None)
        self._clip_item = rumps.MenuItem("Clip…", callback=self.copy_clip)
        self._prompts_item = rumps.MenuItem("Prompts…", callback=self.copy_prompt)
        self._devices_item = rumps.MenuItem("List Audio Devices", callback=self.list_devices)
        self._jev_pref_item = rumps.MenuItem("Jev verwenden", callback=self.toggle_jev_preference)
        self.menu = [
            self._start_stop_item,
            self._prompt_mark_item,
            self._clip_item,
            self._prompts_item,
            None,  # Separator
            rumps.MenuItem("Open Recordings Folder", callback=self.open_recordings),
            rumps.MenuItem("Open Transcripts Folder", callback=self.open_transcripts),
            None,  # Separator
            self._jev_pref_item,
            rumps.MenuItem("Preferences...", callback=self.open_preferences),
            self._devices_item,
            None,  # Separator
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._sync_jev_menu_state()

    def _set_prompt_mark_available(self, available: bool) -> None:
        """Enable marking only while there is a persisted recording identity."""
        if self._prompt_mark_item is not None:
            self._prompt_mark_item.set_callback(self.mark_prompt if available else None)

    def _install_prompt_mark_shortcut(self) -> None:
        """Install local + global monitors for the documented mark shortcut.

        AppKit's global monitor does not receive this app's own events, so the
        local monitor complements it.  The global half needs macOS
        Accessibility permission; the menu item remains available regardless.
        """
        try:
            import AppKit

            mask = _appkit_constant(AppKit, "NSEventMaskKeyDown", "NSKeyDownMask", 1 << 10)
            self._appkit = AppKit
            self._event_monitor_api = AppKit.NSEvent
            self._local_hotkey_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                mask, self._handle_local_prompt_mark_shortcut
            )
            self._global_hotkey_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask, self._handle_global_prompt_mark_shortcut
            )
        except Exception as exc:
            # The recorder still works through its menu on installations where
            # PyObjC/AppKit is unavailable or the event monitor cannot start.
            print(f"Prompt-mark global shortcut unavailable: {exc}", file=sys.stderr)

    def _remove_prompt_mark_shortcut(self) -> None:
        """Remove every installed monitor exactly once before quitting."""
        if self._event_monitor_api is None:
            return
        for attribute in ("_global_hotkey_monitor", "_local_hotkey_monitor"):
            monitor = getattr(self, attribute, None)
            if monitor is None:
                continue
            try:
                self._event_monitor_api.removeMonitor_(monitor)
            except Exception as exc:
                print(f"Could not remove prompt-mark shortcut: {exc}", file=sys.stderr)
            finally:
                setattr(self, attribute, None)

    def _handle_global_prompt_mark_shortcut(self, event) -> None:
        if self._appkit and is_prompt_mark_shortcut(event, self._appkit):
            self.mark_prompt(None)

    def _handle_local_prompt_mark_shortcut(self, event):
        self._handle_global_prompt_mark_shortcut(event)
        return event

    def toggle_recording(self, sender):
        """Start or stop recording."""
        if self.recorder.is_recording:
            self._stop_recording(sender)
        else:
            self._start_recording(sender)

    def _gemini_api_key(self) -> str | None:
        """Read the recorder's configured Gemini key without exposing it."""
        config = getattr(self, "config", {})
        gemini = config.get("gemini", {}) if isinstance(config, dict) else {}
        env_name = gemini.get("api_key_env", "GEMINI_API_KEY") if isinstance(gemini, dict) else "GEMINI_API_KEY"
        if not isinstance(env_name, str) or not env_name.strip():
            env_name = "GEMINI_API_KEY"
        return os.environ.get(env_name)

    def _calendar_context_for_start(
        self, output_file: Path, *, timeout_seconds: float = 3.0
    ) -> CalendarRecordingContext:
        """Resolve the current event off the capture path with a short timeout.

        The existing resolver keys its lookup from the timestamped filename;
        it does not need a completed transcript. Any lookup failure is an
        unresolved meeting and is persisted fail-closed for the Jev gate.
        """
        try:
            from calendar_resolve import resolve

            return calendar_context_from_resolution(
                resolve(output_file.with_suffix(".json"), timeout_seconds=timeout_seconds)
            )
        except Exception as exc:
            print(f"Calendar lookup unavailable for Jev gate: {exc}", file=sys.stderr)
            return CalendarRecordingContext(external_attendees=None, attendance_resolved=False)

    def _jev_preference(self) -> bool:
        """Read the Preferences switch. Missing config stays off."""
        return jev_preference_enabled(getattr(self, "config_path", None))

    def _sync_jev_menu_state(self) -> None:
        item = getattr(self, "_jev_pref_item", None)
        if item is None:
            return
        try:
            item.state = 1 if self._jev_preference() else 0
        except Exception as exc:
            print(f"Could not show Jev preference state: {exc}", file=sys.stderr)

    def toggle_jev_preference(self, sender):
        """Flip the Jev switch. This is not shown when a recording starts."""
        enabled = not self._jev_preference()
        try:
            write_jev_preference(self.config_path, enabled)
        except Exception as exc:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Jev verwenden",
                message=f"Die Einstellung konnte nicht gespeichert werden: {exc}",
            )
            return
        try:
            sender.state = 1 if enabled else 0
        except Exception:
            self._sync_jev_menu_state()

    def _start_recording(self, sender):
        """Start capture before optional calendar/UI/sidecar work."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.recordings_dir / f"{timestamp}.wav"

        try:
            recording_started_monotonic = time.monotonic()
            if not self.recorder.start(output_file):
                raise RuntimeError("recorder refused to start")
        except Exception as e:
            self._recording_stem = None
            self._recording_started_monotonic = None
            self._set_prompt_mark_available(False)
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Error",
                message=str(e)
            )
            return

        # Everything below is optional sidecar/UI work.  It must never route
        # back through the capture-start exception path once the recorder is
        # active, even if local persistence or a background thread fails.
        self.recording_start_time = datetime.now()
        self._recording_stem = timestamp
        self._recording_started_monotonic = recording_started_monotonic
        self.title = TITLE_RECORDING
        sender.title = "Stop Recording"
        self._set_prompt_mark_available(True)
        self._set_modal_menu_enabled(False)
        try:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording started",
                message=f"Saving to: {output_file.name}",
            )
        except Exception as exc:
            print(f"Recording-start notification failed: {exc}", file=sys.stderr)

        # The preference is a local file read. It does not wait on calendar
        # or on a dialog. A failed write still forces Jev off.
        self._initialise_recording_choice(timestamp)
        self._start_live_session()
        try:
            self._start_sidecar_worker(
                name="meeting-sidecar-mic-origin",
                target=lambda: self._persist_mic_first_sample_offset(
                    timestamp, recording_started_monotonic
                ),
            )
            self._start_sidecar_worker(
                name="meeting-sidecar-calendar",
                target=lambda: self._resolve_calendar_after_capture(timestamp, output_file),
            )
        except Exception as exc:
            self._sidecar_write_failed(timestamp, exc)

    def _sidecar_write_failed(self, stem: str, exc: Exception) -> None:
        """Keep recording and permanently force Gemini after any write failure."""
        print(f"Sidecar write failed for {stem}; Jev forced off: {exc}", file=sys.stderr)
        failures = getattr(self, "_sidecar_write_failed_stems", None)
        if failures is None:
            failures = set()
            self._sidecar_write_failed_stems = failures
        failures.add(stem)
        # An initial sidecar is false by construction; this best-effort amend
        # also turns off a choice made just before a later timing write failed.
        try:
            amend_recording_sidecar(
                stem,
                root=self.recordings_dir,
                jev=False,
                jev_external_acknowledged=False,
            )
        except Exception:
            pass
        try:
            self._notify_from_worker(
                subtitle="Sidecar nicht gespeichert",
                message="Die Aufnahme läuft weiter; Jev bleibt ausgeschaltet.",
            )
        except Exception as notify_exc:
            print(f"Sidecar failure notification failed: {notify_exc}", file=sys.stderr)

    def _initialise_recording_choice(self, stem: str) -> None:
        """Write the Preferences switch into the sidecar as soon as capture starts."""
        try:
            initialise_recording_sidecar(
                stem,
                jev=self._jev_preference(),
                external_attendees=None,
                jev_external_acknowledged=False,
                root=self.recordings_dir,
                metadata=sidecar_metadata_from_calendar(
                    CalendarRecordingContext(None, False)
                ),
            )
        except Exception as exc:
            self._sidecar_write_failed(stem, exc)

    def _close_previous_live_session(self) -> None:
        """Stop the previous worker and close its panel before a new recording."""
        session = getattr(self, "_live_session", None)
        if session is None:
            return
        stopper = getattr(session, "stop", None)
        if callable(stopper):
            try:
                stopper()
            except Exception as exc:
                print(f"Previous live sidecar stop failed: {exc}", file=sys.stderr)
        panel = getattr(session, "panel", None)
        closer = getattr(panel, "close", None) if panel is not None else None
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                print(f"Previous live panel close failed: {exc}", file=sys.stderr)
        self._live_session = None

    def _notify_live_unavailable(self, message: str) -> None:
        """One non-modal notice when the live session cannot start."""
        if getattr(self, "_live_unavailable_notified", False):
            return
        self._live_unavailable_notified = True
        self._notify_from_worker(subtitle="Live-Sitzung nicht gestartet", message=message)

    def _start_live_session(self) -> None:
        """Open the side window and start the live worker after capture is up.

        Tests build the app without a run loop. They skip the real panel unless
        they inject ``_live_session_factory``. A failure here cannot stop
        recording. Gemini clients are built on the worker, not here.
        """
        self._close_previous_live_session()
        self._live_unavailable_notified = False
        factory = getattr(self, "_live_session_factory", None)
        if factory is None and not (self._gemini_api_key() or "").strip():
            self._notify_live_unavailable(
                "Gemini-Schlüssel fehlt. Die Aufnahme läuft weiter."
            )
        if factory is None and os.environ.get("PYTEST_CURRENT_TEST"):
            return
        try:
            if factory is None:
                from sidecar.live_session import RecordingLiveSession

                self._live_session = RecordingLiveSession(
                    self.recorder,
                    api_key=self._gemini_api_key(),
                    copy_text=self._copy_to_clipboard,
                    on_unavailable=self._notify_live_unavailable,
                    recordings_dir=self.recordings_dir,
                    stem=self._recording_stem,
                )
                self._live_session.start()
            else:
                self._live_session = factory(self)
                start = getattr(self._live_session, "start", None)
                if callable(start):
                    start()
        except Exception as exc:
            print(f"Live sidecar unavailable; recording continues: {exc}", file=sys.stderr)
            self._notify_live_unavailable(
                f"Die Aufnahme läuft weiter ({type(exc).__name__})."
            )

    def _set_modal_menu_enabled(self, enabled: bool) -> None:
        """Clip, Prompts, and device list open modals. They stay off while recording."""
        for name, callback in (
            ("_clip_item", self.copy_clip),
            ("_prompts_item", self.copy_prompt),
            ("_devices_item", self.list_devices),
        ):
            item = getattr(self, name, None)
            if item is None or not hasattr(item, "set_callback"):
                continue
            item.set_callback(callback if enabled else None)

    def _recording_blocks_modals(self) -> bool:
        recorder = getattr(self, "recorder", None)
        return bool(getattr(recorder, "is_recording", False))

    def _reveal_live_panel(self) -> None:
        session = getattr(self, "_live_session", None)
        panel = getattr(session, "panel", None) if session is not None else None
        reveal = getattr(panel, "order_front", None) if panel is not None else None
        if not callable(reveal):
            return
        try:
            reveal()
        except Exception as exc:
            print(f"Could not show the live panel: {exc}", file=sys.stderr)

    def _run_window(self, **kwargs):
        """Modal text prompt. Never while recording; activate the app first otherwise."""
        if self._recording_blocks_modals():
            self._reveal_live_panel()
            return SimpleNamespace(clicked=False, text="")
        activate_app_for_modal()
        return rumps.Window(**kwargs).run()

    def _run_alert(self, **kwargs):
        """Modal alert. Never while recording; activate the app first otherwise."""
        if self._recording_blocks_modals():
            self._reveal_live_panel()
            return None
        activate_app_for_modal()
        return rumps.alert(**kwargs)

    def _persist_mic_first_sample_offset(
        self, stem: str, mark_zero_monotonic: float
    ) -> None:
        wait = getattr(self.recorder, "wait_for_mic_first_sample", None)
        if not callable(wait):
            return
        first_sample = wait(timeout_seconds=15.0)
        if first_sample is None:
            return
        if stem in getattr(self, "_sidecar_write_failed_stems", set()):
            return
        try:
            amend_recording_sidecar(
                stem,
                root=self.recordings_dir,
                mic_first_sample_offset_seconds=max(
                    0.0, float(first_sample) - float(mark_zero_monotonic)
                ),
            )
        except Exception as exc:
            self._sidecar_write_failed(stem, exc)

    def _resolve_calendar_after_capture(self, stem: str, output_file: Path) -> None:
        """Store calendar metadata. This never opens a dialog and never changes ``jev``."""
        context = self._calendar_context_for_start(output_file, timeout_seconds=3.0)
        if stem in getattr(self, "_sidecar_write_failed_stems", set()):
            return
        try:
            amend_recording_sidecar(
                stem,
                root=self.recordings_dir,
                external_attendees=context.external_attendees,
                metadata=sidecar_metadata_from_calendar(context),
            )
        except Exception as exc:
            self._sidecar_write_failed(stem, exc)

    def _stop_recording(self, sender):
        """Stop the current recording. The live window stays open."""
        session = getattr(self, "_live_session", None)
        if session is not None:
            try:
                session.stop()
            except Exception as exc:
                print(f"Live sidecar stop failed: {exc}", file=sys.stderr)
        output_file = self.recorder.stop()

        # Calculate duration
        duration = ""
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            duration = f" ({minutes}m {seconds}s)"

        # Update UI
        self.title = TITLE_IDLE
        sender.title = "Start Recording"
        self.recording_start_time = None
        self._recording_stem = None
        self._recording_started_monotonic = None
        self._set_prompt_mark_available(False)
        self._set_modal_menu_enabled(True)

        if output_file and output_file.exists():
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording saved" + duration,
                message=f"File: {output_file.name}\nTranscription will start automatically."
            )
        else:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording stopped",
                message="No audio was captured."
            )

    def mark_prompt(self, _):
        """Append a monotonic prompt mark without interacting with audio buffers."""
        stem = self._recording_stem
        started = self._recording_started_monotonic
        if not self.recorder.is_recording or stem is None or started is None:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Prompt markieren",
                message="Keine aktive Aufnahme.",
            )
            return
        try:
            mark = append_prompt_mark(
                stem,
                recording_started_monotonic=started,
                recordings_root=self.recordings_dir,
                clock=time.monotonic,
            )
        except Exception as exc:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Prompt-Markierung fehlgeschlagen",
                message=str(exc),
            )
            return
        rumps.notification(
            title="MeetingRecorder",
            subtitle="Prompt markiert",
            message=f"Marke bei {format_timestamp(mark)} gespeichert.",
        )

    def _dispatch_ui(self, callback) -> None:
        """Return worker results through AppKit's main run loop when available."""
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(callback)
        except Exception:
            # Tests and installations without PyObjC still get the result; the
            # production rumps path has AppHelper through its dependency.
            callback()

    def _notify_from_worker(self, *, subtitle: str, message: str) -> None:
        self._dispatch_ui(
            lambda: rumps.notification(
                title="MeetingRecorder", subtitle=subtitle, message=message
            )
        )

    @staticmethod
    def _copy_to_clipboard(text: str) -> None:
        """Copy finished sidecar text from a worker, never from audio capture."""
        result = subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "pbcopy failed")

    def _start_sidecar_worker(self, *, name: str, target) -> None:
        """Run completed-transcript work off the rumps and audio callback paths."""
        thread: threading.Thread
        threads = getattr(self, "_sidecar_threads", None)
        if threads is None:
            threads = set()
            self._sidecar_threads = threads

        def run() -> None:
            try:
                target()
            finally:
                threads.discard(thread)

        thread = threading.Thread(target=run, name=name, daemon=True)
        threads.add(thread)
        thread.start()

    def _recent_transcript_choice(self, action: str):
        """Ask for one of the newest valid finished transcripts by number."""
        try:
            transcripts = recent_transcripts(self.transcripts_dir, limit=10)
        except Exception as exc:
            rumps.notification(
                title="MeetingRecorder",
                subtitle=action,
                message=f"Transkripte konnten nicht gelesen werden: {exc}",
            )
            return None
        if not transcripts:
            rumps.notification(
                title="MeetingRecorder",
                subtitle=action,
                message="Noch kein fertiges Transkript gefunden.",
            )
            return None
        choices = "\n".join(
            f"{number}. {recent_transcript_title(transcript, recordings_root=self.recordings_dir)}"
            for number, transcript in enumerate(transcripts, start=1)
        )
        response = self._run_window(
            message=f"Wähle ein fertiges Meeting:\n\n{choices}",
            title=action,
            default_text="1",
            ok="Weiter",
            cancel="Abbrechen",
        )
        if not response.clicked:
            return None
        try:
            selected = int(response.text.strip())
        except (AttributeError, TypeError, ValueError):
            selected = 0
        if not 1 <= selected <= len(transcripts):
            self._run_alert(
                title=action,
                message=f"Bitte eine Zahl von 1 bis {len(transcripts)} eingeben.",
                ok="OK",
            )
            return None
        return transcripts[selected - 1]

    def _prompt_text(self, title: str, message: str):
        response = self._run_window(
            message=message,
            title=title,
            default_text="",
            ok="Weiter",
            cancel="Abbrechen",
        )
        if not response.clicked:
            return None
        text = response.text.strip()
        return text or None

    def copy_clip(self, _):
        """Select a finished transcript and create a clipboard-ready topic clip."""
        if self._recording_blocks_modals():
            self._reveal_live_panel()
            return
        transcript = self._recent_transcript_choice("Clip…")
        if transcript is None:
            return
        topic = self._prompt_text("Clip…", "Welches Thema soll der Clip abdecken?")
        if topic is None:
            return

        def make_clip() -> None:
            try:
                clip = clip_for_stem(
                    transcript.stem,
                    topic,
                    transcripts_root=self.transcripts_dir,
                    recordings_root=self.recordings_dir,
                    gemini_api_key=self._gemini_api_key(),
                )
                self._copy_to_clipboard(clip.text)
            except Exception as exc:
                self._notify_from_worker(
                    subtitle="Clip fehlgeschlagen",
                    message=str(exc),
                )
                return
            if clip.has_fallback:
                self._copy_to_clipboard(clip.text)
                self._notify_from_worker(
                    subtitle="Clip…",
                    message="Nichts Passendes gefunden; Top-Kandidaten wurden kopiert.",
                )
                return
            self._notify_from_worker(
                subtitle="Clip kopiert",
                message=f"{clip.line_count} Zeilen in die Zwischenablage kopiert.",
            )

        self._start_sidecar_worker(name="meeting-sidecar-clip", target=make_clip)

    def copy_prompt(self, _):
        """Find marked/suggested prompts in a finished transcript off the UI thread."""
        if self._recording_blocks_modals():
            self._reveal_live_panel()
            return
        transcript = self._recent_transcript_choice("Prompts…")
        if transcript is None:
            return

        def find_prompts() -> None:
            try:
                prompts = prompts_for_stem(
                    transcript.stem,
                    transcripts_root=self.transcripts_dir,
                    recordings_root=self.recordings_dir,
                    gemini_api_key=self._gemini_api_key(),
                )
            except Exception as exc:
                self._notify_from_worker(
                    subtitle="Prompts fehlgeschlagen",
                    message=str(exc),
                )
                return
            if not prompts:
                self._notify_from_worker(
                    subtitle="Prompts…",
                    message="Keine markierten oder vorgeschlagenen Prompts gefunden.",
                )
                return
            self._dispatch_ui(lambda: self._choose_prompt_to_copy(prompts))

        self._start_sidecar_worker(name="meeting-sidecar-prompts", target=find_prompts)

    def _choose_prompt_to_copy(self, prompts) -> None:
        """Let the user choose a marked or suggested prompt without showing its text."""
        choices: list[str] = []
        for number, prompt in enumerate(prompts, start=1):
            choices.append(f"{number}. {prompt_choice_label(prompt)}")
        response = self._run_window(
            message="Welchen Prompt kopieren?\n\n" + "\n".join(choices),
            title="Prompts…",
            default_text="1",
            ok="Kopieren",
            cancel="Abbrechen",
        )
        if not response.clicked:
            return
        try:
            selected = int(response.text.strip())
        except (AttributeError, TypeError, ValueError):
            selected = 0
        if not 1 <= selected <= len(prompts):
            self._run_alert(
                title="Prompts…",
                message=f"Bitte eine Zahl von 1 bis {len(prompts)} eingeben.",
                ok="OK",
            )
            return
        prompt = prompts[selected - 1]

        def copy_selected_prompt() -> None:
            try:
                self._copy_to_clipboard(prompt.copy_text)
            except Exception as exc:
                self._notify_from_worker(
                    subtitle="Prompt konnte nicht kopiert werden",
                    message=str(exc),
                )
                return
            self._notify_from_worker(
                subtitle="Prompt kopiert",
                message=f"{len(prompt.lines)} Zeilen in die Zwischenablage kopiert.",
            )

        self._start_sidecar_worker(
            name="meeting-sidecar-copy-prompt", target=copy_selected_prompt
        )

    def open_recordings(self, _):
        """Open the recordings folder in Finder."""
        subprocess.run(["open", str(self.recordings_dir)])

    def open_transcripts(self, _):
        """Open the transcripts folder in Finder."""
        subprocess.run(["open", str(self.transcripts_dir)])

    def open_preferences(self, _):
        """Open the config file in the default editor."""
        subprocess.run(["open", str(self.config_path)])

    def list_devices(self, _):
        """Show available audio devices. Disabled while a recording is open."""
        if self._recording_blocks_modals():
            return
        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " (current)" if i == self.recorder.device else ""
                input_devices.append(f"• {dev['name']}{marker}")

        device_list = "\n".join(input_devices[:10])  # Limit to 10
        if len(input_devices) > 10:
            device_list += f"\n... and {len(input_devices) - 10} more"

        self._run_alert(
            title="Available Audio Input Devices",
            message=device_list or "No input devices found",
            ok="OK"
        )

    def quit_app(self, _):
        """Quit the application, stopping any active recording."""
        self._remove_prompt_mark_shortcut()
        if self.recorder.is_recording:
            self.recorder.stop()
        rumps.quit_application()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Menu bar app for recording meetings"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create and run app
    app = MeetingRecorderApp(config, args.config)
    app.run()


if __name__ == "__main__":
    main()
