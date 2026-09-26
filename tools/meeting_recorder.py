#!/usr/bin/env python3
"""
MeetingRecorder - macOS menu bar app for recording meetings

A simple menu bar app that records audio from your microphone (or combined
mic + system audio via BlackHole) and saves it for automatic transcription.

Usage:
    python meeting_recorder.py [--config PATH]
"""

import os
import sys
import time
import shutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import sounddevice as sd
import soundfile as sf
import rumps

from capture_provenance import archive_capture, merge_filter
from sidecar.gate import initialise_recording_sidecar
from sidecar.recorder_support import (
    CalendarRecordingContext,
    append_prompt_mark,
    calendar_context_from_resolution,
    recent_transcript_title,
    sidecar_metadata_from_calendar,
)
from sidecar.service import clip_for_stem, prompts_for_stem
from sidecar.transcript import format_timestamp, recent_transcripts


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
        self._mic_frames = 0
        self._previous_adc_end = None
        self._next_timing_sample = 0
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
            if self._sys_active:
                subprocess.run(["pkill", "-INT", "-f", self._PROC_PATTERN])
            raise RuntimeError(f"Failed to start mic recording: {e}")

    def stop(self) -> Optional[Path]:
        """Stop both captures, merge into one 3-channel WAV (mic + system)."""
        if not self.recording:
            return None
        self.recording = False

        # Stop mic, write mic.wav.
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        mic_ok = False
        if self.audio_data:
            arr = np.concatenate(self.audio_data, axis=0)
            sf.write(str(self._mic_wav), arr, self.sample_rate, subtype='PCM_16')
            mic_ok = self._mic_wav.exists() and self._mic_wav.stat().st_size > 1000

        # Stop system tap via SIGINT (graceful teardown => valid WAV + tap freed).
        if self._sys_active:
            subprocess.run(["pkill", "-INT", "-f", self._PROC_PATTERN])
            for _ in range(60):  # up to ~6s for clean teardown
                if not self._sys_proc_running():
                    break
                time.sleep(0.1)
            time.sleep(0.3)
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
        """Callback for the microphone input stream."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        if self.recording:
            self.audio_data.append(indata.copy())
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

    @property
    def is_recording(self) -> bool:
        return self.recording


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

        # Build menu
        self._build_menu()
        self._install_prompt_mark_shortcut()

    def _build_menu(self):
        """Build the menu bar menu."""
        self._start_stop_item = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self._prompt_mark_item = rumps.MenuItem("Prompt markieren", callback=None)
        self.menu = [
            self._start_stop_item,
            self._prompt_mark_item,
            rumps.MenuItem("Clip…", callback=self.copy_clip),
            rumps.MenuItem("Prompts…", callback=self.copy_prompt),
            None,  # Separator
            rumps.MenuItem("Open Recordings Folder", callback=self.open_recordings),
            rumps.MenuItem("Open Transcripts Folder", callback=self.open_transcripts),
            None,  # Separator
            rumps.MenuItem("Preferences...", callback=self.open_preferences),
            rumps.MenuItem("List Audio Devices", callback=self.list_devices),
            None,  # Separator
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

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

    def _calendar_context_for_start(self, output_file: Path) -> CalendarRecordingContext:
        """Resolve the current event before showing the Jev choice.

        The existing resolver keys its lookup from the timestamped filename;
        it does not need a completed transcript. Any lookup failure is an
        unresolved meeting and is persisted fail-closed for the Jev gate.
        """
        try:
            from calendar_resolve import resolve

            return calendar_context_from_resolution(resolve(output_file.with_suffix(".json")))
        except Exception as exc:
            print(f"Calendar lookup unavailable for Jev gate: {exc}", file=sys.stderr)
            return CalendarRecordingContext(external_attendees=True, attendance_resolved=False)

    @staticmethod
    def _jev_dialog_message(context: CalendarRecordingContext) -> str:
        message = (
            "Standard ist Gemini (vertraglich gedeckt).\n\n"
            "Jev darf nur für eine eindeutig als intern aufgelöste Aufnahme "
            "verwendet werden."
        )
        if context.external_attendees and context.attendance_resolved:
            return message + "\n\nJev: nicht durch Kunden-DPA gedeckt."
        if not context.attendance_resolved:
            return message + "\n\nDer Kalendereintrag konnte nicht eindeutig aufgelöst werden; Jev bleibt aus Sicherheitsgründen gesperrt."
        return message

    def _ask_jev_choice(self, context: CalendarRecordingContext) -> bool | None:
        """Show a checkbox dialog whose default is off, or return cancellation."""
        message = self._jev_dialog_message(context)
        try:
            import AppKit
            from Foundation import NSMakeRect

            alert = AppKit.NSAlert.alloc().init()
            alert.setMessageText_("Jev verwenden?")
            alert.setInformativeText_(message)
            alert.addButtonWithTitle_("Aufnahme starten")
            alert.addButtonWithTitle_("Abbrechen")
            checkbox = AppKit.NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 360, 24))
            checkbox.setButtonType_(getattr(AppKit, "NSSwitchButton", 3))
            checkbox.setTitle_("Jev für diese Aufnahme verwenden")
            checkbox.setState_(getattr(AppKit, "NSControlStateValueOff", 0))
            if not context.jev_can_be_selected:
                checkbox.setEnabled_(False)
            alert.setAccessoryView_(checkbox)
            response = alert.runModal()
            if response != getattr(AppKit, "NSAlertFirstButtonReturn", 1000):
                return None
            return bool(
                context.jev_can_be_selected
                and checkbox.state() == getattr(AppKit, "NSControlStateValueOn", 1)
            )
        except Exception as exc:
            # This fallback never turns Jev on. It preserves the safe default
            # when an AppKit checkbox cannot be created.
            print(f"Jev checkbox unavailable; using Gemini: {exc}", file=sys.stderr)
            response = rumps.alert(
                title="Jev verwenden?",
                message=message,
                ok="Mit Gemini starten",
                cancel="Abbrechen",
            )
            return False if response == 1 else None

    def _start_recording(self, sender):
        """Start a new recording."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.recordings_dir / f"{timestamp}.wav"

        calendar_context = self._calendar_context_for_start(output_file)
        jev_choice = self._ask_jev_choice(calendar_context)
        if jev_choice is None:
            return

        try:
            # Persist policy before recording begins. If this write failed the
            # recorder must not start, because a missing policy fails closed but
            # cannot faithfully record the requested decision.
            initialise_recording_sidecar(
                timestamp,
                jev=jev_choice,
                external_attendees=calendar_context.external_attendees,
                root=self.recordings_dir,
                metadata=sidecar_metadata_from_calendar(calendar_context),
            )
            recording_started_monotonic = time.monotonic()
            if not self.recorder.start(output_file):
                raise RuntimeError("recorder refused to start")
            self.recording_start_time = datetime.now()
            self._recording_stem = timestamp
            self._recording_started_monotonic = recording_started_monotonic

            # Update UI
            self.title = TITLE_RECORDING
            sender.title = "Stop Recording"
            self._set_prompt_mark_available(True)

            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording started",
                message=f"Saving to: {output_file.name}"
            )

        except Exception as e:
            self._recording_stem = None
            self._recording_started_monotonic = None
            self._set_prompt_mark_available(False)
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Error",
                message=str(e)
            )

    def _stop_recording(self, sender):
        """Stop the current recording."""
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

        def run() -> None:
            try:
                target()
            finally:
                self._sidecar_threads.discard(thread)

        thread = threading.Thread(target=run, name=name, daemon=True)
        self._sidecar_threads.add(thread)
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
        response = rumps.Window(
            message=f"Wähle ein fertiges Meeting:\n\n{choices}",
            title=action,
            default_text="1",
            ok="Weiter",
            cancel="Abbrechen",
        ).run()
        if not response.clicked:
            return None
        try:
            selected = int(response.text.strip())
        except (AttributeError, TypeError, ValueError):
            selected = 0
        if not 1 <= selected <= len(transcripts):
            rumps.alert(
                title=action,
                message=f"Bitte eine Zahl von 1 bis {len(transcripts)} eingeben.",
                ok="OK",
            )
            return None
        return transcripts[selected - 1]

    @staticmethod
    def _prompt_text(title: str, message: str):
        response = rumps.Window(
            message=message,
            title=title,
            default_text="",
            ok="Weiter",
            cancel="Abbrechen",
        ).run()
        if not response.clicked:
            return None
        text = response.text.strip()
        return text or None

    def copy_clip(self, _):
        """Select a finished transcript and create a clipboard-ready topic clip."""
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
                )
                self._copy_to_clipboard(clip.text)
            except Exception as exc:
                self._notify_from_worker(
                    subtitle="Clip fehlgeschlagen",
                    message=str(exc),
                )
                return
            self._notify_from_worker(
                subtitle="Clip kopiert",
                message=f"{clip.line_count} Zeilen in die Zwischenablage kopiert.",
            )

        self._start_sidecar_worker(name="meeting-sidecar-clip", target=make_clip)

    def copy_prompt(self, _):
        """Find marked/suggested prompts in a finished transcript off the UI thread."""
        transcript = self._recent_transcript_choice("Prompts…")
        if transcript is None:
            return

        def find_prompts() -> None:
            try:
                prompts = prompts_for_stem(
                    transcript.stem,
                    transcripts_root=self.transcripts_dir,
                    recordings_root=self.recordings_dir,
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
            if prompt.source == "mark":
                label = f"Markierung bei {format_timestamp(prompt.mark_seconds or 0)}"
            else:
                label = f"Vorschlag bei {format_timestamp(prompt.start_seconds)}"
            choices.append(f"{number}. {label}")
        response = rumps.Window(
            message="Welchen Prompt kopieren?\n\n" + "\n".join(choices),
            title="Prompts…",
            default_text="1",
            ok="Kopieren",
            cancel="Abbrechen",
        ).run()
        if not response.clicked:
            return
        try:
            selected = int(response.text.strip())
        except (AttributeError, TypeError, ValueError):
            selected = 0
        if not 1 <= selected <= len(prompts):
            rumps.alert(
                title="Prompts…",
                message=f"Bitte eine Zahl von 1 bis {len(prompts)} eingeben.",
                ok="OK",
            )
            return
        prompt = prompts[selected - 1]

        def copy_selected_prompt() -> None:
            try:
                self._copy_to_clipboard(prompt.text)
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
        """Show available audio devices."""
        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " (current)" if i == self.recorder.device else ""
                input_devices.append(f"• {dev['name']}{marker}")

        device_list = "\n".join(input_devices[:10])  # Limit to 10
        if len(input_devices) > 10:
            device_list += f"\n... and {len(input_devices) - 10} more"

        rumps.alert(
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
