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


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"

# Menu bar icons (using emoji as fallback)
ICON_IDLE = None  # Will use title instead
ICON_RECORDING = None
TITLE_IDLE = "🎙️"
TITLE_RECORDING = "🔴"


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

        # Build menu
        self._build_menu()

    def _build_menu(self):
        """Build the menu bar menu."""
        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            None,  # Separator
            rumps.MenuItem("Open Recordings Folder", callback=self.open_recordings),
            rumps.MenuItem("Open Transcripts Folder", callback=self.open_transcripts),
            None,  # Separator
            rumps.MenuItem("Preferences...", callback=self.open_preferences),
            rumps.MenuItem("List Audio Devices", callback=self.list_devices),
            None,  # Separator
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

    def toggle_recording(self, sender):
        """Start or stop recording."""
        if self.recorder.is_recording:
            self._stop_recording(sender)
        else:
            self._start_recording(sender)

    def _start_recording(self, sender):
        """Start a new recording."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.recordings_dir / f"{timestamp}.wav"

        try:
            self.recorder.start(output_file)
            self.recording_start_time = datetime.now()

            # Update UI
            self.title = TITLE_RECORDING
            sender.title = "Stop Recording"

            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording started",
                message=f"Saving to: {output_file.name}"
            )

        except Exception as e:
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
