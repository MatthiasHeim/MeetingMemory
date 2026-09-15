"""Menu-bar adapter for the native recorder; publishes only finalized artifacts."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from native_capture_bridge import NativeCaptureSession, NativeCaptureReadinessTimeout
from source_inputs import prepare_native_sources, _sha
from speaker_integrity import atomic_json


class NativeAudioRecorder:
    def __init__(self, config):
        self.config = config
        audio = config.get('audio', {})
        self.bundle = Path(audio.get('native_bundle', '~/Documents/MeetingRecorder/bin/MeetingNativeCapture.app')).expanduser()
        self.archive_dir = Path(audio.get('archive_dir', '~/Documents/MeetingRecorder/CaptureArchive')).expanduser()
        self.activity_file = self.archive_dir.parent / 'active-native-capture.json'
        self.start_timeout = float(audio.get('native_start_timeout_seconds', 30))
        self.session = None
        self.recording = False
        self.output_file = None
        self.last_status = {}

    @property
    def is_recording(self):
        return self.recording

    def start(self, output_file):
        if self.recording:
            return False
        self.output_file = Path(output_file)
        session_dir = self.archive_dir / (self.output_file.stem + '-native')
        if session_dir.exists():
            raise RuntimeError(f'Capture directory already exists: {session_dir}')
        self.session = NativeCaptureSession(self.bundle, session_dir)
        atomic_json(self.activity_file, {'recorder_pid': os.getpid(),
                    'native_manifest_path': str(self.session.manifest_path), 'status': 'starting'})
        try:
            self.last_status = self.session.start(timeout_seconds=self.start_timeout)
            if self.last_status.get('status') not in ('recording', 'degraded'):
                raise RuntimeError('No audio source is ready; capture retained for diagnosis')
        except NativeCaptureReadinessTimeout:
            # A late source may still become ready. Keep an observable session
            # and its Stop action instead of discarding the capture at a timer.
            self.last_status = self.session.status()
            self.recording = True
            return True
        except Exception:
            # Stop only this owned recorder. Its finalized segments remain recoverable.
            try:
                self.last_status = self.session.stop(timeout_seconds=10)
                atomic_json(self.activity_file, {'recorder_pid': os.getpid(),
                    'native_manifest_path': str(self.session.manifest_path), 'status': 'stopped'})
            except Exception:
                pass
            raise
        self.recording = True
        return True

    def status(self):
        if self.session:
            self.last_status = self.session.status()
        return self.last_status

    def stop(self):
        if not self.recording or not self.session:
            return None
        self.last_status = self.session.stop(timeout_seconds=30)
        self.recording = False
        manifest = self.session.manifest_path
        # The process has stopped. Publication failure must leave raw segments,
        # but must not claim the microphone is still running.
        atomic_json(self.activity_file, {'recorder_pid': os.getpid(),
                    'native_manifest_path': str(manifest), 'status': 'stopped'})
        if not manifest.is_file():
            raise RuntimeError('Native capture manifest missing; retained segments need recovery')
        return publish_native_capture(manifest, self.output_file)


def publish_native_capture(manifest_path, output_file):
    manifest_path, output_file = Path(manifest_path), Path(output_file)
    if output_file.exists() or output_file.with_suffix('.capture.json').exists():
        raise FileExistsError('Capture output or binding already exists; choose a fresh output filename')
    sources, meta = prepare_native_sources(manifest_path, manifest_path.parent / 'derived')
    by_id = {s['source_id']: s for s in sources}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.parent / 'derived' / 'published.wav'
    cmd = ['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-y']
    if 'mic' in by_id and 'system' in by_id:
        # Output physical c0=mic, c1/c2=system, consistent with downstream layout.
        for name in ('mic', 'system'):
            cmd += ['-i', by_id[name]['path']]
        filters = '[0:a]pan=mono|c0=c0[m];[1:a]aformat=channel_layouts=stereo[s];[m][s]join=inputs=2:channel_layout=3.0:map=0.0-FL|1.0-FR|1.1-FC[a]'
        cmd += ['-filter_complex', filters, '-map', '[a]']
    else:
        cmd += ['-i', sources[0]['path']]
    cmd += ['-c:a', 'pcm_f32le', str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    # Publish binding before the atomic watched-WAV rename. Archive originals persist.
    atomic_json(output_file.with_suffix('.capture.json'), {
        'schema_version': 2, 'native_manifest_path': str(manifest_path.resolve()),
        'native_manifest_sha256': meta['native_manifest_sha256'],
        'compatibility_audio_sha256': _sha(tmp),
        'timeline': 'host_clock', 'capture_gaps': meta['capture_gaps'],
        'capture_status': meta['capture_status'],
    })
    os.replace(tmp, output_file)
    return output_file
