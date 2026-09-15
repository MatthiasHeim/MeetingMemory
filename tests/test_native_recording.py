import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import native_recording as nr
from source_inputs import prepare_recording_sources
from test_source_inputs import make_manifest


def test_degraded_start_keeps_usable_capture_running(tmp_path, monkeypatch):
    class Session:
        def __init__(self, bundle, session_dir):
            self.manifest_path = session_dir / 'capture-manifest.json'
        def start(self, **kwargs):
            return {'status': 'degraded', 'readiness': {'mic': True, 'system': False}}
        def stop(self, **kwargs):
            pytest.fail('Usable degraded capture must not be stopped at startup')
    monkeypatch.setattr(nr, 'NativeCaptureSession', Session)
    recorder = nr.NativeAudioRecorder({'audio': {'archive_dir': str(tmp_path / 'archive')}})
    assert recorder.start(tmp_path / 'meeting.wav')
    assert recorder.is_recording
    assert recorder.last_status['status'] == 'degraded'


def test_readiness_timeout_keeps_stop_available(tmp_path, monkeypatch):
    class Session:
        def __init__(self, bundle, session_dir):
            self.manifest_path = session_dir / 'capture-manifest.json'
        def start(self, **kwargs):
            raise nr.NativeCaptureReadinessTimeout('sources still starting')
        def status(self):
            return {'status': 'starting'}
        def stop(self, **kwargs):
            pytest.fail('Readiness timeout must not cancel a late source')
    monkeypatch.setattr(nr, 'NativeCaptureSession', Session)
    recorder = nr.NativeAudioRecorder({'audio': {'archive_dir': str(tmp_path / 'archive')}})
    assert recorder.start(tmp_path / 'meeting.wav')
    assert recorder.is_recording and recorder.last_status['status'] == 'starting'


def test_failed_stop_does_not_offer_a_second_recording(tmp_path):
    class Session:
        def stop(self, **kwargs):
            raise TimeoutError('capture remains active')
    recorder = nr.NativeAudioRecorder({'audio': {'archive_dir': str(tmp_path / 'archive')}})
    recorder.session = Session()
    recorder.recording = True
    with pytest.raises(TimeoutError):
        recorder.stop()
    assert recorder.is_recording
    assert not recorder.start(tmp_path / 'second.wav')


def test_publish_preserves_physical_channels_and_validates_binding(tmp_path):
    if not Path('/opt/homebrew/bin/ffmpeg').is_file():
        pytest.skip('ffmpeg unavailable')
    manifest = make_manifest(tmp_path, offset=2)
    data = json.loads(manifest.read_text())
    system = data['tracks'][1]['segments'][0]
    sf.write(tmp_path / system['path'], np.tile([.3, -.4], (8000, 1)), 8000, subtype='FLOAT')
    system['channels'] = 2
    manifest.write_text(json.dumps(data))
    output = tmp_path / 'recordings' / 'meeting.wav'
    nr.publish_native_capture(manifest, output)
    audio, rate = sf.read(output, always_2d=True)
    assert rate == 8000 and audio.shape == (24000, 3)
    assert np.allclose(audio[:8000, 0], .2)
    assert np.allclose(audio[:16000, 1:], 0)
    assert np.allclose(audio[16000:, 1], .3)
    assert np.allclose(audio[16000:, 2], -.4)
    sources, meta = prepare_recording_sources(output, tmp_path / 'recovery')
    assert len(sources) == 2 and meta['timeline_verified']
    with pytest.raises(FileExistsError):
        nr.publish_native_capture(manifest, output)
    original = output.read_bytes()
    output.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='audio hash mismatch'):
        prepare_recording_sources(output, tmp_path / 'recovery')
    output.write_bytes(original)
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        prepare_recording_sources(output, tmp_path / 'recovery')


def test_native_gap_reaches_durable_partial_draft(tmp_path):
    from transcription_jobs import SourceTranscriptionPipeline
    manifest = make_manifest(tmp_path, offset=2)
    data = json.loads(manifest.read_text())
    data['tracks'][1]['gaps'] = [{'begin_seconds': 0, 'end_seconds': 2, 'reason': 'late_first_packet'}]
    manifest.write_text(json.dumps(data))
    sources, capture = nr.prepare_native_sources(manifest, tmp_path / 'derived')
    # Exact-zero evidence prevents any provider request. Capture gaps must still
    # remain missing evidence even when the supplied speech detector says silent.
    pipeline = SourceTranscriptionPipeline(tmp_path / 'jobs', 'test',
        speech_detector=lambda *a, **k: {'status': 'digital_silence', 'detector': 'test'})
    result = pipeline.run(sources, session_id='gap-test', capture_gaps=capture['capture_gaps'])
    assert not result['_meta']['structural_completeness']['transcript_structurally_complete']
    assert result['_meta']['speaker_attribution']['speaker_dependent_actions'] == 'hold'
    assert any(r['source_id'] == 'system' for r in result['_meta']['missing_ranges'])
