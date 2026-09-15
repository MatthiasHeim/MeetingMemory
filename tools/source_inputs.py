"""Prepare auditable ASR sources without inventing capture synchronization.

Original native segments are immutable. Timeline WAVs are derived FLOAT files;
zero padding represents missing capture, recorded separately as gaps, not silence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from speaker_integrity import atomic_json


def _finite(value, name, *, signed=False):
    number = float(value)
    if not math.isfinite(number) or (number < 0 and not signed):
        raise ValueError(f"Invalid {name}: {value}")
    return number


def _segment_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Native segment must remain inside its capture directory")
    if not path.is_file():
        raise ValueError(f"Missing native segment: {value}")
    return path


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _verify_timing(root, segment, source_id, info):
    if not segment.get('timing_path'):
        raise ValueError('Native segment has no packet timing sidecar')
    path = _segment_path(root, segment['timing_path'])
    timing = json.loads(path.read_text())
    entries = timing.get('entries')
    if timing.get('schema_version') != 1 or timing.get('source_id') != source_id or not entries:
        raise ValueError('Invalid native packet timing sidecar')
    count, previous = 0, None
    origin = float(segment['start_seconds'])
    for entry in entries:
        start = _finite(entry['start_seconds'], 'packet start', signed=True)
        frames = entry['written_frames']
        if type(frames) is not int:
            raise ValueError('Native timing frame counts must be integers')
        if frames <= 0 or (previous is not None and start < previous):
            raise ValueError('Invalid or reordered native packet timing')
        if abs(start - (origin + count / info.samplerate)) > .020 + 2 / info.samplerate:
            raise ValueError('Packet timing disagrees with contiguous segment samples')
        count += frames
        previous = start
    if count != info.frames:
        raise ValueError('Native packet timing frame count mismatch')
    return {'path': str(path), 'sha256': _sha(path), 'source_id': source_id}


def prepare_native_sources(manifest_path: Path, output_dir: Path, *, recover_stopped=False) -> tuple[list[dict], dict]:
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 2 or manifest.get('capture_backend') != 'screencapturekit':
        raise ValueError('Unsupported native capture manifest')
    if manifest.get('status') in ('starting', 'recording') or not manifest.get('finalized', False):
        if not recover_stopped:
            raise ValueError('Capture is still active or not finalized; recover stopped segments explicitly')
        pid = int((manifest.get('process') or {}).get('pid', 0))
        if pid <= 0:
            raise ValueError('Cannot prove native recorder has exited: no process identity')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise ValueError('Cannot prove native recorder has exited') from exc
        else:
            raise ValueError('Native recorder PID is still live; stop it before recovery')
        # Recover only manifest-checkpointed, closed segments. Active partial
        # CAFs remain untouched and the duration of the lost tail is unknown.
        manifest['status'] = 'crash_recovered'
        manifest['errors'] = [*(manifest.get('errors') or []), {
            'code': 'unfinalized_capture_recovery',
            'message': 'Only closed checkpointed segments recovered; final tail duration is unknown.',
        }]
    if manifest.get('timeline') != 'host_clock':
        raise ValueError('Native capture has no certified common timeline')
    tracks = manifest.get('tracks')
    if not isinstance(tracks, list) or not tracks:
        raise ValueError('Capture contains no source tracks')
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, all_gaps, seen, timing_evidence = [], [], set(), []
    raw_starts = [_finite(seg['start_seconds'], 'segment start', signed=True)
                  for track in tracks for seg in (track.get('segments') or [])]
    epoch_offset = min([0.0, *raw_starts])
    # One shared shift includes any pre-roll without destroying relative timing.
    # Original native PTS and manifest remain authoritative and untouched.
    max_end = _finite(manifest.get('final_duration_seconds') or 0, 'capture duration') - epoch_offset
    prepared = []
    for track in tracks:
        source_id = track.get('source_id')
        if source_id not in ('mic', 'system') or source_id in seen:
            raise ValueError('Native source IDs must uniquely identify mic/system')
        seen.add(source_id)
        segments = track.get('segments') or []
        if not segments:
            all_gaps.append({'source_id': source_id, 'start_seconds': 0.0, 'reason': 'source_missing'})
            continue
        items, rate, channels, prior_end = [], None, None, 0.0
        for seg in sorted(segments, key=lambda x: float(x['start_seconds'])):
            start = _finite(seg['start_seconds'], 'segment start', signed=True) - epoch_offset
            duration = _finite(seg['duration_seconds'], 'segment duration')
            path = _segment_path(manifest_path.parent, seg['path'])
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError('Empty native segment')
            if type(seg['frames']) is not int or type(seg['channels']) is not int or info.frames != seg['frames'] or info.samplerate != _finite(seg['sample_rate'], 'sample rate') or info.channels != seg['channels']:
                raise ValueError(f'Segment metadata disagrees with audio: {path.name}')
            if abs(info.duration - duration) > 2 / info.samplerate:
                raise ValueError('Native segment duration mismatch')
            if rate is not None and (info.samplerate != rate or info.channels != channels):
                raise ValueError('Native source format changed; preserve and review segments before resampling')
            rate, channels = info.samplerate, info.channels
            if start + 2 / rate < prior_end:
                raise ValueError('Overlapping native segments; refusing to silently overwrite samples')
            timing_evidence.append(_verify_timing(manifest_path.parent, seg, source_id, info))
            if start - prior_end > 2 / rate:
                all_gaps.append({'source_id': source_id, 'start_seconds': prior_end, 'end_seconds': start, 'reason': 'capture_gap'})
            end = start + info.duration
            prior_end, max_end = end, max(max_end, end)
            items.append((start, path, info.frames, _sha(path)))
        for gap in track.get('gaps') or []:
            shifted = {**gap, 'source_id': source_id}
            # Native manifests use begin_seconds; ASR consumes a canonical range.
            if 'start_seconds' not in shifted and 'begin_seconds' in shifted:
                shifted['start_seconds'] = shifted['begin_seconds']
            shifted.pop('begin_seconds', None)
            if 'start_seconds' not in shifted or 'end_seconds' not in shifted:
                raise ValueError('Native gap has no bounded time range')
            for key in ('start_seconds', 'end_seconds'):
                if key in shifted:
                    shifted[key] = _finite(shifted[key], key, signed=True) - epoch_offset
                    if shifted[key] < 0:
                        raise ValueError('Native gap precedes the recoverable timeline')
            if shifted['end_seconds'] < shifted['start_seconds']:
                raise ValueError('Native gap ends before it begins')
            shifted['duration_seconds'] = shifted['end_seconds'] - shifted['start_seconds']
            all_gaps.append(shifted)
        prepared.append((source_id, rate, channels, items, prior_end))
    if not prepared:
        raise ValueError('No captured audio can be recovered')
    for gap in all_gaps:
        if gap.get('reason') == 'source_missing':
            gap['end_seconds'] = max_end
    # Full common duration, including explicit silence/gaps. No sample-zero resets.
    for source_id, rate, channels, items, last_end in prepared:
        fingerprint = hashlib.sha256(json.dumps({'items': [(x[0], x[2], x[3]) for x in items], 'duration': max_end, 'rate': rate, 'channels': channels}).encode()).hexdigest()[:20]
        target = output_dir / f'{source_id}-{fingerprint}.wav'
        tmp = target.with_suffix('.building.wav')
        with sf.SoundFile(tmp, 'w', samplerate=rate, channels=channels, subtype='FLOAT', format='WAV') as dest:
            position = 0
            zeros = np.zeros((min(rate, 65536), channels), dtype=np.float32)
            for start, path, frames, digest in items:
                target_frame = round(start * rate)
                if target_frame < position:
                    # Subsample rounding differences may touch, never crop a segment.
                    if position - target_frame > 2:
                        raise ValueError('Rounded segment overlap')
                    target_frame = position
                while position < target_frame:
                    n = min(len(zeros), target_frame-position); dest.write(zeros[:n]); position += n
                with sf.SoundFile(path) as src:
                    for block in src.blocks(blocksize=65536, dtype='float32', always_2d=True):
                        dest.write(block); position += len(block)
            total = round(max_end * rate)
            while position < total:
                n = min(len(zeros), total-position); dest.write(zeros[:n]); position += n
        os.replace(tmp, target)
        if max_end - last_end > 2 / rate:
            all_gaps.append({'source_id': source_id, 'start_seconds': last_end, 'end_seconds': max_end, 'reason': 'source_ended_early'})
        sources.append({'source_id': source_id, 'path': str(target), 'start_seconds': 0.0, 'timing_basis': 'host_clock'})
    metadata = {'native_manifest_path': str(manifest_path.resolve()), 'native_manifest_sha256': _sha(manifest_path), 'capture_status': manifest.get('status'), 'capture_errors': manifest.get('errors') or [], 'capture_gaps': all_gaps, 'duration_seconds': max_end, 'timeline_verified': True, 'native_epoch_offset_seconds': epoch_offset, 'native_timing_evidence': timing_evidence}
    atomic_json(output_dir / 'source-inputs.json', {'sources': sources, '_meta': metadata})
    return sources, metadata


def prepare_recording_sources(audio_path: Path, output_dir: Path, *, manifest_path: Path | None = None, offsets: dict | None = None, recover_stopped=False) -> tuple[list[dict], dict]:
    audio_path, output_dir = Path(audio_path), Path(output_dir)
    sidecar = audio_path.with_suffix('.capture.json')
    if manifest_path is None and sidecar.exists():
        binding = json.loads(sidecar.read_text())
        if binding.get('schema_version') != 2:
            raise ValueError('Unsupported native capture binding')
        manifest_path = Path(binding['native_manifest_path'])
        if binding.get('native_manifest_sha256') != _sha(manifest_path):
            raise ValueError('Native capture binding manifest hash mismatch')
        if binding.get('compatibility_audio_sha256') != _sha(audio_path):
            raise ValueError('Native capture binding audio hash mismatch')
    if manifest_path:
        if offsets:
            raise ValueError('Explicit offsets cannot override native timestamps')
        return prepare_native_sources(manifest_path, output_dir, recover_stopped=recover_stopped)
    if recover_stopped:
        raise ValueError('Stopped-native recovery requires an explicit native manifest')
    from audio_converter import extract_source_tracks
    sources = extract_source_tracks(audio_path, output_dir)
    offsets = offsets or {}
    if set(offsets) - {s['source_id'] for s in sources}:
        raise ValueError('Offset specified for absent audio source')
    for source in sources:
        if source['source_id'] in offsets:
            source['start_seconds'] = _finite(offsets[source['source_id']], 'source offset')
            source['timing_basis'] = 'manual_offset_unverified'
    metadata = {'source_audio_path': str(audio_path.resolve()), 'source_audio_sha256': _sha(audio_path), 'timeline_verified': False, 'capture_gaps': [], 'timing_warning': 'Legacy files have no shared first-sample timestamp; offsets are not identity evidence.'}
    atomic_json(output_dir / 'source-inputs.json', {'sources': sources, '_meta': metadata})
    return sources, metadata
