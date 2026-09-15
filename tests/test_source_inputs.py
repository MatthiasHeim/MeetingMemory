import json
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from source_inputs import prepare_native_sources


def make_manifest(tmp_path, offset=2):
    segments=[]
    for name,start in [('mic',0),('system',offset)]:
        path=tmp_path/f'{name}.wav';sf.write(path,np.full((8000,1),.2),8000,subtype='FLOAT')
        segments.append({'source_id':name,'segments':[{'path':path.name,'start_seconds':start,'duration_seconds':1,'sample_rate':8000,'channels':1,'frames':8000}]})
    p=tmp_path/'capture-manifest.json';p.write_text(json.dumps({'schema_version':2,'capture_backend':'screencapturekit','timeline':'host_clock','status':'complete','finalized':True,'tracks':segments}))
    write_timing(p)
    return p


def write_timing(path):
    data = json.loads(path.read_text())
    for track in data['tracks']:
        for index, seg in enumerate(track['segments']):
            name = f"{track['source_id']}-{index}.timing.json"
            seg['timing_path'] = name
            (path.parent / name).write_text(json.dumps({'schema_version': 1, 'source_id': track['source_id'],
                'segment_index': index, 'entries': [{'start_seconds': seg['start_seconds'], 'written_frames': seg['frames']}]}))
    path.write_text(json.dumps(data))


def test_delayed_source_is_padded_at_start_not_relabelled_zero(tmp_path):
    p=make_manifest(tmp_path);sources,meta=prepare_native_sources(p,tmp_path/'derived')
    mic,_=sf.read(sources[0]['path']);system,_=sf.read(sources[1]['path'])
    assert len(mic)==len(system)==24000
    assert np.allclose(system[:16000],0) and np.allclose(system[16000:],.2)
    assert np.allclose(mic[:8000],.2) and np.allclose(mic[8000:],0)
    assert meta['capture_gaps'][0]['end_seconds']==2
    assert all(s['timing_basis']=='host_clock' for s in sources)


def test_reject_active_and_path_escape(tmp_path):
    p=make_manifest(tmp_path);d=json.loads(p.read_text());d['status']='recording';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='active'):prepare_native_sources(p,tmp_path/'out')
    d['status']='complete';d['tracks'][0]['segments'][0]['path']='../outside.wav';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='inside'):prepare_native_sources(p,tmp_path/'out')


def test_reject_overlap_or_wrong_frame_count(tmp_path):
    p=make_manifest(tmp_path);d=json.loads(p.read_text());d['tracks'][0]['segments'][0]['frames']=1;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='metadata'):prepare_native_sources(p,tmp_path/'out')
    d['tracks'][0]['segments'][0]['frames']=8000;d['tracks'][0]['segments'].append(dict(d['tracks'][0]['segments'][0],start_seconds=.5));p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Overlapping'):prepare_native_sources(p,tmp_path/'out')


def test_negative_preroll_uses_one_shared_origin(tmp_path):
    p=make_manifest(tmp_path,offset=0)
    d=json.loads(p.read_text());d['tracks'][0]['segments'][0]['start_seconds']=-.25
    p.write_text(json.dumps(d))
    write_timing(p)
    sources,meta=prepare_native_sources(p,tmp_path/'out')
    mic,_=sf.read(sources[0]['path']);system,_=sf.read(sources[1]['path'])
    assert meta['native_epoch_offset_seconds']==-.25
    assert len(mic)==len(system)==10000
    assert np.allclose(mic[:8000],.2)
    assert np.allclose(system[:2000],0) and np.allclose(system[2000:],.2)


def test_native_gap_alias_shifts_with_common_epoch(tmp_path):
    p = make_manifest(tmp_path)
    data = json.loads(p.read_text())
    data['tracks'][0]['segments'][0]['start_seconds'] = -.25
    data['tracks'][1]['gaps'] = [{'begin_seconds': 0, 'end_seconds': 2,
                                 'reason': 'late_first_packet'}]
    p.write_text(json.dumps(data))
    write_timing(p)
    _, meta = prepare_native_sources(p, tmp_path / 'out')
    gap = next(g for g in meta['capture_gaps'] if g['reason'] == 'late_first_packet')
    assert gap['start_seconds'] == .25
    assert gap['end_seconds'] == 2.25
    assert gap['duration_seconds'] == 2
    assert 'begin_seconds' not in gap


def test_crash_recovery_requires_dead_process_and_keeps_unknown_tail(tmp_path, monkeypatch):
    import source_inputs
    p = make_manifest(tmp_path)
    data = json.loads(p.read_text())
    data.update(status='recording', finalized=False, process={'pid': 123})
    p.write_text(json.dumps(data))
    original = p.read_bytes()
    monkeypatch.setattr(source_inputs.os, 'kill', lambda *a: None)
    with pytest.raises(ValueError, match='still live'):
        prepare_native_sources(p, tmp_path / 'out', recover_stopped=True)
    monkeypatch.setattr(source_inputs.os, 'kill', lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    sources, meta = prepare_native_sources(p, tmp_path / 'out', recover_stopped=True)
    assert len(sources) == 2 and meta['capture_status'] == 'crash_recovered'
    assert meta['capture_errors'][0]['code'] == 'unfinalized_capture_recovery'
    assert p.read_bytes() == original


def test_timing_sidecar_corruption_is_not_certified(tmp_path):
    p = make_manifest(tmp_path)
    timing = tmp_path / 'mic-0.timing.json'
    data = json.loads(timing.read_text())
    data['entries'][0]['written_frames'] = 1
    timing.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='timing frame count'):
        prepare_native_sources(p, tmp_path / 'out')
    data['entries'][0]['written_frames'] = 8000.5
    timing.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='must be integers'):
        prepare_native_sources(p, tmp_path / 'out')
    timing.unlink()
    with pytest.raises(ValueError, match='Missing native segment'):
        prepare_native_sources(p, tmp_path / 'out')
