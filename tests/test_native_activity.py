import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from recorder_status import native_activity_status


def test_native_recording_never_claims_idle_python_parent(tmp_path, monkeypatch):
    import recorder_status
    monkeypatch.setattr(recorder_status, '_native_manifest_process_alive', lambda m: True)
    manifest=tmp_path/'capture.json';manifest.write_text('{"status":"recording"}')
    activity=tmp_path/'active.json';activity.write_text(json.dumps({'recorder_pid':123,'native_manifest_path':str(manifest),'status':'starting'}))
    assert native_activity_status(activity,123)['state']=='active'
    assert native_activity_status(activity,456)['state']=='active'
    manifest.unlink();assert native_activity_status(activity,123)['state']=='unknown'


def test_stale_manifest_does_not_claim_live_capture(tmp_path):
    manifest = tmp_path / 'capture.json'
    manifest.write_text('{"status":"recording"}')
    activity = tmp_path / 'active.json'
    activity.write_text(json.dumps({'native_manifest_path': str(manifest)}))
    assert native_activity_status(activity, 123)['state'] == 'unknown'


def test_owner_death_finalized_child_does_not_leave_permanent_active_state(tmp_path, monkeypatch):
    import recorder_status
    manifest = tmp_path / 'capture.json'
    manifest.write_text(json.dumps({'status': 'degraded', 'finalized': True, 'process': {'pid': 234}}))
    activity = tmp_path / 'active.json'
    activity.write_text(json.dumps({'native_manifest_path': str(manifest), 'status': 'starting'}))
    monkeypatch.setattr(recorder_status.os, 'kill', lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    assert native_activity_status(activity, 456) is None
