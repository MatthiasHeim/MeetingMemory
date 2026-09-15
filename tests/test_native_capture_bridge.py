import json
import os
import stat
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import native_capture_bridge  # noqa: E402
from native_capture_bridge import (  # noqa: E402
    NativeCaptureError,
    NativeCaptureSession,
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def test_status_only_reads_atomic_manifest(tmp_path, monkeypatch):
    session = NativeCaptureSession(tmp_path / "MeetingNativeCapture.app", tmp_path / "session")
    session.session_dir.mkdir()
    atomic_json(
        session.manifest_path,
        {
            "schema_version": 2,
            "capture_backend": "screencapturekit",
            "status": "degraded",
            "finalized": False,
            "session_token": session._session_token,
            "tracks": [
                {
                    "source_id": "system",
                    "segments": [{"path": "segments/system-0001.caf", "start_seconds": 46.28, "duration_seconds": 20}],
                    "gaps": [{"begin_seconds": 0, "end_seconds": 46.28, "duration_seconds": 46.28}],
                }
            ],
        },
    )

    def forbidden_process_probe(*_args, **_kwargs):
        raise AssertionError("status() must not poll processes")

    monkeypatch.setattr("native_capture_bridge.subprocess.run", forbidden_process_probe)
    status = session.status()

    assert status["status"] == "degraded"
    assert status["finalized"] is False
    assert status["manifest_path"] == str(session.manifest_path)
    # Segment paths deliberately remain session-relative for portable manifests.
    assert status["tracks"][0]["segments"][0]["path"] == "segments/system-0001.caf"


def test_session_token_cannot_be_mistaken_for_an_option(tmp_path, monkeypatch):
    monkeypatch.setattr(native_capture_bridge.secrets, "token_urlsafe", lambda _size: "-rare-leading-dash")

    session = NativeCaptureSession(tmp_path / "MeetingNativeCapture.app", tmp_path / "session")

    assert session._session_token == "session--rare-leading-dash"


def write_fake_bundle(bundle: Path) -> None:
    executable = bundle / "Contents" / "MacOS" / "native_capture"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        """#!{python}
import argparse
import json
import os
import signal
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True)
parser.add_argument('--session-token', required=True)
parser.add_argument('--owner-pid', required=True, type=int)
parser.add_argument('--duration')
args = parser.parse_args()
root = Path(args.output_dir)
root.mkdir(parents=True, exist_ok=True)
manifest = root / 'capture-manifest.json'

def publish(finalized):
    payload = {{
        'schema_version': 2,
        'capture_backend': 'screencapturekit',
        'status': 'degraded',
        'finalized': finalized,
        'session_id': 'fake-session',
        'session_token': args.session_token,
        'process': {{'pid': os.getpid(), 'executable': __file__}},
        'owner_process': {{'pid': args.owner_pid, 'monitor': 'watching'}},
        'readiness': {{'mic': True, 'system': False, 'all_required_sources_ready': False}},
        'tracks': [
            {{'source_id': 'mic', 'first_buffer_ready': True, 'segments': [], 'gaps': []}},
            {{'source_id': 'system', 'first_buffer_ready': False, 'segments': [], 'gaps': []}},
        ],
        'errors': [{{'code': 'source_not_ready'}}],
    }}
    temporary = manifest.with_name(manifest.name + '.tmp')
    temporary.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(temporary, manifest)

publish(False)
def stop(_signum, _frame):
    publish(True)
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.05)
""".format(python=sys.executable),
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)


def test_direct_bridge_waits_for_finalized_manifest_and_reaps_child(tmp_path):
    bundle = tmp_path / "MeetingNativeCapture.app"
    write_fake_bundle(bundle)
    session = NativeCaptureSession(bundle, tmp_path / "session", launch_mode="direct")

    started = session.start(timeout_seconds=3)
    assert started["status"] == "degraded"
    assert started["finalized"] is False
    assert started["readiness"]["mic"] is True
    assert started["readiness"]["system"] is False
    assert started["owner_process"]["pid"] == os.getpid()

    stopped = session.stop(timeout_seconds=3)
    assert stopped["status"] == "degraded"
    assert stopped["finalized"] is True
    assert session._direct_process is not None
    assert session._direct_process.poll() == 0


def test_stop_refuses_manifest_from_a_different_session(tmp_path):
    bundle = tmp_path / "MeetingNativeCapture.app"
    executable = bundle / "Contents" / "MacOS" / "native_capture"
    executable.parent.mkdir(parents=True)
    executable.write_text("placeholder", encoding="utf-8")
    session = NativeCaptureSession(bundle, tmp_path / "session")
    session.session_dir.mkdir()
    session._started = True
    atomic_json(
        session.manifest_path,
        {
            "schema_version": 2,
            "status": "recording",
            "finalized": False,
            "session_id": "another-session",
            "session_token": "not-this-bridge",
            "process": {"pid": os.getpid()},
        },
    )

    with pytest.raises(NativeCaptureError, match="does not belong"):
        session.stop(timeout_seconds=1)


def test_synthetic_gap_manifest_preserves_common_coordinates_for_recovery(tmp_path):
    session = NativeCaptureSession(tmp_path / "MeetingNativeCapture.app", tmp_path / "session")
    session.session_dir.mkdir()
    atomic_json(
        session.manifest_path,
        {
            "schema_version": 2,
            "capture_backend": "screencapturekit",
            "status": "degraded",
            "finalized": True,
            "session_id": "synthetic-recovery",
            "session_token": session._session_token,
            "timeline": "host_clock",
            "timeline_metadata": {"epoch_raw_pts": {"value": 2_000_000_000, "timescale": 1_000_000_000}},
            "final_duration_seconds": 66.28,
            "tracks": [
                {
                    "source_id": "mic",
                    "first_buffer_ready": True,
                    "segments": [{"path": "segments/mic-0001.caf", "start_seconds": -0.1, "duration_seconds": 66.38}],
                    "gaps": [],
                },
                {
                    "source_id": "system",
                    "first_buffer_ready": True,
                    "segments": [{"path": "segments/system-0001.caf", "start_seconds": 46.28, "duration_seconds": 20}],
                    "gaps": [
                        {
                            "begin_seconds": 0.0,
                            "end_seconds": 46.28,
                            "duration_seconds": 46.28,
                            "reason": "initial_no_samples",
                        }
                    ],
                },
            ],
            "errors": [],
        },
    )

    manifest = session.status()
    system = next(track for track in manifest["tracks"] if track["source_id"] == "system")
    assert manifest["finalized"] is True
    assert manifest["final_duration_seconds"] == 66.28
    assert system["segments"][0]["start_seconds"] == 46.28
    assert system["gaps"][0]["begin_seconds"] == 0.0
    assert system["gaps"][0]["end_seconds"] == 46.28
    # A caller can apply one global +0.1 shift; individual source zeroing
    # would incorrectly erase the proven 46.28-second system-source delay.
    assert manifest["tracks"][0]["segments"][0]["start_seconds"] == -0.1
