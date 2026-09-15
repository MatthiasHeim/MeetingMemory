#!/usr/bin/env python3
"""Read the recorder's CoreAudio input activity without touching capture.

Exit 0 = recorder input idle, 1 = active, 2 = unknown. Never infer recorder
activity from another app using the microphone or from the system tap alone.
Selectors are defined by Apple's AudioHardware.h (macOS 14.2+ process API).
"""
import ctypes as c
import json
import os
import re
import subprocess
from pathlib import Path


class Address(c.Structure):
    _fields_ = [("selector", c.c_uint32), ("scope", c.c_uint32), ("element", c.c_uint32)]


def _native_manifest_process_alive(manifest):
    from native_capture_bridge import _process_command
    process = manifest.get('process') or {}
    pid = int(process.get('pid', 0))
    executable = process.get('executable')
    token = manifest.get('session_token')
    if pid <= 0 or not executable or not token:
        return False
    command = _process_command(pid) or ''
    return command.startswith(executable + ' ') and ('--session-token ' + token) in command


def native_activity_status(activity_file, recorder_pid):
    """A native child owns audio; querying only the Python PID would lie idle."""
    activity_file = Path(activity_file)
    if not activity_file.exists():
        return None
    try:
        activity = json.loads(activity_file.read_text())
        manifest = json.loads(Path(activity['native_manifest_path']).read_text())
        state = manifest.get('status')
        if state in ('starting', 'recording', 'degraded') and not manifest.get('finalized'):
            if not _native_manifest_process_alive(manifest):
                return {'pid': recorder_pid, 'state': 'unknown',
                        'reason': 'unfinalized native capture has no matching live process; recover retained segments'}
            return {'pid': recorder_pid, 'state': 'active', 'capture_backend': 'screencapturekit',
                    'reason': 'native capture has not finalized'}
        if activity.get('status') == 'stopped' and manifest.get('finalized'):
            return None  # Native is finalized; still inspect current mic activity.
        if manifest.get('finalized'):
            native_pid = int((manifest.get('process') or {}).get('pid', 0))
            if native_pid > 0:
                try:
                    os.kill(native_pid, 0)
                except ProcessLookupError:
                    return None  # Owner died, native finalized and exited.
                except PermissionError:
                    pass  # An inaccessible process is not proof of inactivity.
        return {'pid': recorder_pid, 'state': 'unknown', 'reason': 'native capture state is unresolved'}
    except (OSError, ValueError, KeyError, TypeError):
        return {'pid': recorder_pid, 'state': 'unknown', 'reason': 'native activity evidence is incomplete'}


def recorder_status():
    job = subprocess.check_output(["/bin/launchctl", "print", f"gui/{os.getuid()}/com.user.meetingrecorder"], text=True)
    match = re.search(r"\bpid = (\d+)", job)
    if not match:
        return {"state": "unknown", "reason": "recorder process not running"}
    pid = int(match.group(1))
    native = native_activity_status(Path.home() / 'Documents/MeetingRecorder/active-native-capture.json', pid)
    if native is not None:
        return native
    library = c.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    def get(obj, selector, many=False):
        four = lambda s: int.from_bytes(s.encode(), "big")
        address = Address(four(selector), four("glob"), 0)
        size = c.c_uint32()
        status = library.AudioObjectGetPropertyDataSize(obj, c.byref(address), 0, None, c.byref(size))
        if status:
            return None
        data = (c.c_uint32 * (size.value//4))()
        status = library.AudioObjectGetPropertyData(obj, c.byref(address), 0, None, c.byref(size), c.byref(data))
        if status or not len(data):
            return None
        return list(data) if many else data[0]
    for obj in get(1, "prs#", True) or []:
        if get(obj, "ppid") == pid:
            active = get(obj, "piri")
            return {"pid": pid, "state": "unknown" if active is None else ("active" if active else "idle")}
    return {"pid": pid, "state": "unknown", "reason": "no process activity object"}


if __name__ == "__main__":
    try:
        report = recorder_status()
    except Exception as exc:
        report = {"state": "unknown", "reason": str(exc)}
    print(json.dumps(report))
    raise SystemExit({"idle": 0, "active": 1, "unknown": 2}[report["state"]])
