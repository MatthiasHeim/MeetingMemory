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


class Address(c.Structure):
    _fields_ = [("selector", c.c_uint32), ("scope", c.c_uint32), ("element", c.c_uint32)]


def recorder_status():
    job = subprocess.check_output(["/bin/launchctl", "print", f"gui/{os.getuid()}/com.user.meetingrecorder"], text=True)
    match = re.search(r"\bpid = (\d+)", job)
    if not match:
        return {"state": "unknown", "reason": "recorder process not running"}
    pid = int(match.group(1))
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
