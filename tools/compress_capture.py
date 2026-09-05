#!/usr/bin/env python3
"""Verified transparent APFS compression; file bytes and paths stay unchanged."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from speaker_integrity import atomic_json
from speaker_trial import DEFAULT_STATE, sha256


def compress(path: Path) -> dict:
    before = path.stat()
    tmp = path.with_name(path.name + ".compression.tmp")
    if tmp.exists():
        raise RuntimeError(f"Unfinished compression exists: {tmp}")
    if shutil.disk_usage(path).free < before.st_size + 150_000_000:
        return {"path": str(path), "skipped": "insufficient_copy_headroom"}
    try:
        subprocess.run(["/usr/bin/ditto", "--hfsCompression", str(path), str(tmp)], check=True, timeout=600)
        # ditto declines some large files. Force compression on the disposable
        # COPY only; afsctool verifies by default, and we independently hash the
        # decoded bytes below before any original is replaced.
        compressor = Path("/opt/homebrew/bin/afsctool")
        if compressor.exists() and tmp.stat().st_blocks >= before.st_blocks:
            subprocess.run([str(compressor), "-c", "-T", "LZFSE", str(tmp)],
                           check=True, capture_output=True, timeout=600)
        digest = sha256(path)
        if digest != sha256(tmp):
            raise RuntimeError("Compression byte verification failed")
        current = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (current.st_ino, current.st_size, current.st_mtime_ns):
            raise RuntimeError("Source changed during compression")
        saved = (before.st_blocks - tmp.stat().st_blocks)*512
        if saved <= 0:
            return {"path": str(path), "skipped": "no_storage_saving", "sha256": digest}
        os.replace(tmp, path)
        return {"path": str(path), "sha256": digest, "bytes": before.st_size,
                "bytes_saved": saved, "bytes_verified_identical": True}
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-free-gb", type=float, default=15)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    root = Path.home() / "Documents/MeetingRecorder"
    files = list((root/"Recordings").glob("*.wav")) + list((root/"CaptureArchive").rglob("*.wav")) + list((root/".tmp").glob("*.wav"))
    eligible = [p for p in files if time.time()-p.stat().st_mtime > 3600
                and p.stat().st_size > 1_000_000 and p.stat().st_blocks*512 > p.stat().st_size*0.8]
    # Small files first so early savings create safe headroom for large ones.
    records = []
    for path in sorted(eligible, key=lambda p:p.stat().st_size):
        if len(records) >= args.max_files or shutil.disk_usage(root).free >= args.target_free_gb*1e9:
            break
        rec = compress(path)
        records.append(rec)
        print(json.dumps(rec), flush=True)
    atomic_json(args.state / f"compression-{time.time_ns()}.json", {"files": records,
                "free_disk_gb": shutil.disk_usage(root).free/1e9})


if __name__ == "__main__":
    main()
