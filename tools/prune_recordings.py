#!/usr/bin/env python3
"""Delete raw WAV captures once their MP3 + transcript exist and they are old enough.

Retention policy (see README "Disk retention"):
  * Recordings/<stem>.wav is deleted when ALL hold:
      - mtime older than --days (default 7)
      - Transcripts/<stem>.mp3 exists and is > 1 MB (the long-term audio copy)
      - Transcripts/<stem>.json exists (the transcript was actually produced)
      - no Recordings/<stem>.wav.hold marker (manual hold wins)
  * CaptureArchive/<stem>/*.wav (raw mic/sys tracks) follow the same rule.
  * .tmp/ leftovers are never touched: they are evidence of failed captures.

Prints one JSON line per action to stdout (the recorder pipes this into
logs/capture-housekeeping.log) and never deletes anything under --dry-run.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

MIN_MP3_BYTES = 1_000_000


def _stem(path: Path) -> str:
    # "2026-09-10_14-01-11.mic.wav" -> "2026-09-10_14-01-11"
    return path.name.split(".")[0]


def eligible(wav: Path, transcripts: Path, recordings: Path, days: float, now: float) -> tuple[bool, str]:
    stem = _stem(wav)
    if (recordings / f"{stem}.wav.hold").exists():
        return False, "hold_marker"
    if now - wav.stat().st_mtime < days * 86400:
        return False, "too_recent"
    mp3 = transcripts / f"{stem}.mp3"
    if not mp3.exists() or mp3.stat().st_size < MIN_MP3_BYTES:
        return False, "no_mp3"
    if not (transcripts / f"{stem}.json").exists():
        return False, "no_transcript_json"
    return True, "ok"


def prune(root: Path, days: float, dry_run: bool, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    recordings, transcripts, archive = root / "Recordings", root / "Transcripts", root / "CaptureArchive"
    candidates = sorted(recordings.glob("*.wav")) + sorted(archive.rglob("*.wav"))
    records = []
    for wav in candidates:
        ok, reason = eligible(wav, transcripts, recordings, days, now)
        rec = {"action": "prune_wav", "path": str(wav), "bytes": wav.stat().st_size,
               "deleted": False, "reason": reason}
        if ok and not dry_run:
            wav.unlink()
            rec["deleted"] = True
            parent = wav.parent
            # Drop an archive folder once only capture.json remains.
            if parent.parent == archive and not any(parent.glob("*.wav")):
                shutil.rmtree(parent)
                rec["archive_dir_removed"] = str(parent)
        elif ok:
            rec["reason"] = "dry_run"
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.home() / "Documents/MeetingRecorder")
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = prune(args.root, args.days, args.dry_run)
    for rec in records:
        if rec["deleted"] or rec["reason"] == "dry_run":
            print(json.dumps(rec), flush=True)
    freed = sum(r["bytes"] for r in records if r["deleted"])
    print(json.dumps({"action": "prune_summary", "deleted": sum(r["deleted"] for r in records),
                      "bytes_logical": freed, "dry_run": args.dry_run,
                      "free_disk_gb": round(shutil.disk_usage(args.root).free / 1e9, 1)}), flush=True)


if __name__ == "__main__":
    main()
