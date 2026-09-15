#!/usr/bin/env python3
"""Conservatively retire raw captures only after a certified FLAC replacement.

Raw audio is source evidence. An MP3 and transcript JSON are useful derived
artifacts, but neither is a lossless replacement for the original merged WAV
or its separately archived microphone/system tracks. This pruner therefore
preserves every candidate by default.

The sole deletion path requires all of the following:

* the recording is old enough and has a complete, non-partial transcript;
* no associated ``.hold`` marker exists;
* ``Transcripts/<stem>.source-replacement.json`` is a complete v1 manifest;
* the manifest has an entry for this exact original path and SHA-256;
* its replacement is a separate, existing FLAC outside ``Recordings`` and
  ``CaptureArchive``; and
* a fresh streaming decode verifies matching PCM hash, sample count, sample
  rate, and channel count for original and replacement.

Use ``--verify-lossless-replacement ORIGINAL REPLACEMENT`` to create one
manifest entry during a deliberate migration. It never writes or deletes
anything. Operators must review and assemble those entries into the manifest
described in ``docs/source-preservation.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

MIN_MP3_BYTES = 1_000_000
REPLACEMENT_MANIFEST_VERSION = 1
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
FFPROBE_PATH = "/opt/homebrew/bin/ffprobe"


def _stem(path: Path) -> str:
    # "2026-09-10_14-01-11.mic.wav" -> "2026-09-10_14-01-11"
    return path.name.split(".")[0]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    """Hash one file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_info(path: Path) -> dict[str, Any]:
    """Return the one audio stream's immutable decode parameters."""
    result = subprocess.run(
        [
            FFPROBE_PATH, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_fmt,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-400:]}")
    try:
        streams = json.loads(result.stdout).get("streams")
        stream = streams[0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No usable audio stream in {path}") from exc
    if sample_rate <= 0 or channels <= 0:
        raise RuntimeError(f"Invalid audio stream metadata in {path}")
    return {
        "codec_name": str(stream.get("codec_name") or ""),
        "sample_fmt": str(stream.get("sample_fmt") or ""),
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _decoded_audio_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint decoded integer PCM while streaming it through ffmpeg.

    FLAC can preserve PCM integer samples exactly. Canonical s32le output lets
    a WAV and FLAC representation of the same 16- or 24-bit samples compare
    byte-for-byte without trusting metadata supplied by a manifest. Float
    sources are intentionally rejected: this verifier must not call a rounded
    float conversion an independently proven lossless replacement.
    """
    info = _stream_info(path)
    sample_fmt = info["sample_fmt"].lower()
    if "flt" in sample_fmt or "dbl" in sample_fmt:
        raise RuntimeError(
            f"Cannot certify floating-point source losslessly: {path} ({sample_fmt})"
        )

    digest = hashlib.sha256()
    decoded_bytes = 0
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(
            [
                FFMPEG_PATH, "-nostdin", "-v", "error", "-i", str(path),
                "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-c:a", "pcm_s32le", "-f", "s32le", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr,
            bufsize=0,
        )
        assert process.stdout is not None
        try:
            while True:
                chunk = process.stdout.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                decoded_bytes += len(chunk)
        finally:
            process.stdout.close()
        return_code = process.wait()
        stderr.seek(0)
        error = stderr.read(400).decode(errors="replace")
    if return_code != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {error}")

    bytes_per_frame = 4 * info["channels"]
    if decoded_bytes % bytes_per_frame:
        raise RuntimeError(f"Decoded audio has an incomplete sample frame: {path}")
    return {
        "sample_count": decoded_bytes // bytes_per_frame,
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "pcm_sha256": digest.hexdigest(),
    }


def _same_fingerprint(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in ("sample_count", "sample_rate", "channels", "pcm_sha256")
    )


def verify_lossless_replacement(original: Path, replacement: Path) -> dict[str, Any]:
    """Return a v1 certified-replacement entry after independent verification.

    This is intentionally an explicit, read-only migration action. It does
    not create a manifest, move data, or permit pruning by itself. Callers
    must preserve the returned entry verbatim in the per-recording manifest.
    """
    original = Path(original).resolve(strict=True)
    replacement = Path(replacement).resolve(strict=True)
    if os.path.samefile(original, replacement):
        raise ValueError("Replacement must be a separately stored file")
    replacement_info = _stream_info(replacement)
    if replacement_info["codec_name"] != "flac":
        raise ValueError("Certified replacement must be FLAC")

    original_decode = _decoded_audio_fingerprint(original)
    replacement_decode = _decoded_audio_fingerprint(replacement)
    if not _same_fingerprint(original_decode, replacement_decode):
        raise ValueError(
            "Replacement decoded PCM differs from the original "
            "(sample count, rate, channels, or PCM hash)"
        )
    return {
        "original_path": str(original),
        "original_sha256": _sha256(original),
        "replacement_path": str(replacement),
        "replacement_sha256": _sha256(replacement),
        "lossless_decode": {
            "verified": True,
            "original": original_decode,
            "replacement": replacement_decode,
        },
    }


def _replacement_manifest_path(transcripts: Path, stem: str) -> Path:
    return transcripts / f"{stem}.source-replacement.json"


def _read_replacement_manifest(path: Path) -> tuple[Optional[dict[str, Any]], str]:
    if not path.is_file():
        return None, "no_verified_lossless_replacement"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "malformed_replacement_manifest"
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != REPLACEMENT_MANIFEST_VERSION
        or not isinstance(data.get("replacements"), list)
        or not data["replacements"]
        or not all(isinstance(entry, dict) for entry in data["replacements"])
    ):
        return None, "malformed_replacement_manifest"
    return data, "ok"


def _resolve_manifest_path(
    value: Any,
    root: Path,
    *,
    require_exists: bool = True,
) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=require_exists)
    except (OSError, RuntimeError):
        return None


def _associated_hold_exists(wav: Path, root: Path, recordings: Path) -> bool:
    """Any clear hold marker wins for the merged file and its companions."""
    stem = _stem(wav)
    markers = {
        wav.with_name(wav.name + ".hold"),
        wav.parent / f"{stem}.hold",
        wav.parent / ".hold",
        recordings / f"{stem}.wav.hold",
        recordings / f"{stem}.hold",
    }
    archive = root / "CaptureArchive"
    if _is_within(wav, archive):
        markers.add(archive / stem / ".hold")
        markers.add(archive / stem / f"{stem}.hold")
    return any(marker.exists() for marker in markers)


def _transcript_complete(transcripts: Path, stem: str) -> tuple[bool, str]:
    """Return whether the ordinary derived transcript was a verified success."""
    mp3 = transcripts / f"{stem}.mp3"
    if not mp3.is_file() or mp3.stat().st_size < MIN_MP3_BYTES:
        return False, "no_mp3"

    transcript = transcripts / f"{stem}.json"
    if not transcript.is_file():
        return False, "no_transcript_json"
    try:
        data = json.loads(transcript.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "malformed_transcript_json"
    if not isinstance(data, dict):
        return False, "malformed_transcript_json"

    meta = data.get("_meta")
    if not isinstance(meta, dict):
        return False, "unverified_transcript_json"
    if meta.get("partial") is not False or meta.get("missing_time_ranges"):
        return False, "partial_transcript"
    validation = meta.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        return False, "unverified_transcript_json"
    return True, "ok"


def _certified_replacement(
    wav: Path,
    root: Path,
    transcripts: Path,
) -> tuple[bool, str]:
    """Validate the manifest entry and re-check its evidence from disk."""
    manifest, reason = _read_replacement_manifest(
        _replacement_manifest_path(transcripts, _stem(wav))
    )
    if manifest is None:
        return False, reason

    candidate = wav.resolve()
    entries = []
    for entry in manifest["replacements"]:
        # Entries for earlier siblings may already have been deleted in this
        # prune pass. Their absent original must not invalidate the remaining
        # independently certified entry; the matching candidate is still
        # hashed and decoded below before it can be removed.
        original_path = _resolve_manifest_path(
            entry.get("original_path"), root, require_exists=False
        )
        if original_path is None:
            return False, "malformed_replacement_manifest"
        if original_path == candidate:
            entries.append(entry)
    if not entries:
        return False, "no_verified_lossless_replacement"
    if len(entries) != 1:
        return False, "ambiguous_replacement_manifest"

    entry = entries[0]
    required = ("original_sha256", "replacement_path", "replacement_sha256", "lossless_decode")
    if any(field not in entry for field in required):
        return False, "malformed_replacement_manifest"
    if not isinstance(entry["lossless_decode"], dict):
        return False, "malformed_replacement_manifest"
    proof = entry["lossless_decode"]
    if proof.get("verified") is not True:
        return False, "no_verified_lossless_replacement"
    if not isinstance(proof.get("original"), dict) or not isinstance(proof.get("replacement"), dict):
        return False, "malformed_replacement_manifest"

    replacement = _resolve_manifest_path(entry["replacement_path"], root)
    if replacement is None or not replacement.is_file():
        return False, "replacement_path_invalid"
    candidate_roots = (root / "Recordings", root / "CaptureArchive")
    if any(_is_within(replacement, candidate_root) for candidate_root in candidate_roots):
        return False, "replacement_inside_prune_scope"
    try:
        if os.path.samefile(candidate, replacement):
            return False, "replacement_not_independent"
    except OSError:
        return False, "replacement_path_invalid"

    try:
        if _stream_info(replacement)["codec_name"] != "flac":
            return False, "replacement_not_flac"
        current_original_sha = _sha256(candidate)
        current_replacement_sha = _sha256(replacement)
        if current_original_sha != entry["original_sha256"]:
            return False, "original_hash_mismatch"
        if current_replacement_sha != entry["replacement_sha256"]:
            return False, "replacement_hash_mismatch"
        current_original_decode = _decoded_audio_fingerprint(candidate)
        current_replacement_decode = _decoded_audio_fingerprint(replacement)
    except (OSError, RuntimeError):
        return False, "lossless_decode_verification_failed"

    if (
        not _same_fingerprint(proof["original"], current_original_decode)
        or not _same_fingerprint(proof["replacement"], current_replacement_decode)
        or not _same_fingerprint(current_original_decode, current_replacement_decode)
    ):
        return False, "lossless_decode_mismatch"
    return True, "ok"


def eligible(wav: Path, transcripts: Path, recordings: Path, days: float, now: float) -> tuple[bool, str]:
    """Return whether one raw source can be removed under the strict policy."""
    root = recordings.parent
    if _associated_hold_exists(wav, root, recordings):
        return False, "hold_marker"
    if now - wav.stat().st_mtime < days * 86400:
        return False, "too_recent"
    complete, reason = _transcript_complete(transcripts, _stem(wav))
    if not complete:
        return False, reason
    return _certified_replacement(wav, root, transcripts)


def prune(root: Path, days: float, dry_run: bool, now: float | None = None) -> list[dict]:
    """Prune only individually certified files; never recursively remove folders."""
    root = Path(root)
    now = time.time() if now is None else now
    recordings, transcripts, archive = (
        root / "Recordings",
        root / "Transcripts",
        root / "CaptureArchive",
    )
    candidates = sorted(recordings.glob("*.wav")) + sorted(archive.rglob("*.wav"))
    records = []
    for wav in candidates:
        try:
            size = wav.stat().st_size
            ok, reason = eligible(wav, transcripts, recordings, days, now)
        except FileNotFoundError:
            continue
        rec = {
            "action": "prune_wav",
            "path": str(wav),
            "bytes": size,
            "deleted": False,
            "reason": reason,
        }
        if ok and not dry_run:
            try:
                wav.unlink()
                rec["deleted"] = True
            except FileNotFoundError:
                rec["reason"] = "source_disappeared"
        elif ok:
            rec["reason"] = "dry_run"
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.home() / "Documents/MeetingRecorder")
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-lossless-replacement",
        metavar=("ORIGINAL", "REPLACEMENT"),
        nargs=2,
        help="Print a read-only certified-replacement manifest entry; never deletes files.",
    )
    args = parser.parse_args()
    if args.verify_lossless_replacement:
        original, replacement = map(Path, args.verify_lossless_replacement)
        print(json.dumps(verify_lossless_replacement(original, replacement), indent=2, sort_keys=True))
        return

    records = prune(args.root, args.days, args.dry_run)
    for rec in records:
        if rec["deleted"] or rec["reason"] == "dry_run":
            print(json.dumps(rec), flush=True)
    freed = sum(r["bytes"] for r in records if r["deleted"])
    print(json.dumps({
        "action": "prune_summary",
        "deleted": sum(r["deleted"] for r in records),
        "bytes_logical": freed,
        "dry_run": args.dry_run,
        "free_disk_gb": round(shutil.disk_usage(args.root).free / 1e9, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
