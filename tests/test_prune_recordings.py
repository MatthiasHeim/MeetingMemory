"""Retention tests use only synthetic audio below tmp_path.

They exercise the fail-closed source-preservation contract; no test points at
the live MeetingRecorder archive or invokes the pruning CLI against it.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import prune_recordings
from audio_converter import FFMPEG_PATH, FFPROBE_PATH, extract_source_tracks

OLD = time.time() - 10 * 86400
BIG_MP3 = b"m" * (prune_recordings.MIN_MP3_BYTES + 1)
needs_ffmpeg = pytest.mark.skipif(
    not os.path.exists(FFMPEG_PATH) or not os.path.exists(FFPROBE_PATH),
    reason="ffmpeg/ffprobe not installed",
)


def _write_wav(path: Path, *, channels: int = 1, frames: int = 800, sample_rate: int = 8000) -> Path:
    """Create a tiny, valid integer PCM WAV with distinguishable samples."""
    payload = bytearray()
    for frame in range(frames):
        for channel in range(channels):
            value = int(12_000 * math.sin((frame + channel * 13) / 19))
            payload.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(payload)
    return path


def _complete_transcript() -> dict:
    return {
        "transcript": "Synthetic verified transcript.",
        "_meta": {
            "partial": False,
            "missing_time_ranges": [],
            "validation": {"passed": True},
        },
    }


def _setup(
    tmp_path: Path,
    stem: str = "2026-09-01_10-00-00",
    *,
    mp3: bool = True,
    transcript: object = None,
    hold: bool = False,
    age: float = OLD,
) -> tuple[Path, Path, Path, Path]:
    root = tmp_path
    for directory in ("Recordings", "Transcripts", f"CaptureArchive/{stem}", ".tmp"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    wav = _write_wav(root / "Recordings" / f"{stem}.wav", channels=1)
    mic = _write_wav(root / "CaptureArchive" / stem / f"{stem}.mic.wav", channels=1)
    (root / "CaptureArchive" / stem / "capture.json").write_text("{}")
    tmp = root / ".tmp" / f"{stem}.sys.wav"
    _write_wav(tmp, channels=1)
    for path in (wav, mic, tmp):
        os.utime(path, (age, age))
    if mp3:
        (root / "Transcripts" / f"{stem}.mp3").write_bytes(BIG_MP3)
    if transcript is None:
        transcript = _complete_transcript()
    if transcript is not False:
        transcript_path = root / "Transcripts" / f"{stem}.json"
        if isinstance(transcript, str):
            transcript_path.write_text(transcript)
        else:
            transcript_path.write_text(json.dumps(transcript))
    if hold:
        (root / "Recordings" / f"{stem}.wav.hold").write_text("")
    return root, wav, mic, tmp


def _write_certified_manifest(root: Path, stem: str, *originals: Path) -> Path:
    """Build a real FLAC certificate from synthetic sources only."""
    replacement_dir = root / "SourceTracks" / stem
    entries = []
    for original in originals:
        tracks = extract_source_tracks(original, replacement_dir)
        assert [track["source_id"] for track in tracks] == ["audio"]
        entries.append(
            prune_recordings.verify_lossless_replacement(
                original, Path(tracks[0]["path"])
            )
        )
    manifest = root / "Transcripts" / f"{stem}.source-replacement.json"
    manifest.write_text(json.dumps({
        "schema_version": prune_recordings.REPLACEMENT_MANIFEST_VERSION,
        "replacements": entries,
    }, indent=2))
    return manifest


def _records_for(records: list[dict], stem: str) -> list[dict]:
    return [record for record in records if stem in record["path"]]


def test_old_success_without_lossless_replacement_is_preserved(tmp_path):
    root, wav, mic, tmp = _setup(tmp_path)

    records = prune_recordings.prune(root, 7, dry_run=False)

    assert wav.exists() and mic.exists() and tmp.exists()
    assert {record["reason"] for record in records} == {"no_verified_lossless_replacement"}


def test_malformed_and_partial_transcript_json_fail_closed(tmp_path):
    malformed_root, malformed_wav, malformed_mic, _ = _setup(
        tmp_path / "malformed", transcript="{not valid json"
    )
    partial_root, partial_wav, partial_mic, _ = _setup(
        tmp_path / "partial",
        transcript={"_meta": {"partial": True, "missing_time_ranges": []}},
    )

    malformed = prune_recordings.prune(malformed_root, 7, dry_run=False)
    partial = prune_recordings.prune(partial_root, 7, dry_run=False)

    assert malformed_wav.exists() and malformed_mic.exists()
    assert {record["reason"] for record in malformed} == {"malformed_transcript_json"}
    assert partial_wav.exists() and partial_mic.exists()
    assert {record["reason"] for record in partial} == {"partial_transcript"}


def test_malformed_replacement_manifest_fails_closed(tmp_path):
    root, wav, mic, _ = _setup(tmp_path)
    (root / "Transcripts" / f"{wav.stem}.source-replacement.json").write_text("{")

    records = prune_recordings.prune(root, 7, dry_run=False)

    assert wav.exists() and mic.exists()
    assert {record["reason"] for record in records} == {"malformed_replacement_manifest"}


@needs_ffmpeg
def test_hold_wins_even_over_a_verified_replacement(tmp_path):
    root, wav, mic, _ = _setup(tmp_path, hold=True)
    _write_certified_manifest(root, wav.stem, wav, mic)

    records = prune_recordings.prune(root, 7, dry_run=False)

    assert wav.exists() and mic.exists()
    assert {record["reason"] for record in records} == {"hold_marker"}


@needs_ffmpeg
def test_verified_replacement_deletes_only_certified_originals(tmp_path):
    root, wav, mic, tmp = _setup(tmp_path)
    archive = mic.parent
    _write_certified_manifest(root, wav.stem, wav, mic)

    records = prune_recordings.prune(root, 7, dry_run=False)

    assert not wav.exists() and not mic.exists()
    assert tmp.exists()  # .tmp leftovers remain evidence of failed capture.
    # The pruner must not recursively remove an archive folder after the WAVs.
    assert archive.is_dir()
    assert (archive / "capture.json").exists()
    assert all(record["deleted"] for record in records)


@needs_ffmpeg
def test_tampered_replacement_protects_the_original(tmp_path):
    root, wav, mic, _ = _setup(tmp_path)
    manifest = _write_certified_manifest(root, wav.stem, wav)
    replacement = Path(json.loads(manifest.read_text())["replacements"][0]["replacement_path"])
    with replacement.open("ab") as stream:
        stream.write(b"tampered-after-certification")

    records = prune_recordings.prune(root, 7, dry_run=False)

    root_record = next(record for record in records if Path(record["path"]) == wav)
    assert wav.exists() and mic.exists()
    assert root_record["reason"] == "replacement_hash_mismatch"


@needs_ffmpeg
def test_root_deletion_keeps_unmapped_archive_tracks_and_companions(tmp_path):
    root, wav, mic, _ = _setup(tmp_path)
    archive = mic.parent
    unmapped = _write_wav(archive / f"{wav.stem}.system.wav", channels=2)
    os.utime(unmapped, (OLD, OLD))
    notes = archive / "operator-notes.txt"
    notes.write_text("keep this forensic note")
    _write_certified_manifest(root, wav.stem, wav)

    records = prune_recordings.prune(root, 7, dry_run=False)

    assert not wav.exists()
    assert mic.exists() and unmapped.exists()
    assert archive.is_dir() and notes.exists() and (archive / "capture.json").exists()
    remaining = _records_for(records, wav.stem)
    assert {record["reason"] for record in remaining if not record["deleted"]} == {
        "no_verified_lossless_replacement"
    }


@needs_ffmpeg
def test_dry_run_never_deletes_a_verified_source(tmp_path):
    root, wav, mic, _ = _setup(tmp_path)
    _write_certified_manifest(root, wav.stem, wav, mic)

    records = prune_recordings.prune(root, 7, dry_run=True)

    assert wav.exists() and mic.exists()
    assert {record["reason"] for record in records} == {"dry_run"}


def test_recent_missing_mp3_missing_json_or_held_are_preserved(tmp_path):
    cases = {
        "too_recent": dict(age=time.time() - 86400),
        "no_mp3": dict(mp3=False),
        "no_transcript_json": dict(transcript=False),
        "hold_marker": dict(hold=True),
    }
    for index, (reason, kwargs) in enumerate(cases.items()):
        stem = f"2026-08-0{index + 1}_10-00-00"
        root, wav, mic, _ = _setup(tmp_path / reason, stem=stem, **kwargs)
        records = _records_for(prune_recordings.prune(root, 7, dry_run=False), stem)
        assert wav.exists() and mic.exists(), reason
        assert {record["reason"] for record in records} == {reason}


def test_tiny_mp3_counts_as_missing(tmp_path):
    root, wav, mic, _ = _setup(tmp_path, mp3=False)
    (root / "Transcripts" / f"{wav.stem}.mp3").write_bytes(b"x")

    prune_recordings.prune(root, 7, dry_run=False)

    assert wav.exists() and mic.exists()
