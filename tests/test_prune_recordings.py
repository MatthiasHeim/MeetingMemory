import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prune_recordings

OLD = time.time() - 10 * 86400
BIG_MP3 = b"m" * (prune_recordings.MIN_MP3_BYTES + 1)


def _setup(tmp_path, stem="2026-09-01_10-00-00", mp3=True, js=True, hold=False, age=OLD):
    root = tmp_path
    for d in ("Recordings", "Transcripts", "CaptureArchive/" + stem, ".tmp"):
        (root / d).mkdir(parents=True, exist_ok=True)
    wav = root / "Recordings" / f"{stem}.wav"
    wav.write_bytes(b"w" * 2000)
    mic = root / "CaptureArchive" / stem / f"{stem}.mic.wav"
    mic.write_bytes(b"w" * 2000)
    (root / "CaptureArchive" / stem / "capture.json").write_text("{}")
    tmp = root / ".tmp" / f"{stem}.sys.wav"
    tmp.write_bytes(b"w")
    for p in (wav, mic, tmp):
        import os; os.utime(p, (age, age))
    if mp3:
        (root / "Transcripts" / f"{stem}.mp3").write_bytes(BIG_MP3)
    if js:
        (root / "Transcripts" / f"{stem}.json").write_text("{}")
    if hold:
        (root / "Recordings" / f"{stem}.wav.hold").write_text("")
    return root, wav, mic, tmp


def test_deletes_old_wav_and_archive_when_mp3_and_json_exist(tmp_path):
    root, wav, mic, tmp = _setup(tmp_path)
    recs = prune_recordings.prune(root, 7, dry_run=False)
    assert not wav.exists() and not mic.exists() and not mic.parent.exists()
    assert tmp.exists()  # .tmp leftovers never touched
    assert (root / "Transcripts").is_dir() and len(list((root / "Transcripts").iterdir())) == 2
    assert all(r["deleted"] for r in recs)


def test_dry_run_deletes_nothing(tmp_path):
    root, wav, mic, _ = _setup(tmp_path)
    recs = prune_recordings.prune(root, 7, dry_run=True)
    assert wav.exists() and mic.exists()
    assert {r["reason"] for r in recs} == {"dry_run"}


def test_keeps_recent_missing_mp3_missing_json_or_held(tmp_path):
    cases = {
        "too_recent": dict(age=time.time() - 86400),
        "no_mp3": dict(mp3=False),
        "no_transcript_json": dict(js=False),
        "hold_marker": dict(hold=True),
    }
    for i, (reason, kw) in enumerate(cases.items()):
        stem = f"2026-08-0{i+1}_10-00-00"
        root, wav, mic, _ = _setup(tmp_path, stem=stem, **kw)
        recs = [r for r in prune_recordings.prune(root, 7, dry_run=False) if stem in r["path"]]
        assert wav.exists() and mic.exists(), reason
        assert {r["reason"] for r in recs} == {reason}


def test_tiny_mp3_counts_as_missing(tmp_path):
    root, wav, _, _ = _setup(tmp_path, mp3=False)
    (root / "Transcripts" / f"{wav.stem}.mp3").write_bytes(b"x")
    prune_recordings.prune(root, 7, dry_run=False)
    assert wav.exists()
