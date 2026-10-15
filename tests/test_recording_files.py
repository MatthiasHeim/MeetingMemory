"""Live transcript snapshot and recorder-confirmed speaker roster."""
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sidecar.live import LiveLine
from sidecar.live_session import RecordingLiveSession
from sidecar.recording_files import (
    calendar_speaker_names,
    load_speaker_names,
    parse_speaker_names,
    write_live_transcript,
    write_speaker_names,
)

STEM = "2026-10-15_11-30-00"


def test_parse_speaker_names_splits_trims_and_dedupes():
    assert parse_speaker_names(" Anna  Muster, beat; Anna muster\nBeat\n,") == ["Anna Muster", "beat"]
    assert parse_speaker_names("") == []


def test_speaker_roster_roundtrip_keeps_other_keys(tmp_path):
    (tmp_path / f"{STEM}.diarization.json").write_text(json.dumps({"num_speakers": 3}))
    write_speaker_names(STEM, ["Anna Muster", "Beat"], root=tmp_path)
    data = json.loads((tmp_path / f"{STEM}.diarization.json").read_text())
    assert data["num_speakers"] == 3 and data["authoritative"] is True
    assert [p["name"] for p in data["roster"]] == ["Anna Muster", "Beat"]
    assert load_speaker_names(STEM, root=tmp_path) == ["Anna Muster", "Beat"]
    assert load_speaker_names("2026-01-01_00-00-00", root=tmp_path) == []


def test_calendar_proposal_excludes_owner():
    resolution = {"participant_details": [
        {"name": "Matthias Heim", "role": "self"}, {"name": "Anna Muster", "role": "participant"}, "junk"]}
    assert calendar_speaker_names(resolution) == ["Anna Muster"]
    assert calendar_speaker_names(None) == []


def test_live_transcript_written_on_change_and_at_stop(tmp_path):
    session = RecordingLiveSession(SimpleNamespace(), api_key=None, copy_text=lambda _: None,
                                   open_window=False, recordings_root=tmp_path, stem=STEM)
    path = tmp_path / f"{STEM}.live.json"
    lines = [LiveLine(0, 3, "Ich", "Hallo."), LiveLine(4, 8, "Andere", "Weiter", provisional=True)]
    session._on_update(lines, (), "")
    data = json.loads(path.read_text())
    assert data["recording_active"] is True and data["model"]
    assert data["lines"][1] == {"start": 4, "end": 8, "speaker": "Andere", "text": "Weiter", "provisional": True}
    session.engine.lines = lines
    session.stop()
    assert json.loads(path.read_text())["recording_active"] is False


def test_live_transcript_not_written_without_stem(tmp_path):
    session = RecordingLiveSession(SimpleNamespace(), api_key=None, copy_text=lambda _: None,
                                   open_window=False, recordings_root=tmp_path)
    session._on_update([LiveLine(0, 3, "Ich", "Hallo.")], (), "")
    assert not list(tmp_path.iterdir())


def test_watcher_authoritative_roster_drops_absent_invitees(tmp_path):
    from transcribe_watcher import TranscribeWatcher

    fake = SimpleNamespace(config={}, logger=logging.getLogger("test"))
    merge = TranscribeWatcher._merge_configured_diarization_roster
    calendar = [{"name": "Matthias Heim", "role": "self"},
                {"name": "Anna Muster", "role": "participant", "company": "ACME"},
                {"name": "Absent Person", "role": "participant"}]
    wav = tmp_path / f"{STEM}.wav"
    write_speaker_names(STEM, ["Anna Muster", "Carla Neu"], root=tmp_path)
    merged = merge(fake, wav, calendar)
    assert [p["name"] for p in merged] == ["Matthias Heim", "Anna Muster", "Carla Neu"]
    assert merged[1]["company"] == "ACME"

    # Hand-written roster files without the flag keep the old additive merge.
    (tmp_path / f"{STEM}.diarization.json").write_text(json.dumps({"roster": ["Carla Neu"]}))
    assert len(merge(fake, wav, calendar)) == 4
