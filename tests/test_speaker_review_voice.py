"""speaker_review x voice matching: evidence in the proposal, enrolment on confirm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import speaker_review as sr  # noqa: E402
import voice_match as vm  # noqa: E402
from system_track_attribution import RemoteActivity  # noqa: E402
from voice_library import BASIS_HUMAN, BASIS_SELF, VoiceLibrary  # noqa: E402
from voice_synth import base_vectors, noisy  # noqa: E402

B = base_vectors(["Person A", "Person B", "host"])
T = ("[00:00] Matthias: Hoi zäme, schön dass ihr da sind und wir starten jetzt.\n"
     "[00:20] Speaker 1: Ich bi de Person A, und ich leite s Application Management.\n"
     "[00:40] Matthias: Super, dann fange mer a.\n"
     "[01:00] Speaker 2: Und ich bi neu im Team und mache d Integration.\n")


def _labels(text=T):
    return sr.label_stats(text)


def _voice(label="Speaker 1", name="Person A", score=0.91, verdict="clear", crm=None, role="other", reason=None):
    best = {"name": name, "score": score, "role": role}
    if crm is not None:
        best["crm_contact_id"] = crm
    return {"ran": True, "ok": True, "labels": {label: {
        "segments": 9, "best": best, "runner_up": {"name": "Person B", "score": 0.31, "role": "other"},
        "verdict": verdict, "reason": reason}}, "renames": []}


def _model(kind="unknown", conf="low", name=None, evidence="No suggestion."):
    p = {"kind": kind, "confidence": conf, "evidence": evidence}
    if name:
        p["name"] = name
    return p


# --- apply_voice_evidence ----------------------------------------------------

def test_clear_voice_match_becomes_the_proposal_when_the_model_has_nothing():
    props = {"Speaker 1": _model(), "Speaker 2": _model(), "Matthias": _model("self", "high")}
    extra = sr.apply_voice_evidence(props, _labels(), _voice(crm=77))
    p = props["Speaker 1"]
    assert (p["kind"], p["name"], p["contact_id"], p["confidence"]) == ("crm", "Person A", 77, "medium")
    assert p["evidence"].startswith("Voice matches Person A (0.91)")
    assert p["voice"] == {"name": "Person A", "score": 0.91, "verdict": "clear", "crm_contact_id": 77}
    assert extra["Speaker 1"][0]["id"] == 77 and extra["Speaker 1"][0]["voice_score"] == 0.91
    assert props["Speaker 2"]["kind"] == "unknown"                 # untouched


def test_voice_without_crm_id_proposes_a_new_person_by_name():
    props = {"Speaker 1": _model()}
    sr.apply_voice_evidence(props, _labels(), _voice())
    assert (props["Speaker 1"]["kind"], props["Speaker 1"]["name"]) == ("new", "Person A")
    assert "contact_id" not in props["Speaker 1"]


def test_voice_replaces_a_low_or_medium_model_guess_and_says_so():
    props = {"Speaker 1": _model("new", "medium", "Person B", "Looks like a B.")}
    sr.apply_voice_evidence(props, _labels(), _voice())
    p = props["Speaker 1"]
    assert p["name"] == "Person A" and "replaces the transcript-based guess Person B" in p["evidence"]


def test_voice_that_agrees_with_the_model_raises_confidence_and_keeps_the_proposal():
    props = {"Speaker 1": _model("crm", "medium", "Person A", "Introduced himself.")}
    props["Speaker 1"]["contact_id"] = 12
    sr.apply_voice_evidence(props, _labels(), _voice(crm=77))
    p = props["Speaker 1"]
    assert p["confidence"] == "high" and p["contact_id"] == 12 and p["name"] == "Person A"
    assert "Introduced himself." in p["evidence"] and p["evidence"].startswith("Voice matches")


def test_voice_that_contradicts_high_confidence_transcript_evidence_does_not_win():
    props = {"Speaker 1": _model("new", "high", "Person B", "Said 'I am Person B'.")}
    sr.apply_voice_evidence(props, _labels(), _voice())
    p = props["Speaker 1"]
    assert p["name"] == "Person B" and p["confidence"] == "high" and p["voice_conflict"] is True
    assert "disagrees with the transcript evidence for Person B" in p["evidence"]


def test_possible_match_is_evidence_only():
    props = {"Speaker 1": _model()}
    sr.apply_voice_evidence(props, _labels(), _voice(score=0.62, verdict="possible", reason="margin_below_threshold", crm=5))
    p = props["Speaker 1"]
    assert p["kind"] == "unknown" and "possible only" in p["evidence"] and p["voice"]["verdict"] == "possible"


def test_host_labels_none_matches_and_host_matches_are_ignored():
    props = {"Speaker 1": _model(), "Matthias": _model("self", "high")}
    assert sr.apply_voice_evidence(props, _labels(), _voice(verdict="none")) == {}
    assert sr.apply_voice_evidence(props, _labels(), _voice(name="Matthias Heim", role="self")) == {}
    assert sr.apply_voice_evidence(props, _labels(), _voice(label="Matthias")) == {}
    assert sr.apply_voice_evidence(props, _labels(), {"ran": False}) == {}
    assert sr.apply_voice_evidence(props, _labels(), None) == {}
    assert props["Speaker 1"]["kind"] == "unknown" and "voice" not in props["Matthias"]


def test_voice_entry_found_after_an_auto_apply_rename():
    log = _voice(label="Speaker 1")
    log["renames"] = [{"label": "Speaker 1", "to": "Person A"}]
    assert sr._voice_entry_for(log, "Person A") is log["labels"]["Speaker 1"]
    assert sr._voice_entry_for(log, "Nobody") is None


# --- propose end to end (DB / CRM / model stubbed) -----------------------------

def test_propose_shows_voice_evidence_in_the_stored_review(tmp_path, monkeypatch):
    tpath = tmp_path / "2026-11-01_10-00-00.json"
    tpath.write_text(json.dumps({"transcript": T, "_meta": {"voice_match": _voice(crm=77)}}))
    stored = {}
    monkeypatch.setattr(sr, "fetch_source", lambda sid: {
        "title": "t", "company": None, "metadata": {}, "participant_details": [],
        "participant_resolution_log": {}, "insight_count": 0, "transcript_path": str(tpath)})
    monkeypatch.setattr(sr, "_find_person", lambda *a, **k: [])
    monkeypatch.setattr(sr, "_set_metadata", lambda sid, key, value: stored.update({key: value}))
    runner = lambda prompt: json.dumps({"speakers": []})          # model has no opinion
    out = sr.propose(1, runner=runner)
    row = next(lb for lb in out["review"]["labels"] if lb["label"] == "Speaker 1")
    assert row["proposal"]["name"] == "Person A" and row["proposal"]["evidence"].startswith("Voice matches Person A (0.91)")
    assert [c["id"] for c in row["candidates"]] == [77]
    assert stored["speaker_review"]["status"] == "proposed"
    other = next(lb for lb in out["review"]["labels"] if lb["label"] == "Speaker 2")
    assert other["proposal"]["kind"] == "unknown"


def test_propose_without_a_voice_log_is_unchanged(tmp_path, monkeypatch):
    tpath = tmp_path / "2026-11-01_10-00-00.json"
    tpath.write_text(json.dumps({"transcript": T, "_meta": {}}))
    monkeypatch.setattr(sr, "fetch_source", lambda sid: {
        "title": "t", "company": None, "metadata": {}, "participant_details": [],
        "participant_resolution_log": {}, "insight_count": 0, "transcript_path": str(tpath)})
    monkeypatch.setattr(sr, "_find_person", lambda *a, **k: [])
    monkeypatch.setattr(sr, "_set_metadata", lambda *a: None)
    out = sr.propose(1, runner=lambda p: json.dumps({"speakers": []}), dry_run=True)
    assert all("voice" not in lb["proposal"] for lb in out["review"]["labels"])


# --- enrol on confirm ----------------------------------------------------------

def _mapping():
    return sr.validate_mapping([
        {"label": "Matthias", "kind": "self"},
        {"label": "Speaker 1", "kind": "crm", "name": "Person A", "contact_id": 77},
        {"label": "Speaker 2", "kind": "unknown"}], ["Matthias", "Speaker 1", "Speaker 2"])


def _prepare_archive(tmp_path, stem, with_tracks=True):
    root = tmp_path / "rec"
    (root / "Transcripts").mkdir(parents=True)
    tpath = root / "Transcripts" / f"{stem}.json"
    if with_tracks:
        arch = root / "CaptureArchive" / stem
        arch.mkdir(parents=True)
        for n in ("mic", "sys"):
            (arch / f"{stem}.{n}.wav").write_bytes(b"")
    return tpath


def _gem_after_rename(mapping):
    gem = {"transcript": T, "_meta": {"audio_duration_seconds": 80.0}, "participants": []}
    rename = sr.target_names(mapping)
    sr.rename_labels(gem, rename)
    return gem, rename


def _cfg(monkeypatch, tmp_path, **voice):
    cfg = {"voice_matching": dict({"library_dir": str(tmp_path / "lib")}, **voice)}
    monkeypatch.setattr(sr, "_pipeline_config", lambda: cfg)


def _stub_audio(monkeypatch):
    t0 = [(0, 20), (20, 40), (40, 60), (60, 80)]
    remote = RemoteActivity.from_segments([(20, 39, "remote"), (60, 79, "remote")], 80)
    monkeypatch.setattr(vm, "remote_activity_for_source", lambda s: remote)
    monkeypatch.setattr(vm, "embed_segments", lambda source, plans, **kw: (
        [noisy(B["Person A"], 1, seed=i)[0] for i, _ in enumerate(plans)], {"ok": True}))
    monkeypatch.setattr(vm, "MIN_ENROL_SEGMENTS", 1)
    monkeypatch.setattr(vm, "MIN_TURN_SEC", 7.0)


def test_confirm_enrols_non_host_people_from_archived_tracks(tmp_path, monkeypatch):
    stem = "2026-11-01_10-00-00"
    tpath = _prepare_archive(tmp_path, stem)
    _cfg(monkeypatch, tmp_path)
    _stub_audio(monkeypatch)
    mapping = _mapping()
    gem, rename = _gem_after_rename(mapping)
    rep = sr.enrol_voices(str(tpath), gem, mapping, rename)
    assert [e["name"] for e in rep["enrolled"]] == ["Person A"], rep
    lib = VoiceLibrary(tmp_path / "lib")
    people = lib.people()
    assert [(p["name"], p["crm_contact_id"], p["samples"][0]["basis"]) for p in people] == \
        [("Person A", 77, BASIS_HUMAN)]                                # host and unknown are not enrolled


def test_confirm_enrols_nothing_when_store_others_is_off_or_tracks_are_pruned(tmp_path, monkeypatch):
    stem = "2026-11-01_10-00-00"
    mapping = _mapping()
    gem, rename = _gem_after_rename(mapping)
    tpath = _prepare_archive(tmp_path, stem, with_tracks=True)
    _cfg(monkeypatch, tmp_path, store_others=False)
    _stub_audio(monkeypatch)
    rep = sr.enrol_voices(str(tpath), gem, mapping, rename)
    assert rep["enrolled"] == [] and "store_others" in rep["reason"]
    assert not (tmp_path / "lib").exists()
    # tracks pruned
    stem2 = "2026-11-02_10-00-00"
    tpath2 = _prepare_archive(tmp_path / "x", stem2, with_tracks=False)
    _cfg(monkeypatch, tmp_path)
    rep = sr.enrol_voices(str(tpath2), gem, mapping, rename)
    assert rep["enrolled"] == [] and rep["reason"] == "archived tracks no longer on disk"


def test_enrolment_error_never_raises_out_of_enrol_voices(tmp_path, monkeypatch):
    stem = "2026-11-01_10-00-00"
    tpath = _prepare_archive(tmp_path, stem)
    _cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(vm, "enrol_people", lambda *a, **k: 1 / 0)
    mapping = _mapping()
    gem, rename = _gem_after_rename(mapping)
    rep = sr.enrol_voices(str(tpath), gem, mapping, rename)
    assert rep["enrolled"] == [] and "ZeroDivisionError" in rep["reason"]


class _Cur:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append(sql)


class _Conn:
    def __init__(self):
        self.log = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _Cur(self.log)

    def close(self):
        pass


def _confirm_env(tmp_path, monkeypatch, with_tracks=True):
    stem = "2026-11-01_10-00-00"
    tpath = _prepare_archive(tmp_path, stem, with_tracks)
    tpath.write_text(json.dumps({"transcript": T, "_meta": {"audio_duration_seconds": 80.0}, "participants": []}))
    monkeypatch.setattr(sr, "fetch_source", lambda sid: {
        "title": "t", "company": None, "metadata": {}, "participant_details": [],
        "participant_resolution_log": {}, "insight_count": 0, "transcript_path": str(tpath), "has_decisions": False})
    monkeypatch.setattr(sr, "_conn", lambda: _Conn())
    return tpath


def test_confirm_succeeds_and_enrols_when_tracks_exist(tmp_path, monkeypatch):
    tpath = _confirm_env(tmp_path, monkeypatch)
    _cfg(monkeypatch, tmp_path)
    _stub_audio(monkeypatch)
    out = sr.confirm(1, [{"label": "Matthias", "kind": "self"},
                         {"label": "Speaker 1", "kind": "crm", "name": "Person A", "contact_id": 77},
                         {"label": "Speaker 2", "kind": "unknown"}], trigger=False)
    assert out["ok"] and [e["name"] for e in out["voice_enrolment"]["enrolled"]] == ["Person A"]
    assert "] Person A:" in json.loads(tpath.read_text())["transcript"]
    assert [p["name"] for p in VoiceLibrary(tmp_path / "lib").people()] == ["Person A"]


def test_confirm_still_succeeds_when_enrolment_fails_or_archive_is_pruned(tmp_path, monkeypatch):
    tpath = _confirm_env(tmp_path, monkeypatch, with_tracks=False)
    _cfg(monkeypatch, tmp_path)
    mapping = [{"label": "Matthias", "kind": "self"},
               {"label": "Speaker 1", "kind": "crm", "name": "Person A", "contact_id": 77},
               {"label": "Speaker 2", "kind": "unknown"}]
    out = sr.confirm(1, mapping, trigger=False)                    # tracks pruned
    assert out["ok"] and out["voice_enrolment"]["reason"] == "archived tracks no longer on disk"
    assert "] Person A:" in json.loads(tpath.read_text())["transcript"]
    # an exploding enrolment must not undo or fail the confirmation
    tpath.write_text(json.dumps({"transcript": T, "_meta": {}, "participants": []}))
    monkeypatch.setattr(sr, "enrol_voices", lambda *a, **k: 1 / 0)
    out = sr.confirm(1, mapping, trigger=False)
    assert out["ok"] and any("voice enrolment failed" in w for w in out["warnings"])
    assert "] Person A:" in json.loads(tpath.read_text())["transcript"]
