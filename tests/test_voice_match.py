"""voice_match: segment selection, matching, host check, conflicts, process().

Synthetic unit vectors only: no audio, no real embeddings, generic names.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import voice_match as vm  # noqa: E402
from system_track_attribution import RemoteActivity  # noqa: E402
from voice_library import BASIS_HUMAN, BASIS_SELF, Profile, VoiceLibrary  # noqa: E402
from voice_synth import base_vectors, mix, noisy  # noqa: E402

NAMES = ["Person A", "Person B", "Person C", "host", "stranger"]
B = base_vectors(NAMES)


def prof(name, vecs, role="other", crm=None):
    return Profile(name.lower().replace(" ", "-"), name, role, crm, np.vstack(vecs), ["2026-10-01_10-00-00"])


def emb(label, vecs, role="remote"):
    vecs = np.asarray(vecs, dtype=np.float32)
    plans = [vm.SegmentPlan(label, role, i * 20.0, i * 20.0 + 10, i, 0.95 if role == "remote" else 0.0)
             for i in range(len(vecs))]
    return vm.LabelEmbeddings(label, role, vecs, plans)


LIB = [prof("Person A", noisy(B["Person A"], 3, seed=1)),
       prof("Person B", noisy(B["Person B"], 3, seed=2)),
       prof("Matthias Heim", noisy(B["host"], 4, seed=3), role="self")]


# --- generic labels ----------------------------------------------------------

@pytest.mark.parametrize("label,generic", [
    ("Speaker 1", True), ("speaker A", True), ("Andere", True), ("Sprecher 2", True),
    ("Unidentified participant 3", True), ("Philipp", False), ("Matthias", False),
    ("Dr. Speaker", False), ("Person A", False),
])
def test_is_generic_label(label, generic):
    assert vm.is_generic_label(label) is generic


# --- segment planning --------------------------------------------------------

def _transcript(turns):
    """turns: (start, label) -> '[MM:SS] Label: text' lines."""
    return "\n".join(f"[{t // 60:02d}:{t % 60:02d}] {l}: Das ist ein ausreichend langer Satz hier."
                     for t, l in turns)


def test_plan_uses_system_track_for_remote_and_mic_for_host():
    turns = [(i * 20, "Matthias" if i % 2 == 0 else "Speaker 1") for i in range(10)]
    remote = RemoteActivity.from_segments([(t, t + 19, "remote") for t, l in turns if l != "Matthias"], 200)
    plans = vm.plan_segments(_transcript(turns), remote, duration_sec=200)
    assert {p.role for p in plans if p.label == "Speaker 1"} == {"remote"}
    assert {p.role for p in plans if p.label == "Matthias"} == {"host"}
    assert len(plans) == 10
    p = plans[0]                                    # turn 0-20: 1 s in, no cap hit
    assert (p.start, p.end) == (1.0, 19.0)


def test_plan_skips_short_turns_and_wrong_activity_and_caps_length():
    turns = [(0, "Speaker 2"), (6, "Speaker 1"),        # Speaker 2: a 6 s turn, too short
             (20, "Speaker 1"),                          # 60 s long turn: capped at 25 s
             (80, "Matthias"),                           # host turn while system active: skipped
             (100, "Speaker 2")]                         # remote label, system silent: skipped
    remote = RemoteActivity.from_segments([(0, 79, "remote"), (80, 99, "remote")], 120)
    plans = vm.plan_segments(_transcript(turns), remote, duration_sec=120)
    by = {(p.label, p.start): p for p in plans}
    assert not [p for p in plans if p.label in ("Matthias", "Speaker 2")]
    long = by[("Speaker 1", 21.0)]
    assert long.end - long.start == vm.MAX_SEGMENT_SEC


def test_plan_thresholds_are_inclusive_at_the_documented_values():
    # 70 % system share is remote, 12 % is host, between is neither.
    for share, label, expect in ((0.70, "Speaker 1", "remote"), (0.69, "Speaker 1", None),
                                 (0.12, "Matthias", "host"), (0.13, "Matthias", None)):
        remote = RemoteActivity.from_segments([(0, 20 * share, "remote")], 40)
        plans = vm.plan_segments(_transcript([(0, label), (20, "Speaker 9")]), remote, duration_sec=40)
        got = next((p.role for p in plans if p.label == label), None)
        assert got == expect, (share, label)


def test_plan_spreads_evenly_when_more_than_the_cap():
    turns = [(i * 20, "Speaker 1") for i in range(40)]
    remote = RemoteActivity.from_segments([(0, 800, "remote")], 800)
    plans = vm.plan_segments(_transcript(turns), remote, duration_sec=800)
    assert len(plans) == vm.MAX_SEGMENTS_PER_LABEL
    starts = [p.start for p in plans]
    assert starts[0] < 30 and starts[-1] > 700            # whole meeting, not the first 14


def test_plan_without_system_activity_is_empty():
    assert vm.plan_segments(_transcript([(0, "Speaker 1")]), None, duration_sec=30) == []


# --- matching ----------------------------------------------------------------

def test_clear_match():
    m = vm.match_label(emb("Speaker 1", noisy(B["Person A"], 6, seed=10)), LIB)
    assert m["verdict"] == "clear" and m["best"]["name"] == "Person A"
    assert m["best"]["score"] >= vm.CLEAR_MIN_SCORE
    assert m["runner_up"]["name"] != "Person A" and m["margin"] >= vm.CLEAR_MIN_MARGIN


def test_margin_too_small_is_possible():
    lib = LIB + [prof("Person A twin", noisy(B["Person A"], 3, seed=11))]     # same voice, other entry
    m = vm.match_label(emb("Speaker 1", noisy(B["Person A"], 6, seed=10)), lib)
    assert m["verdict"] == "possible" and m["reason"] == "margin_below_threshold"


def test_score_between_possible_and_clear_is_possible():
    v = mix(B["Person A"], B["stranger"], 0.45)           # cosine to A ~0.63
    sims = float(v @ B["Person A"])
    assert vm.POSSIBLE_MIN_SCORE < sims < vm.CLEAR_MIN_SCORE, sims
    m = vm.match_label(emb("Speaker 1", np.vstack([v] * 5)), LIB)
    assert m["verdict"] == "possible" and m["reason"] == "score_below_clear_threshold"


def test_below_threshold_is_none():
    m = vm.match_label(emb("Speaker 1", noisy(B["stranger"], 6, seed=13)), LIB)
    assert m["verdict"] == "none" and m["reason"] == "below_possible_threshold"


def test_too_few_segments_is_never_clear():
    m = vm.match_label(emb("Speaker 1", noisy(B["Person A"], 2, seed=14)), LIB)
    assert m["verdict"] == "possible" and m["reason"] == "too_few_segments"
    m = vm.match_label(emb("Speaker 1", noisy(B["stranger"], 2, seed=14)), LIB)
    assert m["verdict"] == "none"


def test_empty_library():
    assert vm.match_label(emb("Speaker 1", noisy(B["Person A"], 5)), [])["reason"] == "empty_library"


def test_thresholds_are_the_documented_constants():
    assert (vm.CLEAR_MIN_SCORE, vm.CLEAR_MIN_MARGIN, vm.MIN_LABEL_SEGMENTS) == (0.70, 0.20, 3)


def test_best_centroid_is_used_across_a_persons_meetings():
    # One meeting of this person sounds different (new microphone): still found.
    p = prof("Person C", [noisy(B["Person C"], 1, seed=1)[0], noisy(B["stranger"], 1, seed=2)[0]])
    m = vm.match_label(emb("Speaker 1", noisy(B["stranger"], 5, seed=3)), [p] + LIB)
    assert m["best"]["name"] == "Person C" and m["verdict"] == "clear"


# --- conflicts ---------------------------------------------------------------

def test_two_labels_matching_one_person_conflict():
    matches = vm.match_labels({"Speaker 1": emb("Speaker 1", noisy(B["Person A"], 6, seed=1)),
                               "Speaker 2": emb("Speaker 2", noisy(B["Person A"], 6, seed=2))}, LIB)
    conflicts = vm.find_conflicts(matches)
    assert conflicts == [{"type": "same_person_two_labels", "person": "Person A",
                          "labels": ["Speaker 1", "Speaker 2"]}]


def test_label_split_between_two_people_conflict():
    vecs = np.vstack([noisy(B["Person A"], 5, seed=1), noisy(B["Person B"], 5, seed=2)])
    matches = vm.match_labels({"Speaker 1": emb("Speaker 1", vecs)}, LIB)
    conflicts = vm.find_conflicts(matches)
    assert [c["type"] for c in conflicts] == ["label_split_between_people"]
    assert set(conflicts[0]["votes"]) == {"Person A", "Person B"}


def test_clean_meeting_has_no_conflict():
    matches = vm.match_labels({"Speaker 1": emb("Speaker 1", noisy(B["Person A"], 6, seed=1)),
                               "Speaker 2": emb("Speaker 2", noisy(B["Person B"], 6, seed=2))}, LIB)
    assert vm.find_conflicts(matches) == []


def test_remote_label_matching_the_host_and_named_label_disagreeing():
    matches = vm.match_labels({"Speaker 1": emb("Speaker 1", noisy(B["host"], 6, seed=1)),
                               "Person B": emb("Person B", noisy(B["Person A"], 6, seed=2))}, LIB)
    types = {c["type"] for c in vm.find_conflicts(matches)}
    assert types == {"remote_label_matches_host", "name_disagrees_with_voice"}
    # A named label whose name agrees with the voice is not a conflict.
    ok = vm.match_labels({"Person A": emb("Person A", noisy(B["Person A"], 6, seed=2))}, LIB)
    assert vm.find_conflicts(ok) == []


# --- host check --------------------------------------------------------------

def test_host_check_passes_genuine_and_flags_an_impostor_turn():
    selfp = LIB[2]
    genuine = noisy(B["host"], 8, seed=5)
    impostor = noisy(B["stranger"], 1, seed=6)[0]
    ok = vm.host_check(emb("Matthias", genuine, "host"), selfp)
    assert ok["ran"] and ok["suspicious"] == [] and ok["reason"] is None and ok["centroid_similarity"] > 0.9
    vecs = np.vstack([genuine, impostor])                  # last segment = another voice
    bad = vm.host_check(emb("Matthias", vecs, "host"), selfp)
    assert [s["line"] for s in bad["suspicious"]] == [8]
    assert bad["suspicious"][0]["similarity"] < vm.HOST_SUSPICIOUS_MAX_SIM
    assert bad["reason"] is None                           # the meeting's host is still the host


def test_host_check_whole_meeting_mismatch_and_no_profile():
    out = vm.host_check(emb("Matthias", noisy(B["stranger"], 6, seed=1), "host"), LIB[2])
    assert out["reason"] == "host_voice_does_not_match_self_profile"
    assert vm.host_check(emb("Matthias", noisy(B["host"], 6), "host"), None)["reason"] == "no_self_profile"
    assert vm.host_check(None, LIB[2])["reason"] == "no_host_segments"


def test_host_enrolment_gate():
    good = emb("Matthias", noisy(B["host"], 8, seed=1), "host")
    assert vm.host_enrolment(good, LIB[2])[1] == "ok"
    assert vm.host_enrolment(emb("Matthias", noisy(B["host"], 3), "host"), None)[1] == "too_few_host_segments"
    scattered = emb("Matthias", np.vstack([B[n] for n in NAMES]), "host")       # incoherent segments
    assert vm.host_enrolment(scattered, None)[1] == "host_segments_not_coherent"
    other = emb("Matthias", noisy(B["stranger"], 8, seed=2), "host")
    assert vm.host_enrolment(other, LIB[2])[1] == "host_voice_does_not_match_self_profile"
    assert vm.host_enrolment(other, None)[1] == "ok"           # bootstrap: nothing to contradict


# --- renames -----------------------------------------------------------------

def _clear(name, role="other"):
    return {"verdict": "clear", "best": {"name": name, "score": 0.9, "role": role}, "margin": 0.5}


def test_plan_renames_only_generic_clear_and_conflict_free():
    matches = {"Speaker 1": _clear("Person A"), "Philipp": _clear("Person B"),
               "Speaker 2": {"verdict": "possible", "best": {"name": "Person C", "score": 0.6, "role": "other"}}}
    rename, log = vm.plan_renames({"Speaker 1", "Philipp", "Speaker 2", "Matthias"}, matches, [])
    assert rename == {"Speaker 1": "Person A"}               # real name untouched, possible ignored
    assert log == [{"label": "Speaker 1", "to": "Person A", "score": 0.9, "margin": 0.5}]


def test_plan_renames_refuses_on_conflicts_collisions_and_host():
    m = {"Speaker 1": _clear("Person A")}
    assert vm.plan_renames({"Speaker 1"}, m, [{"type": "same_person_two_labels"}])[0] == {}
    assert vm.plan_renames({"Speaker 1", "Person A"}, m, [])[0] == {}                 # name already a label
    two = {"Speaker 1": _clear("Person A"), "Speaker 2": _clear("Person A")}
    assert len(vm.plan_renames({"Speaker 1", "Speaker 2"}, two, [])[0]) == 1          # never two onto one name
    assert vm.plan_renames({"Speaker 1"}, {"Speaker 1": _clear("Matthias Heim", "self")}, [])[0] == {}


# --- process() ---------------------------------------------------------------

def _meeting(n=12):
    turns = [(i * 20, "Matthias" if i % 2 == 0 else "Speaker 1") for i in range(n)]
    remote = RemoteActivity.from_segments([(t, t + 19, "remote") for t, l in turns if l != "Matthias"], n * 20)
    return _transcript(turns), remote, n * 20


def stub_embedder(per_label):
    """embed_segments replacement: vectors from `per_label[(label, role)]`."""
    def fake(source, plans, **kw):
        counts: dict = {}
        out = []
        for p in plans:
            base = per_label[(p.label, p.role)]
            i = counts[(p.label, p.role)] = counts.get((p.label, p.role), -1) + 1
            out.append(noisy(base, 1, seed=100 + i)[0])
        return out, {"ok": True, "segments": len(plans), "device": "stub"}
    return fake


def _enrolled_lib(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem="2026-09-01_10-00-00", segments=9, basis=BASIS_HUMAN)
    lib.add_sample("Matthias Heim", B["host"], stem="2026-09-01_10-00-00", segments=9, basis=BASIS_SELF, role="self")
    return lib


def _run(tmp_path, monkeypatch, per_label, cfg=None, lib=None, host_verified=True, source="x"):
    text, remote, dur = _meeting()
    monkeypatch.setattr(vm, "embed_segments", stub_embedder(per_label))
    return vm.process(text, source=vm.three_channel_source(source) if source else None, remote=remote,
                      duration_sec=dur, participants=None, stem="2026-11-01_10-00-00",
                      config=cfg or {"voice_matching": {"library_dir": str(tmp_path / "lib")}},
                      host_verified=host_verified, library=lib or _enrolled_lib(tmp_path))


PER_LABEL = {("Matthias", "host"): B["host"], ("Speaker 1", "remote"): B["Person A"]}


def test_process_matches_checks_host_and_enrols_self(tmp_path, monkeypatch):
    lib = _enrolled_lib(tmp_path)
    log, rename = _run(tmp_path, monkeypatch, PER_LABEL, lib=lib)
    assert log["ran"] and log["ok"]
    assert log["labels"]["Speaker 1"]["verdict"] == "clear"
    assert log["labels"]["Speaker 1"]["best"]["name"] == "Person A"
    assert log["host_check"]["ran"] and log["host_check"]["suspicious"] == []
    assert log["self_enrolment"]["enrolled"] is True
    assert rename == {}                                            # auto_apply is off
    assert [s["stem"] for p in lib.people() if p["role"] == "self" for s in p["samples"]] == \
        ["2026-09-01_10-00-00", "2026-11-01_10-00-00"]
    json.dumps(log)                                                # serialisable
    assert log["runtime_seconds"] is not None


def test_process_log_carries_no_vectors(tmp_path, monkeypatch):
    log, _ = _run(tmp_path, monkeypatch, PER_LABEL)

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, (list, tuple)):
            assert len(x) < 16, "a long list in the log could be a vector"
            for v in x:
                yield from walk(v)
        else:
            assert not isinstance(x, (np.ndarray, np.floating)), type(x)
            yield x
    list(walk(log))


def test_process_auto_apply_returns_rename_for_generic_clear_match(tmp_path, monkeypatch):
    cfg = {"voice_matching": {"auto_apply": True, "library_dir": str(tmp_path / "lib")}}
    log, rename = _run(tmp_path, monkeypatch, PER_LABEL, cfg)
    assert rename == {"Speaker 1": "Person A"} and log["auto_apply"] is True
    assert log["renames"][0]["to"] == "Person A"


def test_process_self_enrolment_needs_a_verified_host_label(tmp_path, monkeypatch):
    lib = _enrolled_lib(tmp_path)
    log, _ = _run(tmp_path, monkeypatch, PER_LABEL, lib=lib, host_verified=False)
    assert log["self_enrolment"] == {"enrolled": False, "reason": "host_label_not_verified"}
    assert len([s for p in lib.people() if p["role"] == "self" for s in p["samples"]]) == 1


def test_process_skips_single_source_and_disabled(tmp_path, monkeypatch):
    log, rename = _run(tmp_path, monkeypatch, {}, source=None)
    assert (log["ran"], log["reason"], rename) == (False, "single_source_or_no_system_track", {})
    cfg = {"voice_matching": {"enabled": False}}
    log, _ = _run(tmp_path, monkeypatch, {}, cfg)
    assert (log["ran"], log["reason"]) == (False, "disabled_in_config")


def test_process_worker_failure_is_ok_false_and_changes_nothing(tmp_path, monkeypatch):
    text, remote, dur = _meeting()
    monkeypatch.setattr(vm, "embed_segments", lambda *a, **k: (None, {"ok": False, "error": "TimeoutError: slow"}))
    lib = _enrolled_lib(tmp_path)
    log, rename = vm.process(text, source=vm.three_channel_source("x"), remote=remote, duration_sec=dur,
                             participants=None, stem="2026-11-01_10-00-00", config={}, library=lib)
    assert log["ran"] is False and log["ok"] is False and "TimeoutError" in log["reason"] and rename == {}


def test_process_never_raises(tmp_path, monkeypatch):
    text, remote, dur = _meeting()
    monkeypatch.setattr(vm, "embed_segments", lambda *a, **k: 1 / 0)
    log, rename = vm.process(text, source=vm.three_channel_source("x"), remote=remote, duration_sec=dur,
                             participants=None, stem="2026-11-01_10-00-00", config={},
                             library=_enrolled_lib(tmp_path))
    assert log["ok"] is False and log["reason"].startswith("error:") and rename == {}


def test_process_skips_remote_embedding_when_the_library_has_no_other_people(tmp_path, monkeypatch):
    seen = []

    def fake(source, plans, **kw):
        seen.extend({p.role for p in plans})
        return [noisy(B["host"], 1, seed=i)[0] for i, _ in enumerate(plans)], {"ok": True}
    monkeypatch.setattr(vm, "embed_segments", fake)
    text, remote, dur = _meeting()
    lib = VoiceLibrary(tmp_path / "lib")
    log, _ = vm.process(text, source=vm.three_channel_source("x"), remote=remote, duration_sec=dur,
                        participants=None, stem="2026-11-01_10-00-00", config={}, library=lib, host_verified=True)
    assert set(seen) == {"host"} and log["reason"] == "library_has_no_other_people"
    assert log["self_enrolment"]["enrolled"] is True               # bootstrap of the self profile


def test_voice_config_defaults_when_key_absent():
    assert vm.voice_config({}) == vm.voice_config(None) == {
        "enabled": True, "store_others": True, "auto_apply": False, "device": "",
        "timeout_seconds": vm.EMBED_TIMEOUT_SECONDS}
    assert vm.voice_config({"voice_matching": {"auto_apply": True}})["auto_apply"] is True


# --- enrol_people -------------------------------------------------------------

def test_enrol_people_enrols_and_skips_on_pruned_archive_or_conflict(tmp_path, monkeypatch):
    base = tmp_path / "rec"
    stem = "2026-11-02_10-00-00"
    text, remote, dur = _meeting()
    lib = _enrolled_lib(tmp_path)
    who = {"Speaker 1": {"name": "Person C", "crm_contact_id": 5}}
    # Archive pruned: nothing enrolled, no exception.
    rep = vm.enrol_people(text, who, stem=stem, base=base, duration_sec=dur, library=lib)
    assert rep["enrolled"] == [] and rep["reason"] == "archived tracks no longer on disk"
    # Archive present.
    arch = base / "CaptureArchive" / stem
    arch.mkdir(parents=True)
    for n in ("mic", "sys"):
        (arch / f"{stem}.{n}.wav").write_bytes(b"")
    monkeypatch.setattr(vm, "remote_activity_for_source", lambda s: remote)
    monkeypatch.setattr(vm, "embed_segments", stub_embedder({("Speaker 1", "remote"): B["Person C"]}))
    rep = vm.enrol_people(text, who, stem=stem, base=base, duration_sec=dur, library=lib)
    assert [e["name"] for e in rep["enrolled"]] == ["Person C"]
    person = next(p for p in lib.people() if p["name"] == "Person C")
    assert person["crm_contact_id"] == 5 and person["samples"][0]["basis"] == BASIS_HUMAN
    # A confirmation that contradicts a CLEAR voice match to someone else is not enrolled.
    monkeypatch.setattr(vm, "embed_segments", stub_embedder({("Speaker 1", "remote"): B["Person A"]}))
    rep = vm.enrol_people(text, {"Speaker 1": {"name": "Person D"}}, stem=stem, base=base,
                          duration_sec=dur, library=lib)
    assert rep["enrolled"] == [] and "voice_matches_Person A_clearly" in rep["skipped"][0]["reason"]
    assert not any(p["name"] == "Person D" for p in lib.people())
    # Too few segments.
    monkeypatch.setattr(vm, "plan_segments", lambda *a, **k: [])
    rep = vm.enrol_people(text, who, stem=stem, base=base, duration_sec=dur, library=lib)
    assert rep["skipped"][0]["reason"] == "too_few_system_track_segments"


def test_locate_source_prefers_archived_tracks_then_raw_wav(tmp_path):
    stem = "2026-11-02_10-00-00"
    assert vm.locate_source(tmp_path, stem) is None
    (tmp_path / "Recordings").mkdir()
    (tmp_path / "Recordings" / f"{stem}.wav").write_bytes(b"")
    assert vm.locate_source(tmp_path, stem)["layout"] == "three_channel"
    arch = tmp_path / "CaptureArchive" / stem
    arch.mkdir(parents=True)
    for n in ("mic", "sys"):
        (arch / f"{stem}.{n}.wav").write_bytes(b"")
    src = vm.locate_source(tmp_path, stem, mic_offset_sec=15.4)
    assert src["layout"] == "tracks" and src["mic_offset_sec"] == 15.4


# --- the worker boundary (real process, no model needed for the timeout) ------

def test_embed_segments_timeout_returns_failure_not_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(vm, "embedding_available", lambda: (True, None))
    plans = [vm.SegmentPlan("Speaker 1", "remote", 1.0, 9.0, 0, 0.9)]
    vectors, info = vm.embed_segments(vm.three_channel_source(tmp_path / "x.wav"), plans, timeout_seconds=0)
    assert vectors is None and info["ok"] is False and "Timeout" in info["error"]


def test_embed_segments_reports_unavailable_model(monkeypatch):
    monkeypatch.setattr(vm, "embedding_available", lambda: (False, "torch not installed"))
    vectors, info = vm.embed_segments({}, [vm.SegmentPlan("a", "remote", 0, 9, 0, 1)])
    assert vectors is None and info == {"ok": False, "error": "torch not installed"}


_model_ok, _ = vm.embedding_available()


@pytest.mark.skipif(not _model_ok, reason="pyannote/torch or the embedding model is unavailable")
def test_real_model_embeds_synthetic_audio_in_a_worker_process(tmp_path):
    """Smoke test of the real worker on generated tones (no recordings)."""
    sr = 16000
    t = np.arange(sr * 12) / sr
    sig = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    pcm = (sig * 32767).astype("<i2")
    mic = tmp_path / "t.mic.wav"
    sys_wav = tmp_path / "t.sys.wav"
    for path in (mic, sys_wav):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
    plans = [vm.SegmentPlan("Speaker 1", "remote", 1.0, 9.0, 0, 0.9),
             vm.SegmentPlan("Matthias", "host", 1.0, 9.0, 1, 0.0)]
    vectors, info = vm.embed_segments(vm.tracks_source(mic, sys_wav), plans, timeout_seconds=300)
    assert info["ok"], info
    assert len(vectors) == 2 and all(v.shape == (256,) for v in vectors)
    assert all(np.isclose(np.linalg.norm(v), 1.0, atol=1e-3) for v in vectors)
