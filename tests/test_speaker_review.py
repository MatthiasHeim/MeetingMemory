"""Tests for speaker_review — the pure parts of propose/confirm.

DB, subprocess and model plumbing are covered by the live run against a held
source; these pin the logic that decides what the transcript becomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from attribution_gate import gate  # noqa: E402
from speaker_review import (  # noqa: E402
    companies_to_check,
    confirmed_attribution,
    handoff_plan,
    label_stats,
    parse_proposal,
    participant_details_for,
    is_speaker_label,
    relabelled_review,
    rename_labels,
    target_names,
    validate_mapping,
)

T = ("[00:00] Matthias: Hoi zäme, schön dass ihr da sind.\n"
     "[00:04] Tomi: Ich bi de Dominic, Anwendungsmanager.\n"
     "[00:09] Ruedi: Und ich leite s Application Management.\n"
     "[00:15] Tomi: Genau.\n"
     "[00:20] Matthias: Tomi, chasch du das erkläre?\n")


def _gem(transcript=T):
    return {"transcript": transcript,
            "participants": [{"name": "Matthias", "role": "host"}, {"name": "Tomi", "role": "participant"},
                             {"name": "Ruedi", "role": "participant"}],
            "speaker_emotions": [{"speaker": "Tomi", "arc": ["calm"]}, {"speaker": "Ruedi", "arc": ["warm"]}],
            "speaker_pacing": {"Tomi": {"wpm": 120, "entries": 2}, "Ruedi": {"wpm": 100, "entries": 1}},
            "interruptions": [{"interrupter": "Matthias", "interruptee": "Tomi"}],
            "energy_levels": {},
            "_meta": {"speaker_attribution": {"status": "needs_review", "speaker_dependent_actions": "hold",
                                              "missing_stages": ["acoustic_identity_review"]}}}


def test_label_stats_counts_and_flags_self():
    stats = {s["label"]: s for s in label_stats(T)}
    assert set(stats) == {"Matthias", "Tomi", "Ruedi"}
    assert stats["Tomi"]["lines"] == 2 and stats["Tomi"]["share"] == 0.4
    assert stats["Matthias"]["self_label"] and not stats["Tomi"]["self_label"]
    assert stats["Tomi"]["samples"][0]["ts"] == "00:04"


def test_swap_is_single_pass():
    """Tomi<->Ruedi must swap, not collapse everyone into one name."""
    g = _gem()
    changed = rename_labels(g, {"Tomi": "Ruedi", "Ruedi": "Tomi"})
    assert changed == 3
    assert "[00:04] Ruedi: Ich bi de Dominic" in g["transcript"]
    assert "[00:09] Tomi: Und ich leite" in g["transcript"]
    assert {p["name"] for p in g["participants"]} == {"Matthias", "Tomi", "Ruedi"}


def test_spoken_text_is_never_rewritten():
    g = _gem()
    rename_labels(g, {"Tomi": "Dominic Schelling"})
    assert "Matthias: Tomi, chasch du das erkläre?" in g["transcript"]
    assert "[00:15] Dominic Schelling: Genau." in g["transcript"]


def test_merge_two_labels_into_one_person():
    g = _gem()
    rename_labels(g, {"Tomi": "Roel Vandewall", "Ruedi": "Roel Vandewall"})
    assert [p["name"] for p in g["participants"]].count("Roel Vandewall") == 1
    assert [e["speaker"] for e in g["speaker_emotions"]] == ["Roel Vandewall"]
    assert g["speaker_emotions"][0]["arc"] == ["calm", "warm"]
    assert list(g["speaker_pacing"]) == ["Roel Vandewall"]


def test_validate_mapping_requires_every_label_once():
    labels = ["Matthias", "Tomi", "Ruedi"]
    ok = [{"label": "Matthias", "kind": "self"},
          {"label": "Tomi", "kind": "crm", "name": "Dominic Schelling", "contact_id": "17", "company": "BlueCare"},
          {"label": "Ruedi", "kind": "unknown"}]
    out = validate_mapping(ok, labels)
    assert out[0]["name"] == "Matthias Heim" and out[1]["contact_id"] == 17
    with pytest.raises(ValueError, match="no answer"):
        validate_mapping(ok[:2], labels)
    with pytest.raises(ValueError, match="twice"):
        validate_mapping(ok + [{"label": "Tomi", "kind": "unknown"}], labels)
    with pytest.raises(ValueError, match="unknown speaker label"):
        validate_mapping(ok + [{"label": "Mallory", "kind": "unknown"}], labels)
    with pytest.raises(ValueError, match="contact_id"):
        validate_mapping([ok[0], {"label": "Tomi", "kind": "crm", "name": "X Y"}, ok[2]], labels)
    with pytest.raises(ValueError, match="real name"):
        validate_mapping([ok[0], {"label": "Tomi", "kind": "new", "name": "Evil: [00:00] x"}, ok[2]], labels)


def test_unknowns_get_neutral_distinct_labels():
    m = validate_mapping([{"label": "Matthias", "kind": "self"}, {"label": "Tomi", "kind": "unknown"},
                          {"label": "Ruedi", "kind": "unknown"}], ["Matthias", "Tomi", "Ruedi"])
    r = target_names(m)
    assert r["Tomi"] != r["Ruedi"] and r["Tomi"].startswith("Unidentified participant")
    single = target_names(validate_mapping([{"label": "Matthias", "kind": "self"}, {"label": "Tomi", "kind": "unknown"}],
                                           ["Matthias", "Tomi"]))
    assert single["Tomi"] == "Unidentified participant"


def test_confirm_opens_the_gate_two_sided():
    g = _gem()
    assert gate(g)["speaker_dependent_actions"] == "hold"          # before: held
    g["_meta"]["speaker_attribution"] = confirmed_attribution(g["_meta"]["speaker_attribution"], "now")
    d = gate(g)
    assert d["speaker_dependent_actions"] == "require_turn_evidence" and d["trigger_claude"]
    assert d["missing_stages"] == []


def test_confirm_overrides_a_calendar_collision():
    g = _gem()
    g["_meta"]["speaker_attribution"]["calendar_search"] = {"match_count": 2, "ambiguous": True}
    assert gate(g)["speaker_dependent_actions"] == "resolve_then_extract"
    g["_meta"]["speaker_attribution"] = confirmed_attribution(g["_meta"]["speaker_attribution"], "now")
    assert gate(g)["speaker_dependent_actions"] == "require_turn_evidence"


def test_participant_details_one_row_per_person():
    m = validate_mapping([{"label": "Matthias", "kind": "self"},
                          {"label": "Tomi", "kind": "new", "name": "Dominic S", "company": "BlueCare"},
                          {"label": "Ruedi", "kind": "new", "name": "dominic s"}], ["Matthias", "Tomi", "Ruedi"])
    d = participant_details_for(m)
    assert [r["role"] for r in d] == ["self", "participant"]


def test_handoff_plan_never_starts_a_second_run():
    assert handoff_plan({"task_decisions": {"decisions": []}}, 12) == "corrected"
    assert handoff_plan({"task_decisions": {"attribution_held": True}}, 12) == "followup_only"
    assert handoff_plan({"pipeline_status": "ready_for_followup"}, 0) == "corrected"   # a run owns it
    assert handoff_plan({}, 16) == "followup_only"
    assert handoff_plan({}, 0) == "full"


def test_parse_proposal_rejects_invented_contact_ids():
    known = {17: {"id": 17, "name": "Dominic Schelling", "company": "BlueCare", "position": "Anwendungsmanager"}}
    text = "Sure:\n" + json.dumps({"speakers": [
        {"label": "Matthias", "kind": "self", "confidence": "high", "evidence": "host"},
        {"label": "Tomi", "kind": "crm", "contact_id": 17, "confidence": "high", "evidence": "introduces himself as Dominic"},
        {"label": "Ruedi", "kind": "crm", "contact_id": 999, "name": "Roel Vandewall", "confidence": "medium", "evidence": "x"},
        {"label": "Ghost", "kind": "self"}]})
    p = parse_proposal(text, ["Matthias", "Tomi", "Ruedi"], known)
    assert p["Tomi"]["name"] == "Dominic Schelling" and p["Tomi"]["contact_id"] == 17
    assert p["Ruedi"]["kind"] == "new" and "contact_id" not in p["Ruedi"]   # id 999 never shown
    assert "Ghost" not in p


def test_parse_proposal_fills_missing_labels_with_unknown():
    p = parse_proposal('{"speakers": []}', ["Tomi"], {})
    assert p["Tomi"]["kind"] == "unknown"
    with pytest.raises(ValueError):
        parse_proposal("no json here", ["Tomi"], {})


def test_relabelled_review_lists_current_labels():
    g = _gem()
    m = validate_mapping([{"label": "Matthias", "kind": "self"},
                          {"label": "Tomi", "kind": "crm", "name": "Dominic Schelling", "contact_id": 17},
                          {"label": "Ruedi", "kind": "unknown"}], ["Matthias", "Tomi", "Ruedi"])
    r = target_names(m)
    rename_labels(g, r)
    rows = {x["label"]: x for x in relabelled_review(g["transcript"], m, r)}
    assert set(rows) == {"Matthias Heim", "Dominic Schelling", "Unidentified participant"}
    assert rows["Dominic Schelling"]["proposal"]["contact_id"] == 17
    assert validate_mapping([{"label": k, "kind": "unknown"} for k in rows], list(rows))  # re-editable


def test_companies_from_calendar_brackets_and_clients():
    cal = {"candidates": [{"title": "[BlueCare] Daten für Claude", "attendees": [{"company": "Lailix"}]}]}
    out = companies_to_check(title="2026-10-29_15-02-46", source_company=None, calendar=cal,
                             transcript="wir bei TWINT", client_names=["TWINT", "Brame"])
    assert out == ["BlueCare", "TWINT"]


def test_fake_labels_from_second_timestamps_are_never_offered_or_rewritten():
    t = ("[00:05:18] Nice to see you so much. [00:12:39] Nice to meet you.\n"
         "[31:30] Tomi: Top. Und letzte Frage\n"
         "[00:00:00] (Background noise, typing until 02:43)\n"
         "[00:01] Speaker 1: Hallo\n")
    labels = [s["label"] for s in label_stats(t)]
    assert labels == ["Tomi", "Speaker 1"]
    g = {"transcript": t, "participants": []}
    rename_labels(g, {"Tomi": "Dominic Schelling", "Speaker 1": "Roel Vandewall"})
    assert "[00:05:18] Nice to see you so much. [00:12:39] Nice to meet you." in g["transcript"]
    assert "(Background noise, typing until 02:43)" in g["transcript"]
    assert "[31:30] Dominic Schelling: Top." in g["transcript"]
    assert is_speaker_label("Christoph M. Richard Huth") and not is_speaker_label("[31")


def test_names_must_stay_readable_as_labels():
    with pytest.raises(ValueError, match="at most 60"):
        validate_mapping([{"label": "Tomi", "kind": "new", "name": "A" * 61}], ["Tomi"])
    with pytest.raises(ValueError, match="real name"):
        validate_mapping([{"label": "Tomi", "kind": "new", "name": "Unidentified participant 2"}], ["Tomi"])
    name = "Christoph Maximilian Richard Huth-Lindorfer"
    g = {"transcript": "[00:01] Tomi: Hallo\n", "participants": []}
    rename_labels(g, target_names(validate_mapping([{"label": "Tomi", "kind": "new", "name": name}], ["Tomi"])))
    assert [s["label"] for s in label_stats(g["transcript"])] == [name]


def test_silent_attendees_and_emails_survive_confirmation():
    existing = [{"name": "Matthias Heim", "role": "self", "email": "matthias@lailix.com"},
                {"name": "Tomi", "role": "participant"},
                {"name": "Dominic Schelling", "email": "dominic@bluecare.ch", "company": "BlueCare"},
                {"name": "Mike Iten", "email": "mike@bluecare.ch", "role": "participant"},
                {"name": "Speaker B", "role": "participant"}]
    m = validate_mapping([{"label": "Matthias", "kind": "self"},
                          {"label": "Tomi", "kind": "new", "name": "Dominic Schelling", "company": "BlueCare"}],
                         ["Matthias", "Tomi"])
    d = {r["name"]: r for r in participant_details_for(m, existing)}
    assert set(d) == {"Matthias Heim", "Dominic Schelling", "Mike Iten"}
    assert d["Dominic Schelling"]["email"] == "dominic@bluecare.ch"


import speaker_review as _sr  # noqa: E402
_REAL_WRITE = _sr._write_json_atomic


class _Cur:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params): self.log.append("update")


class _Conn:
    def __init__(self, fail_commit=False):
        self.log, self.fail_commit = [], fail_commit
    def __enter__(self): return self
    def __exit__(self, exc_type, *a):
        if exc_type is None and self.fail_commit:
            raise RuntimeError("commit failed")
        self.log.append("rollback" if exc_type else "commit")
        return False
    def cursor(self): return _Cur(self.log)
    def close(self): pass


def _confirm_env(tmp_path, monkeypatch, conn, *, fail_write=False):
    import speaker_review as sr
    path = tmp_path / "2026-10-29_15-02-46.json"
    path.write_text(json.dumps(_gem()), encoding="utf-8")
    monkeypatch.setattr(sr, "fetch_source", lambda sid: {
        "title": "t", "company": None, "metadata": {}, "participant_details": [], "participant_resolution_log": {},
        "insight_count": 0, "transcript_path": str(path), "has_decisions": False})
    monkeypatch.setattr(sr, "_conn", lambda: conn)
    monkeypatch.setattr(sr, "_write_json_atomic", (lambda p, d: (_ for _ in ()).throw(OSError("disk full")))
                        if fail_write else _REAL_WRITE)
    mapping = [{"label": "Matthias", "kind": "self"}, {"label": "Tomi", "kind": "unknown"}, {"label": "Ruedi", "kind": "unknown"}]
    return sr, path, mapping


def test_confirm_is_all_or_nothing(tmp_path, monkeypatch):
    original = json.dumps(_gem())
    conn = _Conn()
    sr, path, mapping = _confirm_env(tmp_path, monkeypatch, conn, fail_write=True)
    with pytest.raises(OSError):
        sr.confirm(1, mapping, trigger=False)
    assert conn.log == ["update", "rollback"] and path.read_text() == original

    conn = _Conn(fail_commit=True)
    sr, path, mapping = _confirm_env(tmp_path, monkeypatch, conn)
    with pytest.raises(RuntimeError, match="commit"):
        sr.confirm(1, mapping, trigger=False)
    assert path.read_text() == original                     # JSON put back

    conn = _Conn()
    sr, path, mapping = _confirm_env(tmp_path, monkeypatch, conn)
    out = sr.confirm(1, mapping, trigger=False)
    assert out["ok"] and conn.log == ["update", "commit"]
    assert "Unidentified participant 1:" in path.read_text()
    assert (tmp_path / "2026-10-29_15-02-46.pre-speaker-review.json").read_text() == original
