"""Auto-confirm, non-participant labels, CRM matching and measurement in speaker_review.

Regression cases come from three real meetings. Their voice-match, system-track
and calendar metadata are in fixtures/speaker_auto_cases.json with every name
and address replaced (this repository is public); the transcript lines below
are invented and only reproduce each meeting's label shape:

  1572  host speaks under "Ich" and "Matthias", the client under "Andere" and
        "Lukas", a child talks after the goodbye -> auto-confirms, child cut out
  1583  the client contact was proposed as a NEW person although his CRM record
        was on the invite as "Keller, Marc" -> must resolve to the CRM contact
  1587  "Speaker 1" (46%) was never identified -> must ask
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import speaker_review as sr  # noqa: E402
from attribution_gate import gate  # noqa: E402
from speaker_review import (  # noqa: E402
    AUTO_CONFIRM,
    agreement_report,
    agreement_rows,
    auto_confirm_decision,
    calendar_attendees,
    companies_to_check,
    company_variants,
    confirmed_attribution,
    exclude_labels,
    label_stats,
    parse_proposal,
    participant_details_for,
    person_name,
    resolve_new_to_crm,
    same_answer,
    validate_mapping,
)

CASES = json.loads((Path(__file__).parent / "fixtures" / "speaker_auto_cases.json").read_text())

LUKAS = {"id": 4169, "name": "Lukas Brunner", "company": "Acme Health AG", "position": "Product Owner",
         "email": "lukas.brunner@acme-health.test"}
ACME_PEOPLE = [LUKAS,
               {"id": 4170, "name": "Peer Kunz", "company": "Acme Health AG", "position": "Developer"},
               {"id": 4171, "name": "Roel Vandewall", "company": "Acme Health AG", "position": "Head AM"}]
MARC = {"id": 4248, "name": "Marc Keller", "company": "Example Group (Example Holding)",
        "position": "Senior IT System Engineer", "email": "marc.keller@example-group.test", "score": 1.0}


def _lines(label, n, start_min, text="Wir haben das so umgesetzt und es läuft gut."):
    return [f"[{start_min + i // 6:02d}:{(i * 10) % 60:02d}] {label}: {text}" for i in range(n)]


def transcript_1572() -> str:
    """Shape of 1572: Ich 63, Andere 64, Lukas 16, Matthias 23, Child 7 lines."""
    rows = []
    rows += _lines("Ich", 63, 0, "Mit einem Spec kannst du den Entwicklungsschritt automatisieren.")
    rows += _lines("Andere", 64, 11, "Mir als PO fehlt noch der Überblick über die Pipeline.")
    rows += _lines("Lukas", 16, 30, "Das nehme ich so mit ins Team.")
    rows += _lines("Matthias", 22, 33, "Gerne, wir können auch zwei Stunden dafür reservieren.")
    rows.append("[36:00] Matthias: Super, tschüss Lukas!")
    rows += _lines("Child", 7, 37, "Papi, wie alt kann eine Hummel werden?")
    return "\n".join(rows)


def proposals_1572(child_kind="nonparticipant"):
    lukas = {"kind": "crm", "confidence": "high", "evidence": "PO at Acme, same voice as 'Lukas'.",
             **{k: LUKAS[k] for k in ("name", "company", "email", "position")}, "contact_id": LUKAS["id"]}
    self_ = {"kind": "self", "name": "Matthias Heim", "company": "Lailix", "confidence": "high", "evidence": "host"}
    return {"Ich": dict(self_), "Andere": dict(lukas), "Lukas": dict(lukas), "Matthias": dict(self_),
            "Child": {"kind": child_kind, "confidence": "high",
                      "evidence": "A child calls Matthias 'Papi' after the call."}}


def decide_1572(proposals=None, **kw):
    t = transcript_1572()
    meta = CASES["1572"]
    args = dict(transcript=t, voice_log=meta["voice_match"], system_track=meta["system_track_attribution"],
                attendee_emails=set(), calendar_authoritative=True, involved_companies=["Acme Health"],
                involved_people=ACME_PEOPLE, gate_policy="hold")
    args.update(kw)
    return auto_confirm_decision(label_stats(t), proposals or proposals_1572(), **args)


# ── 1572: auto-confirms, the child is not part of the meeting ────────────────

def test_1572_auto_confirms_with_child_excluded():
    d = decide_1572()
    assert d["auto"], d["reasons"]
    m = {e["label"]: e for e in d["mapping"]}
    assert m["Ich"]["kind"] == m["Matthias"]["kind"] == "self"
    assert m["Andere"]["contact_id"] == m["Lukas"]["contact_id"] == 4169
    assert m["Child"]["kind"] == "nonparticipant"
    # Evidence is independent of the model: voice for "Ich", host channel for "Matthias",
    # the name said in the meeting at an involved company for the contact.
    assert any("voice matches Matthias" in e for e in d["evidence"]["Ich"])
    assert any("host channel" in e for e in d["evidence"]["Matthias"])
    assert any("called 'Lukas'" in e for e in d["evidence"]["Andere"])


def test_1572_stored_proposal_child_unknown_is_minor_and_does_not_block():
    """The proposal stored before this change called the child "unknown"."""
    d = decide_1572(proposals_1572(child_kind="unknown"))
    assert d["auto"]
    assert {e["label"]: e["kind"] for e in d["mapping"]}["Child"] == "unknown"


def test_contact_needs_uniqueness_at_the_company():
    """Two Lukases at the involved company: the spoken name no longer identifies him."""
    people = ACME_PEOPLE + [{"id": 5000, "name": "Lukas Other", "company": "Acme Health AG"}]
    d = decide_1572(involved_people=people)
    assert not d["auto"] and any("no evidence beyond the suggestion" in r for r in d["reasons"])


def test_contact_at_uninvolved_company_has_no_evidence():
    d = decide_1572(involved_companies=["Some Other AG"])
    assert not d["auto"]


def test_only_held_meetings_are_auto_confirmed():
    d = decide_1572(gate_policy="resolve_then_extract")
    assert not d["auto"] and "not waiting" in d["reasons"][0] and d["mapping"] == []


def test_failed_suggestion_never_auto_confirms():
    assert not decide_1572(review_error="timeout")["auto"]


# ── 1587: a substantial unknown label asks; a bare self label never passes ───

def transcript_1587() -> str:
    return "\n".join(_lines("Speaker 1", 53, 0, "Ich würde das auf meinem MacBook machen.")
                     + _lines("Matthias", 62, 6, "Frag einfach Claude, der hat Zugriff auf Railway."))


def proposals_1587(speaker1=None):
    return {"Speaker 1": speaker1 or {"kind": "unknown", "confidence": "low", "evidence": "Never named."},
            "Matthias": {"kind": "self", "name": "Matthias Heim", "confidence": "high",
                         "evidence": "Speaker 1 says 'Tschüss, Matthias'."}}


def test_1587_does_not_auto_confirm():
    meta = CASES["1587"]
    t = transcript_1587()
    d = auto_confirm_decision(label_stats(t), proposals_1587(), transcript=t, voice_log=meta["voice_match"],
                              system_track=meta["system_track_attribution"], gate_policy="hold")
    assert not d["auto"]
    assert any("'Speaker 1' (46%) is unknown at low confidence" in r for r in d["reasons"])


def test_self_label_text_alone_is_not_evidence():
    """Even with the other speaker solid, "Matthias" needs host channel or voice."""
    t = transcript_1587()
    peer = {"kind": "crm", "contact_id": 4170, "name": "Peer Kunz", "company": "Acme Health AG",
            "email": "peer.kunz@acme-health.test", "confidence": "high", "evidence": "x"}
    d = auto_confirm_decision(label_stats(t), proposals_1587(peer), transcript=t, voice_log=None, system_track=None,
                              attendee_emails={"peer.kunz@acme-health.test"}, calendar_authoritative=True,
                              gate_policy="hold")
    assert not d["auto"]
    assert any("'Matthias' (54%) as Matthias has no host-channel or voice evidence" in r for r in d["reasons"])
    # With the real channel evidence of 1587 it passes: the self label sat on the host mic.
    meta = CASES["1587"]
    d = auto_confirm_decision(label_stats(t), proposals_1587(peer), transcript=t, voice_log=meta["voice_match"],
                              system_track=meta["system_track_attribution"],
                              attendee_emails={"peer.kunz@acme-health.test"}, calendar_authoritative=True,
                              gate_policy="hold")
    assert d["auto"], d["reasons"]


def test_self_label_measured_as_remote_is_contradicted():
    t = transcript_1587()
    st = {"ran": True, "windows": [{"index": 0, "orientation": "consistent", "labels": {
        "Matthias": {"class": "remote", "remote_share": 0.9}}}]}
    d = auto_confirm_decision(label_stats(t), proposals_1587(), transcript=t, system_track=st, gate_policy="hold")
    assert any("contradicted: measured as remote audio" in r for r in d["reasons"])


def test_voice_clearly_matching_someone_else_blocks_a_contact():
    d = decide_1572(voice_log={"ran": True, "labels": {"Andere": {
        "verdict": "clear", "best": {"name": "Roel Vandewall", "score": 0.82, "crm_contact_id": 4171}}}})
    assert not d["auto"] and any("clearly matches Roel Vandewall" in r for r in d["reasons"])


# ── 1583: an existing CRM contact is never stored as "new" ───────────────────

def test_1583_calendar_attendee_resolves_new_to_crm():
    attendees, authoritative = calendar_attendees(CASES["1583"]["calendar_search"])
    assert authoritative
    assert {"name": "Marc Keller", "email": "marc.keller@example-group.test", "company": None} in attendees
    assert all(a["email"] != "matthias@lailix.com" for a in attendees)
    props = {"Marc": {"kind": "new", "name": "Marc Keller", "company": "Example Group", "confidence": "high",
                      "evidence": "Addressed as Marc; only Marc on the invite."}}
    extra = resolve_new_to_crm(props, lookup=lambda n: [], attendee_contacts=[MARC],
                               involved_companies=["Example Group"], company_people={})
    assert props["Marc"]["kind"] == "crm" and props["Marc"]["contact_id"] == 4248
    assert "calendar invite" in props["Marc"]["crm_match"] and extra["Marc"][0]["id"] == 4248


def test_1583_full_name_lookup_at_consistent_company():
    unrelated = [{"id": 1403, "name": "Marc Defila", "company": "Other GmbH", "score": 0.75}]
    props = {"Marc": {"kind": "new", "name": "Marc Keller", "company": "Example Group", "confidence": "high"}}
    resolve_new_to_crm(props, lookup=lambda n: [MARC] + unrelated, attendee_contacts=[],
                       involved_companies=["Example Group"], company_people={})
    assert props["Marc"]["contact_id"] == 4248
    # Same name at an unrelated company: stays new, but is offered as a candidate.
    props = {"Marc": {"kind": "new", "name": "Marc Keller", "company": "Elsewhere AG", "confidence": "high"}}
    extra = resolve_new_to_crm(props, lookup=lambda n: [MARC], attendee_contacts=[], involved_companies=[],
                               company_people={})
    assert props["Marc"]["kind"] == "new" and extra["Marc"][0]["id"] == 4248


def test_first_name_only_label_gets_the_involved_companys_contact():
    people = {"Example Group": [MARC, {"id": 4245, "name": "Robert Muster", "company": "Example Group"}]}
    props = {"Marc": {"kind": "new", "name": "Marc", "confidence": "medium"}}
    resolve_new_to_crm(props, lookup=lambda n: [], attendee_contacts=[], involved_companies=["Example Group"],
                       company_people=people)
    assert props["Marc"]["contact_id"] == 4248


def test_attendee_domains_bring_their_company_into_view():
    out = companies_to_check(title="2026-11-26_16-01-36", source_company=None,
                             calendar=CASES["1583"]["calendar_search"], transcript="Wir haben Azure analysiert.",
                             client_names=["Example Group (Example Holding)"],
                             domain_companies=["Example Group (Example Holding)"])
    assert out == ["Example Group (Example Holding)"]
    # A client folder name is found in the transcript under its short form.
    out = companies_to_check(title="t", source_company=None, calendar={}, transcript="bei der Example Group läuft",
                             client_names=["Example Group (Example Holding)", "Acme AG"])
    assert out == ["Example Group"]
    assert company_variants("Example Group (Example Holding)") == [
        "Example Group (Example Holding)", "Example Group", "Example Holding"]


def test_1583_one_line_transcript_never_auto_confirms():
    t = "[00:00] Marc: Also [00:02] Marc: ähm [00:03] Marc: ich habe verschiedene Dinge gemacht"
    props = {"Marc": {"kind": "crm", "contact_id": 4248, "name": "Marc Keller", "confidence": "high",
                      "email": MARC["email"], "company": MARC["company"]}}
    d = auto_confirm_decision(label_stats(t), props, transcript=t,
                              attendee_emails={MARC["email"]}, calendar_authoritative=True, gate_policy="hold")
    assert not d["auto"] and any("labelled lines" in r for r in d["reasons"])


def test_last_first_names_merge_with_the_confirmed_person():
    assert person_name("Keller, Marc") == "Marc Keller" and person_name("Marc Keller") == "Marc Keller"
    existing = [{"name": "Keller, Marc", "email": "marc.keller@example-group.test", "role": "participant"},
                {"name": "Muster, Robert", "email": "robert.muster@example-group.test", "role": "participant"}]
    m = validate_mapping([{"label": "Marc", "kind": "crm", "name": "Marc Keller", "contact_id": 4248}], ["Marc"])
    d = participant_details_for(m, existing)
    assert [r["name"] for r in d] == ["Marc Keller", "Muster, Robert"]
    assert d[0]["email"] == "marc.keller@example-group.test"


# ── Non-participant labels ───────────────────────────────────────────────────

def _gem_1572():
    return {"transcript": transcript_1572(),
            "participants": [{"name": n} for n in ("Ich", "Andere", "Lukas", "Matthias", "Child")],
            "speaker_emotions": [{"speaker": "Child", "arc": ["playful"]}, {"speaker": "Andere", "arc": ["calm"]}],
            "speaker_pacing": {"Child": {"wpm": 90}, "Andere": {"wpm": 120}},
            "interruptions": [{"interrupter": "Child", "interruptee": "Matthias"}],
            "energy_levels": {"Child": 3},
            "_meta": {"speaker_attribution": {"status": "needs_review", "speaker_dependent_actions": "hold",
                                              "missing_stages": ["acoustic_identity_review"]}}}


def test_exclude_labels_cuts_lines_and_keeps_them_recoverable():
    g = _gem_1572()
    before = g["transcript"].split("\n")
    n = exclude_labels(g, {"Child"}, now="now", by="auto")
    assert n == 7 and "Hummel" not in g["transcript"]
    assert "Child" not in {p["name"] for p in g["participants"]}
    assert not g["interruptions"] and "Child" not in g["speaker_pacing"] and "Child" not in g["energy_levels"]
    kept = g["_meta"]["excluded_speech"]
    assert kept["labels"] == ["Child"] and len(kept["lines"]) == 7
    assert all(before[x["line_index"]] == x["raw"] for x in kept["lines"])


def test_nonparticipant_is_a_valid_answer_but_not_for_the_only_speaker():
    labels = ["Matthias", "Child"]
    m = validate_mapping([{"label": "Matthias", "kind": "self"}, {"label": "Child", "kind": "nonparticipant"}], labels)
    assert m[1] == {"label": "Child", "kind": "nonparticipant"}
    with pytest.raises(ValueError, match="only speaker"):
        validate_mapping([{"label": "Child", "kind": "nonparticipant"}], ["Child"])
    assert parse_proposal('{"speakers": [{"label": "Child", "kind": "nonparticipant", "confidence": "high"}]}',
                          ["Child"], {})["Child"]["kind"] == "nonparticipant"


def test_large_nonparticipant_share_asks():
    props = proposals_1572()
    props["Andere"] = {"kind": "nonparticipant", "confidence": "high", "evidence": "?"}
    d = decide_1572(props)
    assert not d["auto"] and any("not in the meeting" in r for r in d["reasons"])


def test_auto_attribution_is_recorded_apart_from_human():
    g = _gem_1572()
    decision = {"rule": "auto-v1", "evidence": {"Ich": ["voice matches Matthias (0.84)"]}}
    g["_meta"]["speaker_attribution"] = confirmed_attribution(g["_meta"]["speaker_attribution"], "now", auto=decision)
    sa = g["_meta"]["speaker_attribution"]
    assert sa["status"] == "auto_confirmed" and sa["identity_basis"] == "auto_rule" and sa["auto_rule"] == "auto-v1"
    assert gate(g)["speaker_dependent_actions"] == "require_turn_evidence"
    human = confirmed_attribution(sa, "later")
    assert human["status"] == "human_confirmed" and "auto_rule" not in human


# ── confirm end to end (DB and hand-off faked) ───────────────────────────────

class _Cur:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params):
        self.log.append("delete_chunks" if "source_chunks" in sql else "update")
        if "UPDATE sources" in sql:
            self.log.append(("review", json.loads(params[5]), json.loads(params[4]), params[0]))


class _Conn:
    def __init__(self): self.log = []
    def __enter__(self): return self
    def __exit__(self, exc_type, *a):
        self.log.append("rollback" if exc_type else "commit")
        return False
    def cursor(self): return _Cur(self.log)
    def close(self): pass


def _env(tmp_path, monkeypatch, review, insights=0):
    path = tmp_path / "2026-11-25_17-58-55.json"
    path.write_text(json.dumps(_gem_1572()), encoding="utf-8")
    conn, calls = _Conn(), []
    monkeypatch.setattr(sr, "fetch_source", lambda sid: {
        "title": "t", "company": None, "metadata": {"speaker_review": review}, "participant_details": [],
        "participant_resolution_log": {}, "insight_count": insights, "transcript_path": str(path),
        "has_decisions": False})
    monkeypatch.setattr(sr, "_conn", lambda: conn)
    monkeypatch.setattr(sr, "hand_off", lambda sid, p, suffix: calls.append(suffix) or {"started": True})
    monkeypatch.setattr(sr, "enrol_voices", lambda *a: calls.append("enrol") or {"enrolled": []})
    monkeypatch.setattr(sr, "_flag_insights", lambda sid: calls.append("flag") or "flagged")
    return path, conn, calls


def test_auto_confirm_applies_the_1572_answer(tmp_path, monkeypatch):
    t = transcript_1572()
    review = {"status": "proposed", "proposed_at": "p", "model": "sonnet",
              "labels": [dict(lb, proposal=proposals_1572()[lb["label"]], candidates=[]) for lb in label_stats(t)]}
    path, conn, calls = _env(tmp_path, monkeypatch, review, insights=15)
    d = decide_1572()
    out = sr.confirm(1572, d["mapping"], auto=d)
    assert out["confirmed_by"] == "auto" and out["lines_excluded"] == 7 and out["handoff"] == "followup_only"
    saved = json.loads(path.read_text())
    assert "Hummel" not in saved["transcript"] and len(saved["_meta"]["excluded_speech"]["lines"]) == 7
    assert saved["_meta"]["speaker_attribution"]["status"] == "auto_confirmed"
    assert {s["label"] for s in label_stats(saved["transcript"])} == {"Matthias Heim", "Lukas Brunner"}
    _, stored, attribution, content = next(x for x in conn.log if isinstance(x, tuple))
    assert "Hummel" not in content and "delete_chunks" in conn.log          # stale chunks dropped
    assert stored["confirmed_by"] == "auto" and stored["excluded"] == ["Child"]
    assert stored["original"]["labels"][4]["proposal"]["kind"] == "nonparticipant"   # kept for measurement
    assert stored["auto"]["decision"] == "auto_confirmed" and attribution["auto_evidence"]["Matthias"]
    assert "enrol" not in calls and "flag" in calls                          # no voice enrolment from the rule
    assert calls[-1].startswith("\nSPEAKERS AUTO-CONFIRMED") and "Step 2" in calls[-1]


def test_human_correction_after_auto_keeps_the_original(tmp_path, monkeypatch):
    t = transcript_1572()
    original = {"proposed_at": "p", "labels": [{"label": lb["label"], "share": lb["share"], "lines": lb["lines"],
                                                "proposal": proposals_1572()[lb["label"]]} for lb in label_stats(t)]}
    first = [{"label": "Ich", "kind": "self", "new_label": "Matthias Heim"},
             {"label": "Matthias", "kind": "self", "new_label": "Matthias Heim"},
             {"label": "Andere", "kind": "crm", "contact_id": 4169, "name": "Lukas Brunner", "new_label": "Lukas Brunner"},
             {"label": "Lukas", "kind": "crm", "contact_id": 4169, "name": "Lukas Brunner", "new_label": "Lukas Brunner"},
             {"label": "Child", "kind": "nonparticipant", "new_label": None}]
    review = {"status": "confirmed", "confirmed_by": "auto", "original": original,
              "history": [{"by": "auto", "mapping": first}], "auto": {"rule": "auto-v1"}}
    # Matthias corrects "Lukas Brunner" to Roel in the portal.
    second = [{"label": "Matthias Heim", "kind": "self", "new_label": "Matthias Heim"},
              {"label": "Lukas Brunner", "kind": "crm", "contact_id": 4171, "name": "Roel Vandewall",
               "new_label": "Roel Vandewall"}]
    rows = agreement_rows(dict(review, history=review["history"] + [{"by": "human", "mapping": second}]))
    by = {r["label"]: r for r in rows}
    assert by["Ich"]["agree"] and not by["Andere"]["agree"] and by["Andere"]["corrected_later"]
    assert by["Child"]["agree"] and by["Andere"]["mode"] == "auto"
    rep = agreement_report(rows)
    assert rep["by_mode"]["auto"]["labels"] == 5 and rep["auto_corrected_later"] == 2
    assert rep["by_confidence"]["high"]["agree"] == 3


def test_same_answer_counts_new_that_became_the_crm_contact():
    assert same_answer({"kind": "new", "name": "Marc Keller"}, {"kind": "crm", "name": "Marc Keller", "contact_id": 1})
    assert not same_answer({"kind": "self"}, {"kind": "crm", "contact_id": 4170, "name": "Peer Kunz"})
    assert same_answer({"kind": "crm", "contact_id": "17"}, {"kind": "crm", "contact_id": 17})


def test_thresholds_live_in_one_place():
    assert set(AUTO_CONFIRM) >= {"minor_max_share", "minor_max_lines", "nonparticipant_max_share",
                                 "min_total_lines", "require_self", "eligible_policies"}
