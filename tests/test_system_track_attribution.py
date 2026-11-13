"""Tests for tools/system_track_attribution.py: deterministic host/remote
attribution from the system track, valid on bleeding recordings.

Two-sided by design: every "it fixes X" test has a sibling asserting that the
correct input is left byte-identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import attribution_gate  # noqa: E402
import system_track_attribution as st  # noqa: E402
from gemini_processor import GeminiResult  # noqa: E402
from speaker_integrity import finalize_attribution  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEXT = "Das ist ein ausreichend langer Satz fuer die Messung hier."  # > 40 chars


def _fmt(sec: int) -> str:
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def build(turns):
    """turns: [(start_sec, label, text)] -> transcript string."""
    return "\n".join(f"{_fmt(t)} {label}: {text}" for t, label, text in turns)


def conversation(host="Matthias", other="Speaker 1", n=10, start=0, step=20,
                 invert=False):
    """Alternating turns, step seconds each. Remote speaks on even turns.

    Returns (turns, remote_intervals). With invert=True the LABELS are swapped
    relative to the audio (the incident: host label on the remote speaker).
    """
    turns, remote = [], []
    for i in range(n):
        t = start + i * step
        remote_speaks = i % 2 == 0
        if remote_speaks:
            remote.append((t, t + step - 1))
        label = other if remote_speaks else host
        if invert:
            label = host if remote_speaks else other
        turns.append((t, label, TEXT))
    return turns, remote


def run(turns, remote, duration=None, offsets=None, **kw):
    text = build(turns)
    duration = duration if duration is not None else turns[-1][0] + 20
    return text, st.run(text, st.RemoteActivity(remote, duration),
                        duration_sec=duration, chunk_offsets=offsets, **kw)


# --- the incident fix --------------------------------------------------------

def test_inverted_window_is_swapped():
    turns, remote = conversation(n=12, invert=True)
    text, (out, log) = run(turns, remote)
    assert log["ran"] and log["ok"]
    assert len(log["swaps"]) == 1 and log["swaps"][0]["lines_changed"] == 12
    expected, _ = conversation(n=12, invert=False)
    assert out == build(expected)
    assert log["labels_changed"] is True
    assert log["names_verified"] is False
    assert log["unresolved"] == 0


def test_correct_transcript_is_left_byte_identical():
    turns, remote = conversation(n=12, invert=False)
    text, (out, log) = run(turns, remote)
    assert log["ran"] and log["swaps"] == [] and log["relabels"] == []
    assert out.encode() == text.encode()
    assert log["labels_changed"] is False
    assert log["verified_coverage"] > 0.9


def test_only_the_inverted_chunk_is_swapped():
    a, ra = conversation(n=15, start=0, step=60, invert=True)       # 0-900
    b, rb = conversation(n=15, start=900, step=60, invert=False)    # 900-1800
    turns, remote = a + b, ra + rb
    text, (out, log) = run(turns, remote, duration=1800, offsets=[0.0, 900.0])
    assert [s["window"] for s in log["swaps"]] == [0]
    fixed = build(conversation(n=15, start=0, step=60)[0] + b)
    assert out == fixed
    # chunk 2 lines byte-identical to input
    assert out.split("\n")[15:] == text.split("\n")[15:]


def test_swap_needs_enough_evidence_on_both_labels():
    # 3 turns of 20 s: host label 40 s scorable (ok), other label 20 s (< 30 s)
    turns, remote = conversation(n=3, invert=True)
    _, (out, log) = run(turns, remote)
    assert log["swaps"] == []


# --- more than one remote label ----------------------------------------------

def three_party():
    # Matthias (host) never overlaps the system track; Philipp and Prisco do.
    turns, remote = [], []
    seq = ["Philipp", "Matthias", "Prisco", "Matthias", "Philipp", "Matthias",
           "Prisco", "Matthias", "Philipp", "Matthias", "Prisco", "Matthias"]
    for i, label in enumerate(seq):
        t = i * 20
        if label != "Matthias":
            remote.append((t, t + 19))
        turns.append((t, label, TEXT))
    return turns, remote


def test_three_label_window_never_merges_or_renames_remote_labels():
    turns, remote = three_party()
    text, (out, log) = run(turns, remote)
    assert out == text and log["swaps"] == [] and log["relabels"] == []
    labels_in = [l.split("] ")[1].split(":")[0] for l in text.split("\n")]
    labels_out = [l.split("] ")[1].split(":")[0] for l in out.split("\n")]
    assert labels_in == labels_out


def test_host_turn_measuring_remote_with_unknown_name_is_flagged_not_guessed():
    turns, remote = three_party()
    # turn 3 (host label) is actually a remote voice; neighbours differ
    # (Prisco before, Matthias... make neighbours different remote labels)
    turns[1] = (20, "Matthias", TEXT)
    remote.append((20, 39))          # system active during a host-labelled turn
    # neighbours of turn 1 are Philipp and Prisco: different -> unknown name
    text, (out, log) = run(turns, remote)
    assert out == text
    assert log["relabels"] == []
    assert any(u["scope"] == "turn" and u["label"] == "Matthias"
               for u in log["uncertain_regions"])
    assert log["unresolved"] >= 1


def test_host_turn_measuring_remote_between_same_remote_label_is_fixed():
    turns, remote = three_party()
    # turn 1 sits between Philipp (turn 0) and Philipp: make turn 2 Philipp
    turns[2] = (40, "Philipp", TEXT)
    remote.append((20, 39))
    text, (out, log) = run(turns, remote)
    assert [r["to"] for r in log["relabels"]] == ["Philipp"]
    assert out.split("\n")[1].startswith("[00:20] Philipp:")
    # no remote label was renamed into another remote label
    assert all(r["from"] == "Matthias" for r in log["relabels"])


# --- per-turn correction, two-party -------------------------------------------

def test_long_contradicting_turn_is_relabelled_weaker_is_only_flagged():
    turns, remote = conversation(n=12)
    # turn 1 (host, t=20..40) is acoustically remote for its whole length
    remote.append((20, 40))
    text, (out, log) = run(turns, remote)
    assert [r["line"] for r in log["relabels"]] == [2]
    assert out.split("\n")[1].startswith("[00:20] Speaker 1:")

    # weaker: 70 % of the turn on the system track -> uncertain, text unchanged
    turns, remote = conversation(n=12)
    remote.append((20, 34))
    text, (out, log) = run(turns, remote)
    assert out == text and log["relabels"] == []
    assert any(u["from_line"] == 2 for u in log["uncertain_regions"])


def test_remote_label_with_silent_system_track_is_relabelled_to_host():
    turns, remote = conversation(n=12)
    # turn 2 (remote label) has no system activity at all -> host spoke
    remote = [r for r in remote if r[0] != 40]
    text, (out, log) = run(turns, remote)
    assert [(r["from"], r["to"]) for r in log["relabels"]] == [("Speaker 1", "Matthias")]


def test_short_turns_and_backchannels_are_ignored():
    turns, remote = conversation(n=12)
    # a 2 s host-labelled "Mhm" inside remote speech, a 5 s long-text turn
    turns = turns[:2] + [(38, "Matthias", "Mhm."), (40, "Speaker 1", TEXT)] + turns[3:]
    remote.append((38, 40))
    text, (out, log) = run(turns, remote)
    assert out == text and log["relabels"] == [] and log["unresolved"] == 0
    # < 8 s but scorable-length turn is also never touched
    turns, remote = conversation(n=12)
    turns = turns[:3] + [(60, "Matthias", TEXT), (66, "Speaker 1", TEXT)] + turns[4:]
    remote.append((60, 79))
    text, (out, log) = run(turns, remote)
    assert out == text and log["relabels"] == [] and log["unresolved"] == 0


# --- skip and failure paths -----------------------------------------------------

def test_insufficient_remote_speech_does_not_run():
    turns, _ = conversation(n=6)
    text, (out, log) = run(turns, [(0, 10)])
    assert log["ran"] is False and log["reason"] == "insufficient_remote_speech"
    assert out == text


def test_no_remote_activity_or_no_self_label_or_namesake_do_not_run():
    turns, remote = conversation(n=6)
    text = build(turns)
    out, log = st.run(text, None, skip_reason="no_three_channel_layout")
    assert (log["ran"], log["reason"], out) == (False, "no_three_channel_layout", text)

    others = build([(t, "Speaker 2" if l == "Matthias" else l, x) for t, l, x in turns])
    out, log = st.run(others, st.RemoteActivity([(0, 200)]))
    assert (log["ran"], log["reason"], out) == (False, "no_self_label", others)

    out, log = st.run(text, st.RemoteActivity([(0, 200)]), namesake_remote=True)
    assert (log["ran"], log["reason"]) == (False, "host_namesake_remote_attendee")
    assert out == text


def test_exception_leaves_text_untouched():
    class Boom(st.RemoteActivity):
        def share(self, t0, t1):
            raise RuntimeError("boom")

    turns, remote = conversation(n=12, invert=True)
    text = build(turns)
    out, log = st.run(text, Boom(remote, 240), duration_sec=240)
    assert out == text
    assert log["ran"] is False and log["ok"] is False
    assert log["reason"].startswith("error: RuntimeError")


# --- VAD adapters ----------------------------------------------------------------

def test_remote_activity_from_vad_segments_counts_remote_and_both():
    segs = [(0, 10, "remote"), (10, 20, "both"), (20, 30, "host"), (40, 50, "remote")]
    r = st.RemoteActivity.from_segments(segs)
    assert r.total_seconds == 30.0
    assert r.share(0, 30) == 20 / 30
    assert r.share(20, 30) == 0.0


def test_remote_activity_from_vad_prefers_window_array():
    import numpy as np

    class Vad:
        window_sec = 0.5
        duration_sec = 5.0
        remote_active = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
        segments = []

    r = st.RemoteActivity.from_vad(Vad())
    assert r.total_seconds == 3.0 and r.share(0, 2) == 1.0 and r.share(2, 4) == 0.0
    assert st.RemoteActivity.from_vad(None) is None


def test_chunk_offsets_from_meta():
    assert st.chunk_offsets_for({"chunked": True, "chunk_count": 3}, 2550) == [0, 900, 1800]
    assert st.chunk_offsets_for({"chunked": False, "chunk_count": 1}, 600) == [0.0]
    assert st.chunk_offsets_for({}, None) == [0.0]


# --- regression: the 2026-11-12 incident, content-free fixture ---------------------

def _fixture():
    fx = json.loads((FIXTURES / "system_track_inverted_chunk1.json").read_text())
    lines = []
    for t, label, n in fx["turns"]:
        body = ("wort " * (n // 5 + 1))[:n] or "x"
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {label}: {body}")
    remote = st.RemoteActivity(fx["remote_intervals"], fx["audio_duration_seconds"])
    offsets = st.chunk_offsets_for(
        {"chunked": True, "chunk_count": fx["chunk_count"]},
        fx["audio_duration_seconds"])
    return fx, "\n".join(lines), remote, offsets


def test_incident_fixture_chunk_one_swapped_chunks_two_three_untouched():
    fx, text, remote, offsets = _fixture()
    out, log = st.run(text, remote, duration_sec=fx["audio_duration_seconds"],
                      chunk_offsets=offsets)
    assert log["ran"] and log["ok"]
    assert [s["window"] for s in log["swaps"]] == [0]
    swapped = {"Matthias": "Speaker 1", "Speaker 1": "Matthias"}
    for (t, label, _), a, b in zip(fx["turns"], text.split("\n"), out.split("\n")):
        if t < 900:
            assert b.split("] ", 1)[1].split(":", 1)[0] == swapped[label]
        else:
            assert a == b
    w = {x["index"]: x for x in log["windows"]}
    assert w[0]["orientation"] == "swapped"
    assert w[1]["orientation"] == "consistent" and w[2]["orientation"] == "consistent"
    assert log["unresolved"] == 0 and log["verified_coverage"] > 0.8


def test_incident_fixture_second_run_is_idempotent():
    fx, text, remote, offsets = _fixture()
    kw = dict(duration_sec=fx["audio_duration_seconds"], chunk_offsets=offsets)
    once, _ = st.run(text, remote, **kw)
    twice, log = st.run(once, remote, **kw)
    assert twice == once and log["swaps"] == []


# --- attribution report / gate ------------------------------------------------------

def _finalized(log, separation=None, coherence_ok=True):
    result = GeminiResult("[00:00] Matthias: Hi.", "de")
    result.channel_separation = separation or {
        "admissible": False, "reason": "mic_hears_remote", "state": "bleeding",
        "reference_uncertain_intervals": [], "host_bleed_rate": 0.88}
    result.speaker_coherence_log = {"ok": coherence_ok, "failed_windows": []}
    result.system_track_attribution = log
    result.partial = False
    finalize_attribution(result, {"transcript": result.transcript})
    return result.parsed_response


def _clean_log():
    turns, remote = conversation(n=12)
    return st.run(build(turns), st.RemoteActivity(remote, 240), duration_sec=240)[1]


def test_bleeding_mic_with_clean_system_track_check_does_not_hold_acoustic():
    payload = _finalized(_clean_log())
    report = payload["_meta"]["speaker_attribution"]
    assert "acoustic_identity_review" not in report["missing_stages"]
    assert report["identity_basis"] == "system_track_checked"
    assert report["system_track"]["remote_names_verified"] is False
    assert report["accuracy_measured"] is False
    decision = attribution_gate.gate(payload)
    assert "acoustic_identity_review" not in decision["missing_stages"]
    assert decision["speaker_dependent_actions"] == "require_turn_evidence"
    assert decision["allow_named_commitments"] is False


def test_bleeding_mic_without_system_track_stage_still_holds():
    payload = _finalized(None)
    report = payload["_meta"]["speaker_attribution"]
    assert "acoustic_identity_review" in report["missing_stages"]
    assert report["identity_basis"] == "inferred"
    assert attribution_gate.gate(payload)["speaker_dependent_actions"] == "hold"


def test_unresolved_system_track_contradiction_keeps_the_hold():
    turns, remote = conversation(n=12)
    remote.append((20, 34))          # weak contradiction -> uncertain region
    log = st.run(build(turns), st.RemoteActivity(remote, 240), duration_sec=240)[1]
    assert log["unresolved"] == 1
    payload = _finalized(log)
    report = payload["_meta"]["speaker_attribution"]
    assert "acoustic_identity_review" in report["missing_stages"]
    assert report["speaker_dependent_actions"] == "hold"
    assert any(u.get("source") == "system_track" for u in report["uncertain_regions"])
    assert attribution_gate.gate(payload)["speaker_dependent_actions"] == "hold"


def test_gate_holds_on_stored_system_track_contradiction_even_if_report_is_clean():
    payload = _finalized(_clean_log())
    payload["_meta"]["system_track_attribution"] = {
        "ran": True, "uncertain_regions": [{"from_line": 3}]}
    decision = attribution_gate.gate(payload)
    assert decision["speaker_dependent_actions"] == "hold"
    assert "acoustic_identity_review" in decision["missing_stages"]


def test_low_coverage_system_track_does_not_satisfy_acoustic():
    log = _clean_log()
    log["verified_coverage"] = 0.2
    report = _finalized(log)["_meta"]["speaker_attribution"]
    assert "acoustic_identity_review" in report["missing_stages"]


def test_other_gaps_are_not_loosened_by_a_clean_system_track():
    # coherence incomplete, partial coverage, mic reference gaps all still hold
    result = GeminiResult("[00:00] Matthias: Hi.", "de")
    result.channel_separation = {"admissible": False, "reference_uncertain_intervals": [[0, 120]]}
    result.speaker_coherence_log = {"ok": False}
    result.system_track_attribution = _clean_log()
    result.partial = True
    finalize_attribution(result, {"transcript": result.transcript})
    report = result.speaker_attribution
    assert set(report["missing_stages"]) == {
        "semantic_audit", "acoustic_identity_review", "coverage_recovery"}
    assert report["speaker_dependent_actions"] == "hold"
