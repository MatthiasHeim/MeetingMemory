"""Tests for speaker_coherence — semantic speaker-attribution repair.

Two-sided by construction. Every repair test has a mirror asserting that a
transcript which is ALREADY correct passes through byte-identical: a gate that
only ever rewrites looks just as healthy when it has been gutted into
"relabel everything", and that is precisely the failure this pipeline has
produced before (`singleton_collapse` on source 434).

The model call is injected, so these exercise the parts that can silently
corrupt a transcript — index handling, the allowed-label universe, the
confidence filter, the runaway cap — without a network round trip. The
end-to-end behaviour of the prompt itself was validated live against source
767; see docs/SPEC-speaker-attribution-2026-08-07.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from speaker_coherence import (  # noqa: E402
    MAX_RELABEL_FRACTION,
    canonical_label_map,
    check_and_repair,
    parse_lines,
    parse_response,
    plan_windows,
    revert_verify_flips,
)

FIXTURE = Path(__file__).parent / "fixtures" / "source767_region.txt"

ATTENDEES = [
    {"name": "Matthias Heim", "role": "self", "company": "Lailix"},
    {"name": "Philipp Baltensperger", "role": "attendee", "company": "BlueCare"},
]


def stub(payload: dict):
    """A runner returning a fixed model payload."""
    return lambda _prompt: json.dumps(payload)


def labels_by_ts(transcript: str) -> dict[str, str]:
    return {ln["ts"]: ln["label"] for ln in parse_lines(transcript)}


# ── the source-767 regression fixture ─────────────────────────────────────
#
# Real transcript region from 2026-08-07_14-11-24 (BlueCare 1:1, Matthias
# Heim + Philipp Baltensperger) as the pipeline actually stored it. Ground
# truth below is the subset Matthias confirmed by hand.

GROUND_TRUTH = {
    "13:26": "Matthias",   # asks BlueCare's release-process question
    "13:58": "Philipp",    # ...and this ANSWERS it — was labelled Matthias
    "15:07": "Matthias",   # his own feature-flag argument — was "Speaker 2"
    "15:29": "Matthias",   # ditto, deploy-opacity — was "Speaker 2"
    "15:36": "Matthias",   # asks whether it relates to the new technology
    "15:39": "Philipp",    # ...and this ANSWERS it — was labelled Matthias
}

# What a correct audit returns for that region.
REPAIR_PAYLOAD = {
    "speakers": [
        {"label": "Speaker 2", "identity": "Philipp Baltensperger",
         "evidence": "describes BlueCare's own release packaging"},
    ],
    "relabels": [
        {"line": 11, "from": "Matthias", "to": "Speaker 2", "confidence": "high",
         "reason": "answers the release-process question Matthias asks on line 10"},
        {"line": 14, "from": "Speaker 2", "to": "Matthias", "confidence": "high",
         "reason": "continues Matthias's own incremental-release argument"},
        {"line": 16, "from": "Speaker 2", "to": "Matthias", "confidence": "high",
         "reason": "same speaker's argument about deploy-process opacity"},
        {"line": 18, "from": "Matthias", "to": "Speaker 2", "confidence": "high",
         "reason": "answers the question Matthias asks on line 17"},
    ],
    "uncertain_regions": [],
}


def load_fixture() -> dict:
    return {"transcript": FIXTURE.read_text(encoding="utf-8"),
            "participants": []}


def test_fixture_line_numbers_match_ground_truth():
    """Guard the fixture itself: the line numbers REPAIR_PAYLOAD references
    must still be the timestamps we think they are. Without this the repair
    test could pass by rewriting the wrong lines after a fixture edit."""
    lines = parse_lines(load_fixture()["transcript"])
    expected = {10: "13:26", 11: "13:58", 14: "15:07",
                16: "15:29", 17: "15:36", 18: "15:39"}
    for n, ts in expected.items():
        assert lines[n - 1]["ts"] == ts, f"line {n} is {lines[n-1]['ts']}, want {ts}"


def test_source767_defects_are_repaired():
    """SIDE A — the confirmed misattributions get fixed."""
    g = load_fixture()
    before = labels_by_ts(g["transcript"])
    # The four confirmed-wrong lines really are wrong going in.
    assert before["13:58"] == "Matthias"
    assert before["15:07"] == "Speaker 2"
    assert before["15:29"] == "Speaker 2"
    assert before["15:39"] == "Matthias"

    log = check_and_repair(g, known_attendees=ATTENDEES,
                           runner=stub(REPAIR_PAYLOAD))
    after = labels_by_ts(g["transcript"])

    assert log["ok"] and log["changed"]
    assert len(log["relabels_applied"]) == 4
    for ts, who in GROUND_TRUTH.items():
        got = after[ts]
        if who == "Matthias":
            assert got == "Matthias", f"[{ts}] became {got!r}, expected Matthias"
        else:
            assert got == "Philipp", f"[{ts}] became {got!r}, expected Philipp"


def test_source767_correct_lines_survive_untouched():
    """SIDE B — the lines that were already right must not move, and neither
    must the text. Only the label field of a repaired line may change."""
    g = load_fixture()
    original = g["transcript"]
    check_and_repair(g, known_attendees=ATTENDEES, runner=stub(REPAIR_PAYLOAD))

    before_lines = parse_lines(original)
    after_lines = parse_lines(g["transcript"])
    assert len(before_lines) == len(after_lines)
    changed = 0
    for b, a in zip(before_lines, after_lines):
        assert b["ts"] == a["ts"]
        assert b["rest"] == a["rest"], f"[{b['ts']}] transcript TEXT was altered"
        if b["label"] != a["label"]:
            changed += 1
    # Exactly 7: five lines end up as Philipp (1, 3, 8 via the identity
    # binding; 11 and 18 relabelled from Matthias then bound) and two as
    # Matthias (14, 16). Every other line keeps the label it came in with.
    assert changed == 7, f"{changed} labels moved; expected 7"
    assert labels_by_ts(original)["13:26"] == labels_by_ts(g["transcript"])["13:26"]
    assert labels_by_ts(g["transcript"])["15:36"] == "Matthias"


def test_already_correct_transcript_passes_through_unchanged():
    """SIDE B, the strong form — a coherent transcript with an empty relabel
    set must come out byte-identical."""
    g = load_fixture()
    original = g["transcript"]
    log = check_and_repair(
        g, known_attendees=None,
        runner=stub({"speakers": [], "relabels": [], "uncertain_regions": []}),
    )
    assert log["ok"] and log["ran"]
    assert log["changed"] is False
    assert g["transcript"] == original


# ── integrity guards: the ways a repair pass corrupts a transcript ────────


def test_stale_line_index_is_rejected():
    """A proposal whose `from` doesn't match the line's current label is a
    wrong index — applying it would relabel an innocent line."""
    g = load_fixture()
    original = g["transcript"]
    log = check_and_repair(g, known_attendees=None, runner=stub({
        "relabels": [{"line": 11, "from": "Speaker 2", "to": "Matthias",
                      "confidence": "high", "reason": "off-by-one"}],
    }))
    assert log["changed"] is False
    assert g["transcript"] == original
    assert log["relabels_rejected"][0]["cause"] == "from_label_mismatch"
    assert log["relabels_rejected"][0]["actual_label"] == "Matthias"


def test_hallucinated_name_is_rejected():
    """`to` must be an existing label, the host, or a calendar attendee."""
    g = load_fixture()
    log = check_and_repair(g, known_attendees=None, runner=stub({
        "relabels": [{"line": 11, "from": "Matthias", "to": "Antonella",
                      "confidence": "high", "reason": "invented"}],
    }))
    assert log["changed"] is False
    assert log["relabels_rejected"][0]["cause"] == "target_not_allowed"


def test_medium_and_low_confidence_do_not_rewrite():
    g = load_fixture()
    original = g["transcript"]
    log = check_and_repair(g, known_attendees=None, runner=stub({
        "relabels": [
            {"line": 11, "from": "Matthias", "to": "Speaker 2",
             "confidence": "medium", "reason": "probably"},
            {"line": 14, "from": "Speaker 2", "to": "Matthias",
             "confidence": "low", "reason": "maybe"},
        ],
        "uncertain_regions": [{"from_line": 11, "to_line": 14,
                               "reason": "rapid exchange"}],
    }))
    assert g["transcript"] == original
    assert log["changed"] is False
    assert {r["cause"] for r in log["relabels_rejected"]} == {"not_high_confidence"}
    # ...and the doubt is recorded rather than swallowed.
    assert log["uncertain_regions"] == [
        {"from_line": 11, "to_line": 14, "reason": "rapid exchange"}
    ]


def test_runaway_relabel_set_is_refused_wholesale():
    """Re-diarizing the meeting is not a repair. Above the cap, apply none."""
    g = load_fixture()
    original = g["transcript"]
    lines = parse_lines(original)
    proposals = [
        {"line": n, "from": ln["label"],
         "to": "Matthias" if ln["label"] != "Matthias" else "Speaker 2",
         "confidence": "high", "reason": "rewrite everything"}
        for n, ln in enumerate(lines, start=1)
    ]
    assert len(proposals) > len(lines) * MAX_RELABEL_FRACTION
    log = check_and_repair(g, known_attendees=None,
                           runner=stub({"relabels": proposals}))
    assert log["refused_runaway"] is True
    assert log["changed"] is False
    assert g["transcript"] == original


def test_runner_failure_leaves_transcript_intact_and_reports_not_ok():
    """Fail loud: the transcript survives, but the log says it is unverified."""
    g = load_fixture()
    original = g["transcript"]

    def boom(_prompt):
        raise RuntimeError("claude exited 1: session limit reached")

    log = check_and_repair(g, known_attendees=None, runner=boom)
    assert log["ok"] is False
    assert log["ran"] is False
    assert "session limit" in log["error"]
    assert g["transcript"] == original


def test_unparseable_response_is_not_ok():
    g = load_fixture()
    original = g["transcript"]
    log = check_and_repair(g, known_attendees=None,
                           runner=lambda _p: "I could not determine speakers.")
    assert log["ok"] is False
    assert g["transcript"] == original


# ── identity binding (the `participants: []` failure on source 767) ───────


def test_generic_label_binds_to_calendar_attendee():
    g = load_fixture()
    log = check_and_repair(g, known_attendees=ATTENDEES, runner=stub({
        "speakers": [{"label": "Speaker 2",
                      "identity": "Philipp Baltensperger",
                      "evidence": "describes BlueCare's release process"}],
        "relabels": [],
    }))
    assert log["identity_bindings"][0]["to"] == "Philipp"
    assert "Speaker 2:" not in g["transcript"]
    assert "Philipp:" in g["transcript"]


def test_identity_not_on_the_calendar_is_not_bound():
    """The audio may suggest a name; only the calendar may introduce one."""
    g = load_fixture()
    original = g["transcript"]
    log = check_and_repair(g, known_attendees=ATTENDEES, runner=stub({
        "speakers": [{"label": "Speaker 2", "identity": "Lukas Meier",
                      "evidence": "guessed from voice"}],
        "relabels": [],
    }))
    assert log["identity_bindings"] == []
    assert g["transcript"] == original


# ── response parsing ──────────────────────────────────────────────────────


def test_parse_response_handles_fenced_and_prefixed_json():
    payload = '{"relabels": [], "speakers": []}'
    assert parse_response(payload)["relabels"] == []
    assert parse_response(f"```json\n{payload}\n```")["relabels"] == []
    assert parse_response(f"Here is the audit:\n{payload}\n")["relabels"] == []


def test_parse_response_survives_braces_inside_strings():
    raw = ('{"relabels": [], "speakers": [{"label": "A", '
           '"identity": null, "evidence": "said \\"} not the end\\" here"}]}')
    assert parse_response(raw)["speakers"][0]["label"] == "A"


# ── flip revert (backfill helper) ─────────────────────────────────────────


def test_revert_verify_flips_restores_the_original_labels():
    g = {
        "transcript": ("[13:26] Matthias: Frag.\n"
                       "[13:58] Matthias: Antwort.\n"
                       "[14:13] Matthias: Okay.\n"),
        "speaker_verification": {"flips": [
            {"time_span": "[13:26]-[13:58]", "from": "Speaker 2",
             "to": "Matthias"},
        ]},
    }
    assert revert_verify_flips(g) == 1
    assert g["transcript"].startswith("[13:26] Speaker 2: Frag.")
    assert "[13:58] Matthias: Antwort." in g["transcript"]


def test_revert_skips_lines_something_else_has_since_rewritten():
    """If the label is no longer the flip's `to`, a later stage owns it —
    reverting would clobber that instead of undoing the flip."""
    g = {
        "transcript": "[13:26] Philipp: Frag.\n[13:58] Matthias: Antwort.\n",
        "speaker_verification": {"flips": [
            {"time_span": "[13:26]-[13:58]", "from": "Speaker 2",
             "to": "Matthias"},
        ]},
    }
    assert revert_verify_flips(g) == 0
    assert "[13:26] Philipp: Frag." in g["transcript"]


def test_revert_is_a_noop_without_a_verification_log():
    g = {"transcript": "[00:00] Matthias: Hallo.\n"}
    assert revert_verify_flips(g) == 0
    assert g["transcript"] == "[00:00] Matthias: Hallo.\n"


# ── windowing ─────────────────────────────────────────────────────────────


def test_windows_tile_the_transcript_without_editable_overlap():
    wins = plan_windows(119)
    editable = []
    for _ctx, start, end in wins:
        editable.extend(range(start, end + 1))
    assert editable == list(range(1, 120)), "editable ranges must tile exactly"
    assert all(c <= s for c, s, _ in wins)


def test_short_transcript_is_a_single_window():
    assert plan_windows(20) == [(1, 1, 20)]


def test_window_may_not_relabel_its_context_lead_in():
    """A window's read-only lead-in belongs to the previous window. A
    proposal there must be dropped, not tie-broken."""
    lines = [f"[{i//60:02d}:{i%60:02d}] Matthias: Zeile {i}." for i in range(70)]
    g = {"transcript": "\n".join(lines) + "\n", "participants": []}
    calls = []

    def runner(prompt):
        calls.append(prompt)
        # Always try to relabel line 1, which only the FIRST window owns.
        return json.dumps({"relabels": [
            {"line": 1, "from": "Matthias", "to": "Speaker 2",
             "confidence": "high", "reason": "claiming a line I may not own"},
        ]})

    log = check_and_repair(g, known_attendees=None, runner=runner)
    assert log["windows"] > 1
    # Every window proposes it; only the owner's proposal survives the filter.
    assert log["relabels_proposed"] == 1


def test_partial_window_failure_is_reported_not_swallowed():
    lines = [f"[{i//60:02d}:{i%60:02d}] Matthias: Zeile {i}." for i in range(70)]
    g = {"transcript": "\n".join(lines) + "\n", "participants": []}
    state = {"n": 0}

    def flaky(prompt):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("claude exited 1: session limit reached")
        return json.dumps({"relabels": []})

    log = check_and_repair(g, known_attendees=None, runner=flaky)
    assert log["ran"] is True
    assert len(log["failed_windows"]) == 1
    assert "were not audited" in log["error"]


# ── one person, one label ─────────────────────────────────────────────────


def test_full_name_and_first_name_do_not_split_one_speaker():
    """Regression, source 767 end-to-end run: a relabel targeted
    'Philipp Baltensperger' while the identity binding wrote 'Philipp', so the
    transcript ended up with two speakers for one person. Both forms are in
    the allowed universe by design, so the write path must canonicalise."""
    g = load_fixture()
    log = check_and_repair(g, known_attendees=ATTENDEES, runner=stub({
        "speakers": [{"label": "Speaker 2",
                      "identity": "Philipp Baltensperger",
                      "evidence": "describes BlueCare's release process"}],
        "relabels": [
            {"line": 11, "from": "Matthias", "to": "Philipp Baltensperger",
             "confidence": "high", "reason": "insider answer"},
        ],
    }))
    labels = set(labels_by_ts(g["transcript"]).values())
    assert "Philipp Baltensperger" not in labels
    assert labels == {"Matthias", "Philipp"}
    # ...and the log reports the label that was actually written.
    assert log["relabels_applied"][0]["to"] == "Philipp"


def test_host_full_name_canonicalises_to_the_transcript_convention():
    g = load_fixture()
    check_and_repair(g, known_attendees=ATTENDEES, runner=stub({
        "relabels": [{"line": 14, "from": "Speaker 2", "to": "Matthias Heim",
                      "confidence": "high", "reason": "his own argument"}],
    }))
    assert "Matthias Heim:" not in g["transcript"]
    assert labels_by_ts(g["transcript"])["15:07"] == "Matthias"


def test_namesakes_are_never_merged_by_canonicalisation():
    """Two known people sharing a first name must keep their full names —
    collapsing them would be the source-434 multi-party merge again."""
    m = canonical_label_map([
        {"name": "Philipp Baltensperger"},
        {"name": "Philipp Meier"},
        {"name": "Lukas Weber"},
    ])
    assert "Philipp Baltensperger" not in m
    assert "Philipp Meier" not in m
    assert m["Lukas Weber"] == "Lukas"
    assert m["Matthias Heim"] == "Matthias"
