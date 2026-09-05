"""Tests for speaker_verify — channel-ground-truth attribution flipping."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from speaker_verify import verify, _parse_turns  # noqa: E402


class StubVAD:
    """Interval-arithmetic stand-in for channel_vad.ChannelVAD."""

    def __init__(self, segments, duration_sec):
        self.segments = segments
        self.duration_sec = duration_sec

    def separation_report(self):
        # Unit fixtures describe an externally established capture route.
        # Their short excerpts need not contain the full isolation sample.
        # End-to-end unknown/routing tests use the actual probe separately.
        from channel_vad import channel_separation_report
        report = channel_separation_report(self.segments)
        if report["host_bleed_rate"] is None:
            report.update(admissible=True, reason="fixture_isolation_established")
        report["reference_uncertain_intervals"] = []
        return report

    def shares(self, t0, t1):
        out = {"host_only": 0.0, "remote_only": 0.0, "both": 0.0}
        for s, e, label in self.segments:
            ov = max(0.0, min(e, t1) - max(s, t0))
            if label == 'host':
                out["host_only"] += ov
            elif label == 'remote':
                out["remote_only"] += ov
            elif label == 'both':
                out["both"] += ov
        out["speech"] = sum(out.values())
        return out


def _gem(transcript: str, participants=None) -> dict:
    return {
        "transcript": transcript,
        "participants": participants or [],
    }


# ── the headline fix: remote-labeled host speech flips to Matthias ────


def test_remote_label_on_host_span_flips_to_matthias():
    """The 2026-06-11 bug: a long turn voiced on the mic channel (host
    speaking, remote silent) but labeled 'Speaker B' must flip to Matthias."""
    g = _gem(
        "[00:00] Speaker B: Hallo zusammen.\n"
        "[00:30] Matthias: Hi.\n"
        "[01:00] Speaker B: Wir sollten auf Opus wechseln und gefährliche "
        "Anfragen blockieren, damit wir nie auf Patientendaten zugreifen.\n"
        "[02:00] Speaker B: Weiter gehts.\n"
    )
    vad = StubVAD(
        [(0, 30, 'remote'), (30, 60, 'host'),
         (60, 120, 'host'),            # [01:00] turn: pure host speech
         (120, 150, 'remote')],
        duration_sec=150,
    )
    log = verify(g, vad)
    assert len(log["flips"]) == 1
    flip = log["flips"][0]
    assert flip["from"] == "Speaker B"
    assert flip["to"] == "Matthias"
    assert flip["rule"] == "remote_label_but_mic_dominant"
    assert flip["host_share"] > 0.85
    assert "[01:00] Matthias: Wir sollten auf Opus wechseln" in g["transcript"]
    # The genuinely-remote turns keep their label.
    assert "[00:00] Speaker B: Hallo zusammen." in g["transcript"]
    assert "[02:00] Speaker B: Weiter gehts." in g["transcript"]


def test_host_label_on_remote_span_flips_to_dominant_remote():
    g = _gem(
        "[00:00] Antonella Borromeo: Hallo.\n"
        "[00:30] Matthias: Eine lange Passage die eigentlich von der "
        "Gegenseite gesprochen wurde.\n"
        "[01:30] Antonella Borromeo: Genau.\n",
        # Named (non-generic) labels are only flippable when listed in
        # participants[] — the known-label universe.
        participants=[{"name": "Matthias", "role": "host"},
                      {"name": "Antonella Borromeo", "role": "participant"}],
    )
    vad = StubVAD(
        [(0, 30, 'remote'), (30, 90, 'remote'), (90, 120, 'remote')],
        duration_sec=120,
    )
    log = verify(g, vad)
    assert log["dominant_remote"] == "Antonella Borromeo"
    assert len(log["flips"]) == 1
    flip = log["flips"][0]
    assert flip["from"] == "Matthias"
    assert flip["to"] == "Antonella Borromeo"
    assert flip["rule"] == "host_label_but_mic_silent"
    assert "[00:30] Antonella Borromeo: Eine lange Passage" in g["transcript"]


# ── conservatism gates ────────────────────────────────────────────────


def test_short_turn_with_ambiguous_evidence_never_flipped():
    """Gemini timestamps are ±few-seconds; a short turn with AMBIGUOUS VAD
    evidence (not extreme host_share) is untouchable — the 8 s floor still
    applies when confidence isn't high."""
    g = _gem(
        "[00:00] Speaker B: Kurz und gemischt.\n"
        "[00:05] Matthias: Lang genug um sicher zu sein, dass hier wirklich "
        "jemand spricht.\n"
    )
    # host_share for [0,5) is 50% (mixed) -- not high confidence.
    vad = StubVAD([(0, 2.5, 'host'), (2.5, 5, 'remote'), (5, 60, 'host')],
                  duration_sec=60)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["skipped"]["short"] == 1
    assert "Speaker B: Kurz und gemischt." in g["transcript"]


def test_short_turn_with_high_confidence_evidence_flips():
    """docs/RELIABILITY_PLAN_2026-07.md Phase 2 fix (429 [01:06]): a short
    turn (< 8 s but >= 2.5 s) whose channel evidence is nearly unanimous
    (host_share > 95%) now flips despite being short -- the ±few-second
    timestamp noise can't plausibly explain away a near-100% mic signal."""
    g = _gem(
        "[00:00] Speaker B: Kurz.\n"
        "[00:05] Matthias: Lang genug um sicher zu sein, dass hier wirklich "
        "jemand spricht.\n"
    )
    # host_share for [0,5) is 100% -- the mic was active the whole turn.
    vad = StubVAD([(0, 5, 'host'), (5, 60, 'host')], duration_sec=60)
    log = verify(g, vad)
    assert len(log["flips"]) == 1
    assert log["flips"][0]["from"] == "Speaker B"
    assert log["flips"][0]["to"] == "Matthias"


def test_very_short_turn_below_high_confidence_floor_never_flipped():
    """Even a maximally-confident VAD signal can't rescue a turn under the
    2.5 s absolute floor -- Gemini's timestamp noise dominates at that
    length regardless of channel evidence."""
    g = _gem(
        "[00:00] Speaker B: Ja.\n"
        "[00:02] Matthias: Lang genug um sicher zu sein, dass hier wirklich "
        "jemand spricht und die Aufnahme weitergeht.\n"
    )
    vad = StubVAD([(0, 2, 'host'), (2, 60, 'host')], duration_sec=60)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["skipped"]["short"] == 1


def test_overlap_dominated_turn_skipped():
    g = _gem(
        "[00:00] Speaker B: Viel Durcheinander hier.\n"
        "[00:30] Matthias: Ende.\n"
    )
    # 60% of the first turn's speech is overlapped → no arbitration.
    vad = StubVAD([(0, 12, 'host'), (12, 30, 'both'), (30, 60, 'host')],
                  duration_sec=60)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["skipped"]["overlap_dominated"] == 1


def test_turn_without_speech_evidence_skipped():
    g = _gem(
        "[00:00] Speaker B: Hier ist es eigentlich still.\n"
        "[00:30] Matthias: Ende.\n"
    )
    # Only 2 s of detected speech inside a 30 s turn — below MIN_SPEECH_SEC.
    vad = StubVAD([(0, 2, 'host'), (30, 60, 'host')], duration_sec=60)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["skipped"]["no_speech"] == 1


def test_untimed_turns_skipped():
    """Turns without a leading [MM:SS] can't be located in the audio."""
    g = _gem(
        "Speaker B: Ohne Zeitstempel.\n"
        "Matthias: Auch ohne.\n"
    )
    vad = StubVAD([(0, 60, 'host')], duration_sec=60)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["turns_total"] == 2
    assert log["skipped"]["untimed"] == 2


def test_moderate_host_share_not_flipped():
    """host_share between the flip thresholds (0.15–0.85) is ambiguous —
    the label stands either way."""
    g = _gem(
        "[00:00] Speaker B: Gemischte Passage mit Anteilen von beiden.\n"
        "[01:00] Matthias: Ende.\n"
    )
    vad = StubVAD([(0, 30, 'host'), (30, 60, 'remote'), (60, 90, 'host')],
                  duration_sec=90)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["turns_checked"] >= 1


def test_no_vad_skips_cleanly():
    g = _gem("[00:00] Speaker B: Hallo.\n")
    before = g["transcript"]
    log = verify(g, None)
    assert log["skipped_no_vad"] is True
    assert g["transcript"] == before


def test_matthias_label_without_remote_speaker_not_flipped():
    """Solo recording: nothing to flip a host label TO."""
    g = _gem(
        "[00:00] Matthias: Notiz an mich selbst.\n"
        "[01:00] Matthias: Noch eine.\n"
    )
    vad = StubVAD([(0, 120, 'remote')], duration_sec=120)
    log = verify(g, vad)
    assert log["flips"] == []
    assert log["dominant_remote"] is None


def test_unknown_capitalized_label_never_flipped():
    """German capitalises nouns, so verbatim speech like '[12:00] Fazit: ...'
    can look like a timed turn. Labels outside the known universe (self /
    generic Speaker N / participants[]) must never be rewritten — flipping
    'Fazit' to 'Matthias' would corrupt transcript text."""
    g = _gem(
        "[00:00] Speaker B: Hallo.\n"
        "[12:00] Fazit: wir machen das so und besprechen es nochmal in Ruhe "
        "im nächsten Termin gemeinsam.\n"
        "[13:00] Speaker B: Genau.\n",
        participants=[{"name": "Matthias"}, {"name": "Speaker B"}],
    )
    # Mic dominant during the 'Fazit' interval — would flip if eligible.
    vad = StubVAD([(0, 720, 'remote'), (720, 780, 'host'),
                   (780, 840, 'remote')], duration_sec=840)
    log = verify(g, vad)
    assert log["flips"] == []
    assert "Fazit: wir machen das so" in g["transcript"]


def test_multiparty_suppresses_flip_to_remote_but_not_to_host():
    """With 2+ remote speakers the mic channel can only prove 'not the
    host' — flipping a host label to a specific remote name risks naming
    the wrong person. Flips TO the host stay safe (the mic proves him)."""
    g = _gem(
        "[00:00] Anna Weber: Hallo.\n"
        "[00:30] Matthias: Lange Passage obwohl das Mikrofon still war.\n"
        "[01:30] Stefan Müller: Hi.\n"
        "[02:00] Anna Weber: Diese Passage kam aber vom Host-Mikrofon.\n"
        "[03:00] Stefan Müller: Ende.\n",
        participants=[{"name": "Matthias"}, {"name": "Anna Weber"},
                      {"name": "Stefan Müller"}],
    )
    vad = StubVAD(
        [(0, 30, 'remote'), (30, 90, 'remote'), (90, 120, 'remote'),
         (120, 180, 'host'), (180, 240, 'remote')],
        duration_sec=240,
    )
    log = verify(g, vad)
    assert log["flip_to_remote_suppressed_multiparty"] is True
    # Host label on a remote span: NOT flipped (ambiguous target).
    assert "[00:30] Matthias: Lange Passage" in g["transcript"]
    # Remote label on a host span: flipped to Matthias (unambiguous).
    assert len(log["flips"]) == 1
    assert log["flips"][0]["to"] == "Matthias"
    assert "[02:00] Matthias: Diese Passage" in g["transcript"]


# ── last-turn handling and bookkeeping ────────────────────────────────


def test_last_turn_uses_recording_end():
    g = _gem(
        "[00:00] Matthias: Hallo.\n"
        "[00:30] Speaker B: Diese letzte Passage gehört eigentlich dem Host "
        "und läuft bis zum Ende der Aufnahme.\n"
    )
    vad = StubVAD([(0, 30, 'host'), (30, 90, 'host')], duration_sec=90)
    log = verify(g, vad)
    assert len(log["flips"]) == 1
    assert log["flips"][0]["from"] == "Speaker B"


def test_speaking_stats_recomputed_after_flip():
    g = _gem(
        "[00:00] Matthias: Hallo.\n"
        "[01:00] Speaker B: Eigentlich der Host hier, lange Passage.\n"
        "[03:00] Speaker B: Echte Gegenseite.\n",
        participants=[
            {"name": "Matthias", "role": "host", "speaking_pct": 25,
             "total_seconds": 60},
            {"name": "Speaker B", "role": "participant", "speaking_pct": 75,
             "total_seconds": 180},
        ],
    )
    vad = StubVAD(
        [(0, 60, 'host'), (60, 180, 'host'), (180, 240, 'remote')],
        duration_sec=240,
    )
    log = verify(g, vad)
    assert len(log["flips"]) == 1
    assert log["speaking_stats_recomputed"] is True
    by_name = {p["name"]: p for p in g["participants"]}
    # Matthias now holds 180 of 240 turn-seconds.
    assert by_name["Matthias"]["speaking_pct"] == 75
    assert by_name["Matthias"]["total_seconds"] == 180
    assert by_name["Speaker B"]["speaking_pct"] == 25


def test_multiple_flips_keep_offsets_straight():
    """Two flips in one transcript must each rewrite their own label only
    (offsets shift after the first rewrite — regression guard)."""
    g = _gem(
        "[00:00] Speaker B: Erste falsch zugeordnete lange Passage.\n"
        "[01:00] Speaker B: Echte Gegenseite spricht hier wirklich.\n"
        "[02:00] Speaker B: Zweite falsch zugeordnete lange Passage.\n"
        "[03:00] Speaker B: Und nochmal echte Gegenseite bis zum Ende.\n"
    )
    vad = StubVAD(
        [(0, 60, 'host'), (60, 120, 'remote'),
         (120, 180, 'host'), (180, 240, 'remote')],
        duration_sec=240,
    )
    log = verify(g, vad)
    assert len(log["flips"]) == 2
    assert "[00:00] Matthias: Erste falsch" in g["transcript"]
    assert "[01:00] Speaker B: Echte Gegenseite spricht" in g["transcript"]
    assert "[02:00] Matthias: Zweite falsch" in g["transcript"]
    assert "[03:00] Speaker B: Und nochmal" in g["transcript"]
    # Forensic log is chronological.
    assert log["flips"][0]["time_span"] < log["flips"][1]["time_span"]


# ── turn parsing ──────────────────────────────────────────────────────


def test_parse_turns_extracts_timestamps_and_spans():
    t = ("[00:00] Matthias: Hallo.\n"
         "[02:15] Speaker B: Hi. Mit Inline-Zeit [03:00] mittendrin.\n"
         "Speaker B: Ohne Zeitstempel.\n"
         "[1:02:03] Anna Weber: Spät.\n")
    turns = _parse_turns(t)
    assert [x["speaker"] for x in turns] == [
        "Matthias", "Speaker B", "Speaker B", "Anna Weber"
    ]
    assert turns[0]["t_start"] == 0.0
    assert turns[0]["t_end"] == 135.0
    assert turns[1]["t_start"] == 135.0
    assert turns[1]["t_end"] is None       # next turn is untimed
    assert turns[2]["t_start"] is None
    assert turns[3]["t_start"] == 3723.0


def test_parse_turns_ignores_sentence_fragments():
    """Mid-sentence colons must not become turns (same shape rules as
    speaker_reconcile)."""
    t = ("[00:00] Matthias: Wir haben Standorte in Niederbipp und eine in "
         "Romont. [01:23] Und es gibt: einen Plan.\n")
    turns = _parse_turns(t)
    assert [x["speaker"] for x in turns] == ["Matthias"]


# ── the source-767 fix: refuse to flip on a bleeding mic (2026-08-07) ─────
#
# Two-sided on purpose. A guard that suppresses flips is indistinguishable
# from a guard that suppresses EVERYTHING unless something also proves the
# working path still works — so each test below has its clean-oracle mirror.


def _bleeding_vad(duration=1200.0):
    """A mic that hears the loudspeakers: nearly all remote speech reads as
    'both' (source 767 measured host_bleed_rate 0.71)."""
    return StubVAD(
        [(0.0, 100.0, 'remote'),
         (100.0, 700.0, 'both'),
         (700.0, duration, 'host')],
        duration_sec=duration,
    )


def _clean_vad(duration=1200.0):
    """Headphones: overlap is confined to genuine backchannel."""
    return StubVAD(
        [(0.0, 600.0, 'remote'),
         (600.0, 660.0, 'both'),
         (660.0, duration, 'host')],
        duration_sec=duration,
    )


REMOTE_LABEL_OVER_HOST_SPAN = (
    "[00:00] Speaker 2: Kurzer Einstieg.\n"
    "[11:40] Speaker 2: Also ich gseh das so, mir sölled da dranblibe und "
    "d'Lösung für d'Ärzt würklich verfügbar mache.\n"
    "[15:00] Matthias: Guet.\n"
)


def test_flip_still_fires_when_the_channels_are_separated():
    """SIDE A — the clean case is untouched by the guard. A turn voiced on
    the mic channel while the remote is silent still flips to Matthias."""
    g = _gem(REMOTE_LABEL_OVER_HOST_SPAN)
    log = verify(g, _clean_vad())
    assert log["skipped_channel_bleed"] is False
    assert log["channel_separation"]["admissible"] is True
    assert len(log["flips"]) == 1
    assert log["flips"][0]["to"] == "Matthias"
    assert "[11:40] Matthias:" in g["transcript"]


def test_no_flips_when_the_mic_hears_the_remote():
    """SIDE B — the same transcript and the same apparent host_share, but the
    oracle is bleeding: nothing may be flipped and the transcript is
    byte-identical. This is the source-767 defect."""
    g = _gem(REMOTE_LABEL_OVER_HOST_SPAN)
    before = g["transcript"]
    log = verify(g, _bleeding_vad())
    assert log["skipped_channel_bleed"] is True
    assert log["channel_separation"]["admissible"] is False
    assert log["channel_separation"]["reason"] == "mic_hears_remote"
    assert log["flips"] == []
    assert g["transcript"] == before


def test_bleed_guard_reports_the_measurement_it_refused_on():
    """The verdict has to be auditable — a silent skip is how the pipeline
    got here. The measured rate and the threshold both land in the log."""
    g = _gem(REMOTE_LABEL_OVER_HOST_SPAN)
    log = verify(g, _bleeding_vad())
    sep = log["channel_separation"]
    assert sep["host_bleed_rate"] > sep["max_host_bleed_rate"]
    assert sep["both_sec"] == 600.0
    assert sep["remote_only_sec"] == 100.0


def test_bleed_guard_does_not_suppress_the_reverse_flip_direction():
    """A clean oracle must still be able to flip host->remote. If the guard
    ever made that unreachable it would recreate the one-directional
    behaviour that caused the damage in the first place."""
    g = _gem(
        "[00:00] Matthias: Einleitung.\n"
        "[00:30] Matthias: Also bi üs im Team mache mir das jede Monet so, "
        "mir bündled alles i es Release-Päckli.\n"
        "[10:30] Speaker 2: Danke.\n"
        "[10:40] Matthias: Guet.\n"
    )
    vad = StubVAD(
        [(0.0, 30.0, 'host'),
         (30.0, 630.0, 'remote'),      # the [00:30] turn: mic silent
         (630.0, 700.0, 'both'),
         (700.0, 1200.0, 'host')],
        duration_sec=1200.0,
    )
    log = verify(g, vad)
    assert log["channel_separation"]["admissible"] is True
    assert len(log["flips"]) == 1
    assert log["flips"][0]["rule"] == "host_label_but_mic_silent"
    assert log["flips"][0]["to"] == "Speaker 2" or log["flips"][0]["to"]
