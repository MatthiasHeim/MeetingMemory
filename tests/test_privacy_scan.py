import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import privacy_scan as ps  # noqa: E402


def test_transcript_segments_parse_minutes_and_hours():
    segs = ps.transcript_segments("[00:05] A: hi\nnoise\n[01:00] B: x\n[1:02:03] A: y", duration=4000)
    assert [(s.start, s.end) for s in segs] == [(5.0, 60.0), (60.0, 3723.0), (3723.0, 4000)]


def test_windows_split_at_size():
    segs = [ps.Segment(i * 10.0, i * 10.0 + 10, f"s{i}") for i in range(7)]
    wins = ps.windows(segs, size=30)
    assert [(w.start, w.end) for w in wins] == [(0, 30), (30, 60), (60, 70)]
    assert wins[0].text == "s0 s1 s2"


def test_spans_flag_merge_and_pad():
    wins = [ps.Window(0, 30, ""), ps.Window(30, 60, ""), ps.Window(60, 90, ""), ps.Window(200, 230, "")]
    scores = [{"a": 0.1}, {"a": 0.7}, {"b": 0.9}, {"a": 0.2}]
    assert ps.spans(wins, scores, threshold=0.5, pad=15) == [{"start": 15.0, "end": 105.0, "max_p": 0.9, "reasons": ["a", "b"]}]


def test_spans_nothing_flagged():
    assert ps.spans([ps.Window(0, 30, "")], [{"a": 0.49}], threshold=0.5) == []


def test_spans_clip_to_duration():
    assert ps.spans([ps.Window(80, 100, "")], [{"a": 0.6}], pad=15, duration=105)[0]["end"] == 105
