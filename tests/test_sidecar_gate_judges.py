from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sidecar.gate import (  # noqa: E402
    JudgeGateError,
    append_mark,
    initialise_recording_sidecar,
    judge_backend_for,
)
from sidecar.judges import GeminiJudge, JevJudge  # noqa: E402
from sidecar.service import create_judge  # noqa: E402
from sidecar.transcript import TranscriptLine  # noqa: E402


def _lines(count: int) -> list[TranscriptLine]:
    return [
        TranscriptLine(index, float(index), f"00:{index:02d}", "Speaker", f"Synthetic line {index}")
        for index in range(count)
    ]


class _FakeGemini:
    backend = "gemini"
    model = "fake-gemini"

    def __init__(self, **_kwargs):
        self.calls = 0

    def judge(self, lines, questions, *, context=6):
        self.calls += 1
        return {question: [0.5] * len(lines) for question in questions}


class _FakeJev:
    backend = "jev"
    model = "fake-jev"
    constructed = 0

    def __init__(self, **_kwargs):
        type(self).constructed += 1

    def judge(self, lines, questions, *, context=6):
        return {question: [0.5] * len(lines) for question in questions}


def test_gate_fails_closed_and_never_constructs_jev_without_opt_in(tmp_path, monkeypatch):
    _FakeJev.constructed = 0

    with pytest.raises(JudgeGateError):
        create_judge(
            "2026-09-08_14-31-16",
            requested="jev",
            recordings_root=tmp_path,
            cache_root=tmp_path / "cache",
            gemini_factory=_FakeGemini,
            jev_factory=_FakeJev,
        )
    assert _FakeJev.constructed == 0
    assert judge_backend_for("2026-09-08_14-31-16", recordings_root=tmp_path) == "gemini"

    initialise_recording_sidecar(
        "2026-09-08_14-31-16", jev=False, external_attendees=False, root=tmp_path
    )
    with pytest.raises(JudgeGateError):
        create_judge(
            "2026-09-08_14-31-16",
            requested="jev",
            recordings_root=tmp_path,
            cache_root=tmp_path / "cache",
            gemini_factory=_FakeGemini,
            jev_factory=_FakeJev,
        )
    assert _FakeJev.constructed == 0


def test_gate_uses_jev_only_for_explicit_internal_stored_choice(tmp_path):
    _FakeJev.constructed = 0
    initialise_recording_sidecar(
        "2026-09-08_14-31-16", jev=True, external_attendees=False, root=tmp_path
    )

    judge = create_judge(
        "2026-09-08_14-31-16",
        requested="jev",
        recordings_root=tmp_path,
        cache_root=tmp_path / "cache",
        gemini_factory=_FakeGemini,
        jev_factory=_FakeJev,
    )

    assert judge.backend == "jev"
    assert _FakeJev.constructed == 1


def test_sidecar_policy_flags_and_marks_reject_ambiguous_values(tmp_path):
    with pytest.raises(ValueError):
        initialise_recording_sidecar(
            "2026-09-08_14-31-16", jev="true", external_attendees=False, root=tmp_path
        )
    with pytest.raises(ValueError):
        initialise_recording_sidecar(
            "2026-09-08_14-31-16", jev=False, external_attendees=0, root=tmp_path
        )
    with pytest.raises(ValueError):
        append_mark("2026-09-08_14-31-16", float("inf"), root=tmp_path)


def test_gate_authorises_jev_from_preference_even_with_external_attendees(tmp_path):
    """2026-09-28: jev true is enough. External attendance is metadata only."""
    _FakeJev.constructed = 0
    initialise_recording_sidecar(
        "2026-09-08_14-31-16", jev=True, external_attendees=True, root=tmp_path
    )

    judge = create_judge(
        "2026-09-08_14-31-16",
        requested="jev",
        recordings_root=tmp_path,
        cache_root=tmp_path / "cache",
        gemini_factory=_FakeGemini,
        jev_factory=_FakeJev,
    )
    assert judge.backend == "jev"
    assert _FakeJev.constructed == 1


@pytest.mark.parametrize(
    "payload",
    (
        {"jev": True},
        {"jev": True, "external_attendees": "false"},
        {"jev": True, "external_attendees": 0},
    ),
)
def test_gate_requires_explicit_boolean_internal_attendee_resolution(tmp_path, payload):
    _FakeJev.constructed = 0
    path = tmp_path / "2026-09-08_14-31-16.sidecar.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JudgeGateError):
        create_judge(
            "2026-09-08_14-31-16",
            requested="jev",
            recordings_root=tmp_path,
            cache_root=tmp_path / "cache",
            gemini_factory=_FakeGemini,
            jev_factory=_FakeJev,
        )
    assert _FakeJev.constructed == 0


def test_jev_adapter_rechecks_gate_when_constructed_directly(tmp_path):
    with pytest.raises(JudgeGateError):
        JevJudge(
            stem="2026-09-08_14-31-16",
            recordings_root=tmp_path,
            helper_path=tmp_path / "not-used-when-gate-fails.py",
        )


def test_gemini_judge_batches_twenty_lines_with_schema_and_context():
    requested = []

    class Models:
        def generate_content(self, *, model, contents, config):
            requested.append((model, contents, config))
            # The request itself contains all twenty expected indices; parsing
            # it would duplicate behavior under test, so score known batches.
            indexes = list(range(0, 20)) if len(requested) == 1 else [20]
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {"line_index": index, "probabilities": {"relevant": index / 100}}
                            for index in indexes
                        ]
                    }
                )
            )

    client = SimpleNamespace(
        models=Models(),
        _api_client=SimpleNamespace(
            _http_options=SimpleNamespace(
                base_url="https://generativelanguage.googleapis.com/"
            )
        ),
    )
    judge = GeminiJudge(client=client, types_module=None, model="synthetic")
    scores = judge.judge(_lines(21), {"relevant": "Is this synthetic line relevant?"})

    assert scores["relevant"] == [index / 100 for index in range(21)]
    assert len(requested) == 2
    assert requested[0][0] == "synthetic"
    assert requested[0][2]["response_schema"]["properties"]["results"]["items"]["properties"]["probabilities"]
    assert '"previous_lines"' in requested[0][1]
    assert 'Synthetic line 0' in requested[0][1]


def test_gemini_judge_runs_batches_concurrently_and_orders_scores():
    """Independent batches may complete out of order without reordering lines."""
    started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    peak_active = 0

    class Models:
        def generate_content(self, *, model, contents, config):
            nonlocal active, peak_active
            indexes = [int(value) for value in __import__("re").findall(r'"line_index":(\d+)', contents)]
            with lock:
                active += 1
                peak_active = max(peak_active, active)
                if peak_active >= 2:
                    started.set()
            assert release.wait(timeout=2)
            # Deliberately make the first batch slower after release.
            if indexes[0] == 0:
                time.sleep(0.03)
            with lock:
                active -= 1
            return SimpleNamespace(
                text=json.dumps(
                    {"results": [
                        {"line_index": index, "probabilities": {"relevant": index / 100}}
                        for index in indexes
                    ]}
                )
            )

    client = SimpleNamespace(
        models=Models(),
        _api_client=SimpleNamespace(
            _http_options=SimpleNamespace(base_url="https://generativelanguage.googleapis.com/")
        ),
    )
    judge = GeminiJudge(
        client=client,
        types_module=None,
        model="synthetic",
        batch_size=2,
        max_batch_workers=2,
    )
    result = []
    worker = threading.Thread(
        target=lambda: result.append(judge.judge(_lines(4), {"relevant": "Synthetic question"})),
        daemon=True,
    )
    worker.start()
    assert started.wait(timeout=1)
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert peak_active == 2
    assert result == [{"relevant": [0.0, 0.01, 0.02, 0.03]}]


def test_cached_judge_cache_contains_scores_and_hashes_not_transcript_text(tmp_path):
    base = _FakeGemini()
    from sidecar.judges import CachedJudge
    from sidecar.cache import ProbabilityCache

    judge = CachedJudge(base, ProbabilityCache(tmp_path), "2026-09-08_14-31-16")
    lines = _lines(2)
    judge.judge(lines, {"relevant": "Synthetic question"})
    judge.judge(lines, {"relevant": "Synthetic question"})

    assert base.calls == 1
    payload = next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")
    assert "Synthetic line 0" not in payload
    assert '"scores"' in payload


def test_cache_identity_changes_for_question_criteria_and_model(tmp_path):
    from sidecar.cache import ProbabilityCache
    from sidecar.judges import CachedJudge

    lines = _lines(2)
    first = _FakeGemini()
    first.model = "first-model"
    first_judge = CachedJudge(first, ProbabilityCache(tmp_path), "2026-09-08_14-31-16")
    first_judge.judge(
        lines,
        {
            "relevant": {
                "instructions": "Synthetic question",
                "criteria": {"true": "First definition", "false": "Not first"},
            }
        },
    )
    first_judge.judge(
        lines,
        {
            "relevant": {
                "instructions": "Synthetic question",
                "criteria": {"true": "Second definition", "false": "Not first"},
            }
        },
    )
    second = _FakeGemini()
    second.model = "second-model"
    CachedJudge(second, ProbabilityCache(tmp_path), "2026-09-08_14-31-16").judge(
        lines,
        {
            "relevant": {
                "instructions": "Synthetic question",
                "criteria": {"true": "First definition", "false": "Not first"},
            }
        },
    )

    # A probability from a one-question provider prompt must not be reused for
    # the same question when another question changes the request schema.
    first_judge.judge(
        lines,
        {
            "relevant": {
                "instructions": "Synthetic question",
                "criteria": {"true": "First definition", "false": "Not first"},
            },
            "another_question": "A co-batched synthetic question",
        },
    )

    # Populate a two-question layout, then remove only one of its entries to
    # simulate a partial cache hit. The next request must reissue both vectors,
    # never combine the remaining cached score with a direct score.
    partial_layout = {
        "relevant": {
            "instructions": "Synthetic question",
            "criteria": {"true": "First definition", "false": "Not first"},
        },
        "third_question": "A second co-batched synthetic question",
    }
    first_judge.judge(
        lines,
        partial_layout,
    )
    next(tmp_path.rglob("third_question-*.json")).unlink()
    calls_after_partial_hit = first.calls
    first_judge.judge(
        lines,
        partial_layout,
    )
    calls_after_reissued_layout = first.calls
    first_judge.judge(lines, partial_layout)

    assert calls_after_partial_hit == 4
    assert calls_after_reissued_layout == 5
    assert first.calls == calls_after_reissued_layout
    assert second.calls == 1
    assert len(list(tmp_path.rglob("*.json"))) == 7


def test_probability_cache_rejects_non_finite_scores(tmp_path):
    from sidecar.cache import ProbabilityCache

    with pytest.raises(ValueError):
        ProbabilityCache(tmp_path).put(
            "2026-09-08_14-31-16",
            "gemini",
            "relevant",
            {"instructions": "Synthetic question"},
            input_fingerprint="synthetic",
            model="synthetic",
            scores=[float("nan")],
        )
