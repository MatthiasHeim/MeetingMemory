"""Contract tests for durable source-separated transcription jobs.

No test calls Gemini or ffmpeg.  The adapter is exercised through the same
subprocess boundary as production, and records calls in a file so assertions
remain valid across forked workers.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
import types as pytypes
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import transcription_jobs  # noqa: E402
from transcription_jobs import (  # noqa: E402
    GeminiTranscriptionAdapter,
    JobDeadlineExceeded,
    SourceTranscriptionPipeline,
    _exclusive_job_lock,
)


class FilePlanAdapter:
    """A deterministic, process-safe adapter plan for durable-worker tests."""

    def __init__(self, state_path: Path, plans: dict[str, list[object]]):
        self.state_path = str(state_path)
        self.plans = plans

    def transcribe(self, audio_path: Path, *, start_seconds: float, **kwargs):
        del audio_path, kwargs
        key = str(int(start_seconds))
        state_file = Path(self.state_path)
        state = json.loads(state_file.read_text()) if state_file.exists() else {"calls": [], "counts": {}}
        count = state["counts"].get(key, 0)
        state["counts"][key] = count + 1
        state["calls"].append(float(start_seconds))
        state_file.write_text(json.dumps(state))
        actions = self.plans.get(key, [self._default(start_seconds)])
        action = actions[min(count, len(actions) - 1)]
        if action == "raise":
            raise RuntimeError(f"planned failure for {key}")
        return action

    @staticmethod
    def _default(start_seconds: float) -> dict:
        return {
            "outcome": "transcribed",
            "segments": [
                {
                    "start_seconds": 0,
                    "end_seconds": 10,
                    "text": f"words at {int(start_seconds)}",
                    "speaker": "speaker_01",
                }
            ],
        }


class SlowAdapter:
    def transcribe(self, audio_path: Path, **kwargs):
        del audio_path, kwargs
        time.sleep(5)
        return {"outcome": "transcribed", "segments": []}


def hold_job_lock(path: str, ready) -> None:
    with _exclusive_job_lock(Path(path)):
        ready.put("locked")
        time.sleep(0.35)


def copy_clipper(source: Path, destination: Path, start_seconds: float, end_seconds: float) -> Path:
    del start_seconds, end_seconds
    destination.write_bytes(source.read_bytes())
    return destination


def evidence_by_start(values: dict[int, str]):
    def detector(audio_path: Path, *, start_seconds: float, end_seconds: float):
        del audio_path, end_seconds
        return {"status": values.get(int(start_seconds), "speech_detected"), "detector": "test"}

    return detector


def make_source(tmp_path: Path, name: str = "source.bin", content: bytes = b"source-audio") -> Path:
    source = tmp_path / name
    source.write_bytes(content)
    return source


def make_pipeline(
    tmp_path: Path,
    adapter,
    *,
    duration: float,
    detector=None,
    max_attempts: int = 1,
    request_timeout_seconds: float = 1,
    job_timeout_seconds: float = 30,
) -> SourceTranscriptionPipeline:
    return SourceTranscriptionPipeline(
        tmp_path / "state",
        "test-transcription-model",
        chunk_seconds=180,
        request_timeout_seconds=request_timeout_seconds,
        job_timeout_seconds=job_timeout_seconds,
        adapter=adapter,
        max_attempts=max_attempts,
        clipper=copy_clipper,
        duration_probe=lambda _: duration,
        speech_detector=detector or evidence_by_start({}),
    )


def source_payload(source: Path, *, source_id: str = "mic", start: float = 0) -> dict:
    return {
        "source_id": source_id,
        "path": str(source),
        "start_seconds": start,
        "timing_basis": "unverified_sample_zero",
    }


def calls(state_path: Path) -> list[float]:
    return json.loads(state_path.read_text())["calls"]


def test_interruption_resume_retries_only_failed_chunk(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(
        state_path,
        {
            "0": [FilePlanAdapter._default(0)],
            "180": ["raise", FilePlanAdapter._default(180)],
        },
    )
    first = make_pipeline(tmp_path, adapter, duration=360)
    first_result = first.run([source_payload(source)], session_id="meeting-1")

    assert first_result["_meta"]["durable_job_status"] == "retryable_failure"
    assert calls(state_path) == [0.0, 180.0]
    assert first_result["_meta"]["retryable_ranges"][0]["source_start_seconds"] == 180

    resumed = make_pipeline(tmp_path, adapter, duration=360)
    resumed_result = resumed.run([source_payload(source)], session_id="meeting-1", retry_failed=True)

    assert resumed_result["_meta"]["durable_job_status"] == "complete"
    assert calls(state_path) == [0.0, 180.0, 180.0]
    assert "words at 0" in resumed_result["transcript"]
    assert "words at 180" in resumed_result["transcript"]


def test_input_hash_creates_a_new_durable_job(tmp_path):
    source = make_source(tmp_path, content=b"first")
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]})
    first = make_pipeline(tmp_path, adapter, duration=180)
    result_one = first.run([source_payload(source)], session_id="meeting-1")

    source.write_bytes(b"changed source bytes")
    second = make_pipeline(tmp_path, adapter, duration=180)
    result_two = second.run([source_payload(source)], session_id="meeting-1")

    assert result_one["_meta"]["job_key"] != result_two["_meta"]["job_key"]
    assert calls(state_path) == [0.0, 0.0]
    assert len(list((tmp_path / "state" / "jobs").iterdir())) == 2


def test_exact_digital_silence_tail_is_a_terminal_no_speech_region(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(
        state_path,
        {
            "0": [FilePlanAdapter._default(0)],
            "180": [{"outcome": "no_speech", "segments": []}],
        },
    )
    pipeline = make_pipeline(
        tmp_path,
        adapter,
        duration=360,
        detector=evidence_by_start({180: "digital_silence"}),
    )
    result = pipeline.run([source_payload(source)], session_id="meeting-1")

    job = next((tmp_path / "state" / "jobs").iterdir())
    manifest = json.loads((job / "manifest.json").read_text())
    statuses = [item["status"] for item in manifest["sources"][0]["chunks"]]
    assert statuses == ["transcribed", "no_speech"]
    assert result["_meta"]["durable_job_status"] == "complete"
    assert result["_meta"]["missing_ranges"] == []


def test_model_no_speech_cannot_erase_detected_speech(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [{"outcome": "no_speech", "segments": []}]})
    pipeline = make_pipeline(tmp_path, adapter, duration=180, detector=evidence_by_start({0: "speech_detected"}))
    result = pipeline.run([source_payload(source)], session_id="meeting-1")

    assert result["_meta"]["durable_job_status"] == "needs_review"
    flags = [item["reason"] for item in result["_meta"]["review_flags"]]
    assert "unexplained_speech" in flags
    assert result["_meta"]["missing_ranges"][0]["retryable"] is False


def test_middle_failure_is_explicit_while_later_text_is_preserved(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(
        state_path,
        {
            "0": [FilePlanAdapter._default(0)],
            "180": ["raise"],
            "360": [FilePlanAdapter._default(360)],
        },
    )
    pipeline = make_pipeline(tmp_path, adapter, duration=540)
    result = pipeline.run([source_payload(source)], session_id="meeting-1")

    missing = result["_meta"]["missing_ranges"]
    assert [(item["source_start_seconds"], item["source_end_seconds"]) for item in missing] == [(180.0, 360.0)]
    assert "words at 360" in result["transcript"]
    assert result["_meta"]["structural_completeness"]["transcript_structurally_complete"] is False


def test_malformed_bounds_keep_raw_attempt_and_mark_region_uncertain(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    raw_response = {
        "outcome": "transcribed",
        "segments": [{"start_seconds": 20, "end_seconds": 999, "text": "keep raw", "speaker": "Ada"}],
    }
    adapter = FilePlanAdapter(state_path, {"0": [raw_response]})
    pipeline = make_pipeline(tmp_path, adapter, duration=180)
    result = pipeline.run([source_payload(source)], session_id="meeting-1")

    assert result["_meta"]["durable_job_status"] == "needs_review"
    assert any(item["reason"] == "malformed_segment_bounds" for item in result["_meta"]["review_flags"])
    raw_attempt = next((tmp_path / "state" / "jobs").glob("*/attempts/mic/mic-0000/attempt-0001.json"))
    stored = json.loads(raw_attempt.read_text())
    assert stored["raw_response"] == raw_response
    assert result["segments"] == []


def test_retries_raw_failure_then_preserves_successful_result(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(
        state_path,
        {"0": ["raise", FilePlanAdapter._default(0)]},
    )
    pipeline = make_pipeline(tmp_path, adapter, duration=180, max_attempts=2)
    result = pipeline.run([source_payload(source)], session_id="meeting-1")

    assert result["_meta"]["durable_job_status"] == "complete"
    raw_attempts = sorted((tmp_path / "state" / "jobs").glob("*/attempts/mic/mic-0000/*.json"))
    assert len(raw_attempts) == 2
    assert "planned failure" in json.loads(raw_attempts[0].read_text())["error"]
    assert "raw_response" in json.loads(raw_attempts[1].read_text())


def test_request_deadline_kills_hanging_worker_and_leaves_retryable_range(tmp_path):
    source = make_source(tmp_path)
    pipeline = make_pipeline(
        tmp_path,
        SlowAdapter(),
        duration=180,
        request_timeout_seconds=0.15,
        job_timeout_seconds=0.4,
    )
    started = time.monotonic()
    result = pipeline.run([source_payload(source)], session_id="meeting-1")
    elapsed = time.monotonic() - started

    assert elapsed < 2
    assert result["_meta"]["durable_job_status"] == "retryable_failure"
    assert result["_meta"]["retryable_ranges"][0]["retryable"] is True


def test_sources_stay_separate_when_offsets_are_unverified(tmp_path):
    mic = make_source(tmp_path, "mic.bin", b"mic")
    system = make_source(tmp_path, "system.bin", b"system")
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]})
    pipeline = make_pipeline(tmp_path, adapter, duration=180)
    result = pipeline.run(
        [
            source_payload(mic, source_id="mic", start=0),
            source_payload(system, source_id="system", start=46.28),
        ],
        session_id="meeting-1",
    )

    assert result["transcript"].index("## mic") < result["transcript"].index("## system")
    assert all(not source["timing_verified"] for source in result["_meta"]["source_provenance"])
    system_segment = next(segment for segment in result["segments"] if segment["source_id"] == "system")
    assert system_segment["start_seconds"] == pytest.approx(46.28)
    assert system_segment["timing_verified"] is False
    assert result["_meta"]["speaker_attribution"]["status"] == "hold"


def test_missing_source_gap_preserves_usable_source_as_partial_draft(tmp_path):
    mic = make_source(tmp_path, "mic.bin", b"mic")
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]})
    pipeline = make_pipeline(tmp_path, adapter, duration=180)
    result = pipeline.run(
        [source_payload(mic, source_id="mic")],
        session_id="meeting-1",
        capture_gaps=[
            {
                "source_id": "system",
                "start_seconds": 0,
                "end_seconds": 180,
                "reason": "source_missing",
            }
        ],
    )

    assert "words at 0" in result["transcript"]
    assert result["_meta"]["durable_job_status"] == "needs_review"
    assert any(
        item["source_id"] == "system" and item["reason"] == "source_missing"
        for item in result["_meta"]["missing_ranges"]
    )
    absent = next(item for item in result["_meta"]["source_provenance"] if item["source_id"] == "system")
    assert absent["flags"] == ["source_missing", "timing_unverified"]


def test_unbounded_source_missing_gap_uses_existing_timeline_only_as_gap_extent(tmp_path):
    mic = make_source(tmp_path, "mic.bin", b"mic")
    state_path = tmp_path / "adapter-state.json"
    pipeline = make_pipeline(
        tmp_path,
        FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]}),
        duration=180,
    )
    result = pipeline.run(
        [source_payload(mic, source_id="mic")],
        session_id="meeting-1",
        capture_gaps=[{"source_id": "system", "reason": "source_missing"}],
    )

    missing = next(item for item in result["_meta"]["missing_ranges"] if item["source_id"] == "system")
    assert (missing["source_start_seconds"], missing["source_end_seconds"]) == (0.0, 180.0)


def test_round_robin_schedule_gives_each_source_its_first_chunk_before_second(tmp_path):
    pipeline = make_pipeline(tmp_path, FilePlanAdapter(tmp_path / "ignored.json", {}), duration=180)
    manifest = {
        "sources": [
            {
                "source_id": "mic",
                "chunks": [
                    {"index": 0, "source_start_seconds": 0, "status": "pending"},
                    {"index": 1, "source_start_seconds": 180, "status": "pending"},
                ],
            },
            {
                "source_id": "system",
                "chunks": [
                    {"index": 0, "source_start_seconds": 0, "status": "pending"},
                    {"index": 1, "source_start_seconds": 180, "status": "pending"},
                ],
            },
        ]
    }
    scheduled = pipeline._round_robin_chunks(manifest, retry_failed=True)

    assert [(source["source_id"], chunk["index"]) for source, chunk in scheduled] == [
        ("mic", 0),
        ("system", 0),
        ("mic", 1),
        ("system", 1),
    ]


def test_builtin_gemini_adapter_uses_spawn_not_watcher_fork(tmp_path):
    pipeline = make_pipeline(tmp_path, FilePlanAdapter(tmp_path / "ignored.json", {}), duration=180)
    context = pipeline._multiprocessing_context(GeminiTranscriptionAdapter("test", 1))

    assert context.get_start_method() == "spawn"


def test_default_clipper_preserves_float_source_samples_and_channels(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    source = tmp_path / "source.wav"
    # Float native timeline samples must not go through a lossy MP3/AAC stage.
    samples = np.column_stack(
        [np.linspace(-0.25, 0.25, 1_000, dtype=np.float32), np.linspace(0.5, -0.5, 1_000, dtype=np.float32)]
    )
    sf.write(source, samples, 1_000, subtype="FLOAT")
    clip = tmp_path / "clip.wav"

    SourceTranscriptionPipeline._extract_lossless_clip(source, clip, 0.2, 0.7)

    info = sf.info(clip)
    recovered, rate = sf.read(clip, dtype="float32", always_2d=True)
    assert rate == 1_000
    assert info.subtype == "FLOAT"
    assert recovered.shape == (500, 2)
    assert np.array_equal(recovered, samples[200:700])


def test_running_manifest_recovers_already_written_result_without_replay(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]})
    pipeline = make_pipeline(tmp_path, adapter, duration=180)
    pipeline.run([source_payload(source)], session_id="meeting-1")
    assert calls(state_path) == [0.0]

    job = next((tmp_path / "state" / "jobs").iterdir())
    manifest_path = job / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Simulate a process death after result.json was atomically written but
    # before the final status write.
    manifest["sources"][0]["chunks"][0]["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))

    resumed = make_pipeline(tmp_path, adapter, duration=180)
    result = resumed.run([source_payload(source)], session_id="meeting-1")

    assert result["_meta"]["durable_job_status"] == "complete"
    assert calls(state_path) == [0.0]


def test_same_job_lock_serializes_concurrent_invocations(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("advisory lock test needs POSIX fork")
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    lock_path = tmp_path / "jobs" / "same-hash" / ".lock"
    holder = context.Process(target=hold_job_lock, args=(str(lock_path), ready))
    holder.start()
    assert ready.get(timeout=2) == "locked"
    started = time.monotonic()
    with _exclusive_job_lock(lock_path):
        waited = time.monotonic() - started
    holder.join(timeout=2)

    assert holder.exitcode == 0
    assert waited >= 0.2


def test_model_name_like_speaker_label_is_canonicalized_to_anonymous_unknown(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(
        state_path,
        {
            "0": [
                {
                    "outcome": "transcribed",
                    "segments": [
                        {
                            "start_seconds": 0,
                            "end_seconds": 10,
                            "text": "hello",
                            "speaker": "Speaker Matthias",
                        }
                    ],
                }
            ]
        },
    )
    result = make_pipeline(tmp_path, adapter, duration=180).run(
        [source_payload(source)], session_id="meeting-1"
    )

    assert result["segments"][0]["speaker"] == "mic:chunk0000:speaker_unknown"
    assert "matthias" not in json.dumps(result).lower()


def test_missing_terminal_result_is_persisted_as_retryable_and_replayed(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]})
    first = make_pipeline(tmp_path, adapter, duration=180)
    first.run([source_payload(source)], session_id="meeting-1")
    job = next((tmp_path / "state" / "jobs").iterdir())
    manifest_path = job / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result_path = job / manifest["sources"][0]["chunks"][0]["result_path"]
    result_path.unlink()

    inspect_only = make_pipeline(tmp_path, adapter, duration=180).run(
        [source_payload(source)], session_id="meeting-1", retry_failed=False
    )
    refreshed = json.loads(manifest_path.read_text())
    chunk = refreshed["sources"][0]["chunks"][0]
    assert chunk["status"] == "failed"
    assert chunk["retryable"] is True
    assert inspect_only["_meta"]["retryable_ranges"]
    assert calls(state_path) == [0.0]

    replayed = make_pipeline(tmp_path, adapter, duration=180).run(
        [source_payload(source)], session_id="meeting-1", retry_failed=True
    )
    assert replayed["_meta"]["durable_job_status"] == "complete"
    assert calls(state_path) == [0.0, 0.0]


def test_explicit_uncertain_range_is_not_lost_from_completeness_accounting(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(
        state_path,
        {
            "0": [
                {
                    "outcome": "transcribed",
                    "segments": [
                        {
                            "start_seconds": 0,
                            "end_seconds": 10,
                            "text": "known words",
                            "speaker": "speaker_01",
                        }
                    ],
                    "uncertain_ranges": [{"start_seconds": 40, "end_seconds": 50}],
                }
            ]
        },
    )
    result = make_pipeline(tmp_path, adapter, duration=180).run(
        [source_payload(source)], session_id="meeting-1"
    )

    assert result["_meta"]["durable_job_status"] == "needs_review"
    assert result["_meta"]["partial"] is True
    assert result["_meta"]["missing_ranges"] == [
        pytest.helpers.anything if False else result["_meta"]["missing_ranges"][0]
    ]
    missing = result["_meta"]["missing_ranges"][0]
    assert (missing["source_start_seconds"], missing["source_end_seconds"]) == (40.0, 50.0)
    assert missing["reason"] == "model_uncertain_range"


def test_clip_deadline_failure_remains_retryable(tmp_path):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"

    def slow_clipper(input_path, output_path, start_seconds, end_seconds):
        del start_seconds, end_seconds
        time.sleep(0.12)
        output_path.write_bytes(input_path.read_bytes())
        return output_path

    pipeline = SourceTranscriptionPipeline(
        tmp_path / "state",
        "test-model",
        chunk_seconds=180,
        request_timeout_seconds=1,
        job_timeout_seconds=0.05,
        adapter=FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]}),
        clipper=slow_clipper,
        duration_probe=lambda _: 180,
        speech_detector=evidence_by_start({}),
    )
    result = pipeline.run([source_payload(source)], session_id="meeting-1")

    assert result["_meta"]["durable_job_status"] == "retryable_failure"
    assert result["_meta"]["retryable_ranges"][0]["retryable"] is True


def test_job_budget_starts_before_duration_probe(tmp_path):
    source = make_source(tmp_path)

    def slow_probe(_):
        time.sleep(0.08)
        return 180

    pipeline = SourceTranscriptionPipeline(
        tmp_path / "state",
        "test-model",
        chunk_seconds=180,
        job_timeout_seconds=0.02,
        adapter=FilePlanAdapter(tmp_path / "adapter-state.json", {}),
        clipper=copy_clipper,
        duration_probe=slow_probe,
        speech_detector=evidence_by_start({}),
    )

    with pytest.raises(JobDeadlineExceeded, match="duration probe"):
        pipeline.run([source_payload(source)], session_id="meeting-1")


def test_gemini_adapter_submits_the_structured_response_schema(tmp_path, monkeypatch):
    """JSON MIME mode alone can still return two concatenated JSON objects."""
    calls = {}

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            calls["http_options"] = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            calls["generation_config"] = kwargs

    class FakeFiles:
        def upload(self, **kwargs):
            calls["upload"] = kwargs
            return pytypes.SimpleNamespace(
                name="uploaded-audio",
                state=pytypes.SimpleNamespace(name="ACTIVE"),
            )

        def delete(self, **kwargs):
            calls["delete"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["generate_content"] = kwargs
            return pytypes.SimpleNamespace(text='{"outcome":"uncertain","segments":[]}')

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.files = FakeFiles()
            self.models = FakeModels()

    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    sdk_types = pytypes.ModuleType("google.genai.types")
    sdk_types.HttpOptions = FakeHttpOptions
    sdk_types.GenerateContentConfig = FakeGenerateContentConfig
    genai.Client = FakeClient
    genai.types = sdk_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", sdk_types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = GeminiTranscriptionAdapter("test-model", 12.345).transcribe(
        tmp_path / "clip.wav",
        source_id="mic",
        start_seconds=0,
        end_seconds=180,
        prompt="transcribe only",
        request_timeout_seconds=12.345,
    )

    config = calls["generation_config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] == transcription_jobs.TRANSCRIPTION_SCHEMA
    assert result == {"raw_text": '{"outcome":"uncertain","segments":[]}'}
    assert calls["delete"] == {"name": "uploaded-audio"}


def test_response_schema_declares_properties_for_every_required_field():
    """Keep server-side structured-output validation failures out of retries."""

    def check(node):
        properties = node.get("properties", {})
        assert set(node.get("required", [])).issubset(properties)
        for child in properties.values():
            check(child)
        if isinstance(node.get("items"), dict):
            check(node["items"])

    check(transcription_jobs.TRANSCRIPTION_SCHEMA)
    assert transcription_jobs.TRANSCRIPTION_SCHEMA["properties"]["outcome"] == {
        "type": "string",
        "enum": ["transcribed", "no_speech", "uncertain", "failed"],
    }


def test_request_contract_version_creates_a_new_job_without_deleting_old_cache(
    tmp_path, monkeypatch
):
    source = make_source(tmp_path)
    state_path = tmp_path / "adapter-state.json"
    adapter = FilePlanAdapter(state_path, {"0": [FilePlanAdapter._default(0)]})
    first = make_pipeline(tmp_path, adapter, duration=180).run(
        [source_payload(source)], session_id="meeting-1"
    )

    next_version = "gemini-structured-output-v2-test"
    monkeypatch.setattr(transcription_jobs, "REQUEST_CONTRACT_VERSION", next_version)
    monkeypatch.setattr(
        transcription_jobs,
        "GEMINI_REQUEST_CONTRACT",
        {**transcription_jobs.GEMINI_REQUEST_CONTRACT, "version": next_version},
    )
    second = make_pipeline(tmp_path, adapter, duration=180).run(
        [source_payload(source)], session_id="meeting-1"
    )

    assert first["_meta"]["job_key"] != second["_meta"]["job_key"]
    assert first["_meta"]["request_contract_version"] != second["_meta"][
        "request_contract_version"
    ]
    assert calls(state_path) == [0.0, 0.0]
    assert len(list((tmp_path / "state" / "jobs").iterdir())) == 2
