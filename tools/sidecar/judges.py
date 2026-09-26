"""Judge interface and its Gemini/Jev implementations.

The judge only supplies calibrated-looking probabilities. Selection, threshold
policy, cache reuse, and all clipboard actions stay deterministic in code.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .cache import ProbabilityCache, line_fingerprint
from .gate import judge_backend_for
from .transcript import TranscriptLine


class JudgeError(RuntimeError):
    """A provider did not return usable probabilities."""


QuestionInput = str | Mapping[str, Any]
Questions = Mapping[str, QuestionInput]

# Bump whenever the provider system prompt, JSON schema, batching semantics, or
# score interpretation changes. It is part of each question cache key.
JUDGE_PROMPT_VERSION = "sidecar-judge-v3-2026-09-26"
GEMINI_BATCH_SIZE = 20


@dataclass(frozen=True)
class Question:
    """Normalised binary-question definition shared by both backends."""

    id: str
    instructions: str
    true_criteria: str
    false_criteria: str


def normalise_questions(questions: Questions) -> dict[str, Question]:
    """Accept concise strings or explicit Jev-compatible question objects."""
    if not questions:
        raise ValueError("at least one judge question is required")
    result: dict[str, Question] = {}
    for question_id, supplied in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question ids must be non-empty strings")
        if isinstance(supplied, str):
            instructions = supplied.strip()
            criteria: Mapping[str, Any] = {}
        elif isinstance(supplied, Mapping):
            instructions = str(supplied.get("instructions", "")).strip()
            raw_criteria = supplied.get("criteria", {})
            criteria = raw_criteria if isinstance(raw_criteria, Mapping) else {}
        else:
            raise ValueError(f"question {question_id!r} must be a string or mapping")
        if not instructions:
            raise ValueError(f"question {question_id!r} needs instructions")
        result[question_id] = Question(
            id=question_id,
            instructions=instructions,
            true_criteria=str(criteria.get("true") or "The condition is present in the current line."),
            false_criteria=str(criteria.get("false") or "The condition is absent from the current line."),
        )
    return result


def _question_definition(question: Question) -> dict[str, Any]:
    """The complete binary definition for one model question."""
    return {
        "instructions": question.instructions,
        "criteria": {"true": question.true_criteria, "false": question.false_criteria},
    }


def _cache_question_definition(
    question: Question,
    request_questions: Mapping[str, Question],
    *,
    batch_size: int | None,
) -> dict[str, Any]:
    """Include the full ordered provider request layout in a cache key."""
    return {
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_protocol_sha256": judge_protocol_fingerprint(
            tuple(request_questions), batch_size=batch_size
        ),
        "question": _question_definition(question),
        "request_question_order": list(request_questions),
        "request_questions": {
            question_id: _question_definition(candidate)
            for question_id, candidate in request_questions.items()
        },
    }


class Judge(Protocol):
    """One probability vector per requested question, indexed like ``lines``."""

    backend: str
    model: str | None

    def judge(
        self,
        lines: Sequence[TranscriptLine],
        questions: Questions,
        *,
        context: int = 6,
    ) -> dict[str, list[float]]: ...


def _line_payload(line: TranscriptLine) -> dict[str, Any]:
    return {
        "line_index": line.index,
        "timestamp": line.timestamp,
        "speaker": line.speaker,
        "text": line.text,
    }


def _batch_prompt(
    items: list[dict[str, Any]], questions: Mapping[str, Question]
) -> str:
    question_text = "\n".join(
        f"- {question.id}: {question.instructions}\n"
        f"  true: {question.true_criteria}\n"
        f"  false: {question.false_criteria}"
        for question in questions.values()
    )
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return f"""You are a careful meeting-transcript classifier. Return a probability from 0 to 1 for each requested binary question and each item. Assess only the current line; previous_lines are context for resolving pronouns or references, not separate evidence. Be conservative with greetings, acknowledgements, filler, process talk, and vague semantic overlap.

For a dictated-prompt question, count text that is itself a reusable instruction/prompt the speaker dictates. Do not count a sentence merely talking about prompts, a lead-in such as 'I will record this prompt', or an explanation around a prompt. English prompt text embedded in Swiss German still counts.

Questions:
{question_text}

Items (each has previous_lines followed by current_line):
{payload}

Return JSON only, matching this shape exactly:
{{"results":[{{"line_index":0,"probabilities":{{"question_id":0.0}}}}]}}
"""


def _response_schema(question_ids: Sequence[str]) -> dict[str, Any]:
    probabilities = {
        "type": "object",
        "properties": {
            question_id: {"type": "number", "minimum": 0, "maximum": 1}
            for question_id in question_ids
        },
        "required": list(question_ids),
    }
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_index": {"type": "integer"},
                        "probabilities": probabilities,
                    },
                    "required": ["line_index", "probabilities"],
                },
            }
        },
        "required": ["results"],
    }


def judge_protocol_fingerprint(
    question_ids: Sequence[str], *, batch_size: int | None = GEMINI_BATCH_SIZE
) -> str:
    """Hash the rendered batch protocol and its exact JSON schema.

    The cache must invalidate if either the batch template or structured-output
    contract changes, even when a maintainer forgets to bump a human version.
    Question wording/criteria are independently included by the caller.
    """
    probe_questions = {
        question_id: Question(
            id=question_id,
            instructions="Protocol fingerprint probe instruction.",
            true_criteria="Probe true criterion.",
            false_criteria="Probe false criterion.",
        )
        for question_id in question_ids
    }
    rendered = _batch_prompt(
        [
            {
                "previous_lines": [],
                "current_line": {
                    "line_index": 0,
                    "timestamp": "00:00",
                    "speaker": "Probe",
                    "text": "Protocol fingerprint probe text.",
                },
            }
        ],
        probe_questions,
    )
    payload = json.dumps(
        {
            "batch_prompt": rendered,
            "response_schema": _response_schema(question_ids),
            "batch_size": batch_size,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


JUDGE_PROTOCOL_FINGERPRINT = judge_protocol_fingerprint(
    ("__protocol_probe__",), batch_size=GEMINI_BATCH_SIZE
)


def question_protocol_fingerprint(
    question_id: str,
    supplied: QuestionInput,
    *,
    batch_size: int | None = GEMINI_BATCH_SIZE,
) -> str:
    """Bind a calibrated threshold to one canonical question and its protocol."""
    question = normalise_questions({question_id: supplied})[question_id]
    payload = json.dumps(
        {
            "judge_protocol_sha256": judge_protocol_fingerprint(
                (question_id,), batch_size=batch_size
            ),
            "question": _question_definition(question),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if hasattr(parsed, "model_dump"):
        dumped = parsed.model_dump()
        if isinstance(dumped, dict):
            return dumped
    raw = getattr(response, "text", response)
    if not isinstance(raw, str):
        raise JudgeError("Gemini returned no JSON text")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgeError("Gemini returned invalid judge JSON") from exc
    if not isinstance(value, dict):
        raise JudgeError("Gemini judge JSON must be an object")
    return value


def _bounded_probability(value: Any, *, question_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise JudgeError(f"judge omitted a numeric probability for {question_id!r}") from exc
    if not 0 <= result <= 1:
        raise JudgeError(f"judge probability for {question_id!r} is outside 0..1")
    return result


@dataclass
class GeminiMetrics:
    requests: int = 0
    latencies_seconds: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def p50_seconds(self) -> float | None:
        if not self.latencies_seconds:
            return None
        values = sorted(self.latencies_seconds)
        return values[len(values) // 2]


class GeminiJudge:
    """Contract-covered text judge using the recorder's Gemini key/client route."""

    backend = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gemini-3.8-flash",
        client: Any | None = None,
        types_module: Any | None = None,
        batch_size: int = GEMINI_BATCH_SIZE,
        timeout_seconds: float = 45,
        retry_attempts: int = 2,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be positive")
        self.model = model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.metrics = GeminiMetrics()
        if client is not None:
            self.client = client
            self.types = types_module
            return
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise JudgeError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on local installation
            raise JudgeError("google-genai is required for GeminiJudge") from exc
        self.client = genai.Client(api_key=key)
        self.types = types

    def _generate(self, prompt: str, schema: dict[str, Any]) -> Any:
        started = time.monotonic()
        try:
            if self.types is None:
                config: Any = {
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "http_options": {
                        "timeout": int(self.timeout_seconds * 1000),
                        "retry_options": {"attempts": self.retry_attempts},
                    },
                }
            else:
                config = self.types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=schema,
                    http_options=self.types.HttpOptions(
                        timeout=int(self.timeout_seconds * 1000),
                        retry_options=self.types.HttpRetryOptions(attempts=self.retry_attempts),
                    ),
                )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise JudgeError(f"Gemini judge request failed: {type(exc).__name__}") from exc
        finally:
            self.metrics.requests += 1
            self.metrics.latencies_seconds.append(time.monotonic() - started)
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        if usage is not None:
            self.metrics.input_tokens += int(
                getattr(usage, "prompt_token_count", None)
                or getattr(usage, "promptTokenCount", None)
                or 0
            )
            self.metrics.output_tokens += int(
                getattr(usage, "candidates_token_count", None)
                or getattr(usage, "candidatesTokenCount", None)
                or 0
            )
        return response

    def judge(
        self,
        lines: Sequence[TranscriptLine],
        questions: Questions,
        *,
        context: int = 6,
    ) -> dict[str, list[float]]:
        normalised = normalise_questions(questions)
        if context < 0:
            raise ValueError("context must be non-negative")
        positions = {line.index: position for position, line in enumerate(lines)}
        if len(positions) != len(lines):
            raise JudgeError("line indexes must be unique")
        result = {question_id: [0.0] * len(lines) for question_id in normalised}
        for batch_start in range(0, len(lines), self.batch_size):
            batch = lines[batch_start : batch_start + self.batch_size]
            items: list[dict[str, Any]] = []
            expected_indexes = {line.index for line in batch}
            for line in batch:
                position = positions.get(line.index)
                # Lines supplied by this package always have contiguous indexes,
                # but position-based context preserves the public interface for
                # callers that construct line objects themselves.
                if position is None:
                    raise JudgeError("line indexes must be unique")
                previous = lines[max(0, position - context) : position]
                items.append(
                    {"previous_lines": [_line_payload(item) for item in previous], "current_line": _line_payload(line)}
                )
            response = _load_json_response(self._generate(_batch_prompt(items, normalised), _response_schema(list(normalised))))
            rows = response.get("results")
            if not isinstance(rows, list):
                raise JudgeError("Gemini judge JSON has no results array")
            seen: set[int] = set()
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                try:
                    line_index = int(row.get("line_index"))
                except (TypeError, ValueError):
                    continue
                probabilities = row.get("probabilities")
                if line_index not in expected_indexes or line_index in seen or not isinstance(probabilities, Mapping):
                    continue
                for question_id in normalised:
                    result[question_id][positions[line_index]] = _bounded_probability(
                        probabilities.get(question_id), question_id=question_id
                    )
                seen.add(line_index)
            if seen != expected_indexes:
                raise JudgeError("Gemini judge response did not score every requested line exactly once")
        return result


def _extract_jev_probability(value: Any) -> float:
    """Tolerate the helper's typed-answer wrappers while rejecting ambiguity."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return _bounded_probability(value, question_id="Jev answer")
    if isinstance(value, Mapping):
        for key in ("probability", "value", "yes_probability", "true_probability"):
            if key in value:
                return _bounded_probability(value[key], question_id="Jev answer")
        distribution = value.get("distribution")
        if isinstance(distribution, Mapping):
            for key in (True, "true", "yes"):
                if key in distribution:
                    return _bounded_probability(distribution[key], question_id="Jev answer")
    raise JudgeError("Jev returned an answer without a yes probability")


class JevJudge:
    """Optional audited Jev adapter. Construction is guarded by ``create_judge``."""

    backend = "jev"

    def __init__(
        self,
        *,
        stem: str,
        recordings_root: str | Path | None = None,
        helper_path: str | Path | None = None,
        python: str | None = None,
        timeout_seconds: float = 30,
        model: str = "~typesafe/jev-latest",
    ):
        # The service already checks this, but the adapter must not be a
        # bypassable public OpenRouter entry point. This rejects an
        # unresolved/malformed attendance field before locating the helper.
        judge_backend_for(stem, "jev", recordings_root=recordings_root)
        self.stem = stem
        self.recordings_root = recordings_root
        self.model = model
        self.helper_path = Path(helper_path or Path.home() / ".claude" / "skills" / "jev" / "scripts" / "jev_decide.py")
        self.python = python or os.environ.get("PYTHON", "python3")
        self.timeout_seconds = timeout_seconds
        if not self.helper_path.is_file():
            raise JudgeError(f"Jev helper not found: {self.helper_path}")

    def _call(self, body: dict[str, Any]) -> dict[str, Any]:
        # Recheck immediately before a request in case the sidecar policy was
        # amended after construction.
        judge_backend_for(self.stem, "jev", recordings_root=self.recordings_root)
        command = [
            self.python,
            str(self.helper_path),
            "--classification",
            "internal",
            "--external-processing-approved",
            "--model",
            self.model,
            "--request",
            "-",
            "--answers-only",
            "--timeout",
            str(self.timeout_seconds),
        ]
        try:
            process = subprocess.run(
                command,
                input=json.dumps(body, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds + 5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JudgeError(f"Jev helper failed: {type(exc).__name__}") from exc
        if process.returncode:
            raise JudgeError("Jev helper rejected the gated request")
        try:
            output = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise JudgeError("Jev helper returned invalid JSON") from exc
        if not isinstance(output, dict):
            raise JudgeError("Jev helper returned a non-object answer")
        return output

    def judge(
        self,
        lines: Sequence[TranscriptLine],
        questions: Questions,
        *,
        context: int = 6,
    ) -> dict[str, list[float]]:
        normalised = normalise_questions(questions)
        values = {question_id: [] for question_id in normalised}
        for position, line in enumerate(lines):
            previous = lines[max(0, position - context) : position]
            body = {
                "state": {
                    "previous_lines": [_line_payload(item) for item in previous],
                    "current_line": _line_payload(line),
                },
                "questions": {
                    question_id: {
                        "type": "noul",
                        "instructions": question.instructions,
                        "criteria": {"true": question.true_criteria, "false": question.false_criteria},
                    }
                    for question_id, question in normalised.items()
                },
            }
            answers = self._call(body)
            for question_id in normalised:
                if question_id not in answers:
                    raise JudgeError(f"Jev omitted answer {question_id!r}")
                values[question_id].append(_extract_jev_probability(answers[question_id]))
        return values


class CachedJudge:
    """Caching wrapper that asks a backend only for probability vectors not on disk."""

    def __init__(self, delegate: Judge, cache: ProbabilityCache, stem: str):
        self.delegate = delegate
        self.cache = cache
        self.stem = stem
        self.backend = delegate.backend
        self.model = delegate.model

    def judge(
        self,
        lines: Sequence[TranscriptLine],
        questions: Questions,
        *,
        context: int = 6,
    ) -> dict[str, list[float]]:
        normalised = normalise_questions(questions)
        fingerprint = line_fingerprint(lines, context)
        delegate_batch_size = getattr(self.delegate, "batch_size", None)
        if type(delegate_batch_size) is not int or delegate_batch_size < 1:
            delegate_batch_size = None
        result: dict[str, list[float]] = {}
        missing: dict[str, QuestionInput] = {}
        for question_id, question in normalised.items():
            cache_definition = _cache_question_definition(
                question, normalised, batch_size=delegate_batch_size
            )
            cached = self.cache.get(
                self.stem,
                self.backend,
                question_id,
                cache_definition,
                input_fingerprint=fingerprint,
                model=self.model,
                line_count=len(lines),
            )
            if cached is None:
                missing[question_id] = questions[question_id]
            else:
                result[question_id] = cached
        if missing:
            # Provider probabilities can depend on the complete question list
            # and response schema. If one member of a requested set misses, do
            # not splice it into vectors cached from a different layout: ask
            # for the full original set and cache all of it under that layout.
            fresh = self.delegate.judge(lines, questions, context=context)
            for question_id, scores in fresh.items():
                if question_id not in normalised or len(scores) != len(lines):
                    raise JudgeError("judge returned an invalid probability vector")
                normalised_scores = [_bounded_probability(score, question_id=question_id) for score in scores]
                result[question_id] = normalised_scores
                self.cache.put(
                    self.stem,
                    self.backend,
                    question_id,
                    _cache_question_definition(
                        normalised[question_id], normalised, batch_size=delegate_batch_size
                    ),
                    input_fingerprint=fingerprint,
                    model=self.model,
                    scores=normalised_scores,
                )
            if set(fresh) != set(normalised):
                raise JudgeError("judge did not return every requested probability vector")
        return {question_id: result[question_id] for question_id in normalised}
