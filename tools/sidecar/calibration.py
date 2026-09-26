"""Local-only calibration harness for Gemini sidecar probabilities.

The evaluation directory intentionally contains client transcript text. This
module reads it in place and writes only aggregate metrics/score caches under
``~/.local/share/meeting-sidecar``; it never creates a fixture or repository
artifact from that text.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration_policy import (
    OWNER_APPROVED_CLIP_GROW_THRESHOLD,
    OWNER_APPROVED_CLIP_SEED_THRESHOLD,
    OWNER_APPROVED_MODEL,
    OWNER_APPROVED_PROMPT_THRESHOLD,
    OWNER_OVERRIDE_STATUS,
    owner_override_payload,
    unwaived_owner_criteria_pass,
)
from .judges import (
    JUDGE_PROMPT_VERSION,
    JUDGE_PROTOCOL_FINGERPRINT,
    GeminiJudge,
    Judge,
    question_protocol_fingerprint,
)
from .questions import DICTATING_PROMPT_QUESTION
from .service import relevance_question
from .transcript import TranscriptLine, timestamp_seconds


DEFAULT_EVAL_DIR = Path.home() / ".local" / "share" / "meeting-sidecar" / "eval-2026-09-26"
DEFAULT_REPORT_DIR = Path.home() / ".local" / "share" / "meeting-sidecar" / "calibration"
ORIGINAL_VERSION = {"A": "en", "B": "gsw", "C": "gsw"}


@dataclass(frozen=True)
class Metric:
    model: str
    dataset: str
    version: str
    question: str
    positives: int
    negatives: int
    auc: float | None
    threshold: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    jev_auc: float | None
    jev_precision: float | None
    jev_recall: float | None
    jev_false_positives: int | None


@dataclass(frozen=True)
class ModelSummary:
    model: str
    prompt_threshold: float
    relevance_seed_threshold: float
    relevance_grow_threshold: float
    pass_relevance_a: bool
    pass_relevance_b: bool
    pass_prompt_c: bool
    request_count: int
    p50_request_seconds: float | None
    input_tokens: int
    output_tokens: int


def auc(labels: Sequence[bool], probabilities: Sequence[float]) -> float | None:
    positive = [score for label, score in zip(labels, probabilities) if label]
    negative = [score for label, score in zip(labels, probabilities) if not label]
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def precision_recall(labels: Sequence[bool], probabilities: Sequence[float], threshold: float) -> tuple[float, float, int, int, int]:
    true_positives = sum(label and score >= threshold for label, score in zip(labels, probabilities))
    false_positives = sum(not label and score >= threshold for label, score in zip(labels, probabilities))
    false_negatives = sum(label and score < threshold for label, score in zip(labels, probabilities))
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    return precision, recall, true_positives, false_positives, false_negatives


def choose_prompt_threshold(labels: Sequence[bool], probabilities: Sequence[float]) -> float:
    """Choose a C-Swiss-German threshold meeting the stated recall/FP bar if possible."""
    candidates = sorted({0.0, 0.45, 1.0, *(max(0.0, min(1.0, float(value))) for value in probabilities)})
    scored: list[tuple[tuple[float, ...], float]] = []
    target_recall = 5 / 6
    for threshold in candidates:
        precision, recall, true_positives, false_positives, false_negatives = precision_recall(labels, probabilities, threshold)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        meets = recall >= target_recall and false_positives <= 2
        # Passing the acceptance envelope outranks everything. Within it,
        # choose the highest F1, then fewer false positives, then the more
        # conservative (higher) threshold.
        rank = (float(meets), f1, recall, -float(false_positives), threshold, float(true_positives), -float(false_negatives))
        scored.append((rank, threshold))
    return max(scored, key=lambda item: item[0])[1]


def _dataset(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation dataset {path.name}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("lines"), list):
        raise ValueError(f"evaluation dataset {path.name} has no lines")
    return value


def _json_object(path: Path) -> dict[str, Any]:
    """Read a generic local-only JSON object such as the stored Jev scores."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calibration reference {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"calibration reference {path.name} must be an object")
    return value


def _version_lines(dataset: Mapping[str, Any], version: str) -> list[TranscriptLine]:
    source_lines = dataset["lines"]
    translations = dataset.get("versions", {}).get(version)
    if not isinstance(translations, Mapping):
        raise ValueError(f"evaluation dataset has no {version!r} version")
    result: list[TranscriptLine] = []
    for position, raw in enumerate(source_lines):
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation line is not an object")
        text = translations.get(str(position))
        if not isinstance(text, str):
            raise ValueError(f"evaluation version missing text for line {position}")
        timestamp = str(raw.get("ts", "00:00"))
        try:
            offset_seconds = timestamp_seconds(timestamp)
        except (ValueError, AttributeError):
            offset_seconds = float(position)
        result.append(
            TranscriptLine(
                index=position,
                seconds=float(offset_seconds),
                timestamp=timestamp,
                speaker=str(raw.get("spk", "Speaker")),
                text=text,
            )
        )
    return result


def _labels(dataset: Mapping[str, Any], question: str) -> list[bool]:
    gold = dataset.get("gold")
    if not isinstance(gold, Mapping):
        raise ValueError("evaluation dataset has no gold labels")
    labels: list[bool] = []
    for position in range(len(dataset["lines"])):
        row = gold.get(str(position))
        if not isinstance(row, Mapping):
            raise ValueError(f"evaluation dataset has no gold label for line {position}")
        labels.append(bool(row.get(question)))
    return labels


def _jev_probabilities(
    reference: Mapping[str, Any], dataset: str, version: str, question: str, count: int
) -> list[float] | None:
    values = reference.get(f"{dataset}_{version}")
    if not isinstance(values, Mapping):
        return None
    result: list[float] = []
    for position in range(count):
        row = values.get(str(position))
        if not isinstance(row, Mapping) or question not in row:
            return None
        try:
            result.append(float(row[question]))
        except (TypeError, ValueError):
            return None
    return result


def _metric(
    *,
    model: str,
    dataset: str,
    version: str,
    question: str,
    labels: Sequence[bool],
    probabilities: Sequence[float],
    threshold: float,
    jev_probabilities: Sequence[float] | None,
) -> Metric:
    precision, recall, tp, fp, fn = precision_recall(labels, probabilities, threshold)
    if jev_probabilities is None:
        jev_auc = jev_precision = jev_recall = None
        jev_fp = None
    else:
        jev_auc = auc(labels, jev_probabilities)
        jev_precision, jev_recall, _, jev_fp, _ = precision_recall(labels, jev_probabilities, threshold)
    return Metric(
        model=model,
        dataset=dataset,
        version=version,
        question=question,
        positives=sum(labels),
        negatives=len(labels) - sum(labels),
        auc=auc(labels, probabilities),
        threshold=threshold,
        precision=precision,
        recall=recall,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        jev_auc=jev_auc,
        jev_precision=jev_precision,
        jev_recall=jev_recall,
        jev_false_positives=jev_fp,
    )


def _scores_for_dataset(
    judge: Judge,
    dataset_name: str,
    dataset: Mapping[str, Any],
    version: str,
) -> dict[str, list[float]]:
    lines = _version_lines(dataset, version)
    # Match the production request layouts exactly. A provider can condition a
    # probability on the other questions/schema in the same request, so a
    # convenience co-batch would not validate either CLI operation.
    relevance = judge.judge(
        lines,
        {"relevant": relevance_question(str(dataset.get("topic", "the selected meeting topic")))},
    )["relevant"]
    dictating_prompt = judge.judge(
        lines,
        {"dictating_prompt": DICTATING_PROMPT_QUESTION},
    )["dictating_prompt"]
    return {"relevant": relevance, "dictating_prompt": dictating_prompt}


def calibrate_model(
    model: str,
    *,
    eval_dir: str | Path = DEFAULT_EVAL_DIR,
    judge: Judge | None = None,
) -> tuple[list[Metric], ModelSummary]:
    """Evaluate one Gemini model over every A/B/C language version.

    Calibration deliberately bypasses the probability cache. A warm cache
    would make latency/token telemetry look like zero and could hide a changed
    provider prompt. Runtime clip and prompt calls still use the local cache.
    """
    root = Path(eval_dir).expanduser()
    datasets = {name: _dataset(root / f"ds_{name}.json") for name in ("A", "B", "C")}
    reference = _json_object(root / "jev_results.json")
    raw_judge = judge or GeminiJudge(model=model)
    metrics_before = getattr(raw_judge, "metrics", None)
    all_scores: dict[tuple[str, str], dict[str, list[float]]] = {}
    for dataset_name, dataset in datasets.items():
        versions = dataset.get("versions", {})
        if not isinstance(versions, Mapping):
            raise ValueError(f"ds_{dataset_name}.json has no versions")
        for version in versions:
            all_scores[(dataset_name, str(version))] = _scores_for_dataset(
                raw_judge, dataset_name, dataset, str(version)
            )

    c_labels = _labels(datasets["C"], "dictating_prompt")
    c_probabilities = all_scores[("C", ORIGINAL_VERSION["C"])]["dictating_prompt"]
    # Matthias's calibration decision fixes the runtime prompt threshold at
    # 0.90.  The B waiver cannot conceal a regression in prompt recall at a
    # conveniently lower threshold, so criterion C is measured at that exact
    # operating point.
    prompt_threshold = OWNER_APPROVED_PROMPT_THRESHOLD
    metrics: list[Metric] = []
    for dataset_name, dataset in datasets.items():
        for version in dataset["versions"]:
            for question in ("relevant", "dictating_prompt"):
                labels = _labels(dataset, question)
                # AUC is undefined with only one class; retain the row for an
                # audit trail but it will carry ``None``.
                threshold = prompt_threshold if question == "dictating_prompt" else 0.60
                probs = all_scores[(dataset_name, str(version))][question]
                metrics.append(
                    _metric(
                        model=model,
                        dataset=dataset_name,
                        version=str(version),
                        question=question,
                        labels=labels,
                        probabilities=probs,
                        threshold=threshold,
                        jev_probabilities=_jev_probabilities(reference, dataset_name, str(version), question, len(probs)),
                    )
                )

    def lookup(dataset: str, version: str, question: str) -> Metric:
        return next(
            metric
            for metric in metrics
            if (metric.dataset, metric.version, metric.question) == (dataset, version, question)
        )

    a = lookup("A", ORIGINAL_VERSION["A"], "relevant")
    b = lookup("B", ORIGINAL_VERSION["B"], "relevant")
    c = lookup("C", ORIGINAL_VERSION["C"], "dictating_prompt")
    model_metrics = getattr(raw_judge, "metrics", metrics_before)
    summary = ModelSummary(
        model=model,
        prompt_threshold=prompt_threshold,
        relevance_seed_threshold=0.60,
        relevance_grow_threshold=0.25,
        pass_relevance_a=(a.auc or 0.0) >= 0.88,
        pass_relevance_b=(b.auc or 0.0) >= 0.88,
        pass_prompt_c=c.recall >= 5 / 6 and c.false_positives <= 2,
        request_count=int(getattr(model_metrics, "requests", 0)),
        p50_request_seconds=getattr(model_metrics, "p50_seconds", None),
        input_tokens=int(getattr(model_metrics, "input_tokens", 0)),
        output_tokens=int(getattr(model_metrics, "output_tokens", 0)),
    )
    return metrics, summary


def choose_model(summaries: Sequence[ModelSummary], metrics: Sequence[Metric]) -> ModelSummary:
    """Choose accuracy first, then lower observed token volume/latency."""
    if not summaries:
        raise ValueError("no model summaries to choose from")
    by_model: dict[str, dict[tuple[str, str, str], Metric]] = {}
    for metric in metrics:
        by_model.setdefault(metric.model, {})[(metric.dataset, metric.version, metric.question)] = metric

    def rank(summary: ModelSummary) -> tuple[float, ...]:
        model_metrics = by_model[summary.model]
        a = model_metrics[("A", ORIGINAL_VERSION["A"], "relevant")].auc or 0.0
        b = model_metrics[("B", ORIGINAL_VERSION["B"], "relevant")].auc or 0.0
        c = model_metrics[("C", ORIGINAL_VERSION["C"], "dictating_prompt")]
        passes = float(summary.pass_relevance_a and summary.pass_relevance_b and summary.pass_prompt_c)
        # Higher accuracy wins. Costs are not returned by the AI Studio API, so
        # token count then observed p50 are the reproducible cost/latency tie-breakers.
        return (
            passes,
            a + b + c.recall,
            -float(c.false_positives),
            -float(summary.input_tokens + summary.output_tokens),
            -float(summary.p50_request_seconds or float("inf")),
        )

    return max(summaries, key=rank)


def apply_owner_override(report: Mapping[str, Any]) -> dict[str, Any]:
    """Record the dated Slice-1 owner override without making provider calls.

    The underlying aggregate measurement stays intact.  This merely records
    the explicit decision that B Swiss-German relevance is a known limitation,
    while retaining the measured 0.788--0.884 repeat range in the local report.
    """
    if not isinstance(report, Mapping):
        raise ValueError("calibration report must be an object")
    summaries = report.get("models")
    if not isinstance(summaries, list) or not any(
        isinstance(summary, Mapping) and summary.get("model") == OWNER_APPROVED_MODEL
        for summary in summaries
    ):
        raise ValueError(f"calibration report has no {OWNER_APPROVED_MODEL!r} measurement")

    amended = dict(report)
    if not unwaived_owner_criteria_pass(report):
        # Preserve the raw measurements but refuse to turn an A/C regression
        # into approval.  Prompt extraction then remains disabled because the
        # runtime accepts only `passed` or a valid owner override.
        amended["status"] = "failed_unwaived_acceptance_criteria"
        amended["chosen"] = None
        amended.pop("owner_override", None)
        return amended
    amended["schema_version"] = max(3, int(report.get("schema_version", 0) or 0))
    # The final fingerprints bind the approved thresholds to today's exact
    # runtime prompt/schema.  This is a metadata binding, not a claim that the
    # provider measurement was rerun after the decision.
    amended["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    amended["judge_protocol_sha256"] = JUDGE_PROTOCOL_FINGERPRINT
    amended["dictating_prompt_protocol_sha256"] = question_protocol_fingerprint(
        "dictating_prompt", DICTATING_PROMPT_QUESTION
    )
    amended["status"] = OWNER_OVERRIDE_STATUS
    amended["owner_override"] = owner_override_payload()
    amended["chosen"] = {
        "model": OWNER_APPROVED_MODEL,
        "prompt_threshold": OWNER_APPROVED_PROMPT_THRESHOLD,
        "relevance_seed_threshold": OWNER_APPROVED_CLIP_SEED_THRESHOLD,
        "relevance_grow_threshold": OWNER_APPROVED_CLIP_GROW_THRESHOLD,
    }
    return amended


def read_calibration_report(path: str | Path | None = None) -> dict[str, Any]:
    """Read the local aggregate-only report; evaluation text never enters it."""
    location = Path(path or DEFAULT_REPORT_DIR / "latest.json").expanduser()
    return _json_object(location)


def calibrate(
    *,
    eval_dir: str | Path = DEFAULT_EVAL_DIR,
    models: Sequence[str] = (OWNER_APPROVED_MODEL,),
) -> dict[str, Any]:
    """Run the single approved Gemini judge and return aggregate-only results."""
    all_metrics: list[Metric] = []
    summaries: list[ModelSummary] = []
    started = time.time()
    for model in models:
        metrics, summary = calibrate_model(model, eval_dir=eval_dir)
        all_metrics.extend(metrics)
        summaries.append(summary)
    passing = [
        summary
        for summary in summaries
        if summary.pass_relevance_a and summary.pass_relevance_b and summary.pass_prompt_c
    ]
    chosen = choose_model(passing, all_metrics) if passing else None
    report = {
        "schema_version": 2,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_protocol_sha256": JUDGE_PROTOCOL_FINGERPRINT,
        "dictating_prompt_protocol_sha256": question_protocol_fingerprint(
            "dictating_prompt", DICTATING_PROMPT_QUESTION
        ),
        "generated_at_epoch": round(time.time(), 3),
        "elapsed_seconds": round(time.time() - started, 3),
        "eval_dir": str(Path(eval_dir).expanduser()),
        "models": [asdict(summary) for summary in summaries],
        "chosen": asdict(chosen) if chosen else None,
        "status": "passed" if chosen else "failed_acceptance_criterion_2",
        "metrics": [asdict(metric) for metric in all_metrics],
    }
    # This is the explicit owner-approved B-only exception, not a model
    # router.  The raw metrics above remain in the report for the later
    # human-reference-set remeasurement.  A fully passing report needs no
    # waiver; a failed A or C remains failed.
    return report if chosen else apply_owner_override(report)


def write_calibration_report(report: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Persist aggregate metrics only; never write the evaluation transcript text."""
    output = Path(path).expanduser() if path else DEFAULT_REPORT_DIR / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def markdown_table(report: Mapping[str, Any]) -> str:
    """Compact PR/docs table for the three acceptance measurements per model."""
    metric_index = {
        (item["model"], item["dataset"], item["version"], item["question"]): item
        for item in report.get("metrics", [])
    }
    rows = [
        "| Gemini model | A EN relevance AUC | B Swiss German relevance AUC | C Swiss German prompt recall / FP | Prompt threshold | Input / output tokens | p50 batch request |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in report.get("models", []):
        model = summary["model"]
        a = metric_index[(model, "A", "en", "relevant")]
        b = metric_index[(model, "B", "gsw", "relevant")]
        c = metric_index[(model, "C", "gsw", "dictating_prompt")]
        def auc_value(metric: Mapping[str, Any]) -> str:
            return "—" if metric["auc"] is None else f"{metric['auc']:.3f}"
        latency = summary.get("p50_request_seconds")
        latency_value = "—" if latency is None else f"{latency:.2f}s"
        rows.append(
            f"| {model} | {auc_value(a)} | {auc_value(b)} | "
            f"{c['recall']:.2f} / {c['false_positives']} | {summary['prompt_threshold']:.3f} | "
            f"{summary['input_tokens']:,} / {summary['output_tokens']:,} | {latency_value} |"
        )
    return "\n".join(rows)
