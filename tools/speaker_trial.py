#!/usr/bin/env python3
"""A bounded, resumable production trial. Shadow runs never ingest or notify.

All evidence lives outside Git. Review labels are explicit observations,
never inferred from model confidence, number of repairs or API success.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from speaker_integrity import atomic_json, coherence_complete

DEFAULT_STATE = Path.home() / ".local/share/meeting-pipeline-trial/2026-09-05"
PRODUCTION = Path.home() / "Repos/MeetingMemory"
CONFIG = Path.home() / "Documents/MeetingRecorder/config.yaml"
RULES = {
    "version": 1, "minimum_reviewed_meetings": 5, "minimum_reviewed_seconds": 1200,
    "speaker_confusion_improvement_target": 0.50,
    "speaker_error_regression_pp": 2.0, "word_error_regression_pp": 2.0,
    "overlap_recall_regression_pp": 5.0,
    "paired_partial_regression_pp": 10.0, "minimum_operational_pairs": 5,
    "new_critical_owner_errors_allowed": 0, "new_confirmed_omissions_allowed": 0,
    "policy": "Revert demonstrated material regressions. Otherwise keep; insufficient evidence is inconclusive, not improvement.",
    "truth_sources": ["human_confirmed", "channel_confirmed"],
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_baseline(manifest: dict) -> str:
    """A detached checkout is mutable. Check code and config before import."""
    repo = str(manifest["baseline_repo"])
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all"], text=True)
    if commit != manifest["baseline_commit"] or dirty.strip():
        raise RuntimeError("Frozen baseline is dirty or at a different commit; refusing misleading replay")
    if sha256(Path(manifest["config_backup"])) != manifest["config_sha256"]:
        raise RuntimeError("Frozen baseline configuration changed; refusing misleading replay")
    return commit


def metrics(data: dict) -> dict:
    import re
    meta = data.get("_meta") or {}
    transcript = data.get("transcript") or ""
    duration = float(meta.get("audio_duration_seconds") or 0)
    timestamps = [sum(int(v) * 60**i for i, v in enumerate(reversed(t.split(":"))))
                  for t in re.findall(r"\[(\d{1,2}(?::\d{2}){1,2})\]", transcript)]
    ranges = meta.get("missing_time_ranges") or []
    return {
        "duration_seconds": duration, "word_count": len(transcript.split()),
        "partial": bool(meta.get("partial") or ranges),
        "missing_chunk_seconds": sum(max(0, float(b)-float(a)) for a,b in ranges),
        "invalid_timestamps": sum(t > duration + 2 for t in timestamps),
        "nonmonotonic_timestamps": sum(b < a for a,b in zip(timestamps, timestamps[1:])),
        "semantic_complete": coherence_complete(data.get("speaker_coherence")),
        "channel_checked_turns": (data.get("speaker_verification") or {}).get("turns_checked", 0),
        "channel_flips": len((data.get("speaker_verification") or {}).get("flips") or []),
        "attribution": meta.get("speaker_attribution") or {},
        "no_speech": meta.get("speech_gate") == "digital_silence",
        "processing_seconds": float(meta.get("processing_time_seconds") or 0),
    }


def initialize(state: Path, start: str, end: str) -> dict:
    path = state / "manifest.json"
    if path.exists():
        raise RuntimeError("Trial already initialized; refusing to overwrite the baseline or rules")
    state.mkdir(parents=True, exist_ok=True)
    backup = state / "config.before.yaml"
    shutil.copy2(CONFIG, backup)
    backup.chmod(0o600)
    cfg = yaml.safe_load(backup.read_text())
    transcripts = Path(cfg["paths"]["transcripts"]).expanduser()
    cohort = []
    for path in sorted(transcripts.glob("2026-09-0[1-4]_*.json")):
        data = read(path)
        if metrics(data)["duration_seconds"] < 600:
            continue
        dest = state / "baseline_cohort" / path.name
        atomic_json(dest, data)
        cohort.append({"file": str(dest), "sha256": sha256(dest), "metrics": metrics(data)})
    manifest = {
        "schema_version": 1, "state": "prepared", "start": start, "end": end,
        "timezone": "Asia/Hong_Kong", "production_repo": str(PRODUCTION),
        "baseline_repo": str(state / "baseline"),
        "baseline_commit": subprocess.check_output(["git", "-C", str(state / "baseline"), "rev-parse", "HEAD"], text=True).strip(),
        "config_backup": str(backup), "config_sha256": sha256(backup),
        "rules": RULES, "baseline_cohort": cohort,
        "recordings_dir": str(Path(cfg["paths"]["recordings"]).expanduser()),
        "transcripts_dir": str(transcripts), "review_complete": False,
    }
    verify_baseline(manifest)
    atomic_json(state / "manifest.json", manifest)
    return manifest


def in_window(stem: str, manifest: dict) -> bool:
    from zoneinfo import ZoneInfo
    try:
        dt = datetime.strptime(stem, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=ZoneInfo(manifest["timezone"]))
        return datetime.fromisoformat(manifest["start"]) <= dt < datetime.fromisoformat(manifest["end"])
    except ValueError:
        return False


def collect(state: Path) -> dict:
    manifest = read(state / "manifest.json")
    rows = []
    for wav in sorted(Path(manifest["recordings_dir"]).glob("*.wav")):
        if not in_window(wav.stem, manifest):
            continue
        path = Path(manifest["transcripts_dir"]) / (wav.stem + ".json")
        row = {"stem": wav.stem, "audio_path": str(wav), "transcript_path": str(path),
               "audio_bytes": wav.stat().st_size,
               "awaiting_transcript_seconds": max(0, time.time()-wav.stat().st_mtime)}
        if path.exists():
            try:
                data = read(path)
                revisions = sorted((state / "recordings" / wav.stem).glob("candidate-*.json"), key=lambda p:p.stat().st_mtime)
                revision = read(revisions[-1]) if revisions else None
                if revision:
                    data = revision["payload"]
                row["candidate"] = metrics(data)
                row["candidate_sha256"] = revision["payload_sha256"] if revision else sha256(path)
                row["candidate_revision"] = str(revisions[-1]) if revisions else str(path)
                if revision:
                    row["candidate"]["pipeline_elapsed_seconds"] = revision.get("context", {}).get("elapsed_seconds")
                row["eligible_for_quality_sample"] = row["candidate"]["duration_seconds"] >= 600 and not row["candidate"]["no_speech"]
            except (ValueError, OSError) as exc:
                row["error"] = str(exc)
        shadow = state / "shadows" / wav.stem / "baseline" / (wav.stem + ".json")
        if shadow.exists():
            run = shadow.parent / "run.json"
            certificate = read(run) if run.exists() else {}
            digest = sha256(shadow)
            if (certificate.get("baseline_commit") == manifest["baseline_commit"]
                    and certificate.get("transcript_sha256") == digest
                    and certificate.get("config_sha256") == manifest["config_sha256"]):
                row["baseline"] = metrics(read(shadow))
                row["baseline_sha256"] = digest
                row["baseline"]["pipeline_elapsed_seconds"] = certificate.get("elapsed_seconds")
            else:
                row["baseline_error"] = "missing or invalid provenance certificate"
        queue = state / "recordings" / wav.stem / "verification_queue.json"
        if queue.exists():
            row["verification_queue"] = read(queue)
        rows.append(row)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "rows": rows,
              "raw_recordings": len(rows), "completed_transcripts": sum("candidate" in x for x in rows),
              "paired_baselines": sum("baseline" in x for x in rows),
              "free_disk_gb": round(shutil.disk_usage(state).free/1e9, 2),
              "limitations": "Word count, label changes, VAD overlap and completed audits are not accuracy measurements."}
    atomic_json(state / "metrics.latest.json", report)
    return report


def evaluate(rows: list[dict], annotations: list[dict], rules: dict) -> dict:
    """Use confirmed paired evidence; automatically generated labels are proxies."""
    confirmed = [a for a in annotations if a.get("truth_source") in rules["truth_sources"]]
    valid = []
    seen = set()
    for a in confirmed:
        key = (a.get("meeting"), a.get("start"), a.get("end"))
        # Require audio evidence and immutable revision hashes, avoiding
        # accidental comparisons against a different transcript revision.
        if (key in seen or type(a.get("baseline_speaker_wrong")) is not bool
                or type(a.get("candidate_speaker_wrong")) is not bool
                or not a.get("audio_evidence") or not a.get("baseline_sha256")
                or not a.get("candidate_sha256") or a.get("end", 0) <= a.get("start", 0)):
            continue
        if any(v["meeting"] == a["meeting"] and max(v["start"], a["start"]) < min(v["end"], a["end"]) for v in valid):
            continue
        pair = next((r for r in rows if r["stem"] == a["meeting"]), None)
        if (pair is None or pair.get("baseline_sha256") != a["baseline_sha256"]
                or pair.get("candidate_sha256") != a["candidate_sha256"]):
            continue
        seen.add(key)
        valid.append(a)
    seconds = sum(a["end"]-a["start"] for a in valid)
    meetings = len({a["meeting"] for a in valid})
    b_error = sum(a["end"]-a["start"] for a in valid if a.get("baseline_speaker_wrong"))
    c_error = sum(a["end"]-a["start"] for a in valid if a.get("candidate_speaker_wrong"))
    regressions = []
    if seconds and 100*(c_error-b_error)/seconds > rules["speaker_error_regression_pp"]:
        regressions.append("confirmed speaker-confusion time increased by more than 2 percentage points")
    for a in valid:
        for field in ("critical_owner_error", "omitted_speech", "duplicate_echo_text"):
            if a.get("candidate_"+field) and not a.get("baseline_"+field):
                regressions.append("new confirmed " + field + " in " + a["meeting"])
    words = [a for a in valid if a.get("reference_words", 0) > 0
             and "baseline_word_errors" in a and "candidate_word_errors" in a]
    if words:
        denominator = sum(a["reference_words"] for a in words)
        delta = sum(a["candidate_word_errors"]-a["baseline_word_errors"] for a in words)
        if 100*delta/denominator > rules["word_error_regression_pp"]:
            regressions.append("confirmed word-error rate worsened by more than 2 percentage points")
    overlaps = [a for a in valid if a.get("overlapping_host_seconds", 0) > 0
                and "baseline_host_recalled_seconds" in a and "candidate_host_recalled_seconds" in a]
    if overlaps:
        denominator = sum(a["overlapping_host_seconds"] for a in overlaps)
        delta = sum(a["baseline_host_recalled_seconds"]-a["candidate_host_recalled_seconds"] for a in overlaps)
        if 100*delta/denominator > rules["overlap_recall_regression_pp"]:
            regressions.append("confirmed overlapping-host recall fell by more than 5 percentage points")
    pairs = [r for r in rows if r.get("candidate") and r.get("baseline")
             and r.get("eligible_for_quality_sample")]
    for row in rows:
        candidate = row.get("candidate") or {}
        attribution = candidate.get("attribution") or {}
        if candidate.get("no_speech") and candidate.get("word_count"):
            regressions.append("speech generated from digital silence: " + row["stem"])
        if attribution.get("labels_changed") and attribution.get("speaker_statistics") != "invalidated_after_relabel":
            regressions.append("stale statistics after relabel: " + row["stem"])
    if len(pairs) >= rules["minimum_operational_pairs"]:
        delta = sum(int(r["candidate"]["partial"])-int(r["baseline"]["partial"]) for r in pairs)
        if 100*delta/len(pairs) > rules["paired_partial_regression_pp"]:
            regressions.append("paired partial-transcript rate worsened by more than 10 percentage points")
    # These require the reviewer to establish change causality, not merely
    # notice an old malformed timestamp or an unrelated disk/quota failure.
    operational = [a for a in annotations if a.get("kind") == "confirmed_operational_regression"
                   and a.get("evidence") and a.get("introduced_by_trial") is True]
    for item in operational:
        regressions.append(str(item["evidence"]))
    enough = meetings >= rules["minimum_reviewed_meetings"] and seconds >= rules["minimum_reviewed_seconds"]
    improved = enough and b_error > 0 and c_error <= b_error*(1-rules["speaker_confusion_improvement_target"])
    return {"decision": "revert" if regressions else ("keep_improved" if improved else ("keep_nonregressed" if enough else "keep_inconclusive")),
            "regressions": sorted(set(regressions)), "reviewed_meetings": meetings,
            "reviewed_seconds": seconds, "confirmed_baseline_speaker_error_seconds": b_error,
            "confirmed_candidate_speaker_error_seconds": c_error,
            "unconfirmed_or_stale_annotations": len(annotations)-len(valid),
            "accuracy_claim_permitted": enough,
            "note": "An inconclusive keep follows the requested no-demonstrated-regression policy; it is not evidence of improvement."}


def baseline_worker(state: Path, stem: str) -> None:
    """Import the frozen code in a fresh process and prohibit all publication."""
    manifest = read(state / "manifest.json")
    repo = Path(manifest["baseline_repo"])
    actual_commit = verify_baseline(manifest)
    from dotenv import load_dotenv
    load_dotenv(PRODUCTION / ".env")
    # speaker_integrity is this harness only; the actual pipeline siblings
    # are imported from the immutable baseline, never the candidate folder.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tools"))
    import transcribe_watcher as tw
    cfg = yaml.safe_load(Path(manifest["config_backup"]).read_text())
    output = state / "shadows" / stem / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    # A failed retry must not certify an older leftover as its own result.
    for previous in (output / (stem + ".json"), output / "run.json"):
        if previous.exists():
            previous.rename(previous.with_name(f"previous-{time.time_ns()}-{previous.name}"))
    cfg["paths"].update(recordings=str(output / "unused"), transcripts=str(output), logs=str(output))
    cfg["claude_trigger"] = {**cfg.get("claude_trigger", {}), "enabled": False}
    cfg["webhook"] = {"enabled": False}
    cfg["webhook_gemini"] = {"enabled": False}
    cfg["attribution_trial"] = {"enabled": False}
    watcher = tw.TranscribeWatcher(cfg, logging.getLogger("baseline-shadow"))
    evidence = sorted((state / "recordings" / stem).glob("before_attribution-*.json"), key=lambda p:p.stat().st_mtime)
    attendees = read(evidence[-1]).get("context", {}).get("known_attendees", []) if evidence else []
    watcher._resolve_calendar = lambda _: {"participant_details": attendees}
    # Forbid DB writes, downstream extraction and all outbound notifications.
    for name in ("_seed_insightbase_source", "_enrich_with_gemini", "_persist_calendar", "_trigger_claude",
                 "_notify_telegram_meeting_captured", "_notify_telegram_failure", "_notify_telegram_partial",
                 "_notify_telegram_giveup", "_notify_telegram_coherence_unverified"):
        setattr(watcher, name, lambda *a, **k: None)
    begun = time.time()
    watcher._process_with_gemini(Path(manifest["recordings_dir"]) / (stem + ".wav"))
    outcome = output / (stem + ".json")
    # A concurrent checkout/edit also invalidates this run's provenance.
    verify_baseline(manifest)
    atomic_json(output / "run.json", {"baseline_commit": actual_commit,
                "config_sha256": manifest["config_sha256"],
                "transcript_sha256": sha256(outcome) if outcome.exists() else None,
                "elapsed_seconds": time.time()-begun, "output_exists": outcome.exists(),
                "calendar_source": "frozen candidate attendee list", "published": False})
    if not outcome.exists():
        raise RuntimeError("Baseline replay produced no transcript")


def shadow_next(state: Path, limit: int = 2) -> dict:
    report = collect(state)
    outcomes = []
    for row in report["rows"]:
        if len(outcomes) >= limit:
            break
        if not row.get("eligible_for_quality_sample") or row.get("baseline"):
            continue
        job = state / "shadows" / row["stem"]
        job.mkdir(parents=True, exist_ok=True)
        attempts = read(job / "attempts.json") if (job / "attempts.json").exists() else {"attempts": 0}
        if attempts["attempts"] >= 2:
            continue
        attempts["attempts"] += 1
        atomic_json(job / "attempts.json", attempts)
        with (job / f"attempt-{attempts['attempts']}.log").open("w") as log:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "baseline-worker", "--state", str(state), "--stem", row["stem"]],
                       stdout=log, stderr=subprocess.STDOUT, timeout=7200)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
        outcomes.append({"stem": row["stem"], "exit_code": code})
    collect(state)
    return {"shadows": outcomes}


def retry_audits(state: Path, limit: int = 2) -> dict:
    """Dedicated existing-provider API runtime; advisory only, no label writes.

    Never bypass the Claude reserve guard. A Gemini text audit can identify
    contradictions while that reserve is unavailable; acoustic adjudication
    remains a separate required stage.
    """
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from speaker_coherence import check_and_repair
    load_dotenv(PRODUCTION / ".env")
    manifest = read(state / "manifest.json")
    cfg = yaml.safe_load(Path(manifest["config_backup"]).read_text())
    key = os.environ.get(cfg.get("gemini", {}).get("api_key_env", "GEMINI_API_KEY"))
    if not key:
        raise RuntimeError("Configured Gemini API key unavailable")
    model = cfg.get("gemini", {}).get("model", "gemini-2.5-pro")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180000))
    outcomes = []
    for queue_path in sorted((state / "recordings").glob("*/verification_queue.json")):
        q = read(queue_path)
        if len(outcomes) >= limit:
            break
        if "semantic_audit" not in q.get("missing_stages", []) or q.get("api_audit_attempts", 0) >= 3:
            continue
        q["api_audit_attempts"] = q.get("api_audit_attempts", 0) + 1
        atomic_json(queue_path, q)
        revision = read(Path(q["revision"]))
        def runner(prompt):
            response = client.models.generate_content(model=model, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=8192, response_mime_type="application/json"))
            return response.text
        proposal = copy.deepcopy(revision["payload"])
        log = check_and_repair(proposal, known_attendees=revision.get("context", {}).get("known_attendees"), runner=runner, max_parallel=1)
        dest = queue_path.parent / f"api-audit-{q['api_audit_attempts']}.json"
        atomic_json(dest, {"model": model, "input_revision": q["revision"], "advisory_only": True, "log": log, "proposed_transcript": proposal.get("transcript")})
        if coherence_complete(log):
            q["missing_stages"] = [s for s in q["missing_stages"] if s != "semantic_audit"]
            if (log.get("changed") or log.get("uncertain_regions")) and "acoustic_identity_review" not in q["missing_stages"]:
                q["missing_stages"].append("acoustic_identity_review")
            q["supplemental_semantic_audit"] = str(dest)
        q["state"] = "pending" if q["missing_stages"] else "checks_complete"
        atomic_json(queue_path, q)
        outcomes.append({"stem": q["transcript_stem"], "complete": coherence_complete(log)})
    client.close()
    return {"api_audits": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "collect", "evaluate", "shadow-next", "baseline-worker", "retry-audits"])
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--start", default="2026-09-05T10:00:00+08:00")
    parser.add_argument("--end", default="2026-09-12T09:00:00+08:00")
    parser.add_argument("--stem")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / (args.command + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "init":
            output = initialize(args.state, args.start, args.end)
        elif args.command == "collect":
            output = collect(args.state)
        elif args.command == "evaluate":
            report = collect(args.state)
            labels = args.state / "adjudications.jsonl"
            annotations = [json.loads(s) for s in labels.read_text().splitlines() if s.strip()] if labels.exists() else []
            output = evaluate(report["rows"], annotations, read(args.state / "manifest.json")["rules"])
            atomic_json(args.state / "verdict.latest.json", output)
        elif args.command == "shadow-next":
            output = shadow_next(args.state, max(0, min(args.limit, 8)))
        elif args.command == "retry-audits":
            output = retry_audits(args.state, max(0, min(args.limit, 8)))
        else:
            if not args.stem or Path(args.stem).name != args.stem:
                raise ValueError("A recording stem is required")
            baseline_worker(args.state, args.stem)
            output = {"baseline": "complete", "stem": args.stem}
    # Avoid printing transcripts, credentials or the historical cohort.
    if args.command == "init":
        output = {"state": output["state"], "cohort_size": len(output["baseline_cohort"]), "rules": output["rules"]}
    if args.command == "collect":
        output = {k:v for k,v in output.items() if k != "rows"}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
