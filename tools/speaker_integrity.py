"""Attribution provenance and durable trial evidence, independent of ASR success."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def coherence_complete(log: dict | None) -> bool:
    log = log or {}
    return bool(log.get("ok") and not log.get("failed_windows")
                and not log.get("error") and not log.get("refused_runaway"))


def digital_silence(path: Path) -> bool:
    """Only exact digital zero is a hard no-speech verdict; quiet is unknown."""
    import numpy as np
    import soundfile as sf
    try:
        with sf.SoundFile(path) as source:
            if not len(source):
                return False
            for block in source.blocks(blocksize=65536, dtype="float32"):
                if np.any(block != 0):
                    return False
        return True
    except (OSError, RuntimeError):
        return False


def finalize_attribution(result, before: dict) -> dict:
    """Never equate a text check, an API response or a partial VAD with truth.

    Gap-based speaking totals cannot be reconstructed as speech time. Clear
    those and the audio-derived speaker signals after label changes; retain
    the original estimates in the revision archive for later reanalysis.
    """
    changed = result.transcript != before.get("transcript", "")
    if changed:
        for participant in result.participants or []:
            for key in ("speaking_pct", "total_seconds"):
                participant.pop(key, None)
        result.speaker_emotions = []
        result.speaker_pacing = {}
        result.interruptions = []
        result.energy_levels = {}
    separation = getattr(result, "channel_separation", None) or {}
    verification = getattr(result, "speaker_verification_log", None) or {}
    coherence = getattr(result, "speaker_coherence_log", None) or {}
    acoustic = bool(separation.get("admissible")
                    and verification.get("turns_checked", 0) > 0
                    and not verification.get("skipped_channel_bleed"))
    semantic = coherence_complete(coherence)
    unresolved = coherence.get("uncertain_regions") or []
    # Even admissible VAD only checks a subset of turns. Explicit audio or
    # human adjudication is needed before claiming whole-meeting accuracy.
    missing = []
    if not semantic:
        missing.append("semantic_audit")
    if not acoustic or unresolved or separation.get("reference_uncertain_intervals"):
        missing.append("acoustic_identity_review")
    if getattr(result, "partial", False):
        missing.append("coverage_recovery")
    report = {
        "schema_version": 1,
        "status": "needs_review" if missing else "partially_checked",
        "identity_basis": "channel_checked_subset" if acoustic else "inferred",
        "semantic_audit_complete": semantic,
        "channel_checked_turns": verification.get("turns_checked", 0),
        "missing_stages": missing,
        "uncertain_regions": unresolved,
        "speaker_dependent_actions": "hold" if missing else "require_turn_evidence",
        "labels_changed": changed,
        "speaker_statistics": "invalidated_after_relabel" if changed else "model_estimates",
        "accuracy_measured": False,
    }
    result.speaker_attribution = report
    return report


def save_trial_stage(config: dict, audio_file: Path, stage: str,
                     payload: dict, **context) -> None:
    """Immutable per-attempt evidence. Never let a disk failure lose capture.

    This hook has no model calls, DB writes or notifications. It remains
    active through the trial deadline so overdue recordings are reviewable.
    """
    cfg = config.get("attribution_trial") or {}
    if not cfg.get("enabled") or not cfg.get("state_dir"):
        return
    try:
        root = Path(cfg["state_dir"]).expanduser() / "recordings" / audio_file.stem
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        dest = root / f"{stage}-{digest[:16]}.json"
        if not dest.exists():
            atomic_json(dest, {"stage": stage, "audio_path": str(audio_file),
                              "recorded_at": datetime.now(timezone.utc).isoformat(),
                              "payload_sha256": digest, "payload": payload,
                              "context": context})
        if stage == "candidate":
            status = payload.get("_meta", {}).get("speaker_attribution") or {}
            atomic_json(root / "verification_queue.json", {
                "transcript_stem": audio_file.stem, "revision": str(dest),
                "missing_stages": status.get("missing_stages", []),
                "state": "pending" if status.get("missing_stages") else "checks_complete",
            })
    except Exception:
        logging.getLogger(__name__).exception("Could not archive attribution trial stage")
