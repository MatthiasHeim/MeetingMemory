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


# The system-track stage must have scored at least this share of the timed
# speech for the acoustic stage to count as satisfied. Scored AND consistent
# (host label measures host, remote labels measure remote): a window whose
# labels the system track could not classify adds nothing.
SYSTEM_TRACK_MIN_VERIFIED_COVERAGE = 0.5


def system_track_verdict(log: dict | None) -> dict:
    """Summarise a system_track_attribution log for the attribution report.

    `satisfied` means: the stage ran cleanly, verified a meaningful share of
    the speech and left no unresolved host/remote contradiction. It speaks
    only about host-vs-remote. Which remote person is which is never
    established by this stage (`remote_names_verified` stays False).
    """
    log = log or {}
    ran = bool(log.get("ran") and log.get("ok"))
    unresolved = list(log.get("uncertain_regions") or [])
    coverage = float(log.get("verified_coverage") or 0.0) if ran else 0.0
    return {
        "ran": ran,
        "reason": log.get("reason"),
        "verified_coverage": coverage,
        "turns_checked": int(log.get("turns_checked") or 0) if ran else 0,
        "swaps": len(log.get("swaps") or []),
        "relabels": len(log.get("relabels") or []),
        "unresolved": unresolved,
        "satisfied": bool(ran and not unresolved
                          and coverage >= SYSTEM_TRACK_MIN_VERIFIED_COVERAGE),
        "remote_names_verified": False,
    }


def trial_input_digest(config: dict, path: Path) -> str | None:
    if not (config.get("attribution_trial") or {}).get("enabled"):
        return None
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        logging.getLogger(__name__).exception("Could not fingerprint trial audio")
        return None


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
    mic_acoustic = bool(separation.get("admissible")
                        and verification.get("turns_checked", 0) > 0
                        and not verification.get("skipped_channel_bleed"))
    # The system track is digital, so it stays valid when the mic bleeds.
    system = system_track_verdict(getattr(result, "system_track_attribution", None))
    acoustic = bool(mic_acoustic or system["satisfied"])
    semantic = coherence_complete(coherence)
    unresolved = coherence.get("uncertain_regions") or []
    # Even admissible VAD only checks a subset of turns. Explicit audio or
    # human adjudication is needed before claiming whole-meeting accuracy.
    missing = []
    if not semantic:
        missing.append("semantic_audit")
    # Unresolved system-track contradictions keep the hold even if the stage
    # otherwise covered the speech.
    if (not acoustic or unresolved or system["unresolved"]
            or separation.get("reference_uncertain_intervals")):
        missing.append("acoustic_identity_review")
    if getattr(result, "partial", False):
        missing.append("coverage_recovery")
    report = {
        "schema_version": 1,
        "status": "needs_review" if missing else "partially_checked",
        "identity_basis": ("system_track_checked" if system["satisfied"]
                           else "channel_checked_subset" if mic_acoustic
                           else "inferred"),
        "semantic_audit_complete": semantic,
        "channel_checked_turns": verification.get("turns_checked", 0),
        "missing_stages": missing,
        "uncertain_regions": list(unresolved) + system["unresolved"],
        "speaker_dependent_actions": "hold" if missing else "require_turn_evidence",
        "labels_changed": changed,
        "speaker_statistics": "invalidated_after_relabel" if changed else "model_estimates",
        "accuracy_measured": False,
    }
    if system["ran"]:
        # Visible limits: host-vs-remote only, names still unverified.
        report["system_track"] = {
            "verified_coverage": system["verified_coverage"],
            "turns_checked": system["turns_checked"],
            "swaps": system["swaps"],
            "relabels": system["relabels"],
            "unresolved": len(system["unresolved"]),
            "remote_names_verified": False,
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
        audio_digest = context.get("audio_sha256")
        verified = bool(audio_digest and (stage != "candidate" or
                        trial_input_digest(config, audio_file) == audio_digest))
        dest = root / f"{stage}-{digest[:16]}-{(audio_digest or 'unknown')[:16]}.json"
        if not dest.exists():
            atomic_json(dest, {"stage": stage, "audio_path": str(audio_file),
                              "recorded_at": datetime.now(timezone.utc).isoformat(),
                              "payload_sha256": digest, "payload": payload,
                              "audio_sha256": audio_digest, "audio_input_verified": verified,
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
