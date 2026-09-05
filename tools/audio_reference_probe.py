#!/usr/bin/env python3
"""Offline delay/echo pilot. Outputs derived audio only; never routes production.

Fixed-delay alignment and a static room filter are diagnostic baselines, not
a validated adaptive AEC. ERLE on remote-only audio says nothing about host
recall during double talk. Absolute capture-clock timestamps are still needed.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from speaker_integrity import atomic_json


def estimate_delay(mic, remote, sr):
    mic = mic - mic.mean(); remote = remote - remote.mean()
    energy = np.sqrt((mic@mic)*(remote@remote))
    if energy < 1e-10:
        return {"state": "missing_reference", "lag_seconds": None, "correlation": None}
    corr = signal.correlate(mic, remote, method="fft")
    lags = signal.correlation_lags(len(mic), len(remote))
    mask = abs(lags) < 2*sr
    i = int(np.argmax(abs(corr[mask])))
    rho = float(corr[mask][i]/energy)
    return {"state": "measurable" if abs(rho) >= 0.25 else "uncertain",
            "lag_seconds": float(lags[mask][i]/sr), "correlation": rho}


def align_reference(remote, lag):
    """Reference shifted onto the mic timeline; retain length, zero-pad edges."""
    out = np.zeros_like(remote)
    if lag >= len(remote) or lag <= -len(remote):
        return out
    if lag > 0:
        out[lag:] = remote[:-lag]
    elif lag < 0:
        out[:lag] = remote[-lag:]
    else:
        out[:] = remote
    return out


def static_echo_pilot(mic, reference, sr):
    """Fit first half, evaluate second half; caller must establish remote-only.

    A modest spectral floor limits ill-conditioned filter gains. This deliberately
    does not adapt on held-out speech or assume an activity label identifies it.
    """
    middle = len(mic)//2
    _, rr = signal.welch(reference[:middle], fs=sr, nperseg=2048)
    _, rm = signal.csd(reference[:middle], mic[:middle], fs=sr, nperseg=2048)
    transfer = rm / np.maximum(rr, max(rr.max()*0.001, 1e-12))
    impulse = np.fft.irfft(transfer, n=2048)[:1024]
    prediction = signal.fftconvolve(reference, impulse, mode="full")[:len(mic)]
    residual = mic - prediction
    energy_before = float(np.mean(mic[middle:]**2))
    energy_after = float(np.mean(residual[middle:]**2))
    return residual, {"training_seconds": middle/sr,
                      "heldout_energy_reduction_db": 10*np.log10(max(energy_before,1e-12)/max(energy_after,1e-12)),
                      "not_a_speaker_accuracy_metric": True}


def probe(audio: Path, output: Path, start: float, seconds: float, remote_only: bool):
    output.mkdir(parents=True, exist_ok=True)
    sr = 8000
    proc = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-ss", str(start), "-i", str(audio),
        "-t", str(seconds), "-af", "pan=stereo|c0=c0|c1=0.5*c1+0.5*c2,highpass=f=300,lowpass=f=3400",
        "-ar", str(sr), "-f", "f32le", "pipe:1"], check=True, capture_output=True)
    data = np.frombuffer(proc.stdout, dtype="<f4").reshape(-1,2).astype(np.float64)
    if not len(data):
        raise ValueError("No audio in requested interval")
    report = estimate_delay(data[:,0], data[:,1], sr)
    report.update(source=str(audio), original_start_seconds=start, duration_seconds=len(data)/sr,
                  limitation="Correlation and energy reduction do not measure speaker accuracy or overlap recall.")
    if report["state"] == "measurable":
        reference = align_reference(data[:,1], round(report["lag_seconds"]*sr))
        sf.write(output/"aligned-reference.wav", reference, sr, subtype="FLOAT")
        sf.write(output/"original-mic.wav", data[:,0], sr, subtype="FLOAT")
        if remote_only:
            residual, info = static_echo_pilot(data[:,0], reference, sr)
            sf.write(output/"static-echo-residual.wav", residual, sr, subtype="FLOAT")
            report["static_echo_pilot"] = info
    atomic_json(output/"probe.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--confirmed-remote-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(probe(args.audio, args.output, args.start, args.seconds, args.confirmed_remote_only), indent=2))
