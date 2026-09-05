import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_reference_probe import estimate_delay, align_reference


def test_delay_estimate_and_padding_keep_original_timeline():
    sr = 8000
    remote = np.random.default_rng(42).normal(size=sr*3)
    mic = align_reference(remote, 1170)
    report = estimate_delay(mic, remote, sr)
    assert report["state"] == "measurable"
    assert report["lag_seconds"] == 1170/sr
    assert mic.shape == remote.shape and np.all(mic[:1170] == 0)
    assert np.array_equal(mic[1170:], remote[:-1170])


def test_missing_reference_never_becomes_cancellable_echo():
    report = estimate_delay(np.ones(8000), np.zeros(8000), 8000)
    assert report["state"] == "missing_reference"
    assert report["lag_seconds"] is None
