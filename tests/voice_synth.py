"""Synthetic stand-ins for voice work: unit vectors, no audio, no real people.

Everything here is made up from a seeded RNG. Names are generic ("Person A").
"""

from __future__ import annotations

import numpy as np

DIM = 64


def base_vectors(names, seed=7):
    """One well-separated unit vector per name (orthogonalised)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(DIM, max(len(names), 2))))
    return {n: q[:, i].astype(np.float32) for i, n in enumerate(names)}


def noisy(base, n, noise=0.15, seed=0):
    """n unit vectors scattered around `base` (cosine to base ~0.99 at 0.15)."""
    rng = np.random.default_rng(seed)
    out = base[None, :] + noise * rng.normal(size=(n, base.size)) / np.sqrt(base.size) * 3
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def mix(a, b, w):
    """Unit vector w*a + (1-w)*b."""
    return unit(w * np.asarray(a) + (1 - w) * np.asarray(b))
