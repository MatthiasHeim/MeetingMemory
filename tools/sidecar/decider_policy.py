"""Local Decider thresholds, checked on the recorder's 2026-10-10 reference sample.

These are model-specific operating points, not calibrated guarantees. The
reference labels are machine-generated. See docs/local-decider.md.
"""
DECIDER_MODEL = 'pplx-decider-v1.1-27b'
DECIDER_PROMPT_THRESHOLD = 0.92
DECIDER_SEED_THRESHOLD = 0.10
DECIDER_GROW_THRESHOLD = 0.025


def clip_thresholds(backend: str, *, widen: bool = False) -> dict[str, float]:
    if backend != 'decider':
        return {}
    return {
        'seed_threshold': 0.05 if widen else DECIDER_SEED_THRESHOLD,
        'grow_threshold': 0.01 if widen else DECIDER_GROW_THRESHOLD,
    }
