"""Deterministic dataset splitting.

Provides a single, reproducible train/validation/test partition of volume
indices. Centralizing this is what lets the rest of the code avoid evaluation
leakage: the decision threshold is calibrated on the validation split and the
model is reported on a held-out test split that was never used for fitting or
calibration.
"""
from __future__ import annotations

import numpy as np

import config


def train_val_test_indices(
    n: int,
    seed: int = config.RANDOM_SEED,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Split ``range(n)`` into (train, val, test) index lists.

    The split is deterministic for a given ``(n, seed)`` and the three lists are
    a disjoint, exhaustive partition of ``range(n)``. Each list is returned
    sorted for stable, human-readable output.
    """
    if n <= 0:
        return [], [], []
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0):
        raise ValueError("val_frac and test_frac must each be in [0, 1).")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1 (need a non-empty train set).")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    # Guard tiny datasets so train is never empty.
    n_test = min(n_test, max(0, n - 1))
    n_val = min(n_val, max(0, n - 1 - n_test))

    test = sorted(int(i) for i in shuffled[:n_test])
    val = sorted(int(i) for i in shuffled[n_test:n_test + n_val])
    train = sorted(int(i) for i in shuffled[n_test + n_val:])
    return train, val, test
