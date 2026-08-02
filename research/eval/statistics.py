"""Small statistical helpers shared by the evaluation scripts."""

from __future__ import annotations

import numpy as np

__all__ = ["spearman"]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Zero-based ranks, assigning tied values their average rank."""
    _, inverse, counts = np.unique(
        np.asarray(values, dtype=np.float64),
        return_inverse=True,
        return_counts=True,
    )
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    tied_ranks = starts + (counts - 1) / 2.0
    return tied_ranks[inverse]


def spearman(x: np.ndarray, y: np.ndarray,
             min_samples: int = 2) -> float | None:
    """Spearman correlation with finite filtering and tie-aware ranks."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have the same shape")
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < min_samples:
        return None
    rx = _average_ranks(x[ok])
    ry = _average_ranks(y[ok])
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
