"""Reflectance scale checks and vector normalization."""

from __future__ import annotations

import numpy as np

# Plausible median reflectance after conversion to 0-1. Anything outside is flagged.
PLAUSIBLE_MEDIAN = (0.0, 1.5)


def infer_scale(values: np.ndarray) -> float:
    """Guess the factor that converts stored values to 0-1 reflectance.

    Libraries and products store reflectance as fractions, percent, or integers scaled
    by 10,000. Returns 1.0, 0.01 or 1e-4.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    p99 = np.percentile(np.abs(v), 99)
    if p99 <= 2.0:
        return 1.0
    if p99 <= 200.0:
        return 0.01
    return 1e-4


def is_plausible(reflectance: np.ndarray) -> bool:
    """Scale check run on every ingested spectrum."""
    v = np.asarray(reflectance, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    med = float(np.median(v))
    return PLAUSIBLE_MEDIAN[0] <= med <= PLAUSIBLE_MEDIAN[1]


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Unit-length rows. Cosine similarity of unit vectors ranks exactly like SAM."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def sam_degrees(cosine: np.ndarray) -> np.ndarray:
    """Spectral angle in degrees from cosine similarity."""
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
