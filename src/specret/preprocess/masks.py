"""Wavelength-range masks."""

from __future__ import annotations

import numpy as np


def range_mask(wavelengths_nm: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
    """True where a wavelength falls inside any of the inclusive [lo, hi] ranges."""
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    hit = np.zeros(wl.shape, dtype=bool)
    for lo, hi in ranges:
        hit |= (wl >= lo) & (wl <= hi)
    return hit


def window_mask(wavelengths_nm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True inside one inclusive window, e.g. the 2000-2500 nm `swir2` window."""
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    return (wl >= lo) & (wl <= hi)
