"""Candidate plumes from an enhancement map (swath geometry).

Single pixels are too noisy to threshold, so the map is first smoothed with a small Gaussian
and turned into a z-score with each column's own robust spread (the matched filter's noise
differs between detector columns). Candidates are then found by hysteresis: connected regions
above `z_low` that contain at least one pixel above `z_high`, with at least `min_px` pixels.
After 1-px smoothing, Gaussian noise alone gives about 10 seeds above z = 4 per 60,000 pixels
(the Euler characteristic of a smooth random field), but about 0.2 above z = 5, hence the default.

Phase G2 uses the candidates for identification; G3 refines them into plume records
(source location, IME, emission rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

ROBUST = 1.4826


@dataclass
class Candidate:
    id: int
    rows: np.ndarray  # pixel indices in swath geometry
    cols: np.ndarray
    peak_z: float  # highest smoothed z-score
    mean_alpha: float  # mean enhancement (ppm·m) over the candidate
    max_alpha: float

    @property
    def n_px(self) -> int:
        return int(self.rows.size)

    @property
    def centroid(self) -> tuple[float, float]:
        return float(self.rows.mean()), float(self.cols.mean())

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        m[self.rows, self.cols] = True
        return m


def smooth_nan(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN pixels (normalized convolution)."""
    ok = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(ok, a, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(np.float64), sigma)
    out = num / np.where(den > 1e-6, den, np.nan)
    out[~ok] = np.nan
    return out


def column_z(a: np.ndarray) -> np.ndarray:
    """z-score per detector column, using the column's median and MAD."""
    med = np.nanmedian(a, axis=0)
    sig = ROBUST * np.nanmedian(np.abs(a - med), axis=0)
    return (a - med) / np.where(sig > 0, sig, np.nan)


def find_candidates(
    alpha: np.ndarray,
    z_high: float = 5.0,
    z_low: float = 2.5,
    min_px: int = 5,
    smooth_px: float = 1.0,
) -> tuple[list[Candidate], np.ndarray]:
    """Candidates (largest peak first) and the smoothed z map they were found in."""
    z = column_z(smooth_nan(alpha.astype(np.float64), smooth_px))
    low = np.nan_to_num(z, nan=-np.inf) > z_low
    lab, n = ndimage.label(low, structure=np.ones((3, 3)))
    if n == 0:
        return [], z
    zz = np.nan_to_num(z, nan=-np.inf)
    peak = ndimage.maximum(zz, lab, index=np.arange(1, n + 1))
    size = ndimage.sum(low, lab, index=np.arange(1, n + 1))
    keep = np.flatnonzero((peak > z_high) & (size >= min_px)) + 1
    out = []
    objects = ndimage.find_objects(lab)
    for lid in keep:
        sl = objects[lid - 1]
        rr, cc = np.nonzero(lab[sl] == lid)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        a = alpha[rr, cc]
        out.append(
            Candidate(
                id=0,
                rows=rr,
                cols=cc,
                peak_z=float(zz[rr, cc].max()),
                mean_alpha=float(np.nanmean(a)),
                max_alpha=float(np.nanmax(a)),
            )
        )
    out.sort(key=lambda c: -c.peak_z)
    for i, c in enumerate(out):
        c.id = i + 1
    return out, z
