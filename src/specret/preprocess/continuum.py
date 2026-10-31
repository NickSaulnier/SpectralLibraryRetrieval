"""Convex-hull continuum removal.

The continuum is the upper convex hull of the spectrum. Dividing by it leaves absorption
features as dips below 1.0 and removes overall brightness and slope, which dominate the
spectral angle on raw reflectance. Bands that are NaN (masked water bands, missing data) are
ignored when building the hull, and the hull is interpolated across them.
"""

from __future__ import annotations

import numpy as np


def upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the upper convex hull of points (x ascending), left to right."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Drop b if it lies on or below the segment a -> i.
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull)


def continuum(wavelength: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Hull continuum evaluated at every band (NaN where the spectrum is NaN)."""
    wl = np.asarray(wavelength, dtype=np.float64)
    y = np.asarray(spectrum, dtype=np.float64)
    ok = np.isfinite(y)
    out = np.full_like(y, np.nan)
    if ok.sum() < 2:
        return out
    xs, ys = wl[ok], y[ok]
    h = upper_hull(xs, ys)
    out[ok] = np.interp(xs, xs[h], ys[h])
    return out


def continuum_removed(wavelength: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """Spectrum / continuum for one spectrum (n,) or rows of a matrix (m, n)."""
    S = np.atleast_2d(np.asarray(spectra, dtype=np.float64))
    out = np.full_like(S, np.nan)
    for i, row in enumerate(S):
        c = continuum(wavelength, row)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.where(c > 0, row / c, np.nan)
    return out[0] if np.ndim(spectra) == 1 else out


def band_depth(wavelength: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """1 - continuum-removed reflectance: 0 on the continuum, positive inside absorptions."""
    return 1.0 - continuum_removed(wavelength, spectra)
