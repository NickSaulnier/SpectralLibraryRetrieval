"""Convex-hull continuum removal.

The continuum is the upper convex hull of the spectrum. Dividing by it leaves absorption
features as dips below 1.0 and removes overall brightness and slope, which dominate the
spectral angle on raw reflectance. Bands that are NaN (masked water bands, missing data) are
ignored when building the hull, and the hull is interpolated across them.

The row loop is compiled with numba, so a full EMIT scene (~1.5M spectra) takes seconds.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the upper convex hull of points (x ascending), left to right."""
    n = x.size
    hull = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        while m >= 2:
            a, b = hull[m - 2], hull[m - 1]
            # Drop b if it lies on or below the segment a -> i.
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross >= 0:
                m -= 1
            else:
                break
        hull[m] = i
        m += 1
    return hull[:m]


def upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _upper_hull(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


@njit(cache=True)
def _continuum_row(wl: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    ok = np.flatnonzero(np.isfinite(y))
    out[:] = np.nan
    if ok.size < 2:
        return
    xs, ys = wl[ok], y[ok]
    h = _upper_hull(xs, ys)
    out[ok] = np.interp(xs, xs[h], ys[h])


@njit(cache=True, parallel=True)
def _continuum_rows(wl: np.ndarray, S: np.ndarray) -> np.ndarray:
    out = np.empty_like(S)
    for i in prange(S.shape[0]):
        _continuum_row(wl, S[i], out[i])
    return out


def continuum(wavelength: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Hull continuum evaluated at every band (NaN where the spectrum is NaN)."""
    wl = np.ascontiguousarray(wavelength, dtype=np.float64)
    S = np.atleast_2d(np.ascontiguousarray(spectrum, dtype=np.float64))
    c = _continuum_rows(wl, S)
    return c[0] if np.ndim(spectrum) == 1 else c


def continuum_removed(wavelength: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """Spectrum / continuum for one spectrum (n,) or rows of a matrix (m, n)."""
    S = np.atleast_2d(np.ascontiguousarray(spectra, dtype=np.float64))
    c = _continuum_rows(np.ascontiguousarray(wavelength, dtype=np.float64), S)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(c > 0, S / c, np.nan)
    return out[0] if np.ndim(spectra) == 1 else out


def band_depth(wavelength: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """1 - continuum-removed reflectance: 0 on the continuum, positive inside absorptions."""
    return 1.0 - continuum_removed(wavelength, spectra)
