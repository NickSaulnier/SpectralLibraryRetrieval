"""Splice correction for ASD-type field spectrometers.

ASD instruments use three detectors (VNIR to 1000 nm, SWIR1 1001-1800 nm, SWIR2 from
1801 nm). Temperature drift between them leaves small steps at the joins. The correction
keeps SWIR1 fixed and scales the other two detectors so each join is continuous, using a
linear extrapolation of the SWIR1 side across the join.
"""

from __future__ import annotations

import numpy as np

DEFAULT_JOINS_NM = (1000.0, 1800.0)


def _extrapolate(wl: np.ndarray, y: np.ndarray, at: float) -> float:
    coef = np.polyfit(wl, y, 1)
    return float(np.polyval(coef, at))


def splice_correct(
    wavelength_nm: np.ndarray,
    reflectance: np.ndarray,
    joins_nm: tuple[float, float] = DEFAULT_JOINS_NM,
    fit_bands: int = 5,
) -> np.ndarray:
    """Return a copy of `reflectance` with detector steps removed (multiplicative)."""
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    out = np.asarray(reflectance, dtype=np.float64).copy()
    j1, j2 = joins_nm
    vnir = wl <= j1
    swir1 = (wl > j1) & (wl <= j2)
    swir2 = wl > j2
    if not (vnir.any() and swir1.sum() >= fit_bands and swir2.any()):
        return out

    s1_idx = np.flatnonzero(swir1)
    head, tail = s1_idx[:fit_bands], s1_idx[-fit_bands:]
    v_last = np.flatnonzero(vnir)[-1]
    s2_first = np.flatnonzero(swir2)[0]

    target_v = _extrapolate(wl[head], out[head], wl[v_last])
    if out[v_last] != 0:
        out[vnir] *= target_v / out[v_last]

    target_s2 = _extrapolate(wl[tail], out[tail], wl[s2_first])
    if out[s2_first] != 0:
        out[swir2] *= target_s2 / out[s2_first]
    return out
