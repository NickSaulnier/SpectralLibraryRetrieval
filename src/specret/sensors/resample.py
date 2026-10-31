"""Convolve high-resolution spectra to a sensor's bands.

Each sensor band is modelled as a Gaussian spectral response with the published centre and
FWHM. Compared with the simple version in the project brief, this one:

* integrates with trapezoidal weights, because library wavelength grids are not uniform;
* ignores NaN samples and reports a band as invalid (NaN) when less than `min_coverage` of
  its response falls on valid library samples, instead of silently renormalizing;
* optionally subtracts the library's own resolution in quadrature
  (sqrt(fwhm_sensor**2 - fwhm_lib**2)), so an already-blurred spectrum is not blurred twice.
  Where the library is as coarse as the sensor, the band value is linearly interpolated.
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def trapezoid_weights(wavelengths: np.ndarray) -> np.ndarray:
    """Quadrature weight of each sample on a (possibly irregular) ascending grid."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    q = np.zeros_like(wl)
    d = np.diff(wl)
    q[:-1] += d / 2.0
    q[1:] += d / 2.0
    return q


def effective_fwhm(
    centers: np.ndarray,
    sensor_fwhm: np.ndarray,
    lib_wl: np.ndarray | None = None,
    lib_fwhm: np.ndarray | None = None,
) -> np.ndarray:
    """Kernel width still to apply after accounting for the library's own resolution."""
    sensor_fwhm = np.asarray(sensor_fwhm, dtype=np.float64)
    if lib_fwhm is None or lib_wl is None:
        return sensor_fwhm
    lf = np.interp(centers, lib_wl, lib_fwhm)
    return np.sqrt(np.maximum(sensor_fwhm**2 - lf**2, 0.0))


def response_matrix(lib_wl: np.ndarray, centers: np.ndarray, fwhm: np.ndarray) -> np.ndarray:
    """Integration weights W (n_bands x n_samples) so that band = W @ spectrum.

    Rows sum to ~1 when the kernel lies inside the grid. Bands whose kernel is narrower
    than the local sample spacing fall back to linear interpolation weights.
    """
    lib_wl = np.asarray(lib_wl, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    sigma = np.asarray(fwhm, dtype=np.float64) * FWHM_TO_SIGMA
    n = lib_wl.size
    W = np.zeros((centers.size, n))

    spacing = np.interp(centers, lib_wl[1:], np.diff(lib_wl))
    resolved = sigma >= spacing
    if resolved.any():
        q = trapezoid_weights(lib_wl)
        s = sigma[resolved, None]
        z = (lib_wl[None, :] - centers[resolved, None]) / s
        W[resolved] = np.exp(-0.5 * z**2) / (s * np.sqrt(2.0 * np.pi)) * q[None, :]

    for i in np.flatnonzero(~resolved):
        j = int(np.searchsorted(lib_wl, centers[i]))
        if j == 0 or j >= n:
            continue  # outside the grid: no weight, band will be invalid
        t = (centers[i] - lib_wl[j - 1]) / (lib_wl[j] - lib_wl[j - 1])
        W[i, j - 1], W[i, j] = 1.0 - t, t
    return W


def resample(
    lib_wl: np.ndarray,
    lib_refl: np.ndarray,
    centers: np.ndarray,
    fwhm: np.ndarray,
    lib_fwhm: np.ndarray | None = None,
    min_coverage: float = 0.9,
) -> np.ndarray:
    """Resample one spectrum (n,) or a stack sharing one grid (m, n) to sensor bands.

    Returns (n_bands,) or (m, n_bands), with NaN for bands that fail the coverage test.
    """
    lib_wl = np.asarray(lib_wl, dtype=np.float64)
    R = np.asarray(lib_refl, dtype=np.float64)
    single = R.ndim == 1
    R = np.atleast_2d(R)

    order = np.argsort(lib_wl)
    lib_wl, R = lib_wl[order], R[:, order]
    if lib_fwhm is not None:
        lib_fwhm = np.asarray(lib_fwhm, dtype=np.float64)[order]

    width = effective_fwhm(centers, fwhm, lib_wl, lib_fwhm)
    W = response_matrix(lib_wl, centers, width)

    valid = np.isfinite(R)
    mass = valid.astype(np.float64) @ W.T
    num = np.where(valid, R, 0.0) @ W.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(mass >= min_coverage, num / mass, np.nan)
    return out[0] if single else out
