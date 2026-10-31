"""Constrained least-squares unmixing for a handful of endmembers.

`nnls_fit` solves min ||A f - x|| with f >= 0. With `sum_to_one`, a heavily weighted row of
ones is appended (the standard FCLS trick), pushing sum(f) towards 1. Library spectra and
pixels rarely share a brightness scale (grain size, illumination), so the default leaves the
sum free and reports relative fractions plus the overall scale.

`baseline` adds an unconstrained offset and linear slope (as +/- column pairs, since NNLS
needs non-negative weights). It absorbs continuum differences between library samples and
the pixel, so extra endmembers are only chosen to explain absorption features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

SUM_TO_ONE_WEIGHT = 1e3


@dataclass
class Fit:
    coefficients: np.ndarray  # raw non-negative weights, one per endmember
    residual: np.ndarray  # x - A f, on the fitted bands
    rss: float

    @property
    def scale(self) -> float:
        return float(self.coefficients.sum())

    @property
    def fractions(self) -> np.ndarray:
        """Relative abundances (coefficients normalised to sum to 1)."""
        s = self.coefficients.sum()
        return self.coefficients / s if s > 0 else self.coefficients


def baseline_columns(wavelengths_nm: np.ndarray) -> np.ndarray:
    """(n_bands, 4): +offset, -offset, +slope, -slope."""
    t = (np.asarray(wavelengths_nm, dtype=np.float64) - 1500.0) / 1000.0
    one = np.ones_like(t)
    return np.column_stack([one, -one, t, -t])


def nnls_fit(
    endmembers: np.ndarray,
    x: np.ndarray,
    sum_to_one: bool = False,
    baseline: np.ndarray | None = None,
) -> Fit:
    """Fit `x` (n_bands,) with columns of `endmembers` (n_bands, p) plus optional baseline."""
    A = np.asarray(endmembers, dtype=np.float64)
    b = np.asarray(x, dtype=np.float64)
    p = A.shape[1]
    A_all = A if baseline is None else np.hstack([A, baseline])
    if sum_to_one:
        row = np.zeros((1, A_all.shape[1]))
        row[0, :p] = SUM_TO_ONE_WEIGHT
        A_fit = np.vstack([A_all, row])
        b_fit = np.append(b, SUM_TO_ONE_WEIGHT)
    else:
        A_fit, b_fit = A_all, b
    coef, _ = nnls(A_fit, b_fit, maxiter=50 * A_all.shape[1])
    resid = b - A_all @ coef
    return Fit(coef[:p], resid, float(resid @ resid))


def bic(rss: float, n: int, p: int) -> float:
    """Bayesian information criterion for a Gaussian residual with p free weights."""
    return n * np.log(max(rss, 1e-300) / n) + p * np.log(n)
