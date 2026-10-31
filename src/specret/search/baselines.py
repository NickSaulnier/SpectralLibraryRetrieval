"""Classical non-cosine baselines, by brute force (PLAN.md, Phase 2c).

* `sid` - Spectral Information Divergence (Chang, 2000) on raw reflectance over good bands:
  each spectrum is treated as a probability distribution and compared by symmetric
  Kullback-Leibler divergence. Lower is better.
* `cr_euclid` - Euclidean distance between continuum-removed spectra over good bands.
  Unlike `cr`, depth is not normalised away, so abundance differences count.

These are not vector-database friendly (SID is not a metric on unit vectors), so they run as
exact brute force. With a few thousand library spectra that is fast enough to benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from specret.preprocess.continuum import continuum_removed
from specret.search.retrieve import RESULT_FIELDS, filter_library
from specret.sensors.library import SensorLibrary

EPS = 1e-6


def _sid_prepare(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(x, EPS, None)
    p = x / x.sum(axis=1, keepdims=True)
    return p, np.log(p)


def sid_matrix(queries: np.ndarray, library: np.ndarray) -> np.ndarray:
    """SID between each query row and each library row: (nq, nlib)."""
    p, logp = _sid_prepare(queries)
    q, logq = _sid_prepare(library)
    # sum p log p/q + q log q/p = sum (p - q)(log p - log q)
    a = (p * logp).sum(1)[:, None] + (q * logq).sum(1)[None, :]
    b = p @ logq.T + logp @ q.T
    return a - b


def euclid_matrix(queries: np.ndarray, library: np.ndarray) -> np.ndarray:
    d2 = (queries**2).sum(1)[:, None] + (library**2).sum(1)[None, :] - 2 * queries @ library.T
    return np.sqrt(np.maximum(d2, 0.0))


class BruteForceRetriever:
    """Same `query` interface as Retriever, for distance-based baselines."""

    def __init__(
        self,
        lib: SensorLibrary,
        canonical: pd.DataFrame,
        metric: str = "sid",
        chapters: tuple[str, ...] | None = None,
        purity: tuple[str, ...] | None = None,
    ):
        if metric not in ("sid", "cr_euclid"):
            raise ValueError(f"unknown metric {metric!r}")
        self.meta = canonical.set_index("spectrum_id")
        lib = filter_library(lib, self.meta, chapters, purity)
        self.lib, self.metric = lib, metric
        self.mask = lib.sensor.good_mask
        x = self._prepare(lib.matrix)
        self.ok = np.isfinite(x).all(axis=1)
        self.vectors, self.ids = x[self.ok], lib.spectrum_ids[self.ok]

    def _prepare(self, spectra: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(spectra, dtype=np.float64))[:, self.mask]
        if self.metric == "cr_euclid":
            x = continuum_removed(self.lib.sensor.centers_nm[self.mask], x)
        return x

    def check_bands(self, centers_nm: np.ndarray) -> None:
        c = self.lib.sensor.centers_nm
        if centers_nm.shape != c.shape or not np.allclose(centers_nm, c, atol=0.5):
            raise ValueError(f"query bands do not match library sensor '{self.lib.sensor.name}'")

    def query(self, spectra: np.ndarray, k: int = 5, batch: int = 2048) -> list[list[dict]]:
        x = self._prepare(spectra)
        ok = np.isfinite(x).all(axis=1)
        out: list[list[dict]] = [[] for _ in range(len(x))]
        dist_fn = sid_matrix if self.metric == "sid" else euclid_matrix
        idx_ok = np.flatnonzero(ok)
        for start in range(0, idx_ok.size, batch):
            qi = idx_ok[start : start + batch]
            d = dist_fn(x[qi], self.vectors)
            top = np.argsort(d, axis=1)[:, :k]
            for row, q in enumerate(qi):
                matches = []
                for j in top[row]:
                    sid = str(self.ids[j])
                    m = self.meta.loc[sid]
                    matches.append(
                        {
                            "spectrum_id": sid,
                            "distance": float(d[row, j]),
                            "sam_deg": float("nan"),
                            **{f: m.get(f) for f in RESULT_FIELDS},
                        }
                    )
                out[q] = matches
        return out
