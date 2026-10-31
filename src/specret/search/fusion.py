"""Reciprocal rank fusion of several ranked candidate lists."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

RRF_K = 60


@njit(cache=True, parallel=True)
def _rrf(rows: np.ndarray, weights: np.ndarray, n_out: int):
    nq, m = rows.shape
    out_rows = np.full((nq, n_out), -1, dtype=np.int64)
    out_scores = np.zeros((nq, n_out))
    for q in prange(nq):
        order = np.argsort(rows[q])
        uniq = np.empty(m, dtype=np.int64)
        total = np.zeros(m)
        u = -1
        prev = -2
        for j in order:
            r = rows[q, j]
            if r < 0:
                continue
            if r != prev:
                u += 1
                uniq[u] = r
                prev = r
            total[u] += weights[j]
        n_unique = u + 1
        if n_unique == 0:
            continue
        best = np.argsort(-total[:n_unique], kind="mergesort")[:n_out]
        for i in range(best.size):
            out_rows[q, i] = uniq[best[i]]
            out_scores[q, i] = total[best[i]]
    return out_rows, out_scores


def rrf(rank_lists: list[np.ndarray], n_out: int, k: int = RRF_K) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-query rankings of library rows.

    `rank_lists` holds (nq, K) arrays of library rows, best first, -1 for none. Each
    appearance at rank r (0-based) adds 1 / (k + r + 1). Returns (rows, scores), each
    (nq, n_out), best first, padded with -1 / 0. Ties keep the order of first appearance
    by row number.
    """
    rows = np.ascontiguousarray(np.concatenate(rank_lists, axis=1), dtype=np.int64)
    weights = np.concatenate([1.0 / (k + np.arange(r.shape[1]) + 1) for r in rank_lists])
    return _rrf(rows, weights, n_out)
