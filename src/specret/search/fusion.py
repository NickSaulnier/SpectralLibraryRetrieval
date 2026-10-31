"""Reciprocal rank fusion of several ranked candidate lists."""

from __future__ import annotations

import numpy as np

RRF_K = 60


def rrf(rank_lists: list[np.ndarray], n_out: int, k: int = RRF_K) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-query rankings of library rows.

    `rank_lists` holds (nq, K) arrays of library rows, best first, -1 for none. Each
    appearance at rank r (0-based) adds 1 / (k + r + 1). Returns (rows, scores), each
    (nq, n_out), best first, padded with -1 / 0.
    """
    nq = rank_lists[0].shape[0]
    out_rows = np.full((nq, n_out), -1, dtype=np.int64)
    out_scores = np.zeros((nq, n_out))
    weights = [1.0 / (k + np.arange(r.shape[1]) + 1) for r in rank_lists]
    for q in range(nq):
        rows = np.concatenate([r[q] for r in rank_lists])
        w = np.concatenate(weights)
        keep = rows >= 0
        if not keep.any():
            continue
        uniq, inv = np.unique(rows[keep], return_inverse=True)
        total = np.bincount(inv, weights=w[keep])
        order = np.argsort(-total, kind="stable")[:n_out]
        out_rows[q, : order.size] = uniq[order]
        out_scores[q, : order.size] = total[order]
    return out_rows, out_scores
