"""Synthetic mixture benchmark for retrieve-and-unmix (PLAN.md, Phase 4).

Each query mixes 1-3 frozen-split test spectra from different groups with Dirichlet fractions
(each at least `min_fraction`), applies a brightness scale and EMIT-like noise. The reference
library holds train and val samples only, so no true endmember is in the library itself:
the unmixer has to find spectra of the same groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from specret.sensors.library import SensorLibrary
from specret.unmix.pursuit import Unmixer

MIXTURE_SEED = 4_2026


def make_mixtures(
    test: SensorLibrary,
    groups: np.ndarray,
    n: int,
    noise_sigma: np.ndarray,
    min_fraction: float = 0.15,
    seed: int = MIXTURE_SEED,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    full = np.flatnonzero(test.full_coverage)
    queries, truth = [], []
    while len(queries) < n:
        k = rng.integers(1, 4)
        picks: list[int] = []
        for i in rng.permutation(full):
            if groups[i] not in {groups[p] for p in picks}:
                picks.append(int(i))
            if len(picks) == k:
                break
        f = rng.dirichlet(np.full(k, 2.0))
        if f.min() < min_fraction:
            continue
        x = (f[:, None] * test.matrix[picks]).sum(axis=0)
        x = x * rng.uniform(0.7, 1.3) + rng.normal(0.0, 1.0, x.shape) * noise_sigma
        queries.append(x)
        t: dict[str, float] = {}
        for p, fi in zip(picks, f, strict=True):
            t[groups[p]] = t.get(groups[p], 0.0) + float(fi)
        truth.append(t)
    return np.vstack(queries), truth


def _score(pred: dict[str, float], true: dict[str, float], min_fraction: float) -> dict:
    p = {g for g, f in pred.items() if f >= min_fraction}
    t = set(true)
    union = set(pred) | t
    err = [pred.get(g, 0.0) - true.get(g, 0.0) for g in union]
    return {
        "recall": len(p & t) / len(t),
        "precision": len(p & t) / len(p) if p else 0.0,
        "n_true": len(t),
        "n_pred": len(p),
        "abundance_rmse": float(np.sqrt(np.mean(np.square(err)))),
    }


def evaluate(
    unmixer: Unmixer,
    queries: np.ndarray,
    truth: list[dict[str, float]],
    noise_sigma: np.ndarray | None,
    min_fraction: float = 0.1,
) -> pd.DataFrame:
    """Per-query scores for the single best match ("top1") and for pursuit unmixing."""
    ranked = unmixer.r.rank(queries)
    rows = []
    for i, (x, t) in enumerate(zip(queries, truth, strict=True)):
        top = ranked.rows[i, 0]
        single = {unmixer.groups[top]: 1.0} if top >= 0 else {}
        rows.append({"method": "top1", **_score(single, t, min_fraction)})
        u = unmixer.unmix(x, noise_sigma, seed_rows=ranked.rows[i])
        pred: dict[str, float] = {}
        if u is not None:
            for r, f in zip(u.rows, u.fit.fractions, strict=True):
                pred[unmixer.groups[r]] = pred.get(unmixer.groups[r], 0.0) + float(f)
        rows.append({"method": "pursuit", **_score(pred, t, min_fraction)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["method", "n_true"]).agg(
        queries=("recall", "size"),
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        n_pred=("n_pred", "mean"),
        abundance_rmse=("abundance_rmse", "mean"),
    )
