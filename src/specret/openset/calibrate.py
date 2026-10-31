"""Calibrated confidence and unknown-material detection (PLAN.md, Phase 4).

A logistic model predicts P(the top-1 group is right) from cheap retrieval signals:

* `score`    - combined ranking score of the best match
* `margin`   - its lead over the best candidate from another group
* `sff`      - raw feature-fit score of the best match
* `sam_deg`  - spectral angle to the best match on raw reflectance
* `cr_cos`   - cosine to the best match in continuum-removed 2.0-2.5 um space
* `depth`    - the query's strongest absorption depth in any feature window
* `fit_rms`  - relative RMS residual of query ~ a * best match + offset + slope (good bands)
* `consensus`- share of the top five candidates in the best match's group

Queries of materials absent from the library are labelled "not right", so a low probability
means "probably unknown or misidentified". Below the threshold the tool answers "unknown".
Calibration uses held-out groups (the frozen open-set split), never the test split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from specret.preprocess.normalize import sam_degrees
from specret.rerank.sff import compute_features
from specret.search.hybrid import HybridRetriever, RankResult
from specret.search.vectors import represent, vector_masks

FEATURES = ("score", "margin", "sff", "sam_deg", "cr_cos", "depth", "fit_rms", "consensus")


def confidence_features(
    retriever: HybridRetriever, spectra: np.ndarray, ranked: RankResult | None = None
) -> tuple[RankResult, pd.DataFrame]:
    """Rank `spectra` (or reuse `ranked`) and compute the calibration features."""
    spectra = np.atleast_2d(np.asarray(spectra, dtype=np.float64))
    ranked = ranked or retriever.rank(spectra)
    lib = retriever.lib
    c, good = lib.sensor.centers_nm, lib.sensor.good_mask
    top = ranked.rows[:, 0]
    valid = top >= 0
    t = np.where(valid, top, 0)

    groups = retriever.meta.loc[lib.spectrum_ids, "group"].fillna("").to_numpy()
    g = np.where(ranked.rows >= 0, groups[np.maximum(ranked.rows, 0)], "")
    other = (g != g[:, :1]) & (ranked.rows >= 0)
    best_other = np.where(other, ranked.combined, -np.inf).max(axis=1)
    margin = np.where(np.isfinite(best_other), ranked.combined[:, 0] - best_other, 1.0)

    qs, _ = represent(spectra, c, good, "sam")
    ls, _ = represent(lib.matrix[t], c, good, "sam")
    sam = sam_degrees(np.einsum("ij,ij->i", qs, ls))
    m = vector_masks(lib)["cr_swir2"]
    qc, _ = represent(spectra, c, m, "cr_swir2")
    lc, _ = represent(lib.matrix[t], c, m, "cr_swir2")
    cr_cos = np.einsum("ij,ij->i", qc, lc)

    qf = compute_features(spectra, lib.sensor, retriever.features)
    depth = np.max(np.vstack([w.depth for w in qf.windows]), axis=0) if qf.windows else 0.0

    sff = ranked.sff.score[:, 0] if ranked.sff is not None else np.zeros(len(spectra))
    fit_rms = _linear_fit_rms(spectra[:, good], lib.matrix[t][:, good], c[good])
    top5 = g[:, :5]
    consensus = ((top5 == g[:, :1]) & (ranked.rows[:, :5] >= 0)).sum(axis=1) / 5.0
    df = pd.DataFrame(
        {
            "score": ranked.combined[:, 0],
            "margin": margin,
            "sff": sff,
            "sam_deg": sam,
            "cr_cos": cr_cos,
            "depth": depth,
            "fit_rms": fit_rms,
            "consensus": consensus,
        }
    )
    df[~valid] = np.nan
    return ranked, df


def _linear_fit_rms(q: np.ndarray, c: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Relative RMS residual of q ~ a*c + b0 + b1*t, solved for every row at once."""
    t = (wl - 1500.0) / 1000.0
    ok = np.isfinite(q).all(axis=1) & np.isfinite(c).all(axis=1)
    out = np.full(len(q), np.nan)
    if not ok.any():
        return out
    Q, C = q[ok], c[ok]
    X = np.stack([C, np.ones_like(C), np.broadcast_to(t, C.shape)], axis=2)  # (n, b, 3)
    XtX = np.einsum("nbi,nbj->nij", X, X) + 1e-9 * np.eye(3)
    Xty = np.einsum("nbi,nb->ni", X, Q)
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    resid = Q - np.einsum("nbi,ni->nb", X, beta)
    out[ok] = np.sqrt((resid**2).mean(axis=1)) / np.maximum(np.abs(Q).mean(axis=1), 1e-9)
    return out


@dataclass
class ConfidenceModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # intercept first
    threshold: float = 0.5
    features: tuple[str, ...] = FEATURES

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        x = df[list(self.features)].to_numpy(dtype=np.float64)
        x = np.where(np.isfinite(x), x, self.mean)
        return np.hstack([np.ones((len(x), 1)), (x - self.mean) / self.std])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(top-1 group is right); NaN rows (no match) get 0."""
        p = 1.0 / (1.0 + np.exp(-self._x(df) @ self.coef))
        p[df[list(self.features)].isna().all(axis=1).to_numpy()] = 0.0
        return p

    @classmethod
    def fit(cls, df: pd.DataFrame, y: np.ndarray, l2: float = 1e-2) -> ConfidenceModel:
        x = df[list(FEATURES)].to_numpy(dtype=np.float64)
        ok = np.isfinite(x).all(axis=1)
        x, y = x[ok], np.asarray(y, dtype=np.float64)[ok]
        mean, std = x.mean(axis=0), x.std(axis=0) + 1e-9
        X = np.hstack([np.ones((len(x), 1)), (x - mean) / std])

        def loss(w):
            z = X @ w
            nll = np.sum(np.logaddexp(0, z) - y * z) / len(y)
            return nll + l2 * np.sum(w[1:] ** 2)

        def grad(w):
            p = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (p - y) / len(y)
            g[1:] += 2 * l2 * w[1:]
            return g

        w = minimize(loss, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x
        return cls(mean, std, w)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "features": list(self.features),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "threshold": self.threshold,
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceModel:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            np.array(d["mean"]),
            np.array(d["std"]),
            np.array(d["coef"]),
            float(d["threshold"]),
            tuple(d["features"]),
        )


# --- metrics -------------------------------------------------------------------------


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (probability a positive outranks a negative)."""
    s = np.asarray(scores, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    if pos.all() or (~pos).all():
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def accuracy_at_coverage(p: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """Accuracy over the most confident `coverage` fraction of queries."""
    order = np.argsort(-np.asarray(p))
    keep = order[: max(1, int(round(coverage * len(order))))]
    return float(np.asarray(y)[keep].mean())
