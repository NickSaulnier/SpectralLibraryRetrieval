"""Spectral feature fitting: rescore retrieval candidates on diagnostic absorption windows.

For each window in configs/features.yaml, the query's and each candidate's band-depth
profiles are compared by
* shape: Pearson correlation of the profiles (independent of depth, i.e. of abundance);
* position: band-centre difference, through exp(-0.5 (dlambda / sigma)^2).

A window counts when either side has a feature deeper than `min_depth`, weighted by the
deeper of the two. A feature present on only one side scores zero there, so a candidate is
penalised for absorptions the pixel lacks and vice versa. The score is the weighted mean
over windows, in [0, 1]. Per-window values are kept so a match can be explained.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from specret.rerank.features import (
    FeatureConfig,
    Window,
    band_center,
    depth_profiles,
    load_features,
    window_bands,
)
from specret.sensors.sensor import Sensor


@dataclass
class WindowProfiles:
    window: Window
    band_idx: np.ndarray
    wavelengths: np.ndarray
    profiles: np.ndarray  # (m, n_window_bands)
    depth: np.ndarray  # (m,)
    center: np.ndarray  # (m,)


@dataclass
class FeatureSet:
    """Window profiles for a stack of spectra on one sensor."""

    windows: list[WindowProfiles] = field(default_factory=list)


def compute_features(
    spectra: np.ndarray, sensor: Sensor, cfg: FeatureConfig | None = None
) -> FeatureSet:
    cfg = cfg or load_features()
    out = FeatureSet()
    for w in cfg.windows:
        idx = window_bands(sensor, w)
        if idx is None:
            continue
        prof = depth_profiles(spectra, sensor.centers_nm, idx)
        wl = sensor.centers_nm[idx]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (invalid spectra)
            depth = np.nanmax(prof, axis=1)
            center = band_center(prof, wl)
        out.windows.append(WindowProfiles(w, idx, wl, prof, np.nan_to_num(depth, nan=0.0), center))
    return out


@dataclass
class SFFResult:
    score: np.ndarray  # (nq, N)
    per_window: dict[str, dict[str, np.ndarray]]  # window -> {r, center_diff, weight, score}


def _pearson(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Correlation of q (nq, nb) with each row of c (nq, N, nb) along the last axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qc = q - np.nanmean(q, axis=-1, keepdims=True)
        cc = c - np.nanmean(c, axis=-1, keepdims=True)
    qc, cc = np.nan_to_num(qc), np.nan_to_num(cc)
    num = np.einsum("qb,qnb->qn", qc, cc)
    den = np.sqrt((qc**2).sum(-1))[:, None] * np.sqrt((cc**2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def score_candidates(
    query: FeatureSet,
    library: FeatureSet,
    candidates: np.ndarray,
    cfg: FeatureConfig | None = None,
) -> SFFResult:
    """Score library rows `candidates` (nq, N; -1 = none) against each query."""
    cfg = cfg or load_features()
    cand = np.where(candidates >= 0, candidates, 0)
    num = np.zeros(candidates.shape)
    den = np.zeros(candidates.shape)
    per_window = {}
    for qw, lw in zip(query.windows, library.windows, strict=True):
        qd = qw.depth[:, None]
        cd = lw.depth[cand]
        r = _pearson(qw.profiles, lw.profiles[cand])
        dc = qw.center[:, None] - lw.center[cand]
        q_has, c_has = qd > cfg.min_depth, cd > cfg.min_depth
        s = np.where(
            q_has & c_has,
            np.clip(r, 0.0, 1.0) * np.exp(-0.5 * (dc / cfg.center_sigma_nm) ** 2),
            0.0,
        )
        weight = np.where(q_has | c_has, np.maximum(qd, cd), 0.0)
        num += weight * s
        den += weight
        per_window[qw.window.name] = {
            "query_depth": np.broadcast_to(qd, candidates.shape),
            "query_center": np.broadcast_to(qw.center[:, None], candidates.shape),
            "cand_depth": cd,
            "cand_center": lw.center[cand],
            "r": r,
            "weight": weight,
            "score": s,
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(den > 0, num / den, 0.0)
    score[candidates < 0] = -np.inf
    return SFFResult(score, per_window)


def explain(result: SFFResult, qi: int, ci: int, min_weight: float = 0.005) -> list[dict]:
    """Per-window breakdown for query `qi`, candidate column `ci`, strongest first."""
    rows = []
    for name, d in result.per_window.items():
        w = float(d["weight"][qi, ci])
        if w < min_weight:
            continue
        rows.append(
            {
                "window": name,
                "weight": w,
                "score": float(d["score"][qi, ci]),
                "query_depth": float(d["query_depth"][qi, ci]),
                "query_center_nm": float(d["query_center"][qi, ci]),
                "cand_depth": float(d["cand_depth"][qi, ci]),
                "cand_center_nm": float(d["cand_center"][qi, ci]),
                "shape_r": float(d["r"][qi, ci]),
            }
        )
    return sorted(rows, key=lambda r: -r["weight"])
