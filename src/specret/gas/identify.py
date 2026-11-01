"""Which gas (if any) explains a candidate plume: joint gas + confuser fit (PLAN_GAS.md G2).

For a candidate's pixels P, the log radiance ratio y_p = ln(x_p / mu_col) (mu from the
matched filter's background) is averaged: y_bar. A thin plume of gases g with mean enhancement
alpha_g, over a surface that may hold more of confuser material j, gives

    y_bar ≈ -sum_g alpha_g k_g + sum_j beta_j s_j + baseline

with k_g the unit absorption spectra and s_j the confuser signatures (gas/confusers.py).

The noise in y_bar is spatially correlated surface texture, not pixel noise, so its covariance
is measured directly: the candidate's mask is shifted to many places in the scene that hold no
candidate, and y_bar is computed at each (the "null" sample). Everything is whitened with that
covariance (leave-one-out shrinkage), then fitted by non-negative least squares. Terms are
added greedily while they lower the BIC, up to `max_terms`; a per-window offset and slope are
always in the model.

The result names the gas, its fitted enhancement and z-score, the confusers tried, and the
features the confidence model uses:
* mf_z         peak z-score of the candidate in the matched-filter map
* gas_z        z-score of the identified gas in the joint fit
* coherence    lag-1 spatial autocorrelation of the enhancement around the candidate
* vs_confusers chi-square gained by the gas over the best confuser-only model (log1p)
* consistency  disagreement of the gas between fit windows, |z| of alpha_w1 - alpha_w2
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from specret.gas.matched_filter import loo_shrinkage
from specret.gas.plumes import Candidate
from specret.unmix.fcls import nnls_fit


@dataclass
class FitContext:
    """Per-scene inputs, restricted to the fit bands."""

    y: np.ndarray  # (rows, cols, B) relative residual x / mu - 1, NaN where unusable
    usable: np.ndarray  # (rows, cols)
    centers_nm: np.ndarray  # (B,)
    windows: list[tuple[float, float]]
    gas_k: dict[str, np.ndarray]  # gas -> (B,) unit absorption per ppm·m
    confusers: dict[str, np.ndarray]  # name -> (B,) signature
    alpha: np.ndarray  # (rows, cols) matched-filter enhancement of the detection gas
    occupied: np.ndarray  # (rows, cols) pixels in or near any candidate (excluded from nulls)


@dataclass
class Identification:
    candidate_id: int
    gas: str | None  # identified gas, or None
    verdict: str  # "<gas>", "confuser:<name>", "unknown absorber" or "no signal"
    terms: dict[str, float]  # selected term -> coefficient (gases ppm·m, confusers depth)
    gas_z: dict[str, float]
    window_alpha: dict[str, list[float]] = field(default_factory=dict)  # gas -> per window
    chi2_baseline: float = np.nan
    chi2_model: float = np.nan
    chi2_confusers_only: float = np.nan
    n_bands: int = 0
    n_null: int = 0
    features: dict[str, float] = field(default_factory=dict)


def relative_residual(radiance: np.ndarray, mu: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """ln(x / mu_col) per pixel, (rows, cols, B) float32, NaN where not usable. The log keeps
    Beer-Lambert absorption additive, so strong plumes stay closer to linear in alpha."""
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.clip(radiance / mu[None, :, :].astype(np.float32), 1e-6, None))
    y[~usable] = np.nan
    return y.astype(np.float32, copy=False)


def baseline_matrix(centers_nm: np.ndarray, windows) -> np.ndarray:
    """Signed offset and slope per window, as +/- column pairs (B, 4 * windows)."""
    cols = []
    for lo, hi in windows:
        m = ((centers_nm >= lo) & (centers_nm <= hi)).astype(np.float64)
        t = (centers_nm - 0.5 * (lo + hi)) / (hi - lo) * m
        cols += [m, -m, t, -t]
    return np.column_stack(cols)


def null_means(
    ctx: FitContext, cand: Candidate, n: int, seed: int, max_shift: int = 250
) -> np.ndarray:
    """y_bar at up to n shifted copies of the candidate's mask (n, B)."""
    rng = np.random.default_rng(seed)
    rows, cols = ctx.usable.shape
    out = []
    for _ in range(20 * n):
        if len(out) >= n:
            break
        dr, dc = rng.integers(-max_shift, max_shift + 1, size=2)
        r, c = cand.rows + dr, cand.cols + dc
        if r.min() < 0 or c.min() < 0 or r.max() >= rows or c.max() >= cols:
            continue
        if ctx.occupied[r, c].any() or ctx.usable[r, c].mean() < 0.9:
            continue
        out.append(np.nanmean(ctx.y[r, c], axis=0))
    return np.asarray(out)


def whitener(null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, W) with W^T W = C^-1 for the shrunk null covariance."""
    m = null.mean(axis=0)
    d = torch.as_tensor(null - m, dtype=torch.float64)[None]
    a = float(loo_shrinkage(d, torch.tensor([null.shape[0]]))[0])
    s = np.cov(null.T)
    c = (1 - a) * s + a * np.diag(np.diag(s))
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, vals.max() * 1e-10, None)
    return m, (vecs / np.sqrt(vals)).T


def _chi2(A: np.ndarray, z: np.ndarray, base: np.ndarray) -> tuple[float, np.ndarray]:
    """NNLS chi-square and coefficients. Columns are scaled to unit norm for the solver (gas
    columns per ppm·m differ from confuser columns by many orders of magnitude)."""
    if A.shape[1] == 0:
        return nnls_fit(np.zeros((z.size, 0)), z, baseline=base).rss, np.zeros(0)
    norm = np.linalg.norm(A, axis=0)
    norm[norm == 0] = 1.0
    f = nnls_fit(A / norm, z, baseline=base)
    return f.rss, f.coefficients / norm


def _greedy(terms: dict[str, np.ndarray], z, base, max_terms: int, penalty: dict[str, float]):
    """Forward selection: add the term that lowers chi-square most, while the drop exceeds
    that term's penalty (BIC: ln n, more for flag-only gases). Returns (names, coef, chi2)."""
    chosen: list[str] = []
    chi2, coef = _chi2(np.zeros((z.size, 0)), z, base)
    while len(chosen) < max_terms:
        best = None
        for name, col in terms.items():
            if name in chosen:
                continue
            A = np.column_stack([terms[t] for t in chosen] + [col])
            c2, cf = _chi2(A, z, base)
            gain = chi2 - c2 - penalty[name]
            if best is None or gain > best[3]:
                best = (name, c2, cf, gain)
        if best is None or best[3] <= 0:
            break
        chosen.append(best[0])
        chi2, coef = best[1], best[2]
    return chosen, coef[: len(chosen)], chi2


def _ls_z(A: np.ndarray, z: np.ndarray, base: np.ndarray, j: int) -> tuple[float, float]:
    """Unconstrained least-squares estimate and z-score of column j (whitened space)."""
    X = np.hstack([A, base[:, ::2]])  # one column per signed baseline term
    norm = np.linalg.norm(X, axis=0)
    norm[norm == 0] = 1.0
    Xn = X / norm
    coef, *_ = np.linalg.lstsq(Xn, z, rcond=None)
    cov = np.linalg.pinv(Xn.T @ Xn)
    sd = float(np.sqrt(max(cov[j, j], 1e-30)))
    return float(coef[j] / norm[j]), float(coef[j] / sd)


def coherence(alpha: np.ndarray, cand: Candidate, margin: int = 3) -> float:
    """Lag-1 autocorrelation of alpha in the candidate's bounding box plus a margin."""
    r0, r1 = max(cand.rows.min() - margin, 0), cand.rows.max() + margin + 1
    c0, c1 = max(cand.cols.min() - margin, 0), cand.cols.max() + margin + 1
    a = alpha[r0:r1, c0:c1].astype(np.float64)
    pairs = [(a[:, :-1], a[:, 1:]), (a[:-1], a[1:])]
    x = np.concatenate([p[0].ravel() for p in pairs])
    y = np.concatenate([p[1].ravel() for p in pairs])
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def identify(
    ctx: FitContext,
    cand: Candidate,
    detection_gas: str = "CH4",
    max_terms: int = 3,
    n_null: int = 300,
    seed: int = 0,
    named: tuple[str, ...] = ("CH4", "CO2"),
    flag_penalty: float = 3.0,
) -> Identification:
    """Fit the candidate. Only `named` gases can be the identified gas; other gases enter the
    fit at `flag_penalty` times the BIC penalty and are reported as "consistent with", and H2O
    is a nuisance term (a wetter or drier column) that is never named."""
    use = ctx.usable[cand.rows, cand.cols]
    y_bar = np.nanmean(ctx.y[cand.rows[use], cand.cols[use]], axis=0).astype(np.float64)
    null = null_means(ctx, cand, n_null, seed + cand.id)
    nb = ctx.centers_nm.size
    if null.shape[0] < 30 or not np.isfinite(y_bar).all():
        return Identification(cand.id, None, "no signal", {}, {}, n_bands=nb, n_null=len(null))
    m, W = whitener(null)
    z = W @ (y_bar - m)
    base = W @ baseline_matrix(ctx.centers_nm, ctx.windows)
    gas_cols = {g: W @ (-k) for g, k in ctx.gas_k.items()}
    conf_cols = {f"confuser:{n}": W @ s for n, s in ctx.confusers.items()}
    bic = np.log(nb)
    penalty = {
        t: bic * (1 if t in named or t == "H2O" or t in conf_cols else flag_penalty)
        for t in [*gas_cols, *conf_cols]
    }

    chosen, coef, chi2 = _greedy({**gas_cols, **conf_cols}, z, base, max_terms, penalty)
    chi2_base, _ = _chi2(np.zeros((nb, 0)), z, base)
    nuisance = {t: c for t, c in gas_cols.items() if t not in named and t in ("H2O",)}
    _, _, chi2_conf = _greedy({**conf_cols, **nuisance}, z, base, max_terms, penalty)
    terms = dict(zip(chosen, map(float, coef), strict=True))

    A = np.column_stack([({**gas_cols, **conf_cols})[t] for t in chosen]) if chosen else None
    gas_z = {}
    for j, t in enumerate(chosen):
        if t in gas_cols and t != "H2O":
            gas_z[t] = _ls_z(A, z, base, j)[1]
    named_z = {g: v for g, v in gas_z.items() if g in named and v > 0}
    gas = max(named_z, key=named_z.get) if named_z else None

    # The detection gas is always scored per window, so consistency is defined even when the
    # fit did not choose it.
    focus = gas or detection_gas
    window_alpha, window_sd = [], []
    for lo, hi in ctx.windows:
        wm = (ctx.centers_nm >= lo) & (ctx.centers_nm <= hi)
        k = ctx.gas_k[focus][wm]
        if np.abs(k).max() < 1e-12 or wm.sum() < 5:
            continue
        mw, Ww = whitener(null[:, wm])
        zw = Ww @ (y_bar[wm] - mw)
        bw = Ww @ baseline_matrix(ctx.centers_nm[wm], [(lo, hi)])
        a, zz = _ls_z((Ww @ -k)[:, None], zw, bw, 0)
        window_alpha.append(a)
        window_sd.append(abs(a / zz) if zz != 0 else np.inf)
    consistency = np.nan
    if len(window_alpha) >= 2:
        d = window_alpha[0] - window_alpha[1]
        consistency = float(abs(d) / np.hypot(window_sd[0], window_sd[1]))

    flags = [t for t in chosen if t in gas_cols and t not in named and t != "H2O"]
    if gas is not None:
        verdict = gas
    elif flags:
        verdict = f"consistent with {flags[0]}"
    elif any(t in conf_cols for t in chosen):
        verdict = next(t for t in chosen if t in conf_cols)
    elif (chi2_base - nb) / np.sqrt(2 * nb) > 5:
        verdict = "unknown absorber"
    else:
        verdict = "no signal"
    features = {
        "mf_z": cand.peak_z,
        "gas_z": gas_z.get(focus, _ls_z(gas_cols[focus][:, None], z, base, 0)[1]),
        "coherence": coherence(ctx.alpha, cand),
        "vs_confusers": float(np.log1p(max(chi2_conf - chi2, 0.0))),
        "consistency": consistency,
        "log_px": float(np.log(cand.n_px)),
    }
    return Identification(
        candidate_id=cand.id,
        gas=gas,
        verdict=verdict,
        terms=terms,
        gas_z=gas_z,
        window_alpha={focus: window_alpha},
        chi2_baseline=float(chi2_base),
        chi2_model=float(chi2),
        chi2_confusers_only=float(chi2_conf),
        n_bands=nb,
        n_null=int(null.shape[0]),
        features=features,
    )
