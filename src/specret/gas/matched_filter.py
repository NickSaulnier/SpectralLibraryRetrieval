"""Column-wise matched filter for gas enhancement in at-sensor radiance (PLAN_GAS.md 1.1).

For a pushbroom sensor each detector column (crosstrack position) has its own response, so the
background mean mu and covariance Sigma are estimated per column, from that column's pixels down
the track. With the target t = -mu * k (k = unit absorption spectrum, per ppm·m), the estimate is

    alpha = (x - mu)^T Sigma^-1 t / (t^T Sigma^-1 t)       [ppm·m]

Sigma is shrunk towards its diagonal, C = (1 - a) S + a diag(S). The shrinkage a is either fixed
or chosen per column by leave-one-out cross-validation (Theiler 2012, "The incredible shrinking
covariance estimator", eqn 29), as the EMIT GHG products do; with ~200 bands and ~1300 pixels
per column the sample covariance alone is too noisy.

An optional second pass recomputes mu and Sigma without the pixels that look like plume in the
first pass, so strong plumes do not leak into their own background. Runs with torch, on the GPU
when there is one, a chunk of columns at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROBUST = 1.4826  # MAD to standard deviation for a normal distribution
LOO_ALPHAS = 10.0 ** np.arange(-10, 0.05, 0.05)  # candidate shrinkages, as in EMIT GHG


@dataclass
class MatchedFilterResult:
    alpha: np.ndarray  # (rows, cols) enhancement in ppm·m; NaN where invalid
    sigma: np.ndarray  # (cols,) robust background standard deviation of alpha per column
    excluded: np.ndarray  # (rows, cols) pixels left out of the final background statistics
    n_valid: np.ndarray  # (cols,) pixels used for the final statistics
    shrinkage: np.ndarray  # (cols,) covariance shrinkage used per column
    mu: np.ndarray  # (cols, bands) background mean radiance per column, final pass


def loo_shrinkage(d, n, alphas=LOO_ALPHAS):
    """Shrinkage per column minimizing the leave-one-out negative log-likelihood.

    d: (c, R, B) centred data with unused rows set to zero; n: (c,) rows used per column.
    Whitening by the diagonal target and one eigendecomposition per column make every
    candidate cheap: with S~ = D^-1/2 S D^-1/2 = V L V^T and Y = d D^-1/2 V,
        G(a) = n b S + a D,  b = (1 - a) / (n - 1)
        log det G = sum log D + sum log(n b L + a),   r_k = sum_i Y_ki^2 / (n b L_i + a)
        nll(a) = 0.5 log det G + 1/(2n) sum_k [log q_k + r_k / q_k],   q_k = 1 - b r_k
    """
    import torch

    nn = n.to(d.dtype)
    s = d.transpose(1, 2) @ d / (nn - 1)[:, None, None]
    diag = torch.diagonal(s, dim1=1, dim2=2).clamp(min=1e-30)  # (c, B)
    w = diag.rsqrt()
    lam, vec = torch.linalg.eigh(s * w[:, :, None] * w[:, None, :])
    lam = lam.clamp(min=0)
    y2 = ((d * w[:, None, :]) @ vec) ** 2  # (c, R, B)
    a = torch.as_tensor(alphas, dtype=d.dtype, device=d.device)  # (A,)
    beta = (1 - a)[None, :] / (nn - 1)[:, None]  # (c, A)
    denom = (nn[:, None, None] * beta[:, None, :]) * lam[:, :, None] + a[None, None, :]  # (c,B,A)
    logdet = torch.log(diag).sum(1)[:, None] + torch.log(denom).sum(1)  # (c, A)
    r = y2 @ (1 / denom)  # (c, R, A)
    q = 1 - beta[:, None, :] * r
    used = (d != 0).any(dim=-1)[:, :, None]  # rows that are part of the statistics
    term = torch.where(used & (q > 0), torch.log(q.clamp(min=1e-300)) + r / q, 0.0)
    bad = (used & (q <= 0)).any(dim=1)  # (c, A): candidate not valid for this column
    nll = 0.5 * logdet + term.sum(1) / (2 * nn[:, None])
    nll = torch.where(bad | ~torch.isfinite(nll), torch.inf, nll)
    return a[nll.argmin(dim=1)]


def _filter_chunk(x, keep, finite, kk, shrinkage, min_n):
    """One pass for a chunk of columns. x: (c, R, B); keep, finite: (c, R); kk: (c, B)."""
    import torch

    w = keep.to(x.dtype)
    n = keep.sum(dim=1)
    ok = n >= min_n
    mu = (w[..., None] * x).sum(1) / n.clamp(min=1)[:, None]
    d = (x - mu[:, None, :]) * w[..., None]
    if shrinkage == "loo":
        shrink = loo_shrinkage(d, n.clamp(min=2))
    else:
        shrink = torch.full((x.shape[0],), float(shrinkage), dtype=x.dtype, device=x.device)
    s = d.transpose(1, 2) @ d / (n.clamp(min=2) - 1).to(x.dtype)[:, None, None]
    diag = torch.diag_embed(torch.diagonal(s, dim1=1, dim2=2))
    eye = torch.eye(x.shape[2], dtype=x.dtype, device=x.device)
    cov = (1 - shrink)[:, None, None] * s + shrink[:, None, None] * diag
    cov = cov + (~ok)[:, None, None] * eye  # keep empty columns solvable
    t = -mu * kk
    sinv_t = torch.linalg.solve(cov, t[..., None])[..., 0]  # (c, B)
    norm = (t * sinv_t).sum(dim=-1)
    a = ((x - mu[:, None, :]) @ sinv_t[..., None])[..., 0] / norm[:, None]  # (c, R)
    a[~ok] = float("nan")
    a[~finite] = float("nan")
    return a, n, shrink, mu


def column_matched_filter(
    radiance: np.ndarray,
    k: np.ndarray,
    valid: np.ndarray | None = None,
    n_pass: int = 2,
    shrinkage: float | str = "loo",
    exclude_sigma: float = 3.0,
    min_pixels_factor: float = 2.0,
    device: str | None = None,
    chunk: int = 64,
) -> MatchedFilterResult:
    """Enhancement for every pixel of `radiance` (rows, cols, bands).

    k is the unit absorption spectrum, (bands,) for the whole scene or (cols, bands) per column.
    `valid` (rows, cols) marks pixels usable for statistics; pixels with any NaN band are never
    used. Columns with fewer than min_pixels_factor * bands usable pixels are left as NaN.
    `shrinkage` is a fixed value in [0, 1] or "loo" for leave-one-out selection per column.
    """
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, cols, nb = radiance.shape
    k = np.broadcast_to(np.asarray(k, dtype=np.float64), (cols, nb))
    valid = np.ones((rows, cols), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    min_n = max(min_pixels_factor * nb, nb + 2)
    alpha = np.full((rows, cols), np.nan, dtype=np.float32)
    sigma = np.full(cols, np.nan, dtype=np.float32)
    excluded = np.zeros((rows, cols), dtype=bool)
    n_valid = np.zeros(cols, dtype=np.int64)
    shrink_out = np.full(cols, np.nan, dtype=np.float32)
    mu_out = np.full((cols, nb), np.nan, dtype=np.float32)
    for c0 in range(0, cols, chunk):
        sl = slice(c0, min(c0 + chunk, cols))
        xc = torch.as_tensor(np.ascontiguousarray(radiance[:, sl].transpose(1, 0, 2)), device=dev)
        xc = xc.to(torch.float64)
        finite = torch.isfinite(xc).all(dim=-1)
        usable = finite & torch.as_tensor(np.ascontiguousarray(valid[:, sl].T), device=dev)
        xc = torch.nan_to_num(xc, nan=0.0)
        kk = torch.as_tensor(np.array(k[sl], dtype=np.float64), device=dev)
        keep = usable
        for _ in range(n_pass):
            used = keep
            a, n, shrink, mu = _filter_chunk(xc, used, finite, kk, shrinkage, min_n)
            # Robust background spread per column, from the pixels the statistics used
            bg = torch.where(used, a, torch.full_like(a, float("nan")))
            med = torch.nanmedian(bg, dim=1).values
            sig = ROBUST * torch.nanmedian((bg - med[:, None]).abs(), dim=1).values
            keep = usable & ~(a > (med + exclude_sigma * sig)[:, None])
        alpha[:, sl] = a.T.float().cpu().numpy()
        sigma[sl] = sig.float().cpu().numpy()
        excluded[:, sl] = (usable & ~used).T.cpu().numpy()
        n_valid[sl] = n.cpu().numpy()
        shrink_out[sl] = shrink.float().cpu().numpy()
        mu_out[sl] = mu.float().cpu().numpy()
    return MatchedFilterResult(alpha, sigma, excluded, n_valid, shrink_out, mu_out)
