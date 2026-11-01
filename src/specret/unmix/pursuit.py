"""Retrieve-and-unmix: greedy forward selection of library endmembers (PLAN.md, Phase 4).

1. Seed with the hybrid retriever's best candidate for the pixel.
2. Build a pool of possible next endmembers: the retriever's top candidates for the pixel,
   plus the library spectra best correlated with the current residual. (Residuals can be
   negative and are not reflectance, so they are matched by correlation in linear space,
   never through the continuum-removed or learned representations.)
3. Try adding each pool member, keeping the one that lowers the residual most. Accept it only
   if its fraction is meaningful, the BIC improves, and the residual sum of squares drops by at
   least `min_improvement` (a fraction). The last rule matters: the library never holds the
   pixel's exact sample, so any extra endmember lowers the residual a little, and adjacent
   bands are too correlated for the BIC penalty alone. One endmember per group.
4. Stop at `max_components`, when nothing improves the BIC, or when the residual reaches the
   noise floor (reduced chi-square <= 1 when a noise curve is given).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from specret.search.hybrid import HybridRetriever
from specret.unmix.fcls import Fit, baseline_columns, bic, nnls_fit


@dataclass
class Unmixing:
    rows: list[int]  # library rows of the endmembers, in selection order
    fit: Fit
    n_bands: int
    reduced_chi2: float | None
    bic_path: list[float] = field(default_factory=list)

    @property
    def rms(self) -> float:
        return float(np.sqrt(self.fit.rss / self.n_bands))


def _pearson_to_rows(residual: np.ndarray, lib: np.ndarray) -> np.ndarray:
    r = residual - residual.mean()
    L = lib - lib.mean(axis=1, keepdims=True)
    den = np.linalg.norm(L, axis=1) * np.linalg.norm(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, L @ r / den, -np.inf)


class Unmixer:
    def __init__(
        self,
        retriever: HybridRetriever,
        canonical: pd.DataFrame,
        max_components: int = 3,
        pool_size: int = 5,
        min_fraction: float = 0.1,
        min_improvement: float = 0.6,
        sum_to_one: bool = False,
        baseline: bool = True,
    ):
        self.r = retriever
        lib = retriever.lib
        self.good = lib.sensor.good_mask
        M = lib.matrix[:, self.good]
        self.usable = np.isfinite(M).all(axis=1)
        self.M = np.where(np.isfinite(M), M, 0.0)
        groups = canonical.set_index("spectrum_id").loc[lib.spectrum_ids, "group"]
        self.groups = groups.fillna("").to_numpy()
        self.max_components, self.pool_size = max_components, pool_size
        self.min_fraction, self.sum_to_one = min_fraction, sum_to_one
        self.min_improvement = min_improvement
        wl = lib.sensor.centers_nm[self.good]
        self.baseline = baseline_columns(wl) if baseline else None
        self.n_extra = 2 if baseline else 0  # free baseline parameters in the BIC

    def unmix(
        self,
        spectrum: np.ndarray,
        noise_sigma: np.ndarray | None = None,
        seed_rows: np.ndarray | None = None,
    ) -> Unmixing | None:
        x_full = np.asarray(spectrum, dtype=np.float64)
        x = x_full[self.good]
        if not np.isfinite(x).all():
            return None
        n = x.size
        if seed_rows is None:
            seed_rows = self.r.rank(x_full[None, :]).rows[0]
        seeds = [int(r) for r in seed_rows if r >= 0 and self.usable[r]]
        if not seeds:
            return None
        sigma = None if noise_sigma is None else np.asarray(noise_sigma)[self.good]

        def chi2(fit: Fit) -> float | None:
            if sigma is None:
                return None
            return float(np.sum((fit.residual / sigma) ** 2) / max(n - len(active), 1))

        active = [seeds[0]]
        fit = nnls_fit(self.M[active].T, x, self.sum_to_one, self.baseline)
        path = [bic(fit.rss, n, 1 + self.n_extra)]
        while len(active) < self.max_components:
            c2 = chi2(fit)
            if c2 is not None and c2 <= 1.0:
                break
            used = {self.groups[a] for a in active}
            corr = _pearson_to_rows(fit.residual, self.M)
            corr[~self.usable] = -np.inf
            by_residual = np.argsort(-corr)[: self.pool_size]
            pool = [
                int(p)
                for p in dict.fromkeys([*seeds[1:], *by_residual.tolist()])
                if self.groups[p] not in used and self.usable[p]
            ]
            best = None
            for p in pool:
                trial = nnls_fit(self.M[[*active, p]].T, x, self.sum_to_one, self.baseline)
                if trial.fractions[-1] < self.min_fraction:
                    continue
                if best is None or trial.rss < best[1].rss:
                    best = (p, trial)
            if best is None:
                break
            new_bic = bic(best[1].rss, n, len(active) + 1 + self.n_extra)
            if new_bic >= path[-1] or best[1].rss > (1 - self.min_improvement) * fit.rss:
                break
            active.append(best[0])
            fit = best[1]
            path.append(new_bic)

        # Drop endmembers whose fraction collapsed after refitting.
        keep = [i for i, f in enumerate(fit.fractions) if f >= self.min_fraction / 2] or [0]
        if len(keep) < len(active):
            active = [active[i] for i in keep]
            fit = nnls_fit(self.M[active].T, x, self.sum_to_one, self.baseline)
        return Unmixing(active, fit, n, chi2(fit), path)

    def describe(
        self, u: Unmixing, canonical: pd.DataFrame, spectrum: np.ndarray | None = None
    ) -> dict:
        """Components by fraction. Given the unmixed `spectrum`, also the decomposition on the
        good bands: each endmember scaled by its coefficient (in the component order), the
        baseline, the model (their sum) and the residual."""
        meta = canonical.set_index("spectrum_id")
        ids = self.r.lib.spectrum_ids
        comps, scaled = [], []
        order = sorted(range(len(u.rows)), key=lambda i: -u.fit.fractions[i])
        for i in order:
            row, frac = u.rows[i], u.fit.fractions[i]
            m = meta.loc[ids[row]]
            comps.append(
                {
                    "spectrum_id": str(ids[row]),
                    "material_name": m["material_name"],
                    "group": m["group"],
                    "fraction": float(frac),
                }
            )
            scaled.append(u.fit.coefficients[i] * self.M[row])
        out = {
            "components": comps,
            "scale": u.fit.scale,
            "residual_rms": u.rms,
            "reduced_chi2": u.reduced_chi2,
        }
        if spectrum is not None:
            model = np.asarray(spectrum, dtype=np.float64)[self.good] - u.fit.residual
            out["spectra"] = {
                "components": [c.tolist() for c in scaled],
                "baseline": (model - np.sum(scaled, axis=0)).tolist(),
                "model": model.tolist(),
                "residual": u.fit.residual.tolist(),
            }
        return out
