"""Scene-level gas analysis: matched filters, candidates, identification, mineral screen (G2).

For each detection gas (CH4, CO2) the matched filter gives an enhancement map; candidates are
found in each map and every candidate is identified by the joint gas + confuser fit. The
records carry the features for the confidence model (eval/gas_benchmark.py calibrates it).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

from specret.gas.confusers import build_confusers, mineral_screen
from specret.gas.detect import Enhancement, Prepared, enhancement
from specret.gas.identify import FitContext, Identification, identify, relative_residual
from specret.gas.library import plume_config
from specret.gas.plumes import Candidate, find_candidates
from specret.gas.signatures import window_absorption
from specret.io.emit_l1b import sibling

LIBRARY_NPZ = Path("data/resampled/emit/library.npz")
CANONICAL = Path("data/canonical/library.parquet")


@dataclass
class CandidateRecord:
    detection_gas: str
    candidate: Candidate
    ident: Identification
    mineral_fraction: float | None

    def row(self) -> dict:
        c, i = self.candidate, self.ident
        return {
            "detection_gas": self.detection_gas,
            "candidate_id": c.id,
            "n_px": c.n_px,
            "row": c.centroid[0],
            "col": c.centroid[1],
            "mean_alpha": c.mean_alpha,
            "max_alpha": c.max_alpha,
            "verdict": i.verdict,
            "gas": i.gas,
            "terms": ";".join(f"{k}={v:.3g}" for k, v in i.terms.items()),
            "mineral_fraction": self.mineral_fraction,
            **i.features,
        }


@lru_cache(maxsize=1)
def _library_confusers(windows: tuple[tuple[float, float], ...]):
    from specret.schema import read_library
    from specret.sensors.library import SensorLibrary

    lib = SensorLibrary.load(LIBRARY_NPZ)
    return build_confusers(lib, read_library(CANONICAL), [list(w) for w in windows])


def confusers_on_bands(centers_nm: np.ndarray, windows) -> dict[str, np.ndarray]:
    """Confuser signatures at the given band centres (nearest library band)."""
    conf = _library_confusers(tuple(tuple(w) for w in windows))
    j = np.abs(conf.centers_nm[None, :] - centers_nm[:, None]).argmin(axis=1)
    return {n: s[j] for n, s in zip(conf.names, conf.signatures, strict=True)}


def fit_bands(prep: Prepared, windows) -> np.ndarray:
    c = prep.centers_nm
    return np.flatnonzero(np.any([(c >= lo) & (c <= hi) for lo, hi in windows], axis=0))


def analyze(
    prep: Prepared,
    gases: tuple[str, ...] = ("CH4", "CO2"),
    radiance: np.ndarray | None = None,
    seed: int = 0,
    device: str | None = None,
) -> tuple[dict[str, Enhancement], list[CandidateRecord]]:
    """Enhancement maps and identified candidates for one scene (optionally with injected
    radiance of the same shape as prep.radiance)."""
    cfg = plume_config()
    icfg = cfg["identify"]
    rad = prep.radiance if radiance is None else radiance
    enh = {g: enhancement(prep, g, rad, device) for g in gases}
    found = {g: find_candidates(e.alpha, **cfg["candidates"])[0] for g, e in enh.items()}

    windows = [tuple(w) for w in icfg["fit_windows"]]
    fb = fit_bands(prep, windows)
    usable = ~prep.screen & np.isfinite(rad[:, :, fb]).all(axis=-1)
    y = relative_residual(rad[:, :, fb], enh[gases[0]].result.mu[:, fb], usable)
    occupied = np.zeros(usable.shape, dtype=bool)
    for cands in found.values():
        for c in cands:
            occupied[c.rows, c.cols] = True
    occupied = ndimage.binary_dilation(occupied, iterations=3)
    centers = prep.centers_nm[fb]
    gas_k = {
        g: window_absorption(prep.pieces, centers, prep.fwhm_nm[fb], prep.conditions, g)
        for g in icfg["gases"]
    }
    confusers = confusers_on_bands(centers, windows)
    screen = mineral_screen(sibling(prep.scene.path, "L2B_MIN"))

    records = []
    for g, cands in found.items():
        ctx = FitContext(y, usable, centers, windows, gas_k, confusers, enh[g].alpha, occupied)
        for c in cands:
            ident = identify(
                ctx,
                c,
                g,
                icfg["max_terms"],
                icfg["null_samples"],
                seed,
                tuple(icfg["named_gases"]),
                icfg["flag_penalty"],
            )
            frac = screen(c.rows, c.cols) if screen else None
            records.append(CandidateRecord(g, c, ident, frac))
    return enh, records
