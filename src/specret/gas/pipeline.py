"""Scene-level gas analysis: matched filters, candidates, identification, mineral screen (G2).

For each detection gas (CH4, CO2) the matched filter gives an enhancement map; candidates are
found in each map and every candidate is identified by the joint gas + confuser fit. The
records carry the features for the confidence model (eval/gas_benchmark.py calibrates it).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

from specret.gas.confusers import build_confusers, mineral_screen
from specret.gas.detect import Enhancement, Prepared, enhancement
from specret.gas.identify import FitContext, Identification, identify, relative_residual
from specret.gas.library import plume_config
from specret.gas.plumes import Candidate, column_z, find_candidates, smooth_nan
from specret.gas.quantify import Quantification, quantify
from specret.gas.signatures import window_absorption
from specret.io.emit_l1b import sibling

LIBRARY_NPZ = Path("data/resampled/emit/library.npz")
CANONICAL = Path("data/canonical/library.parquet")
MODEL_PATH = Path("configs/calibration/gas_emit.json")


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


@dataclass
class Plume:
    """A candidate the identifier names with enough confidence, quantified."""

    id: int
    record: CandidateRecord
    confidence: float
    quant: Quantification | None  # None when no wind could be had

    def row(self) -> dict:
        c, i = self.record.candidate, self.record.ident
        out = {
            "plume_id": self.id,
            "gas": i.gas,
            "confidence": self.confidence,
            "detection_gas": self.record.detection_gas,
            "candidate_id": c.id,
            "n_px": c.n_px,
            "mean_alpha_ppmm": c.mean_alpha,
            "max_alpha_ppmm": c.max_alpha,
            "terms": ";".join(f"{k}={v:.3g}" for k, v in i.terms.items()),
            "confusers_fitted": ";".join(t for t in i.terms if t.startswith("confuser:")),
            "carbonate_fraction": self.record.mineral_fraction,
        }
        if self.quant is not None:
            q = self.quant.as_dict()
            parts = q.pop("sigma_parts")
            out.update(q)
            out.update({f"rel_sigma_{k}": v for k, v in parts.items()})
        return out


def confidence_model(path: Path = MODEL_PATH):
    from specret.openset.calibrate import ConfidenceModel

    return ConfidenceModel.load(path) if path.exists() else None


def scene_time(prep: Prepared) -> datetime:
    sid = prep.scene.scene_id
    return datetime.strptime(sid.split("_")[0], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def build_plumes(
    prep: Prepared,
    enh: dict[str, Enhancement],
    records: list[CandidateRecord],
    wind=None,
    model=None,
    threshold: float | None = None,
) -> list[Plume]:
    """Plumes from identified candidates, most confident first. `wind` overrides the wind
    lookup (an io.wind.Wind for every plume); otherwise HRRR/MERRA-2 at each plume's source."""
    import pandas as pd

    from specret.io.wind import wind_at

    cfg = plume_config()
    threshold = cfg["identify"]["confidence_threshold"] if threshold is None else threshold
    model = model or confidence_model()
    if not records:
        return []
    feats = pd.DataFrame([r.ident.features for r in records])
    p = model.predict(feats) if model is not None else np.ones(len(records))
    keep = [
        (r, float(pi))
        for r, pi in zip(records, p, strict=True)
        if r.ident.gas is not None and pi >= threshold
    ]
    keep.sort(key=lambda t: -t[1])
    # Two detection maps can find the same plume: keep the more confident record
    taken = np.zeros(prep.scene.shape, dtype=bool)
    occupied = np.zeros(prep.scene.shape, dtype=bool)
    for r in records:
        occupied[r.candidate.rows, r.candidate.cols] = True
    occupied = ndimage.binary_dilation(occupied, iterations=3)
    z = {g: column_z(smooth_nan(e.alpha.astype(np.float64), 1.0)) for g, e in enh.items()}
    qcfg = cfg["quantify"]
    t = scene_time(prep)
    plumes = []
    for r, pi in keep:
        c = r.candidate
        if taken[c.rows, c.cols].mean() > 0.5:
            continue
        taken[c.rows, c.cols] = True
        gas = r.ident.gas
        # Quantify on the named gas's own map (a CH4 plume found in the CO2 map, say)
        g_map = gas if gas in enh else r.detection_gas
        w = wind
        if w is None:
            lat, lon = prep.scene.lat[c.rows, c.cols].mean(), prep.scene.lon[c.rows, c.cols].mean()
            try:
                w = wind_at(float(lat), float(lon), t, min_relative_sd=qcfg["wind_min_relative_sd"])
            except Exception:
                w = None
        quant = None
        if w is not None:
            quant = quantify(
                enh[g_map].alpha,
                z[g_map],
                c,
                gas,
                prep.scene.lat,
                prep.scene.lon,
                prep.conditions.surface_km,
                w,
                occupied,
                tuple(qcfg["u_eff"]),
                cfg["candidates"]["z_low"],
                retrieval_rel_sd=qcfg["retrieval_rel_sd"],
            )
        plumes.append(Plume(len(plumes) + 1, r, pi, quant))
    return plumes
