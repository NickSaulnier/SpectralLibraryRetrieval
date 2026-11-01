"""Surface materials that can mimic gas absorption, and the mineral-map screen (PLAN_GAS.md G2).

Carbonates absorb near 2.3 µm (calcite 2.34, dolomite 2.32), as do hydrocarbons and many
plastics (C-H at 1.7 and 2.3 µm). A field or rooftop of such material can look like a CH4
plume to a matched filter. Each confuser class gets a signature on the sensor's bands: the
continuum-removed log reflectance of its library spectra, averaged over the class. Within a
plume, more of that material changes log radiance by about beta * signature, so the
signature competes with the gas terms in the joint fit (gas/identify.py).

Independently, the paired EMIT L2B mineral map flags candidates whose pixels are mostly
carbonate at ~2 µm.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from specret.gas.library import plume_config
from specret.preprocess.continuum import continuum
from specret.sensors.library import SensorLibrary


@dataclass
class Confusers:
    names: list[str]
    centers_nm: np.ndarray
    signatures: np.ndarray  # (classes, bands); 0 outside the fit windows
    n_spectra: dict[str, int]


def continuum_removed_log(wl: np.ndarray, refl: np.ndarray, windows) -> np.ndarray:
    """ln(R) - ln(hull) within each window; NaN outside the windows or where R is invalid."""
    out = np.full(wl.size, np.nan)
    for lo, hi in windows:
        m = (wl >= lo) & (wl <= hi) & np.isfinite(refl) & (refl > 0)
        if m.sum() < 4:
            continue
        hull = continuum(wl[m], refl[m])
        out[m] = np.log(refl[m]) - np.log(hull)
    return out


def build_confusers(
    lib: SensorLibrary, canonical: pd.DataFrame, windows, config: dict | None = None
) -> Confusers:
    """One mean signature per confuser class in plumes.yaml, on the library's bands."""
    cfg = config or plume_config()["confusers"]["classes"]
    meta = canonical.set_index("spectrum_id").loc[list(lib.spectrum_ids)]
    names = meta["material_name"].astype(str).to_numpy()
    wl = lib.sensor.centers_nm
    sigs, counts = [], {}
    for cls, pattern in cfg.items():
        rows = np.flatnonzero([bool(re.fullmatch(pattern, n)) for n in names])
        crs = [continuum_removed_log(wl, lib.matrix[i], windows) for i in rows]
        full = max((np.isfinite(c).sum() for c in crs), default=0)
        crs = [c for c in crs if np.isfinite(c).sum() >= 0.8 * full]
        counts[cls] = len(crs)
        with warnings.catch_warnings():  # bands outside the windows are NaN in every spectrum
            warnings.simplefilter("ignore", RuntimeWarning)
            sigs.append(np.nan_to_num(np.nanmean(crs, axis=0)) if crs else np.zeros(wl.size))
    keep = [i for i, c in enumerate(cfg) if counts[c] > 0]
    return Confusers(
        names=[list(cfg)[i] for i in keep],
        centers_nm=wl,
        signatures=np.array([sigs[i] for i in keep]),
        n_spectra=counts,
    )


def mineral_screen(l2bmin_path: Path | None, groups: set[str] | None = None):
    """Function (rows, cols) -> fraction of a candidate's pixels whose L2B MIN ~2 µm mineral is
    a confuser group (carbonates by default). Returns None without a mineral map."""
    if l2bmin_path is None:
        return None
    from specret.eval.emit_l2bmin import coarse_group, load_l2bmin

    groups = groups or set(plume_config()["confusers"]["mineral_groups"])
    m = load_l2bmin(l2bmin_path)
    confuser_ids = np.array(
        sorted(i for i, n in m.names.items() if i > 0 and coarse_group(n) in groups)
    )
    is_conf = np.isin(m.group2_id, confuser_ids)

    def fraction(rows: np.ndarray, cols: np.ndarray) -> float:
        return float(is_conf[rows, cols].mean()) if rows.size else 0.0

    return fraction
