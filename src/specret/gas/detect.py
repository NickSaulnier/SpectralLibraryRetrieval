"""Gas enhancement maps from EMIT L1B radiance (Phase G1: CH4 matched filter).

Output per scene in outputs/gas/<scene id>/:
* <gas>_enh.tif     float32 enhancement (ppm·m) on the GLT map grid, nodata -9999
* <gas>_enh.json    conditions, unit absorption spectrum, background spread and timing
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from scipy import ndimage

from specret.gas.library import build_window, lbl_ranges, plume_config
from specret.gas.matched_filter import MatchedFilterResult, column_matched_filter
from specret.gas.signatures import Conditions, window_absorption
from specret.io.emit_ghg import NODATA
from specret.io.emit_l1b import EmitL1B, open_emit_l1b
from specret.scene.outputs import _write

OUT_DIR = Path("outputs/gas")
DEFAULT_H2O = 1.5  # g/cm^2, when the scene has no L2A mask
AMF_NODES = 5  # unit absorption spectra are computed at this many AMFs and interpolated
FLARE_NM = 2389.486


@dataclass
class Enhancement:
    gas: str
    scene_id: str
    alpha: np.ndarray  # (rows, cols) swath geometry, ppm·m
    result: MatchedFilterResult
    conditions: Conditions
    band_centers_nm: np.ndarray
    k: np.ndarray  # (bands,) unit absorption at the scene's mean AMF, per ppm·m
    screened: int  # cloud, water and flare pixels left out
    seconds: float


def column_targets(pieces, centers, fwhm, base: Conditions, amf_col: np.ndarray, gas: str):
    """Unit absorption spectra per column, interpolated between a few AMFs."""
    lo, hi = np.nanmin(amf_col), np.nanmax(amf_col)
    nodes = np.linspace(lo, hi, AMF_NODES) if hi - lo > 1e-3 else np.array([lo])
    ks = np.stack(
        [
            window_absorption(
                pieces, centers, fwhm, Conditions(float(a), base.h2o_gcm2, base.surface_km), gas
            )
            for a in nodes
        ]
    )
    if nodes.size == 1:
        return np.broadcast_to(ks[0], (amf_col.size, centers.size))
    a = np.nan_to_num(amf_col, nan=float(np.nanmean(amf_col)))
    return np.stack([np.interp(a, nodes, ks[:, b]) for b in range(centers.size)], axis=1)


def window_bands(scene: EmitL1B, windows) -> np.ndarray:
    return np.unique(np.concatenate([scene.band_indices(lo, hi) for lo, hi in windows]))


def flare_mask(scene: EmitL1B, rad: np.ndarray, idx: np.ndarray, threshold: float, px: int):
    """Gas flares: radiance at 2389 nm above `threshold`, dilated by `px` pixels (as EMIT GHG)."""
    j = int(np.argmin(np.abs(scene.centers_nm[idx] - FLARE_NM)))
    if abs(scene.centers_nm[idx][j] - FLARE_NM) > 10:
        return np.zeros(rad.shape[:2], dtype=bool)
    hot = np.nan_to_num(rad[:, :, j]) > threshold
    return ndimage.binary_dilation(hot, iterations=px) if px and hot.any() else hot


@dataclass
class Prepared:
    """A scene read once for gas work: radiance in the matched-filter bands and conditions."""

    scene: EmitL1B
    windows: list[list[float]]
    pieces: list  # [((lo, hi), LineByLine)] where line-by-line spectra exist
    idx: np.ndarray  # band indices into the granule
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    radiance: np.ndarray  # (rows, cols, len(idx))
    amf_col: np.ndarray  # (cols,) mean AMF per column
    conditions: Conditions
    screen: np.ndarray  # (rows, cols) cloud, water and flare pixels
    seconds: float


def prepare(scene: EmitL1B, gas: str = "CH4") -> Prepared:
    mf = plume_config()["matched_filter"]
    windows = mf["windows"][gas]
    t0 = time.perf_counter()
    pieces = [(r, build_window(*r)) for r in lbl_ranges(windows)]
    idx = window_bands(scene, windows)
    rad = scene.read_bands(idx)
    amf = scene.amf()
    elev = scene.elev_m[np.isfinite(scene.elev_m) & (scene.elev_m > -500)]
    base = Conditions(
        amf=float(np.nanmean(amf)),
        h2o_gcm2=scene.h2o_gcm2() or DEFAULT_H2O,
        surface_km=max(float(np.median(elev)) / 1e3, 0.0) if elev.size else 0.0,
    )
    # Clouds, water and flares are left out of the statistics and of the output
    screen = scene.cloud_water() | flare_mask(
        scene, rad, idx, mf["flare_radiance"], mf["flare_dilation_px"]
    )
    return Prepared(
        scene=scene,
        windows=windows,
        pieces=pieces,
        idx=idx,
        centers_nm=scene.centers_nm[idx],
        fwhm_nm=scene.fwhm_nm[idx],
        radiance=rad,
        amf_col=np.nanmean(amf, axis=0),
        conditions=base,
        screen=screen,
        seconds=time.perf_counter() - t0,
    )


def enhancement(
    prep: Prepared,
    gas: str = "CH4",
    radiance: np.ndarray | None = None,
    device: str | None = None,
) -> Enhancement:
    """Matched filter for `gas` on the prepared scene (or on `radiance`, e.g. with injected
    plumes, which must have the same shape)."""
    mf = plume_config()["matched_filter"]
    t0 = time.perf_counter()
    rad = prep.radiance if radiance is None else radiance
    k_col = column_targets(
        prep.pieces, prep.centers_nm, prep.fwhm_nm, prep.conditions, prep.amf_col, gas
    )
    res = column_matched_filter(
        rad,
        k_col,
        valid=~prep.screen,
        n_pass=mf["n_pass"],
        shrinkage=mf["shrinkage"],
        exclude_sigma=mf["exclude_sigma"],
        min_pixels_factor=mf["min_pixels_factor"],
        device=device,
    )
    res.alpha[prep.screen] = np.nan
    return Enhancement(
        gas=gas,
        scene_id=prep.scene.scene_id,
        alpha=res.alpha,
        result=res,
        conditions=prep.conditions,
        band_centers_nm=prep.centers_nm,
        k=window_absorption(prep.pieces, prep.centers_nm, prep.fwhm_nm, prep.conditions, gas),
        screened=int(prep.screen.sum()),
        seconds=prep.seconds + time.perf_counter() - t0,
    )


def detect(scene: EmitL1B, gas: str = "CH4", device: str | None = None) -> Enhancement:
    return enhancement(prepare(scene, gas), gas, device=device)


def write_enhancement(scene: EmitL1B, enh: Enhancement, out_dir: Path = OUT_DIR) -> Path:
    folder = out_dir / scene.scene_id
    folder.mkdir(parents=True, exist_ok=True)
    ortho = scene.orthorectify(enh.alpha, fill=np.nan)
    data = np.where(np.isfinite(ortho), ortho, NODATA).astype(np.float32)
    path = folder / f"{enh.gas.lower()}_enh.tif"
    _write(path, data, Affine.from_gdal(*scene.geotransform), scene.crs_wkt, NODATA)
    sigma = enh.result.sigma[np.isfinite(enh.result.sigma)]
    meta = {
        "gas": enh.gas,
        "scene_id": enh.scene_id,
        "units": "ppm m",
        "conditions": asdict(enh.conditions),
        "band_centers_nm": enh.band_centers_nm.round(2).tolist(),
        "unit_absorption_per_ppmm": enh.k.tolist(),
        "background_sigma_ppmm_median": float(np.median(sigma)) if sigma.size else None,
        "columns_without_output": int((~np.isfinite(enh.result.sigma)).sum()),
        "pixels_excluded_as_plume": int(enh.result.excluded.sum()),
        "pixels_screened": enh.screened,
        "shrinkage_median": float(np.nanmedian(enh.result.shrinkage)),
        "seconds": round(enh.seconds, 1),
    }
    (folder / f"{enh.gas.lower()}_enh.json").write_text(json.dumps(meta, indent=1))
    return path


def detect_scene(path: str | Path, gas: str = "CH4", out_dir: Path = OUT_DIR) -> Path:
    scene = open_emit_l1b(path)
    enh = detect(scene, gas)
    return write_enhancement(scene, enh, out_dir)
