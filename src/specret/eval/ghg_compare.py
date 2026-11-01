"""Compare our enhancement maps with EMIT L2B CH4ENH/CO2ENH (Phase G1 exit check).

Both maps are on the scene's GLT grid. Agreement is measured where both have data:
* plume pixels (the union of the scene's L2B plume complexes): Pearson r, OLS slope and
  intercept of ours against EMIT's, mean difference, and the Deming slope. Both maps are
  noisy, so OLS underestimates the slope when EMIT's map is the noisier one; Deming
  regression uses the ratio of the two background variances and is the slope to judge by;
* background (everything outside the plumes, dilated by a few pixels): robust spread of each
  map and the correlation over the whole scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from specret.io.emit_ghg import enhancement_path, plume_mask, read_raster

ROBUST = 1.4826
PLUME_MARGIN_PX = 5


def deming_slope(x: np.ndarray, y: np.ndarray, noise_ratio: float) -> float:
    """Slope of y on x when both are noisy; noise_ratio = var(noise in y) / var(noise in x)."""
    sxx, syy = np.var(x), np.var(y)
    sxy = np.cov(x, y, bias=True)[0, 1]
    d = syy - noise_ratio * sxx
    return float((d + np.sqrt(d * d + 4 * noise_ratio * sxy * sxy)) / (2 * sxy))


def _fit(x: np.ndarray, y: np.ndarray, noise_ratio: float | None = None) -> dict:
    if x.size < 3:
        return {"n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    extra = {}
    if noise_ratio is not None and np.isfinite(noise_ratio):
        extra["deming_slope"] = deming_slope(x, y, noise_ratio)
    return {
        **extra,
        "n": int(x.size),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_diff": float(np.mean(y - x)),
    }


def robust_sigma(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(ROBUST * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def compare(ours_path: Path, scene_folder: Path, scene: str, gas: str = "CH4") -> dict:
    ref_path = enhancement_path(scene_folder, scene, gas)
    if ref_path is None:
        raise ValueError(f"no EMIT L2B {gas}ENH for {scene} in {scene_folder}")
    ours, ref = read_raster(ours_path), read_raster(ref_path)
    if ours.data.shape != ref.data.shape:
        raise ValueError(f"grids differ: ours {ours.data.shape}, EMIT {ref.data.shape}")
    both = np.isfinite(ours.data) & np.isfinite(ref.data)
    plume = plume_mask(scene_folder, scene, ref, gas)
    near = ndimage.binary_dilation(plume, iterations=PLUME_MARGIN_PX)
    x, y = ref.data, ours.data
    p, b = both & plume, both & ~near
    sig_ours, sig_emit = robust_sigma(y[b]), robust_sigma(x[b])
    return {
        "scene": scene,
        "gas": gas,
        "plume_pixels": _fit(x[p], y[p], (sig_ours / sig_emit) ** 2),
        "scene_pixels": _fit(x[both], y[both]),
        "background_sigma_ours": sig_ours,
        "background_sigma_emit": sig_emit,
        "background_median_ours": float(np.median(y[b])),
        "background_median_emit": float(np.median(x[b])),
        "valid_pixels_ours": int(np.isfinite(y).sum()),
        "valid_pixels_emit": int(np.isfinite(x).sum()),
    }
