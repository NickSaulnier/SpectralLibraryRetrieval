"""Sample clean real pixels from EMIT scenes for encoder pretraining (PLAN.md, Phase 5b).

Pixels flagged by the L2A mask granule's aggregate flag (cloud, cirrus, water, spacecraft,
dilated cloud) are dropped, as are fill values. Sampling is spatially stratified: the swath is
cut into tiles and each tile contributes equally, so large uniform areas do not dominate.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from specret.io.emit import FILL_VALUE, read_band_parameters

AGGREGATE_FLAG = "Aggregate Flag"


def clear_mask(mask_path: str | Path) -> np.ndarray:
    """(rows, cols) True where EMIT's aggregate quality flag is clear."""
    with netCDF4.Dataset(mask_path) as ds:
        names = [str(n) for n in ds["sensor_band_parameters"]["mask_bands"][:]]
        band = names.index(AGGREGATE_FLAG)
        flag = np.ma.filled(ds["mask"][:, :, band], 1.0)
    return flag == 0


def sample_pixels(
    rfl_path: str | Path,
    mask_path: str | Path | None,
    n: int,
    tile: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """(n', bands) float16 reflectance of clear pixels (n' <= n), spatially stratified."""
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(rfl_path) as ds:
        refl = ds["reflectance"]
        rows, cols, _ = refl.shape
        ok = clear_mask(mask_path) if mask_path else np.ones((rows, cols), bool)
        tiles = [(r, c) for r in range(0, rows, tile) for c in range(0, cols, tile)]
        per_tile = max(1, n // len(tiles))
        picks = []
        for r, c in tiles:
            rr, cc = np.nonzero(ok[r : r + tile, c : c + tile])
            if rr.size:
                k = rng.choice(rr.size, min(per_tile, rr.size), replace=False)
                picks.append(np.column_stack([rr[k] + r, cc[k] + c]))
        picks = np.vstack(picks)
        rng.shuffle(picks)
        picks = picks[:n]
        out = np.empty((len(picks), refl.shape[2]), dtype=np.float16)
        order = np.argsort(picks[:, 0])
        for start in range(0, len(order), 4096):  # read row-sorted chunks
            idx = order[start : start + 4096]
            r0, r1 = picks[idx, 0].min(), picks[idx, 0].max() + 1
            block = np.ma.filled(refl[r0:r1, :, :], FILL_VALUE)
            out[idx] = block[picks[idx, 0] - r0, picks[idx, 1]]
    out = out.astype(np.float32)
    out[out <= FILL_VALUE + 1] = np.nan
    keep = np.isfinite(out).any(axis=1)
    return out[keep].astype(np.float16)


def save_pool(path: str | Path, pixels: np.ndarray, rfl_path: str | Path) -> None:
    centers, fwhm, good = read_band_parameters(rfl_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pixels=pixels, centers_nm=centers, fwhm_nm=fwhm, good_mask=good)


def load_pool(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate pixel pools that share EMIT's band set: (pixels, centers, fwhm, good)."""
    parts, ref = [], None
    for p in paths:
        z = np.load(p)
        if ref is None:
            ref = (z["centers_nm"], z["fwhm_nm"], z["good_mask"])
        elif not np.allclose(z["centers_nm"], ref[0], atol=0.5):
            raise ValueError(f"{p}: band centres differ from the first pool")
        parts.append(z["pixels"])
    return np.vstack(parts), *ref
