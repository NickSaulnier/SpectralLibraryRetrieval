"""Our gas enhancement against a published map of the same quantity, over the visible area (V4).

The reference is read onto our map grid. Plume pixels follow the recipe each comparison was
validated with: EMIT L2B plume complexes, with the background outside them (+5 px) setting
the noise ratio (G1); or, for the published AVIRIS-NG map, its own strong regions, with every
pixel setting the noise ratio (G5). Masks are made on the whole grid and then cut to the view,
so a view of the whole scene reproduces the published agreement numbers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

from specret.eval.ghg_compare import (
    PLUME_MARGIN_PX,
    compare_arrays,
    published_plume_mask,
)
from specret.io.emit_ghg import plume_mask, read_raster

SAMPLE = 3000  # points returned for the scatter plot
_LOCK = threading.Lock()


@dataclass
class Pair:
    ours: np.ndarray
    ref: np.ndarray
    plume: np.ndarray
    background: np.ndarray | None
    lat: np.ndarray  # pixel centres
    lon: np.ndarray
    recipe: str


def _on_grid(ref_path: Path, like_path: Path) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(like_path) as like, rasterio.open(ref_path) as src:
        a = src.read(1).astype(np.float32)
        nodata = src.nodata
        if (src.width, src.height, src.transform) == (like.width, like.height, like.transform):
            out = a
        else:
            out = np.full((like.height, like.width), np.nan, dtype=np.float32)
            reproject(
                a,
                out,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=like.transform,
                dst_crs=like.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
    if nodata is not None:
        out = np.where(out == nodata, np.nan, out)
    return out


def _centres(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import rasterio
    from rasterio.warp import transform as warp

    with rasterio.open(path) as ds:
        rows, cols = np.mgrid[0 : ds.height, 0 : ds.width]
        xs, ys = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel())
        if ds.crs and not ds.crs.is_geographic:
            xs, ys = warp(ds.crs, "EPSG:4326", xs, ys)
        shape = (ds.height, ds.width)
    return np.asarray(ys).reshape(shape), np.asarray(xs).reshape(shape)


@lru_cache(maxsize=4)
def load_pair(ours_path: Path, ref_path: Path, ref_name: str, scene: str, gas: str) -> Pair:
    with _LOCK:
        ours = read_raster(ours_path)
        ref = _on_grid(ref_path, ours_path)
        if ref_name.startswith("emit_"):
            plume = plume_mask(ref_path.parent, scene, read_raster(ref_path), gas)
            background = ~ndimage.binary_dilation(plume, iterations=PLUME_MARGIN_PX)
            recipe = "EMIT L2B plume complexes; noise ratio from the background outside them (G1)"
        else:
            plume = published_plume_mask(ref)
            background = None
            recipe = (
                "published map smoothed z > 4 in regions of 30+ px; "
                "noise ratio from all pixels (G5)"
            )
        lat, lon = _centres(ours_path)
        return Pair(ours.data, ref, plume, background, lat, lon, recipe)


def compare_view(pair: Pair, bbox: tuple[float, float, float, float] | None, seed: int = 0) -> dict:
    """Agreement inside bbox (west, south, east, north), or over the whole grid, plus a
    sample of points for a scatter plot (plume pixels first)."""
    view = np.ones(pair.ours.shape, dtype=bool)
    if bbox is not None:
        w, s, e, n = bbox
        view = (pair.lon >= w) & (pair.lon <= e) & (pair.lat >= s) & (pair.lat <= n)
    ours = np.where(view, pair.ours, np.nan)
    bg = pair.background
    stats = compare_arrays(ours, pair.ref, pair.plume & view, bg)
    both = view & np.isfinite(pair.ours) & np.isfinite(pair.ref)
    rng = np.random.default_rng(seed)
    p_idx = np.flatnonzero(both & pair.plume)
    b_idx = np.flatnonzero(both & ~pair.plume)
    p_take = rng.choice(p_idx, min(p_idx.size, SAMPLE // 2), replace=False)
    b_take = rng.choice(b_idx, min(b_idx.size, SAMPLE - p_take.size), replace=False)
    pts = lambda idx: {  # noqa: E731
        "ref": pair.ref.ravel()[idx].round(1).tolist(),
        "ours": pair.ours.ravel()[idx].round(1).tolist(),
    }
    return {
        **stats,
        "recipe": pair.recipe,
        "view_pixels": int(both.sum()),
        "sample": {"plume": pts(p_take), "background": pts(b_take)},
    }
