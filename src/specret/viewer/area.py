"""Summaries of a user-drawn area (PLAN_VIEWER.md, Phase V2).

Mineral composition is computed in swath geometry from the scene map, exactly as the MCP
`scene_summary` tool does (scene/mapper.group_fractions), so the two always agree. Continuous
layers are summarized on their own map grid (rasterio mask), and plumes count when their
source point lies inside the area.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def polygon_area_km2(poly: list[tuple[float, float]]) -> float:
    """Area of a small lon/lat polygon (shoelace on a local equirectangular projection)."""
    lon, lat = np.asarray(poly, dtype=np.float64).T
    k = np.cos(np.radians(lat.mean()))
    x = np.radians(lon) * k * EARTH_RADIUS_KM
    y = np.radians(lat) * EARTH_RADIUS_KM
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


@lru_cache(maxsize=4)
def _scene_map(path: str, mtime: int):
    from specret.scene.mapper import SceneMap

    return SceneMap.load(path)


def mineral_composition(ws, mineral_scene: str, poly, min_p: float) -> dict:
    from specret.scene.mapper import group_fractions

    path = ws.map_dir(mineral_scene) / "scene_map.npz"
    smap = _scene_map(str(path), path.stat().st_mtime_ns)
    scene = ws.scene(mineral_scene)
    df = group_fractions(
        smap, getattr(scene, "lat", None), getattr(scene, "lon", None), None, min_p, poly
    )
    return {
        "pixels": int(df["pixels"].sum()),
        "groups": [
            {
                "group": str(g),
                "pixels": int(r.pixels),
                "fraction": float(r.fraction),
                "median_p": None if np.isnan(r.median_p) else float(r.median_p),
            }
            for g, r in df.iterrows()
        ],
    }


def layer_stats(layer, poly, bins: int = 20) -> dict | None:
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import transform_geom

    geom = {"type": "Polygon", "coordinates": [[*poly, poly[0]]]}
    with rasterio.open(layer.path) as ds:
        try:
            g = transform_geom("EPSG:4326", ds.crs, geom)
            data, _ = mask(ds, [g], crop=True, filled=False, indexes=1)
        except ValueError:  # the polygon misses the raster
            return None
    v = np.ma.filled(data.astype(np.float64), np.nan).ravel()
    v = v[np.isfinite(v)]
    lo, hi = layer.rescale
    counts, edges = (
        np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
        if v.size
        else (
            np.zeros(bins, int),
            np.linspace(lo, hi, bins + 1),
        )
    )

    def stat(f):
        return float(f(v)) if v.size else None

    return {
        "id": layer.id,
        "title": layer.title,
        "units": layer.units,
        "pixels": int(v.size),
        "mean": stat(np.mean),
        "median": stat(np.median),
        "p05": stat(lambda a: np.percentile(a, 5)),
        "p95": stat(lambda a: np.percentile(a, 95)),
        "hist_edges": [float(e) for e in edges],
        "hist_counts": [int(c) for c in counts],
    }


def plumes_inside(gas_dir: Path | None, poly) -> dict:
    from matplotlib.path import Path as MplPath

    out = {"count": 0, "by_gas": {}, "ids": []}
    path = gas_dir / "plumes.csv" if gas_dir else None
    if path is None or not path.exists() or not path.stat().st_size:
        return out
    df = pd.read_csv(path)
    if df.empty or "source_lat" not in df:
        return out
    pts = df[["source_lon", "source_lat"]].to_numpy(dtype=np.float64)
    ok = np.isfinite(pts).all(axis=1)
    inside = np.zeros(len(df), dtype=bool)
    inside[ok] = MplPath(poly).contains_points(pts[ok])
    sub = df[inside]
    out["count"] = int(len(sub))
    out["ids"] = [int(i) for i in sub["plume_id"]]
    for gas, g in sub.groupby("gas"):
        q = g["q_kg_h"].dropna() if "q_kg_h" in g else pd.Series(dtype=float)
        out["by_gas"][str(gas)] = {"count": float(len(g)), "total_q_kg_h": float(q.sum())}
    return out
