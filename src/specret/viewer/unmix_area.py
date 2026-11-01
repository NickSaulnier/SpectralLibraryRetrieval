"""Unmix the pixels of a drawn area into library endmembers (PLAN_VIEWER.md, Phase V2).

Unmixing costs about 13 ms per EMIT pixel, about 6 hours for a whole scene, so it runs on
demand for an area rather than as a whole-scene product. The pixels inside the polygon are
taken in swath geometry. Above `max_pixels`, a regular grid of every `step`-th row and column
is unmixed and each result fills its step x step block, so the layer covers the area at a
coarser resolution and says so.

Outputs, in outputs/viewer/<scene key>/unmix/ (each run replaces the last):
* <group>.tif     fraction of each endmember group (0-1) on the map grid, NaN outside the area
* residual.tif    RMS reflectance residual of the fit
* meta.json       polygon, pixels unmixed, step, seconds, mean fraction per group
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from specret.viewer.layers import VIEWER_DIR, _write_cog


def unmix_dir(ws, key: str) -> Path:
    return ws.root / VIEWER_DIR / key / "unmix"


def area_pixels(scene, polygon) -> tuple[np.ndarray, np.ndarray]:
    from matplotlib.path import Path as MplPath

    xs, ys = zip(*polygon, strict=True)
    lat, lon = scene.lat, scene.lon
    box = (lon >= min(xs)) & (lon <= max(xs)) & (lat >= min(ys)) & (lat <= max(ys))
    r, c = np.nonzero(box)
    inside = MplPath(polygon).contains_points(np.column_stack([lon[r, c], lat[r, c]]))
    return r[inside], c[inside]


def unmix_area(
    ws, key: str, mineral_scene: str, polygon, max_pixels: int = 3000, progress=None
) -> dict:
    from affine import Affine

    t0 = time.perf_counter()
    scene = ws.scene(mineral_scene)
    sensor = ws.sensor_for_scene(scene)
    unmixer = ws.unmixer(sensor)
    noise = ws.noise_curve(sensor)
    rows, cols = area_pixels(scene, polygon)
    if not rows.size:
        raise ValueError("the area holds no pixels of this scene")
    step = 1
    while rows.size > max_pixels * step * step:
        step += 1
    keep = (rows % step == 0) & (cols % step == 0)
    rows, cols = rows[keep], cols[keep]
    shape = scene.shape
    fractions: dict[str, np.ndarray] = {}
    residual = np.full(shape, np.nan, dtype=np.float32)
    inside = np.zeros(shape, dtype=bool)
    ar, ac = area_pixels(scene, polygon)
    inside[ar, ac] = True
    for i, (r, c) in enumerate(zip(rows, cols, strict=True)):
        res = unmixer.unmix(scene.pixel(int(r), int(c)), noise)
        block = (slice(r, r + step), slice(c, c + step))
        if res is not None:
            d = unmixer.describe(res, ws.canonical)
            by_group: dict[str, float] = {}
            for comp in d["components"]:
                g = comp.get("group") or "other"
                by_group[g] = by_group.get(g, 0.0) + float(comp["fraction"])
            for g, f in by_group.items():
                fractions.setdefault(g, np.full(shape, np.nan, dtype=np.float32))
                fractions[g][block] = f
            for g, a in fractions.items():
                if g not in by_group:
                    a[block] = 0.0
            residual[block] = float(d["residual_rms"])
        if progress and i % 50 == 0:
            progress(i / rows.size)
    for a in (*fractions.values(), residual):
        a[~inside] = np.nan
    out = unmix_dir(ws, key)
    if out.exists():
        for f in out.glob("*"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    transform = Affine.from_gdal(*scene.geotransform)
    for g, a in fractions.items():
        _write_cog(out / f"{g}.tif", scene.orthorectify(a), transform, scene.crs_wkt, np.nan)
    _write_cog(out / "residual.tif", scene.orthorectify(residual), transform, scene.crs_wkt, np.nan)
    means = {g: float(np.nanmean(a[inside])) for g, a in fractions.items()}
    meta = {
        "polygon": [list(p) for p in polygon],
        "pixels_in_area": int(inside.sum()),
        "pixels_unmixed": int(rows.size),
        "step": step,
        "seconds": round(time.perf_counter() - t0, 1),
        "mean_fraction": dict(sorted(means.items(), key=lambda t: -t[1])),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta
