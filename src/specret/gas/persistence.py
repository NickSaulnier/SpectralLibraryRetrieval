"""Multi-overpass persistence: a source seen again on other days is more likely real (G5).

For each plume in one scene, the other overpasses are checked at the plume's source location:
* covered: the location lies inside the other scene's valid data (within `radius_m` of a pixel
  that has an enhancement value);
* seen: the other scene has a plume, or a candidate named with the same gas, whose source or
  centroid lies within `radius_m`.
Persistence is seen / covered. Emissions are intermittent, so absence on another day is weak
evidence; repeated presence is strong evidence against a surface artifact, which would also
repeat, unless the artifact is a confuser the fit already named.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8


def _xy(lat, lon, lat0, lon0):
    k = np.cos(np.radians(lat0))
    return (
        np.radians(np.asarray(lon) - lon0) * k * EARTH_RADIUS_M,
        np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M,
    )


@dataclass
class SceneResult:
    scene_id: str
    plumes: pd.DataFrame  # plumes.csv
    candidates: pd.DataFrame  # candidates.csv, with lat/lon
    footprint: tuple[np.ndarray, np.ndarray]  # lat, lon of pixels with valid enhancement


def load_scene(folder: Path, scene) -> SceneResult:
    """Read a scene's outputs; `scene` (EmitL1B) gives candidate coordinates and footprint."""
    plumes = (
        pd.read_csv(folder / "plumes.csv")
        if (folder / "plumes.csv").stat().st_size
        else pd.DataFrame()
    )
    cands = pd.read_csv(folder / "candidates.csv")
    r = cands["row"].round().astype(int).clip(0, scene.shape[0] - 1)
    c = cands["col"].round().astype(int).clip(0, scene.shape[1] - 1)
    cands["lat"], cands["lon"] = scene.lat[r, c], scene.lon[r, c]
    step = 4  # a sparse footprint is enough at a few hundred metres
    import rasterio

    with rasterio.open(folder / "ch4_enh.tif") as ds:
        ok = ds.read(1)[::step, ::step] != ds.nodata
        rows, cols = np.nonzero(ok)
        xs, ys = rasterio.transform.xy(ds.transform, rows * step, cols * step)
    return SceneResult(scene.scene_id, plumes, cands, (np.asarray(ys), np.asarray(xs)))


def persistence(results: list[SceneResult], radius_m: float = 300.0) -> pd.DataFrame:
    """One row per plume of every scene, with how many other scenes covered and saw it."""
    rows = []
    for a in results:
        for _, p in a.plumes.iterrows():
            lat0, lon0 = p["source_lat"], p["source_lon"]
            covered = seen = 0
            seen_in = []
            for b in results:
                if b.scene_id == a.scene_id or b.scene_id[:15] == a.scene_id[:15]:
                    continue  # same overpass (adjacent scenes of one orbit) is not a repeat
                x, y = _xy(*b.footprint, lat0, lon0)
                if not len(x) or np.min(np.hypot(x, y)) > radius_m:
                    continue
                covered += 1
                pts = [b.candidates.loc[b.candidates["gas"] == p["gas"], ["lat", "lon"]]]
                if len(b.plumes):
                    bp = b.plumes[b.plumes["gas"] == p["gas"]]
                    pts.append(bp[["source_lat", "source_lon"]].set_axis(["lat", "lon"], axis=1))
                pts = pd.concat(pts)
                if len(pts):
                    px, py = _xy(pts["lat"], pts["lon"], lat0, lon0)
                    if np.min(np.hypot(px, py)) <= radius_m:
                        seen += 1
                        seen_in.append(b.scene_id)
            rows.append(
                {
                    "scene": a.scene_id,
                    "plume_id": p["plume_id"],
                    "gas": p["gas"],
                    "confidence": p["confidence"],
                    "source_lat": lat0,
                    "source_lon": lon0,
                    "q_kg_h": p.get("q_kg_h"),
                    "overpasses_covering": covered,
                    "overpasses_seen": seen,
                    "persistence": seen / covered if covered else np.nan,
                    "seen_in": ";".join(seen_in),
                }
            )
    return pd.DataFrame(rows)
