"""EMIT L2B greenhouse-gas products, used as references (EMITL2BCH4ENH/CO2ENH, CH4PLM/CO2PLM v002).

* EMIT_L2B_<GAS>ENH_..._<scene>.tif: enhancement in ppm·m on the scene's GLT grid, nodata -9999.
  <GAS>SENS and <GAS>UNCERT are the sensitivity and uncertainty on the same grid.
* EMIT_L2B_<GAS>PLM_..._<plume>.tif: one plume complex, cropped to its bounding box on the same
  grid; <GAS>PLMMETA_..._<plume>.json is its GeoJSON record (outline, maximum, emission rate).
Plume files carry the plume id instead of the scene id, so they are found by the shared
acquisition time in the file name (YYYYMMDDTHHMMSS).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from specret.io.emit_l1b import scene_id

NODATA = -9999.0


@dataclass
class Raster:
    data: np.ndarray  # float32, NaN for nodata
    transform: rasterio.Affine
    crs: object


def read_raster(path: Path) -> Raster:
    with rasterio.open(path) as ds:
        v = ds.read(1).astype(np.float32)
        nodata = ds.nodata if ds.nodata is not None else NODATA
        v[v == nodata] = np.nan
        return Raster(v, ds.transform, ds.crs)


def enhancement_path(folder: Path, scene: str, gas: str = "CH4", kind: str = "ENH") -> Path | None:
    hits = sorted(folder.glob(f"EMIT_L2B_{gas}{kind}_*{scene}*.tif"))
    return hits[0] if hits else None


def plume_paths(folder: Path, scene: str, gas: str = "CH4") -> list[Path]:
    stamp = scene.split("_")[0]
    return sorted(folder.glob(f"EMIT_L2B_{gas}PLM_*_{stamp}_*.tif"))


def plume_mask(folder: Path, scene: str, like: Raster, gas: str = "CH4") -> np.ndarray:
    """Union of the scene's plume complexes on the grid of `like` (an enhancement raster)."""
    out = np.zeros(like.data.shape, dtype=bool)
    res = like.transform.a
    for p in plume_paths(folder, scene, gas):
        r = read_raster(p)
        c0 = int(round((r.transform.c - like.transform.c) / res))
        r0 = int(round((like.transform.f - r.transform.f) / res))
        h, w = r.data.shape
        sub = out[max(r0, 0) : r0 + h, max(c0, 0) : c0 + w]
        src = np.isfinite(r.data) & (r.data > 0)
        sub |= src[
            max(-r0, 0) : max(-r0, 0) + sub.shape[0], max(-c0, 0) : max(-c0, 0) + sub.shape[1]
        ]
    return out


def plume_records(folder: Path, scene: str, gas: str = "CH4") -> list[dict]:
    """Properties of each plume complex (GeoJSON features), with the plume id."""
    stamp = scene.split("_")[0]
    out = []
    for p in sorted(folder.glob(f"EMIT_L2B_{gas}PLMMETA_*_{stamp}_*.json")):
        for f in json.loads(p.read_text()).get("features", []):
            out.append({"file": p.name, **f.get("properties", {})})
    return out


def scene_folder(l1b_path: Path) -> tuple[Path, str]:
    return l1b_path.parent, scene_id(l1b_path)
