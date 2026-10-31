"""Write scene maps as cloud-optimised GeoTIFFs plus legend tables.

Layers (orthorectified with the scene's geometry lookup table):
* group.tif       uint8 taxonomy group code of the best match (0 = no data), with a colour table
* top1_row.tif    int32 library row of the best match (-1 = no data); see library.csv
* score.tif       float32 ranking score of the best match, 0-1
* margin.tif      float32 score gap to the best candidate of another group (a confidence cue)
* confidence.tif  float32 calibrated P(best group is right), when the map has it
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.shutil import copy as raster_copy

from specret.scene.mapper import SceneMap


def group_codes(groups: list[str]) -> dict[str, int]:
    """Stable codes 1..n in sorted order; 0 is reserved for no data."""
    return {g: i + 1 for i, g in enumerate(sorted(set(groups) - {""}))}


def group_colors(codes: dict[str, int]) -> dict[int, tuple[int, int, int, int]]:
    """Distinct, evenly spaced hues for the colour table."""
    out = {0: (0, 0, 0, 0)}
    for c in codes.values():
        h = ((c - 1) * 0.618033988749895) % 1.0  # golden-ratio hue spacing
        r, gg, b = colorsys.hsv_to_rgb(h, 0.65, 0.95 if c % 2 else 0.75)
        out[c] = (int(r * 255), int(gg * 255), int(b * 255), 255)
    return out


def _write(path: Path, data: np.ndarray, transform, crs, nodata, colormap=None) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    # The COG driver cannot be written to directly: build a GTiff in memory, then copy.
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(data, 1)
            if colormap:
                tmp.write_colormap(1, colormap)
        with mem.open() as src:
            raster_copy(src, path, driver="COG", compress="DEFLATE")


def write_scene_maps(
    smap: SceneMap, scene, canonical: pd.DataFrame, out_dir: str | Path
) -> dict[str, Path]:
    """Write GeoTIFF layers and legend CSVs; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = group_codes(list(smap.library_groups))
    transform = Affine.from_gdal(*scene.geotransform) if scene.geotransform else Affine.identity()
    crs = scene.crs_wkt or "EPSG:4326"

    top1 = smap.top_rows[..., 0]
    group_code = np.zeros(smap.shape, dtype=np.uint8)
    valid = top1 >= 0
    group_code[valid] = [codes.get(g, 0) for g in smap.library_groups[top1[valid]]]

    layers = {
        "group": (scene.orthorectify(group_code, fill=0).astype(np.uint8), 0, group_colors(codes)),
        "top1_row": (scene.orthorectify(top1, fill=-1).astype(np.int32), -1, None),
        "score": (scene.orthorectify(smap.scores[..., 0]).astype(np.float32), np.nan, None),
        "margin": (scene.orthorectify(smap.margin).astype(np.float32), np.nan, None),
    }
    if smap.p_correct is not None:
        conf = scene.orthorectify(smap.p_correct).astype(np.float32)
        layers["confidence"] = (conf, np.nan, None)
    paths = {}
    for name, (data, nodata, cmap) in layers.items():
        p = out_dir / f"{name}.tif"
        _write(p, data, transform, crs, nodata, cmap)
        paths[name] = p

    meta = canonical.set_index("spectrum_id")
    lib = meta.loc[smap.library_ids, ["material_name", "group", "material_class", "purity_flag"]]
    lib = lib.reset_index().rename_axis("row").reset_index()
    lib.to_csv(out_dir / "library.csv", index=False)
    colors = group_colors(codes)
    legend = pd.DataFrame(
        [
            {"code": c, "group": g, "rgba": colors[c]}
            for g, c in sorted(codes.items(), key=lambda x: x[1])
        ]
    )
    legend.to_csv(out_dir / "groups.csv", index=False)
    paths["library"] = out_dir / "library.csv"
    paths["groups"] = out_dir / "groups.csv"
    return paths
