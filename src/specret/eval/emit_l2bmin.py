"""Compare retrieval with the EMIT L2B mineral product (EMITL2BMIN) over a region.

L2B MIN comes from Tetracorder run on the same USGS library family (splib06), so agreement
measures agreement with Tetracorder, not ground truth (PLAN.md, Decision 7).

The product has two layers: group 1 (~1 um Fe-bearing features) and group 2 (~2 um features
of clays, micas, carbonates and sulfates). Agreement is measured at group and class level.
Groups and classes come from configs/taxonomy.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from specret.search.retrieve import Retriever
from specret.taxonomy import load_taxonomy


def coarse_group(name: str | None) -> str:
    """Taxonomy group of a USGS-style name such as 'Calcite+.33Muscov AMX5'."""
    return load_taxonomy().group_of(name)


def coarse_class(name: str | None) -> str:
    return load_taxonomy().classify(name)[1]


@dataclass
class L2BMin:
    group1_id: np.ndarray  # (rows, cols) int, 0 = no detection
    group1_depth: np.ndarray
    group2_id: np.ndarray
    group2_depth: np.ndarray
    names: dict[int, str]  # mineral index -> name
    lat: np.ndarray
    lon: np.ndarray


def load_l2bmin(path: str | Path) -> L2BMin:
    with netCDF4.Dataset(path) as ds:
        mm = ds["mineral_metadata"]
        names = {int(i): str(n) for i, n in zip(mm["index"][:], mm["name"][:], strict=True)}

        def arr(v, fill):
            return np.ma.filled(ds[v][:], fill)

        return L2BMin(
            group1_id=arr("group_1_mineral_id", 0).astype(int),
            group1_depth=arr("group_1_band_depth", 0.0).astype(float),
            group2_id=arr("group_2_mineral_id", 0).astype(int),
            group2_depth=arr("group_2_band_depth", 0.0).astype(float),
            names=names,
            lat=ds["location"]["lat"][:],
            lon=ds["location"]["lon"][:],
        )


def compare_group2(
    scene,
    l2b: L2BMin,
    retriever: Retriever,
    bbox: tuple[float, float, float, float],
    min_depth: float = 0.05,
    k: int = 5,
    max_pixels: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per pixel in `bbox` where L2B MIN reports a ~2 um mineral above `min_depth`.

    `bbox` None means the whole scene; `max_pixels` takes a fixed random sample.
    """
    sel = (l2b.group2_id > 0) & (l2b.group2_depth >= min_depth)
    if bbox is not None:
        west, south, east, north = bbox
        sel &= (l2b.lon > west) & (l2b.lon < east) & (l2b.lat > south) & (l2b.lat < north)
    rows, cols = np.nonzero(sel)
    if max_pixels is not None and rows.size > max_pixels:
        keep = np.sort(np.random.default_rng(seed).choice(rows.size, max_pixels, replace=False))
        rows, cols = rows[keep], cols[keep]
    if rows.size == 0:
        return pd.DataFrame()

    spectra = np.empty((rows.size, len(scene.centers_nm)))
    for r0 in range(rows.min(), rows.max() + 1, 128):  # read in row bands to bound memory
        m = (rows >= r0) & (rows < r0 + 128)
        if m.any():
            c0, c1 = cols[m].min(), cols[m].max() + 1
            block = scene.block(slice(r0, min(r0 + 128, rows.max() + 1)), slice(c0, c1))
            spectra[m] = block[rows[m] - r0, cols[m] - c0]
    matches = retriever.query(spectra, k=k)

    records = []
    for r, c, m in zip(rows, cols, matches, strict=True):
        ref_name = l2b.names.get(int(l2b.group2_id[r, c]), "")
        top = m[0] if m else {}
        records.append(
            {
                "row": int(r),
                "col": int(c),
                "l2b_name": ref_name,
                "l2b_group": coarse_group(ref_name),
                "l2b_depth": float(l2b.group2_depth[r, c]),
                "top1_id": top.get("spectrum_id"),
                "top1_name": top.get("material_name"),
                "top1_purity": top.get("purity_flag"),
                "top1_group": top.get("group") or coarse_group(top.get("material_name")),
                "l2b_class": coarse_class(ref_name),
                "top1_class": top.get("material_class") or coarse_class(top.get("material_name")),
                "top1_sam_deg": top.get("sam_deg", np.nan),
                "topk_groups": [x.get("group") or coarse_group(x["material_name"]) for x in m],
            }
        )
    df = pd.DataFrame(records)
    df["top1_agree"] = df["top1_group"] == df["l2b_group"]
    df["top1_class_agree"] = df["top1_class"] == df["l2b_class"]
    df["topk_agree"] = [g in ks for g, ks in zip(df["l2b_group"], df["topk_groups"], strict=True)]
    return df


def summarize(df: pd.DataFrame, min_pixels: int = 50) -> pd.DataFrame:
    """Per L2B group: pixel count, top-1 and top-k group agreement, median SAM angle."""
    g = df.groupby("l2b_group")
    out = pd.DataFrame(
        {
            "pixels": g.size(),
            "top1_agree": g["top1_agree"].mean(),
            "top1_class_agree": g["top1_class_agree"].mean(),
            "topk_agree": g["topk_agree"].mean(),
            "median_sam_deg": g["top1_sam_deg"].median(),
            "most_common_top1": g["top1_group"].agg(lambda s: s.value_counts().index[0]),
        }
    )
    return out[out["pixels"] >= min_pixels].sort_values("pixels", ascending=False)


def unmix_group2(
    scene,
    l2b: L2BMin,
    unmixer,
    bbox: tuple[float, float, float, float],
    min_depth: float = 0.05,
    per_group: int = 300,
    noise_sigma: np.ndarray | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Unmix a random sample of L2B ~2 um pixels per group; compare top-1 and any-component."""
    west, south, east, north = bbox
    in_box = (l2b.lon > west) & (l2b.lon < east) & (l2b.lat > south) & (l2b.lat < north)
    sel = in_box & (l2b.group2_id > 0) & (l2b.group2_depth >= min_depth)
    rows, cols = np.nonzero(sel)
    ref = np.array([coarse_group(l2b.names.get(int(i), "")) for i in l2b.group2_id[rows, cols]])
    rng = np.random.default_rng(seed)
    records = []
    for g in np.unique(ref):
        idx = np.flatnonzero(ref == g)
        for i in rng.choice(idx, min(per_group, idx.size), replace=False):
            x = scene.pixel(int(rows[i]), int(cols[i]))
            u = unmixer.unmix(x, noise_sigma)
            if u is None:
                continue
            comp = [unmixer.groups[r] for r in u.rows]
            records.append(
                {
                    "l2b_group": g,
                    "top1_group": comp[0],
                    "components": comp,
                    "n_components": len(comp),
                    "top1_agree": comp[0] == g,
                    "any_agree": g in comp,
                }
            )
    return pd.DataFrame(records)
