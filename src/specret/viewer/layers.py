"""What the viewer's map can show, per scene (PLAN_VIEWER.md, Phase V1).

A *scene* here is one place and time, under one key, whatever produced its layers:
* EMIT: the scene id (20230608T162322_2315911_014), shared by its L2A mineral map and its L1B
  gas outputs;
* AVIRIS: the header stem of the reflectance cube;
* AVIRIS-NG: the flight line id (ang20161001t205456).

A *layer* is one raster (a COG) plus how to draw it: categorical with an embedded colour table,
or continuous with a colormap and a stretch. Layers come from:
* Minerals: outputs/maps/<scene>/group, confidence, score, margin;
* Gas: outputs/gas/<key>/ch4_enh, co2_enh, confidence (plume outlines are vector, plumes.geojson);
* Reference: EMIT L2B CH4ENH/CO2ENH, the published AVIRIS-NG CH4 map, EMIT L2B MIN groups;
* Context: a true-colour image.
The last two kinds are built on first use from the source granules and cached under
outputs/viewer/<key>/ (layer builders below).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

EMIT_ID = re.compile(r"(\d{8}T\d{6}_\d{7}_\d{3})")
VIEWER_DIR = Path("outputs/viewer")
UNMIX_MIN_MEAN = 0.02  # unmixing groups below this mean fraction in the area get no layer
DEPTH_MIN = 0.05  # L2B MIN band depth below which a pixel has no reference mineral


@dataclass
class Layer:
    id: str  # "<scene key>.<name>", unique and URL-safe
    scene: str
    name: str
    title: str
    group: str  # Minerals, Gas, Reference, Context
    kind: str  # "categorical", "continuous" or "rgb"
    path: Path | None  # the COG; None until built
    units: str = ""
    colormap: str = "viridis"
    rescale: tuple[float, float] = (0.0, 1.0)
    legend: list[dict] = field(default_factory=list)  # categorical: [{code, label, rgb}]
    build: Callable[[], Path] | None = field(default=None, repr=False)  # makes path if missing
    note: str = ""

    def public(self) -> dict:
        d = asdict(self)
        d.pop("build")
        d["path"] = None
        d["ready"] = self.path is not None and self.path.exists()
        if self.kind == "categorical" and d["ready"]:
            from specret.viewer.tiles import category_fractions

            frac = category_fractions(self.path)
            d["legend"] = sorted(
                ({**e, "fraction": frac.get(e["code"], 0.0)} for e in d["legend"]),
                key=lambda e: -e["fraction"],
            )
        d["buildable"] = self.build is not None
        d["rescale"] = list(self.rescale)
        return d


@dataclass
class SceneEntry:
    key: str
    sensor: str
    title: str
    bounds: tuple[float, float, float, float] | None  # west, south, east, north (degrees)
    mineral_scene: str | None  # Workspace.scenes() id, for pixel identification
    gas_scene: str | None  # Workspace.gas_scenes() id
    gas_dir: Path | None


def scene_key(scene_id: str) -> str:
    m = EMIT_ID.search(scene_id)
    return m.group(1) if m else scene_id


def _bounds_of(path: Path) -> tuple[float, float, float, float] | None:
    import rasterio
    from rasterio.warp import transform_bounds

    try:
        with rasterio.open(path) as ds:
            return tuple(float(v) for v in transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
    except Exception:
        return None


def _nc_bounds(path: Path) -> tuple[float, float, float, float] | None:
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            a = ds.attrs
            return (
                float(a["westernmost_longitude"]),
                float(a["southernmost_latitude"]),
                float(a["easternmost_longitude"]),
                float(a["northernmost_latitude"]),
            )
    except Exception:
        return None


def _gas_files(folder: Path | None) -> list[Path]:
    return sorted(folder.glob("*.tif")) if folder and folder.is_dir() else []


def scene_entries(ws) -> list[SceneEntry]:
    """Every scene the viewer can show, with what each has."""
    out: dict[str, SceneEntry] = {}
    for sid, info in ws.scenes().items():
        key = scene_key(sid)
        out[key] = SceneEntry(key, info.kind.upper(), sid, None, sid, None, None)
    for gid in ws.gas_scenes():
        e = out.setdefault(gid, SceneEntry(gid, "EMIT", gid, None, None, None, None))
        e.gas_scene, e.gas_dir = gid, ws.gas_dir(gid)
    ang = ws.root / "data/scenes/avirisng"
    for d in sorted(ang.iterdir()) if ang.is_dir() else []:
        e = out.setdefault(d.name, SceneEntry(d.name, "AVIRIS-NG", d.name, None, None, None, None))
        e.gas_dir = ws.gas_dir(d.name)
    for e in out.values():
        e.bounds = _entry_bounds(ws, e)
    return sorted(out.values(), key=lambda e: e.key)


def _entry_bounds(ws, e: SceneEntry):
    for p in _gas_files(e.gas_dir):
        if (b := _bounds_of(p)) is not None:
            return b
    if e.mineral_scene:
        map_dir = ws.map_dir(e.mineral_scene)
        if (map_dir / "group.tif").exists():
            return _bounds_of(map_dir / "group.tif")
        path = ws.scenes()[e.mineral_scene].path
        if path.suffix == ".nc":
            return _nc_bounds(path)
    if e.gas_scene:
        return _nc_bounds(ws.gas_scenes()[e.gas_scene])
    ref = next((ws.root / "data/scenes/avirisng" / e.key).glob("*_cmf_pub_ch4.tif"), None)
    return _bounds_of(ref) if ref else None


# --- layers --------------------------------------------------------------------------------

GAS_STYLE = {
    "CH4": ("magma", (0.0, 1500.0)),
    "CO2": ("magma", (0.0, 60000.0)),
}


def _groups_legend(csv: Path) -> list[dict]:
    import pandas as pd

    out = []
    for _, r in pd.read_csv(csv).iterrows():
        rgba = [int(v) for v in re.findall(r"\d+", str(r["rgba"]))][:3]
        out.append({"code": int(r["code"]), "label": str(r["group"]), "rgb": rgba})
    return out


def scene_layers(ws, e: SceneEntry) -> list[Layer]:
    layers: list[Layer] = []
    key = e.key
    cache = ws.root / VIEWER_DIR / key

    def add(name, title, group, kind, path, **kw):
        layers.append(Layer(f"{key}.{name}", key, name, title, group, kind, path, **kw))

    # Context first, so it draws underneath
    src = _context_source(ws, e)
    if src is not None:
        add(
            "rgb",
            "True colour",
            "Context",
            "rgb",
            cache / "rgb.tif",
            build=lambda: build_rgb(src, cache / "rgb.tif"),
        )

    if e.mineral_scene:
        d = ws.map_dir(e.mineral_scene)
        if (d / "group.tif").exists():
            legend = _groups_legend(d / "groups.csv") if (d / "groups.csv").exists() else []
            add(
                "mineral_group",
                "Mineral group (best match)",
                "Minerals",
                "categorical",
                d / "group.tif",
                legend=legend,
            )
            add(
                "mineral_confidence",
                "P(group correct)",
                "Minerals",
                "continuous",
                d / "confidence.tif",
                colormap="rdylgn",
                rescale=(0.0, 1.0),
            )
            add(
                "mineral_score",
                "Match score",
                "Minerals",
                "continuous",
                d / "score.tif",
                colormap="viridis",
                rescale=(0.0, 1.0),
            )
            add(
                "mineral_margin",
                "Margin to other groups",
                "Minerals",
                "continuous",
                d / "margin.tif",
                colormap="cividis",
                rescale=(0.0, 0.3),
            )

    if e.gas_dir and e.gas_dir.is_dir():
        for gas, (cmap, rng) in GAS_STYLE.items():
            p = e.gas_dir / f"{gas.lower()}_enh.tif"
            if p.exists():
                add(
                    f"{gas.lower()}_enh",
                    f"{gas} enhancement",
                    "Gas",
                    "continuous",
                    p,
                    units="ppm·m",
                    colormap=cmap,
                    rescale=rng,
                )
        if (e.gas_dir / "confidence.tif").exists():
            add(
                "plume_confidence",
                "Plume confidence",
                "Gas",
                "continuous",
                e.gas_dir / "confidence.tif",
                colormap="rdylgn",
                rescale=(0.0, 1.0),
            )

    unmix = cache / "unmix"
    if (unmix / "meta.json").exists():
        meta = json.loads((unmix / "meta.json").read_text(encoding="utf-8"))
        note = f"{meta['pixels_unmixed']} pixels unmixed" + (
            f", every {meta['step']}th row and column" if meta["step"] > 1 else ""
        )
        for g, f in meta["mean_fraction"].items():
            if f < UNMIX_MIN_MEAN:
                continue
            add(
                f"unmix_{g}",
                f"Unmix: {g.replace('_', ' ')} fraction",
                "Unmixing",
                "continuous",
                unmix / f"{g}.tif",
                colormap="viridis",
                rescale=(0.0, 1.0),
                note=note,
            )
        add(
            "unmix_residual",
            "Unmix: fit residual (RMS)",
            "Unmixing",
            "continuous",
            unmix / "residual.tif",
            units="reflectance",
            colormap="magma",
            rescale=(0.0, 0.02),
            note=note,
        )

    for gas, (cmap, rng) in GAS_STYLE.items():
        for p in _emit_ghg(ws, key, gas):
            add(
                f"emit_{gas.lower()}enh",
                f"EMIT L2B {gas}ENH",
                "Reference",
                "continuous",
                p,
                units="ppm·m",
                colormap=cmap,
                rescale=rng,
            )
    for p in (ws.root / "data/scenes/avirisng" / key).glob("*_cmf_pub_ch4.tif"):
        add(
            "published_ch4",
            "Published CH4 (Duren et al. 2019)",
            "Reference",
            "continuous",
            p,
            units="ppm·m",
            colormap="magma",
            rescale=(0.0, 1500.0),
        )
    l2bmin = _l2bmin_source(ws, key)
    if l2bmin is not None:
        out = cache / "l2bmin_group2.tif"
        legend_path = out.with_suffix(".json")
        legend = json.loads(legend_path.read_text()) if legend_path.exists() else []
        add(
            "emit_l2bmin",
            "EMIT L2B MIN (~2 µm group)",
            "Reference",
            "categorical",
            out,
            legend=legend,
            build=lambda: build_l2bmin(l2bmin, _glt_source(ws, key), out),
        )
        ours = ws.map_dir(e.mineral_scene) / "group.tif" if e.mineral_scene else None
        if ours is not None and ours.exists():
            add(
                "agreement",
                "Agreement with EMIT L2B MIN",
                "Reference",
                "categorical",
                cache / "agreement.tif",
                legend=AGREEMENT_LEGEND,
                build=lambda: build_agreement(ours, l2bmin, _glt_source(ws, key), cache),
                note=f"Our best-match group against L2B MIN's ~2 µm group, where its band "
                f"depth is at least {DEPTH_MIN}",
            )
    return layers


def _emit_ghg(ws, key: str, gas: str) -> list[Path]:
    if not EMIT_ID.fullmatch(key):
        return []
    return sorted((ws.root / "data/scenes").rglob(f"EMIT_L2B_{gas}ENH_*{key}*.tif"))[:1]


def _first(ws, pattern: str) -> Path | None:
    return next(iter(sorted((ws.root / "data/scenes").rglob(pattern))), None)


def _l2bmin_source(ws, key: str) -> Path | None:
    return _first(ws, f"EMIT_L2B_MIN_*{key}*.nc") if EMIT_ID.fullmatch(key) else None


def _glt_source(ws, key: str) -> Path | None:
    """A granule of the same EMIT scene that carries the GLT (L2A reflectance or L1B)."""
    return _first(ws, f"EMIT_L2A_RFL_*{key}*.nc") or _first(ws, f"EMIT_L1B_RAD_*{key}*.nc")


def _context_source(ws, e: SceneEntry):
    """("emit", path) for an EMIT L2A or L1B granule, ("avirisng", id) for a flight line."""
    if EMIT_ID.fullmatch(e.key):
        p = _glt_source(ws, e.key)
        return ("emit", p) if p else None
    if e.sensor == "AVIRIS-NG":
        return ("avirisng", e.key)
    return None


# --- builders (run once, cached) -----------------------------------------------------------

AGREEMENT_LEGEND = [
    {"code": 1, "label": "agree", "rgb": [63, 160, 90]},
    {"code": 2, "label": "disagree", "rgb": [200, 70, 50]},
    {"code": 3, "label": "no reference", "rgb": [150, 150, 150]},
]

_BUILD_LOCK = threading.Lock()
RGB_NM = (650.0, 560.0, 470.0)


def _stretch(a: np.ndarray) -> np.ndarray:
    """Per-band 2-98 percentile stretch to uint8; 0 stays no data."""
    out = np.zeros(a.shape, dtype=np.uint8)
    for b in range(a.shape[-1]):
        v = a[..., b]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        lo, hi = np.percentile(v[ok], [2, 98])
        s = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1) * 254 + 1
        out[..., b] = np.where(ok, s, 0).astype(np.uint8)
    return out


def _write_cog(path: Path, data: np.ndarray, transform, crs, nodata, colormap=None) -> None:
    import rasterio
    from rasterio.shutil import copy as raster_copy

    path.parent.mkdir(parents=True, exist_ok=True)
    bands = data[None] if data.ndim == 2 else np.moveaxis(data, -1, 0)
    profile = {
        "driver": "GTiff",
        "height": bands.shape[1],
        "width": bands.shape[2],
        "count": bands.shape[0],
        "dtype": bands.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(bands)
            if colormap:
                ds.write_colormap(1, colormap)
        with mem.open() as ds:
            raster_copy(ds, tmp, driver="COG", compress="DEFLATE")
    tmp.replace(path)


def build_rgb(source, out: Path) -> Path:
    from affine import Affine

    with _BUILD_LOCK:
        if out.exists():
            return out
        kind, ref = source
        if kind == "emit":
            import xarray as xr

            from specret.io.emit import orthorectify

            with xr.open_dataset(ref, group="location") as loc:
                glt_x = loc["glt_x"].values.astype(np.int64)
                glt_y = loc["glt_y"].values.astype(np.int64)
            ds = xr.open_dataset(ref)
            var = "reflectance" if "reflectance" in ds else "radiance"
            with xr.open_dataset(ref, group="sensor_band_parameters") as bp:
                centers = bp["wavelengths"].values
            idx = [int(np.argmin(np.abs(centers - w))) for w in RGB_NM]
            cube = np.stack([ds[var][:, :, i].values for i in idx], axis=-1).astype(np.float32)
            cube[cube <= -9998] = np.nan
            img = orthorectify(cube, glt_x, glt_y, np.nan)
            transform = Affine.from_gdal(*np.asarray(ds.attrs["geotransform"], dtype=float))
            crs = str(ds.attrs.get("spatial_ref", "EPSG:4326"))
            ds.close()
        else:
            from specret.io.avirisng import open_avirisng

            s = open_avirisng(ref)
            idx = [int(np.argmin(np.abs(s.centers_nm - w))) for w in RGB_NM]
            mm = s.img.open_memmap(interleave="bip")
            img = np.asarray(mm[:, :, idx], dtype=np.float32)
            img[img <= -9998] = np.nan
            transform, crs = Affine.from_gdal(*s.geotransform), s.crs_wkt
        _write_cog(out, _stretch(img), transform, crs, 0)
        return out


def build_l2bmin(path: Path, glt_source: Path | None, out: Path) -> Path:
    """EMIT L2B MIN ~2 µm minerals as taxonomy groups on the GLT grid, with a colour table."""
    from affine import Affine

    with _BUILD_LOCK:
        if out.exists():
            return out
        if glt_source is None:
            raise ValueError("no L2A or L1B granule of this scene to orthorectify L2B MIN with")
        import xarray as xr

        from specret.eval.emit_l2bmin import coarse_group, load_l2bmin
        from specret.io.emit import orthorectify
        from specret.scene.outputs import group_codes, group_colors

        m = load_l2bmin(path)
        groups = {i: coarse_group(n) for i, n in m.names.items() if i > 0}
        codes = group_codes([g for g in groups.values() if g])
        lut = np.zeros(max(m.names) + 1, dtype=np.uint8)
        for i, g in groups.items():
            lut[i] = codes.get(g, 0)
        swath = lut[np.clip(m.group2_id, 0, lut.size - 1)]
        with xr.open_dataset(glt_source, group="location") as loc:
            glt_x = loc["glt_x"].values.astype(np.int64)
            glt_y = loc["glt_y"].values.astype(np.int64)
        with xr.open_dataset(glt_source) as ds:
            transform = Affine.from_gdal(*np.asarray(ds.attrs["geotransform"], dtype=float))
            crs = str(ds.attrs.get("spatial_ref", "EPSG:4326"))
        img = orthorectify(swath, glt_x, glt_y, 0).astype(np.uint8)
        colors = group_colors(codes)
        _write_cog(out, img, transform, crs, 0, colormap=colors)
        depth = orthorectify(m.group2_depth.astype(np.float32), glt_x, glt_y, np.nan)
        _write_cog(out.with_name("l2bmin_depth2.tif"), depth, transform, crs, np.nan)
        legend = [{"code": c, "label": g, "rgb": list(colors[c][:3])} for g, c in codes.items()]
        out.with_suffix(".json").write_text(json.dumps(legend), encoding="utf-8")
        return out


def _labels(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    """A categorical raster and its code -> label table (groups.csv or the legend json)."""
    import pandas as pd
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1)
    csv, js = path.with_name("groups.csv"), path.with_suffix(".json")
    if csv.exists():
        t = pd.read_csv(csv)
        return a, dict(zip(t["code"].astype(int), t["group"].astype(str), strict=True))
    return a, {int(e["code"]): e["label"] for e in json.loads(js.read_text(encoding="utf-8"))}


def build_agreement(ours: Path, l2bmin: Path, glt_source: Path | None, cache: Path) -> Path:
    """Per-pixel agreement of our best-match group with L2B MIN's ~2 µm group, and its
    statistics (agreement.json): pixels with a reference, agreement, top confusions."""
    import pandas as pd
    import rasterio

    out = cache / "agreement.tif"
    ref_path = cache / "l2bmin_group2.tif"
    if not ref_path.with_name("l2bmin_depth2.tif").exists():
        ref_path.unlink(missing_ok=True)  # an older build without depth
    build_l2bmin(l2bmin, glt_source, ref_path)
    with _BUILD_LOCK:
        if out.exists():
            return out
        a, a_labels = _labels(ours)
        b, b_labels = _labels(ref_path)
        with rasterio.open(ref_path.with_name("l2bmin_depth2.tif")) as ds:
            depth = ds.read(1)
        with rasterio.open(ours) as ds:
            transform, crs = ds.transform, ds.crs
        if a.shape != b.shape:
            raise ValueError(f"grids differ: ours {a.shape}, L2B MIN {b.shape}")
        lut_a = np.array([a_labels.get(i, "") for i in range(256)], dtype=object)
        lut_b = np.array([b_labels.get(i, "") for i in range(256)], dtype=object)
        la, lb = lut_a[a], lut_b[b]
        valid = a > 0
        has_ref = valid & (b > 0) & (np.nan_to_num(depth) >= DEPTH_MIN)
        code = np.zeros(a.shape, dtype=np.uint8)
        code[valid] = 3
        code[has_ref & (la == lb)] = 1
        code[has_ref & (la != lb)] = 2
        colors = {e["code"]: (*e["rgb"], 255) for e in AGREEMENT_LEGEND}
        _write_cog(out, code, transform, crs, 0, colormap=colors)
        dis = code == 2
        pairs = pd.Series(list(zip(la[dis], lb[dis], strict=True))).value_counts()
        stats = {
            "pixels_mapped": int(valid.sum()),
            "pixels_with_reference": int(has_ref.sum()),
            "agreement": float((code == 1).sum() / max(int(has_ref.sum()), 1)),
            "depth_min": DEPTH_MIN,
            "top_confusions": [
                {"ours": o, "l2bmin": t, "pixels": int(n)} for (o, t), n in pairs.head(10).items()
            ],
        }
        out.with_suffix(".json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        return out
