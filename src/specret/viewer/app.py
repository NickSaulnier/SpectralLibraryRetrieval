"""Local web viewer: click a pixel on a mineral map, drop in a field spectrum, or look at gas
plumes found in EMIT radiance.

Run with `uv run specret view` and open http://127.0.0.1:8765. The page lists every scene
and can map an unmapped one in the background. Clicking a pixel (or entering a lat/lon)
plots its spectrum against the top matches, shades the diagnostic feature windows, and
lists the feature evidence for each match. Everything, Plotly included, is served locally.
"""

from __future__ import annotations

import io
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from specret.scene import jobs
from specret.scene.mapper import SceneMap
from specret.scene.outputs import group_codes, group_colors
from specret.viewer import layers as layers_mod
from specret.viewer import tiles as tiles_mod
from specret.workspace import UNKNOWN_BELOW, Workspace, jsonable

STATIC = Path(__file__).parent / "static"  # app/ holds the React build of viewer/web (committed)
UNKNOWN_CODE = 255  # map code for pixels whose calibrated confidence is below UNKNOWN_BELOW
UNKNOWN_RGB = (128, 128, 128)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def depth_profiles_payload(lib, cfg, query, matches: list[dict], row_of: dict) -> list[dict]:
    """Continuum-removed band depth per diagnostic window for the query and the top matches,
    as the feature-fit score sees them (rerank/sff.compute_features)."""
    from specret.rerank.sff import compute_features

    stack = np.vstack(
        [np.asarray(query, dtype=np.float64)]
        + [lib.matrix[row_of[m["spectrum_id"]]] for m in matches]
    )
    out = []
    for wp in compute_features(stack, lib.sensor, cfg).windows:
        out.append(
            {
                "window": wp.window.name,
                "wavelengths_nm": wp.wavelengths.tolist(),
                "query": wp.profiles[0].tolist(),
                "matches": [p.tolist() for p in wp.profiles[1:]],
                "depth": wp.depth.tolist(),
                "center_nm": wp.center.tolist(),
            }
        )
    return out


def create_app(ws: Workspace | None = None) -> FastAPI:
    ws = ws or Workspace()
    app = FastAPI(title="specret viewer")
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    # Read once, so a running server always serves the page that matches its own endpoints.
    legacy = (STATIC / "index.html").read_text(encoding="utf-8")
    app_dir = STATIC / "app"
    built = app_dir / "index.html"
    page = built.read_text(encoding="utf-8") if built.exists() else legacy
    if (app_dir / "assets").is_dir():
        app.mount("/app/assets", StaticFiles(directory=app_dir / "assets"), name="assets")

    @lru_cache(maxsize=4)
    def scene_map(scene_id: str) -> SceneMap:
        path = ws.map_dir(scene_id) / "scene_map.npz"
        if not path.exists():
            raise HTTPException(404, f"{scene_id} is not mapped; run `specret map-scene`")
        return SceneMap.load(path)

    @lru_cache(maxsize=4)
    def ortho_layers(scene_id: str) -> tuple[np.ndarray, dict[str, int]]:
        """Orthorectified group-code image and the code table."""
        smap, scene = scene_map(scene_id), ws.scene(scene_id)
        codes = group_codes(list(smap.library_groups))
        top1 = smap.top_rows[..., 0]
        code = np.zeros(smap.shape, dtype=np.uint8)
        valid = top1 >= 0
        code[valid] = [codes.get(g, 0) for g in smap.library_groups[top1[valid]]]
        if smap.p_correct is not None:
            code[valid & (smap.p_correct < UNKNOWN_BELOW)] = UNKNOWN_CODE
        return scene.orthorectify(code, fill=0).astype(np.uint8), codes

    def spectrum_payload(sensor: str, query: np.ndarray, result: dict) -> dict:
        lib = ws.library(sensor)
        s = lib.sensor
        row_of = {sid: i for i, sid in enumerate(lib.spectrum_ids)}
        good = s.good_mask
        matches = result["matches"]
        for m in matches[:3]:
            m["spectrum"] = lib.matrix[row_of[m["spectrum_id"]]][good].tolist()
        cfg = ws.retriever(sensor).features
        windows = {w.name: [w.lo_nm, w.hi_nm] for w in cfg.windows}
        return jsonable(
            {
                **result,
                "wavelengths_nm": s.centers_nm[good].tolist(),
                "query": np.asarray(query, dtype=float)[good].tolist(),
                "windows": windows,
                "profiles": depth_profiles_payload(lib, cfg, query, matches[:3], row_of),
            }
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """The React app (built into static/app), or the legacy page if it is not built."""
        return page

    @app.get("/legacy", response_class=HTMLResponse)
    def legacy_page() -> str:
        """The original single-page viewer, kept until the React app reaches parity."""
        return legacy

    @app.get("/static/plotly.min.js")
    def plotly_js() -> Response:
        return Response(
            _plotly_js(),
            media_type="application/javascript",
            headers={"Cache-Control": "max-age=86400"},
        )

    @app.get("/api/scenes")
    def scenes() -> list[dict]:
        return [
            {
                "scene_id": sid,
                "kind": info.kind,
                "mapped": (ws.map_dir(sid) / "scene_map.npz").exists(),
                "job": jobs.running_job(sid),
            }
            for sid, info in ws.scenes().items()
        ]

    @app.post("/api/map/{scene_id}")
    def start_mapping(scene_id: str) -> dict:
        try:
            return {"job_id": jobs.start_map_job(ws, scene_id)}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        try:
            status = jobs.job_status(job_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        if status["state"] == "done":
            scene_map.cache_clear()
            ortho_layers.cache_clear()
            geo_scenes.cache_clear()
        if status.get("error"):
            status["error"] = status["error"].splitlines()[0]
        return status

    @app.get("/api/map/{scene_id}.png")
    def map_png(scene_id: str) -> Response:
        from PIL import Image

        img, codes = ortho_layers(scene_id)
        colors = group_colors(codes)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for c, rgba in colors.items():
            palette[c] = rgba[:3]
        palette[UNKNOWN_CODE] = UNKNOWN_RGB
        pil = Image.fromarray(img, mode="P")
        pil.putpalette(palette.flatten().tolist())
        buf = io.BytesIO()
        pil.save(buf, format="PNG", transparency=0)
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/api/legend/{scene_id}")
    def legend(scene_id: str) -> list[dict]:
        img, codes = ortho_layers(scene_id)
        colors = group_colors(codes)
        counts = np.bincount(img.ravel(), minlength=256)
        total = max(int(counts[1:].sum()), 1)
        rows = [
            {"group": g, "code": c, "rgb": list(colors[c][:3]), "fraction": counts[c] / total}
            for g, c in codes.items()
            if counts[c] > 0
        ]
        if counts[UNKNOWN_CODE]:
            rows.append(
                {
                    "group": f"low confidence (P < {UNKNOWN_BELOW})",
                    "code": UNKNOWN_CODE,
                    "rgb": list(UNKNOWN_RGB),
                    "fraction": counts[UNKNOWN_CODE] / total,
                }
            )
        return jsonable(sorted(rows, key=lambda r: -r["fraction"]))

    @app.get("/api/pixel/{scene_id}")
    def pixel(
        scene_id: str,
        x: int | None = None,
        y: int | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> dict:
        """Identify the pixel at ortho-image coordinates (x, y), or nearest to (lat, lon)."""
        try:
            scene = ws.scene(scene_id)
            if lat is not None and lon is not None:
                x, y = _ortho_xy(ws, scene, lat, lon)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        if x is None or y is None:
            raise HTTPException(400, "give x and y, or lat and lon")
        if hasattr(scene, "glt_y"):  # EMIT: map grid -> swath through the lookup table
            if not (0 <= y < scene.glt_y.shape[0] and 0 <= x < scene.glt_x.shape[1]):
                raise HTTPException(400, "outside the image")
            row, col = int(scene.glt_y[y, x]) - 1, int(scene.glt_x[y, x]) - 1
            if row < 0 or col < 0:
                raise HTTPException(404, "no data at this location")
        else:  # AVIRIS: the cube is already on its map grid
            if not (0 <= y < scene.shape[0] and 0 <= x < scene.shape[1]):
                raise HTTPException(400, "outside the image")
            row, col = y, x
        sensor = ws.sensor_for_scene(scene)
        spectrum = scene.pixel(row, col)
        if not np.isfinite(spectrum).any():
            raise HTTPException(404, "no data at this location")
        result = ws.identify(spectrum, sensor, k=5, unmix=True, unmix_spectra=True)
        result["pixel"] = {**ws.pixel_info(scene, row, col), "x": x, "y": y}
        return spectrum_payload(sensor, spectrum, result)

    @app.post("/api/field")
    async def field(file: UploadFile) -> list[dict]:
        """Identify an uploaded field spectrum (.asd, .csv, .txt)."""
        suffix = Path(file.filename or "upload.csv").suffix.lower()
        if suffix not in (".asd", ".csv", ".txt", ".tsv"):
            raise HTTPException(400, "upload an .asd, .csv, .txt or .tsv file")
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            from specret.io.field import read_field

            spectra = read_field(path)
            results = ws.identify_field(path, k=5)
        out = []
        sensor = "field_asd"
        for spec, res in zip(spectra, results, strict=True):
            query = ws.to_sensor(spec.wavelength_nm, spec.reflectance, sensor)
            out.append({"name": spec.name, **spectrum_payload(sensor, query, res)})
        return out

    # --- gas plumes (EMIT and AVIRIS-NG radiance, HyTES flights) ----------------------

    def gas_folder(scene_id: str) -> Path:
        if scene_id not in ws.gas_sources():
            raise HTTPException(404, f"no radiance for {scene_id}")
        return ws.gas_dir(scene_id)

    @app.get("/api/gas/scenes")
    def gas_scenes() -> list[dict]:
        return [
            {
                "scene_id": sid,
                "sensor": src.sensor,
                "analyzed": (ws.gas_dir(sid) / "plumes.csv").exists(),
                "job": jobs.running_job(sid, "gas"),
            }
            for sid, src in ws.gas_sources().items()
        ]

    @app.get("/api/gas/plumes")
    def all_plumes() -> list[dict]:
        """Every plume of every analyzed scene, line and flight, with its sensor and, for EMIT,
        how many other overpasses saw it again (`specret gas-persistence`)."""
        from specret.gas import tools

        rep = tools.persistence_table(ws.root / "outputs/gas")
        out = []
        for sid, src in ws.gas_sources().items():
            folder = ws.gas_dir(sid)
            if not (folder / "plumes.csv").exists():
                continue
            for row in tools.list_plumes(folder)["plumes"]:
                out.append(
                    {
                        "scene_id": sid,
                        "sensor": src.sensor,
                        **row,
                        **rep.get((sid, row["plume_id"]), {}),
                    }
                )
        return out

    @app.post("/api/gas/{scene_id}")
    def start_gas(scene_id: str) -> dict:
        try:
            return {"job_id": jobs.start_gas_job(ws, scene_id)}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/gas/{scene_id}/{gas}.png")
    def gas_png(scene_id: str, gas: str) -> Response:
        """Enhancement on the map grid (0 to the 99.5th percentile, magma-like ramp), with
        plume outlines in cyan."""
        import rasterio
        from PIL import Image

        folder = gas_folder(scene_id)
        path = folder / f"{gas.lower()}_enh.tif"
        if not path.exists():
            raise HTTPException(404, f"no {gas} map for {scene_id}")
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float32)
            a[a == ds.nodata] = np.nan
        hi = float(np.nanpercentile(a, 99.5)) or 1.0
        t = np.clip(np.nan_to_num(a, nan=0.0) / hi, 0, 1)
        rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
        rgba[..., 0] = (255 * np.clip(1.6 * t, 0, 1)).astype(np.uint8)
        rgba[..., 1] = (255 * np.clip(1.6 * t - 0.6, 0, 1)).astype(np.uint8)
        rgba[..., 2] = (
            255 * np.clip(0.5 + 0.5 * np.sin(np.pi * (t - 0.15)), 0, 1) * (t < 0.8)
        ).astype(np.uint8)
        rgba[..., 3] = np.where(np.isfinite(a), 255, 0)
        mask_path = folder / "plume_mask.tif"
        if mask_path.exists():
            from scipy import ndimage

            with rasterio.open(mask_path) as ds:
                lab = ds.read(1)
            edge = (lab > 0) & ~ndimage.binary_erosion(lab > 0)
            rgba[edge] = (0, 230, 255, 255)
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/api/gas/{scene_id}/plumes")
    def gas_plumes(scene_id: str) -> dict:
        from specret.gas import tools

        folder = gas_folder(scene_id)
        try:
            out = tools.list_plumes(folder)
            summary = tools.scene_gas_summary(folder)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        rep = tools.persistence_table(ws.root / "outputs/gas")
        path = ws.gas_scenes().get(scene_id)  # EMIT: ortho-image pixel of each source
        if path is not None:
            import xarray as xr

            with xr.open_dataset(path) as ds:
                x0, dx, _, y0, _, dy = np.asarray(ds.attrs["geotransform"], dtype=float)
        for pl in out["plumes"]:
            pl["sensor"] = ws.gas_sources()[scene_id].sensor
            pl.update(rep.get((scene_id, pl["plume_id"]), {}))
            if path is not None and pl.get("source_lat") is not None:
                pl["x"] = int(np.floor((pl["source_lon"] - x0) / dx))
                pl["y"] = int(np.floor((pl["source_lat"] - y0) / dy))
        out["summary"] = summary
        return out

    @app.get("/api/gas/{scene_id}/plume/{plume_id}")
    def gas_plume(scene_id: str, plume_id: int) -> dict:
        from specret.gas import tools

        try:
            return tools.explain_plume(gas_folder(scene_id), plume_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None

    # --- map: scenes, layers, tiles (Phase V1) ----------------------------------------

    @lru_cache(maxsize=1)
    def geo_scenes() -> dict:
        return {e.key: e for e in layers_mod.scene_entries(ws)}

    def entry(key: str):
        e = geo_scenes().get(key)
        if e is None:
            geo_scenes.cache_clear()
            e = geo_scenes().get(key)
        if e is None:
            raise HTTPException(404, f"unknown scene '{key}'")
        return e

    def layer_of(layer_id: str) -> layers_mod.Layer:
        key = layer_id.rsplit(".", 1)[0]
        for layer in layers_mod.scene_layers(ws, entry(key)):
            if layer.id == layer_id:
                return layer
        raise HTTPException(404, f"unknown layer '{layer_id}'")

    @app.get("/api/geo/scenes")
    def geo_scene_list() -> list[GeoScene]:
        """Every scene the map can show: its footprint and which kinds of layers it has."""
        out = []
        for e in geo_scenes().values():
            mapped = (
                bool(e.mineral_scene) and (ws.map_dir(e.mineral_scene) / "scene_map.npz").exists()
            )
            analyzed = bool(e.gas_dir) and (e.gas_dir / "plumes.csv").exists()
            out.append(
                GeoScene(
                    key=e.key,
                    sensor=e.sensor,
                    title=e.title,
                    bounds=list(e.bounds) if e.bounds else None,
                    mineral_scene=e.mineral_scene,
                    gas_scene=e.gas_scene,
                    mapped=mapped,
                    analyzed=analyzed,
                )
            )
        return out

    @app.get("/api/geo/{key}/layers")
    def geo_layers(key: str) -> list[LayerInfo]:
        """The layers of one scene, in drawing order (bottom first)."""
        return [LayerInfo(**layer.public()) for layer in layers_mod.scene_layers(ws, entry(key))]

    @app.get("/api/tiles/{layer_id}/{z}/{x}/{y}.png")
    def tile(
        layer_id: str,
        z: int,
        x: int,
        y: int,
        rescale: str | None = None,
        colormap: str | None = None,
        groups: str | None = None,
        min_p: float = 0.0,
    ) -> Response:
        """One 256 px Web Mercator tile. Layers built on first use (true colour, L2B MIN)
        are built by the first request, which takes a few seconds."""
        layer = layer_of(layer_id)
        if layer.path is None or not layer.path.exists():
            if layer.build is None:
                raise HTTPException(404, f"{layer_id} has no data")
            try:
                layer.build()
            except Exception as exc:  # reported to the map as a missing layer
                raise HTTPException(500, f"could not build {layer_id}: {exc}") from None
        if layer.kind == "categorical":
            keep = None
            if groups is not None:
                try:
                    keep = tuple(int(g) for g in groups.split(",") if g)
                except ValueError:
                    raise HTTPException(400, "groups is a comma-separated list of codes") from None
            conf = layer.path.parent / "confidence.tif" if layer.name == "mineral_group" else None
            png = tiles_mod.render_categories(layer.path, z, x, y, keep, conf, min_p)
            return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=300"})
        rng = layer.rescale
        if rescale:
            try:
                lo, hi = (float(v) for v in rescale.split(","))
                rng = (lo, hi)
            except ValueError:
                raise HTTPException(400, "rescale is lo,hi") from None
        cmap = colormap or layer.colormap
        if cmap not in tiles_mod.COLORMAPS:
            raise HTTPException(400, f"colormap must be one of {tiles_mod.COLORMAPS}")
        png = tiles_mod.render(layer.path, layer.kind, z, x, y, rng, cmap)
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=300"})

    @app.get("/api/geo/{key}/values")
    def geo_values(key: str, lat: float, lon: float) -> dict[str, float | str | None]:
        """Each ready layer's value at a location (categorical layers give their label)."""
        out: dict[str, float | str | None] = {}
        for layer in layers_mod.scene_layers(ws, entry(key)):
            if layer.kind == "rgb" or layer.path is None or not layer.path.exists():
                continue
            v = tiles_mod.point_value(layer.path, lon, lat)
            if layer.kind == "categorical" and v is not None:
                label = next((e["label"] for e in layer.legend if e["code"] == int(v)), None)
                out[layer.id] = label
            else:
                out[layer.id] = v
        return out

    @app.get("/api/geo/{key}/plumes.geojson")
    def geo_plumes(key: str) -> dict:
        """Plume outlines with their attributes (an empty collection if not analyzed)."""
        e = entry(key)
        path = e.gas_dir / "plumes.geojson" if e.gas_dir else None
        if path is None or not path.exists():
            return {"type": "FeatureCollection", "features": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/geo/{key}/summary")
    def geo_summary(key: str, req: AreaRequest) -> AreaSummary:
        """What is inside a polygon: mineral composition (the same computation as the MCP
        scene_summary tool), statistics of every continuous layer, and plumes whose source
        lies inside."""
        from specret.viewer import area

        e = entry(key)
        if len(req.polygon) < 3:
            raise HTTPException(400, "a polygon needs at least 3 vertices")
        poly = [(float(x), float(y)) for x, y in req.polygon]
        minerals = None
        if e.mineral_scene and (ws.map_dir(e.mineral_scene) / "scene_map.npz").exists():
            minerals = area.mineral_composition(ws, e.mineral_scene, poly, req.min_p)
        layers = [
            area.layer_stats(layer, poly)
            for layer in layers_mod.scene_layers(ws, e)
            if layer.kind == "continuous" and layer.path is not None and layer.path.exists()
        ]
        return AreaSummary(
            area_km2=area.polygon_area_km2(poly),
            min_p=req.min_p,
            minerals=minerals,
            layers=[s for s in layers if s is not None],
            plumes=area.plumes_inside(e.gas_dir, poly),
        )

    @app.post("/api/geo/{key}/unmix")
    def geo_unmix(key: str, req: UnmixRequest) -> dict[str, str]:
        """Start unmixing an area's pixels in the background; poll /api/jobs/{job_id}. The
        fraction layers appear in the scene's layer list when it is done."""
        e = entry(key)
        if not e.mineral_scene:
            raise HTTPException(400, "this scene has no reflectance to unmix")
        if len(req.polygon) < 3:
            raise HTTPException(400, "a polygon needs at least 3 vertices")
        poly = [(float(x), float(y)) for x, y in req.polygon]
        return {"job_id": jobs.start_unmix_job(ws, key, e.mineral_scene, poly, req.max_pixels)}

    @app.get("/api/geo/{key}/agreement")
    def geo_agreement(key: str) -> dict:
        """Agreement of our mineral groups with EMIT L2B MIN (built on first request)."""
        layer = next(
            (x for x in layers_mod.scene_layers(ws, entry(key)) if x.name == "agreement"), None
        )
        if layer is None:
            raise HTTPException(404, "needs a mineral map and an EMIT L2B MIN granule")
        if not layer.path.exists():
            layer.build()
        return json.loads(layer.path.with_suffix(".json").read_text(encoding="utf-8"))

    @app.get("/api/geo/{key}/compare")
    def geo_compare(key: str, layer: str, bbox: str | None = None) -> dict:
        """Our enhancement layer against its published counterpart (the layer's compare_with)
        inside bbox "west,south,east,north" (default: the whole scene): r, OLS and Deming
        slopes on plume and scene pixels, background spreads, and a sample for a scatter."""
        from specret.viewer import compare

        e = entry(key)
        ls = {x.name: x for x in layers_mod.scene_layers(ws, e)}
        ours = ls.get(layer)
        if ours is None or ours.compare_with is None:
            raise HTTPException(404, f"{layer} has no published counterpart in {key}")
        ref = ls[ours.compare_with]
        box = None
        if bbox:
            try:
                box = tuple(float(v) for v in bbox.split(","))
                assert len(box) == 4
            except (ValueError, AssertionError):
                raise HTTPException(400, "bbox is west,south,east,north") from None
        gas = layer.removesuffix("_enh").upper()
        pair = compare.load_pair(ours.path, ref.path, ref.name, key, gas)
        return jsonable(
            {
                "ours": ours.title,
                "ref": ref.title,
                "units": ours.units,
                **compare.compare_view(pair, box),
            }
        )

    @app.get("/api/colormaps")
    def colormaps() -> dict[str, list[str]]:
        """Each colormap sampled at 11 stops, as hex colours, for legends."""
        from rio_tiler.colormap import cmap

        out = {}
        for name in tiles_mod.COLORMAPS:
            cm = cmap.get(name)
            stops = (cm[int(i)][:3] for i in np.linspace(0, 255, 11))
            out[name] = ["#" + "".join(f"{c:02x}" for c in rgb) for rgb in stops]
        return out

    return app


class GeoScene(BaseModel):
    key: str
    sensor: str
    title: str
    bounds: list[float] | None
    mineral_scene: str | None
    gas_scene: str | None
    mapped: bool
    analyzed: bool


class AreaRequest(BaseModel):
    polygon: list[list[float]]  # [[lon, lat], ...], not necessarily closed
    min_p: float = 0.3  # below this P(group correct), pixels count as likely_unknown


class UnmixRequest(BaseModel):
    polygon: list[list[float]]
    max_pixels: int = 3000  # about 40 s; larger areas are unmixed on a coarser grid


class GroupShare(BaseModel):
    group: str
    pixels: int
    fraction: float
    median_p: float | None


class MineralSummary(BaseModel):
    pixels: int
    groups: list[GroupShare]


class LayerStats(BaseModel):
    id: str
    title: str
    units: str
    pixels: int
    mean: float | None
    median: float | None
    p05: float | None
    p95: float | None
    hist_edges: list[float]
    hist_counts: list[int]


class PlumeSummary(BaseModel):
    count: int
    by_gas: dict[str, dict[str, float]]  # gas -> {count, total_q_kg_h}
    ids: list[int]


class AreaSummary(BaseModel):
    area_km2: float
    min_p: float
    minerals: MineralSummary | None
    layers: list[LayerStats]
    plumes: PlumeSummary


class LegendItem(BaseModel):
    code: int
    label: str
    rgb: list[int]
    fraction: float | None = None  # share of the scene's valid pixels


class LayerInfo(BaseModel):
    id: str
    scene: str
    name: str
    title: str
    group: str
    kind: str
    units: str
    colormap: str
    rescale: list[float]
    legend: list[LegendItem]
    ready: bool
    buildable: bool
    note: str
    compare_with: str | None = None


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def _ortho_xy(ws: Workspace, scene, lat: float, lon: float) -> tuple[int, int]:
    """Ortho-image (x, y) of a location: EMIT's north-up lat/lon grid, or AVIRIS's own grid."""
    if hasattr(scene, "glt_y"):
        x0, dx, _, y0, _, dy = scene.geotransform
        return int(np.floor((lon - x0) / dx)), int(np.floor((lat - y0) / dy))
    row, col = ws.locate(scene, lat=lat, lon=lon)
    return col, row


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
