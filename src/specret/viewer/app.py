"""Local web viewer: click a pixel on a mineral map, drop in a field spectrum, or look at gas
plumes found in EMIT radiance.

Run with `uv run specret view` and open http://127.0.0.1:8765. The page lists every scene
and can map an unmapped one in the background. Clicking a pixel (or entering a lat/lon)
plots its spectrum against the top matches, shades the diagnostic feature windows, and
lists the feature evidence for each match. Everything, Plotly included, is served locally.
"""

from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from specret.scene import jobs
from specret.scene.mapper import SceneMap
from specret.scene.outputs import group_codes, group_colors
from specret.workspace import UNKNOWN_BELOW, Workspace, jsonable

STATIC = Path(__file__).parent / "static"  # app/ holds the React build of viewer/web (committed)
UNKNOWN_CODE = 255  # map code for pixels whose calibrated confidence is below UNKNOWN_BELOW
UNKNOWN_RGB = (128, 128, 128)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


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
        windows = {w.name: [w.lo_nm, w.hi_nm] for w in ws.retriever(sensor).features.windows}
        return jsonable(
            {
                **result,
                "wavelengths_nm": s.centers_nm[good].tolist(),
                "query": np.asarray(query, dtype=float)[good].tolist(),
                "windows": windows,
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
        result = ws.identify(spectrum, sensor, k=5, unmix=True)
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

    # --- gas plumes (EMIT L1B radiance scenes) ---------------------------------------

    def gas_folder(scene_id: str) -> Path:
        if scene_id not in ws.gas_scenes():
            raise HTTPException(404, f"no EMIT L1B radiance for {scene_id}")
        return ws.gas_dir(scene_id)

    @app.get("/api/gas/scenes")
    def gas_scenes() -> list[dict]:
        return [
            {
                "scene_id": sid,
                "analyzed": (ws.gas_dir(sid) / "plumes.csv").exists(),
                "job": jobs.running_job(sid, "gas"),
            }
            for sid in ws.gas_scenes()
        ]

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
        path = ws.gas_scenes()[scene_id]
        import xarray as xr

        with xr.open_dataset(path) as ds:
            x0, dx, _, y0, _, dy = np.asarray(ds.attrs["geotransform"], dtype=float)
        for pl in out["plumes"]:
            if pl.get("source_lat") is not None:
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

    return app


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
