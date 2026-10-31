"""Local web viewer: click a pixel on a mineral map, or drop in a field spectrum.

Run with `uv run specret view` and open http://127.0.0.1:8765. The page shows the mapped
scene; clicking a pixel plots its spectrum against the top matches, shades the diagnostic
feature windows, and lists the feature evidence for each match.
"""

from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from specret.scene.mapper import SceneMap
from specret.scene.outputs import group_codes, group_colors
from specret.workspace import UNKNOWN_BELOW, Workspace, jsonable

STATIC = Path(__file__).parent / "static"
UNKNOWN_CODE = 255  # map code for pixels whose calibrated confidence is below UNKNOWN_BELOW
UNKNOWN_RGB = (128, 128, 128)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(ws: Workspace | None = None) -> FastAPI:
    ws = ws or Workspace()
    app = FastAPI(title="specret viewer")

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
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/scenes")
    def scenes() -> list[dict]:
        return [
            {
                "scene_id": sid,
                "kind": info.kind,
                "mapped": (ws.map_dir(sid) / "scene_map.npz").exists(),
            }
            for sid, info in ws.scenes().items()
        ]

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
    def pixel(scene_id: str, x: int, y: int) -> dict:
        """Identify the pixel under ortho-image coordinates (x, y)."""
        scene = ws.scene(scene_id)
        if not (0 <= y < scene.glt_y.shape[0] and 0 <= x < scene.glt_x.shape[1]):
            raise HTTPException(400, "outside the image")
        gy, gx = int(scene.glt_y[y, x]), int(scene.glt_x[y, x])
        if gy <= 0 or gx <= 0:
            raise HTTPException(404, "no data at this location")
        row, col = gy - 1, gx - 1
        sensor = ws.sensor_for_scene(scene)
        spectrum = scene.pixel(row, col)
        result = ws.identify(spectrum, sensor, k=5, unmix=True)
        result["pixel"] = {
            "row": row,
            "col": col,
            "lat": float(scene.lat[row, col]),
            "lon": float(scene.lon[row, col]),
        }
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

    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
