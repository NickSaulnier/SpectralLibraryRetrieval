"""FastMCP server exposing physics-grounded spectral identification tools.

Run with `uv run specret mcp` (stdio). An LLM agent can then answer questions such as
"what minerals are in this part of the scene, and why" by calling these tools instead of
guessing. Every answer names library spectra and the absorption features that support it.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path

from fastmcp import FastMCP

from specret.workspace import Workspace, jsonable

INSTRUCTIONS = """\
Tools for identifying materials from reflectance spectra (EMIT, AVIRIS, field spectrometers)
against the USGS spectral library. Start with `list_resources`. Scores are 0-1; matches carry
per-window absorption-feature evidence (band centre and depth of query vs. library spectrum).
Confidence is provisional until open-set rejection is added: treat 'low' as "possibly a
material not in the library". Mineral groups follow configs/taxonomy.yaml.
"""

JOBS: dict[str, dict] = {}


def build_server(ws: Workspace | None = None) -> FastMCP:
    ws = ws or Workspace()
    mcp = FastMCP("specret", instructions=INSTRUCTIONS)

    @mcp.tool
    def list_resources() -> dict:
        """List sensors with resampled libraries, available scenes, and which scenes are mapped."""
        scenes = {
            sid: {
                "kind": info.kind,
                "mapped": (ws.map_dir(sid) / "scene_map.npz").exists(),
            }
            for sid, info in ws.scenes().items()
        }
        return {"sensors": ws.sensors(), "scenes": scenes, "library_size": len(ws.canonical)}

    @mcp.tool
    def identify_spectrum(
        wavelengths_nm: list[float],
        reflectance: list[float],
        sensor: str = "emit",
        k: int = 5,
    ) -> dict:
        """Identify the material(s) in one reflectance spectrum (0-1 scale).

        If the wavelengths are not exactly the sensor's bands, the spectrum is resampled onto
        them (treat it as high resolution). Returns ranked matches with feature evidence.
        """
        x = ws.to_sensor(wavelengths_nm, reflectance, sensor)
        return ws.identify(x, sensor, k)

    @mcp.tool
    def identify_pixel(
        scene_id: str,
        row: int | None = None,
        col: int | None = None,
        lat: float | None = None,
        lon: float | None = None,
        k: int = 5,
    ) -> dict:
        """Identify the material at one scene pixel, by (row, col) or by (lat, lon)."""
        scene = ws.scene(scene_id)
        if lat is not None and lon is not None:
            if not hasattr(scene, "nearest_pixel"):
                raise ValueError("lat/lon lookup needs an EMIT scene; give row and col")
            row, col = scene.nearest_pixel(lat, lon)
        if row is None or col is None:
            raise ValueError("give row and col, or lat and lon")
        sensor = ws.sensor_for_scene(scene)
        out = ws.identify(scene.pixel(row, col), sensor, k)
        out["pixel"] = {"row": row, "col": col}
        if hasattr(scene, "lat"):
            out["pixel"].update(lat=float(scene.lat[row, col]), lon=float(scene.lon[row, col]))
        return jsonable(out)

    @mcp.tool
    def explain_match(
        spectrum_id: str,
        wavelengths_nm: list[float] | None = None,
        reflectance: list[float] | None = None,
        scene_id: str | None = None,
        row: int | None = None,
        col: int | None = None,
        sensor: str = "emit",
    ) -> dict:
        """Explain why a library spectrum does or does not match a query.

        The query is either a spectrum (wavelengths_nm + reflectance) or a scene pixel
        (scene_id + row + col). Returns per-window band centres, depths and shape fit, the
        USGS purity grade, and the sample's description.
        """
        if scene_id is not None:
            scene = ws.scene(scene_id)
            sensor = ws.sensor_for_scene(scene)
            x = scene.pixel(row, col)
        elif wavelengths_nm is not None and reflectance is not None:
            x = ws.to_sensor(wavelengths_nm, reflectance, sensor)
        else:
            raise ValueError("give a spectrum or a scene pixel")
        return ws.explain(x, sensor, spectrum_id)

    @mcp.tool
    def identify_field_spectrum(path: str, k: int = 5) -> list[dict]:
        """Identify a field spectrometer file (.asd binary, or CSV/TXT export) on this machine."""
        p = Path(path)
        if not p.is_file() or p.suffix.lower() not in (".asd", ".csv", ".txt", ".tsv"):
            raise ValueError("path must be an existing .asd, .csv, .txt or .tsv file")
        return ws.identify_field(p, k)

    @mcp.tool
    def library_entry(spectrum_id: str) -> dict:
        """Metadata for one library spectrum: material, group, class, purity grade and notes."""
        meta = ws.canonical.set_index("spectrum_id")
        if spectrum_id not in meta.index:
            raise ValueError(f"unknown spectrum_id '{spectrum_id}'")
        m = meta.loc[spectrum_id]
        fields = (
            "material_name",
            "group",
            "material_class",
            "chapter",
            "sample_id",
            "instrument",
            "measurement",
            "purity_flag",
            "purity_note",
        )
        return jsonable({"spectrum_id": spectrum_id, **{f: m.get(f) for f in fields}})

    @mcp.tool
    def map_scene(scene_id: str) -> dict:
        """Start mapping every pixel of a scene in the background (a few minutes for EMIT).

        Returns a job id; poll `job_status`. Results feed `scene_summary`.
        """
        if scene_id not in ws.scenes():
            raise ValueError(f"unknown scene '{scene_id}'")
        job_id = uuid.uuid4().hex[:8]
        JOBS[job_id] = {"scene_id": scene_id, "state": "running", "started": time.time()}

        def work() -> None:
            from specret.scene.mapper import map_scene as run_map
            from specret.scene.outputs import write_scene_maps

            try:
                scene = ws.scene(scene_id)
                r = ws.retriever(ws.sensor_for_scene(scene))
                smap = run_map(scene, r, ws.canonical, progress=False)
                out = ws.map_dir(scene_id)
                smap.save(out / "scene_map.npz")
                paths = write_scene_maps(smap, scene, ws.canonical, out)
                JOBS[job_id].update(
                    state="done",
                    seconds=round(smap.seconds),
                    outputs={k: str(v) for k, v in paths.items()},
                )
            except Exception as exc:  # reported to the agent through job_status
                JOBS[job_id].update(state="failed", error=f"{exc}\n{traceback.format_exc()}")

        threading.Thread(target=work, daemon=True).start()
        return {"job_id": job_id, "state": "running"}

    @mcp.tool
    def job_status(job_id: str) -> dict:
        """State of a background mapping job."""
        if job_id not in JOBS:
            raise ValueError(f"unknown job '{job_id}'")
        job = dict(JOBS[job_id])
        job["elapsed_s"] = round(time.time() - job.pop("started"))
        return job

    @mcp.tool
    def scene_summary(
        scene_id: str,
        bbox: list[float] | None = None,
        min_score: float = 0.6,
        top: int = 12,
    ) -> dict:
        """Mineral-group composition of a mapped scene, optionally inside [W, S, E, N].

        Pixels whose best-match score is below `min_score` are reported as low_confidence.
        """
        from specret.scene.mapper import SceneMap, group_fractions

        path = ws.map_dir(scene_id) / "scene_map.npz"
        if not path.exists():
            raise ValueError(f"scene '{scene_id}' is not mapped yet; call map_scene first")
        smap = SceneMap.load(path)
        lat = lon = None
        if bbox is not None:
            scene = ws.scene(scene_id)
            lat, lon = scene.lat, scene.lon
        df = group_fractions(smap, lat, lon, tuple(bbox) if bbox else None, min_score)
        rows = [
            {
                "group": g,
                "pixels": int(r.pixels),
                "fraction": float(r.fraction),
                "median_score": float(r.median_score),
            }
            for g, r in df.head(top).iterrows()
        ]
        return jsonable(
            {
                "scene_id": scene_id,
                "bbox": bbox,
                "groups": rows,
                "mapped_pixels": int(df["pixels"].sum()),
            }
        )

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
