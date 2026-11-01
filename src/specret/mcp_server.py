"""FastMCP server exposing physics-grounded spectral identification tools.

Run with `uv run specret mcp` (stdio). An LLM agent can then answer questions such as
"what minerals are in this part of the scene, and why" by calling these tools instead of
guessing. Every answer names library spectra and the absorption features that support it.
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from specret.gas import tools as gas
from specret.scene import jobs
from specret.workspace import Workspace, jsonable

INSTRUCTIONS = """\
Tools for identifying materials from reflectance spectra (EMIT, AVIRIS, field spectrometers)
against the USGS spectral library. Start with `list_resources`. Scores are 0-1; matches carry
per-window absorption-feature evidence (band centre and depth of query vs. library spectrum).
Each answer carries `p_group_correct`, a calibrated probability that the best match's
mineral group is right, and a tier: confident (>= 0.5), uncertain (0.3-0.5) or
"likely unknown" (< 0.3, probably a material missing from the library). Pixels are often
mixtures: use `unmix_pixel` to split one into library endmembers with fractions.
Mineral groups follow configs/taxonomy.yaml.

Gas plumes (CH4, CO2) are found in EMIT L1B radiance scenes (`gas_scenes` in
`list_resources`): `detect_plumes` runs detection in the background, then `list_plumes`,
`explain_plume` and `quantify_plume` answer "is anything leaking here, what is it, and how
much?". Enhancements are in ppm·m, rates in kg/h with a 1-sigma uncertainty that is usually
dominated by the wind. Confidence is a calibrated probability that the plume is real and its
gas is right; surface materials that mimic gas (carbonates, hydrocarbons, plastics) are fitted
alongside and reported.
"""


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
        gas = {
            sid: {"analyzed": (ws.gas_dir(sid) / "plumes.csv").exists()} for sid in ws.gas_scenes()
        }
        return {
            "sensors": ws.sensors(),
            "scenes": scenes,
            "gas_scenes": gas,
            "library_size": len(ws.canonical),
        }

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
        row, col = ws.locate(scene, row, col, lat, lon)
        sensor = ws.sensor_for_scene(scene)
        out = ws.identify(scene.pixel(row, col), sensor, k)
        out["pixel"] = ws.pixel_info(scene, row, col)
        return jsonable(out)

    @mcp.tool
    def unmix_pixel(
        scene_id: str,
        row: int | None = None,
        col: int | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> dict:
        """Split a scene pixel into up to three library endmembers with relative fractions.

        Also returns the best single match and its calibrated confidence for comparison.
        """
        scene = ws.scene(scene_id)
        row, col = ws.locate(scene, row, col, lat, lon)
        sensor = ws.sensor_for_scene(scene)
        out = ws.identify(scene.pixel(row, col), sensor, k=3, unmix=True)
        out["pixel"] = ws.pixel_info(scene, row, col)
        return jsonable(out)

    @mcp.tool
    def unmix_spectrum(
        wavelengths_nm: list[float], reflectance: list[float], sensor: str = "emit"
    ) -> dict:
        """Split a reflectance spectrum into up to three library endmembers."""
        x = ws.to_sensor(wavelengths_nm, reflectance, sensor)
        return ws.identify(x, sensor, k=3, unmix=True)

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
        return {"job_id": jobs.start_map_job(ws, scene_id), "state": "running"}

    @mcp.tool
    def job_status(job_id: str) -> dict:
        """State of a background job (mapping or plume detection), with `progress` 0-1."""
        return jobs.job_status(job_id)

    @mcp.tool
    def detect_plumes(scene_id: str, rerun: bool = False) -> dict:
        """Find, identify and quantify CH4 and CO2 plumes in an EMIT L1B scene.

        Runs in the background (about a minute plus wind lookups); poll `job_status`, then use
        `list_plumes`. If the scene was already analyzed and `rerun` is false, returns its
        summary straight away.
        """
        if not rerun and (ws.gas_dir(scene_id) / "plumes.csv").exists():
            return {"state": "done", **gas.scene_gas_summary(ws.gas_dir(scene_id))}
        return {"job_id": jobs.start_gas_job(ws, scene_id), "state": "running"}

    @mcp.tool
    def list_plumes(
        scene_id: str, gas_name: str | None = None, min_q_kg_h: float | None = None
    ) -> dict:
        """Plumes of an analyzed scene, most confident first: gas, confidence, source lat/lon,
        emission rate and its uncertainty, and any confuser materials fitted alongside."""
        return gas.list_plumes(ws.gas_dir(scene_id), gas_name, min_q_kg_h)

    @mcp.tool
    def explain_plume(scene_id: str, plume_id: int) -> dict:
        """Why a plume was called what it was: the fitted terms (gases in ppm·m, confusers),
        the evidence features behind its confidence, the carbonate fraction under it, and
        the rate's uncertainty budget."""
        return gas.explain_plume(ws.gas_dir(scene_id), plume_id)

    @mcp.tool
    def quantify_plume(
        scene_id: str,
        plume_id: int,
        wind_u: float | None = None,
        wind_v: float | None = None,
        wind_sd: float | None = None,
    ) -> dict:
        """Emission rate (kg/h) of one plume, with the stored wind or your own 10 m wind
        (wind_u east, wind_v north, m/s; wind_sd its standard deviation, default half the speed)."""
        return gas.quantify_plume(ws.gas_dir(scene_id), plume_id, wind_u, wind_v, wind_sd)

    @mcp.tool
    def gas_signature(gas_name: str = "CH4", amf: float = 2.2, h2o_gcm2: float = 1.5) -> dict:
        """Unit absorption spectrum of a gas on EMIT bands: the change in ln(radiance) per
        ppm·m of plume, for a two-way air-mass factor and water column."""
        return gas.gas_signature(gas_name, amf, h2o_gcm2)

    @mcp.tool
    def scene_summary(
        scene_id: str,
        bbox: list[float] | None = None,
        min_p: float = 0.3,
        top: int = 12,
    ) -> dict:
        """Mineral-group composition of a mapped scene, optionally inside [W, S, E, N], plus a
        gas section (plumes by gas, total rate) when the scene's radiance has been analyzed.

        Pixels whose calibrated P(group correct) is below `min_p` are reported as
        "likely_unknown".
        """
        from specret.scene.mapper import SceneMap, group_fractions

        path = ws.map_dir(scene_id) / "scene_map.npz"
        gas_part = gas.gas_section(ws, scene_id)
        if not path.exists():
            if gas_part is not None:
                return jsonable({"scene_id": scene_id, "gas": gas_part})
            raise ValueError(f"scene '{scene_id}' is not mapped yet; call map_scene first")
        smap = SceneMap.load(path)
        lat = lon = None
        if bbox is not None:
            scene = ws.scene(scene_id)
            lat, lon = getattr(scene, "lat", None), getattr(scene, "lon", None)
        df = group_fractions(smap, lat, lon, tuple(bbox) if bbox else None, min_p)
        rows = [
            {
                "group": g,
                "pixels": int(r.pixels),
                "fraction": float(r.fraction),
                "median_p_correct": float(r.median_p),
            }
            for g, r in df.head(top).iterrows()
        ]
        return jsonable(
            {
                "scene_id": scene_id,
                "bbox": bbox,
                "groups": rows,
                "mapped_pixels": int(df["pixels"].sum()),
                "gas": gas_part,
            }
        )

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
