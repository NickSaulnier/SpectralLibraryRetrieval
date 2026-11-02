import asyncio
from pathlib import Path

import numpy as np
import pytest
import rasterio

from specret.schema import SpectrumRecord, write_library
from specret.sensors.library import SensorLibrary
from specret.sensors.sensor import Sensor

WL = np.arange(1990.0, 2420.0, 7.5)
SENSOR = Sensor("emit", WL, np.full(WL.size, 8.0), np.ones(WL.size, bool))


def spectrum(*features, base=0.5):
    s = base * np.ones_like(WL)
    for c, d in features:
        s = s * (1 - d * np.exp(-0.5 * ((WL - c) / 12.0) ** 2))
    return s


MATERIALS = {
    "Kaolinite": spectrum((2165, 0.12), (2207, 0.18)),
    "Muscovite": spectrum((2200, 0.2), (2350, 0.06)),
    "Calcite": spectrum((2340, 0.15)),
    "Dolomite": spectrum((2318, 0.15)),
}


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """A miniature project: canonical library, one resampled sensor library."""
    records = [
        SpectrumRecord(f"t:{i}", f"s{i}", "toy", name, name, WL, refl, chapter="M")
        for i, (name, refl) in enumerate(MATERIALS.items())
    ]
    write_library(records, tmp_path / "data/canonical/library.parquet", classify=True)
    ids = np.array([r.spectrum_id for r in records])
    SensorLibrary(SENSOR, ids, np.vstack(list(MATERIALS.values()))).save(
        tmp_path / "data/resampled/emit/library.npz"
    )
    return tmp_path


class FakeScene:
    """Minimal scene: a 3x4 swath with an identity geometry lookup."""

    path = "fake_scene"
    geotransform = (-117.3, 0.001, 0.0, 37.6, 0.0, -0.001)
    crs_wkt = "EPSG:4326"

    def __init__(self):
        cube = np.empty((3, 4, WL.size))
        names = list(MATERIALS)
        for r in range(3):
            for c in range(4):
                cube[r, c] = 0.9 * MATERIALS[names[(r + c) % 4]]
        cube[0, 0] = np.nan  # no data
        self.cube = cube
        self.centers_nm = WL
        self.shape = (3, 4)
        self.lat = 37.6 - np.arange(3)[:, None] * 0.001 + np.zeros((1, 4))
        self.lon = -117.3 + np.arange(4)[None, :] * 0.001 + np.zeros((3, 1))
        self.glt_y, self.glt_x = np.mgrid[1:4, 1:5]

    def block(self, rows, cols):
        return self.cube[rows, cols].astype(np.float32)

    def pixel(self, row, col):
        return self.cube[row, col]

    def orthorectify(self, image, fill=np.nan):
        from specret.io.emit import orthorectify

        return orthorectify(image, self.glt_x, self.glt_y, fill)


def test_map_scene_and_geotiffs(workspace_root, tmp_path):
    from specret.scene.mapper import SceneMap, group_fractions, map_scene
    from specret.scene.outputs import write_scene_maps
    from specret.schema import read_library
    from specret.search.hybrid import HybridRetriever

    canonical = read_library(workspace_root / "data/canonical/library.parquet")
    lib = SensorLibrary.load(workspace_root / "data/resampled/emit/library.npz")
    scene = FakeScene()
    smap = map_scene(
        scene,
        HybridRetriever(lib, canonical, k_recall=4, n_rerank=4),
        canonical,
        k=2,
        progress=False,
    )
    assert not smap.valid[0, 0] and smap.valid.sum() == 11
    assert smap.top1_group()[0, 1] == "white_mica"  # (0 + 1) % 4 -> Muscovite
    assert smap.top1_group()[1, 2] == "dolomite"  # (1 + 2) % 4 -> Dolomite

    smap.save(tmp_path / "m.npz")
    again = SceneMap.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(again.top_rows, smap.top_rows)

    fr = group_fractions(smap)
    assert fr["pixels"].sum() == 11

    paths = write_scene_maps(smap, scene, canonical, tmp_path / "maps")
    with rasterio.open(paths["group"]) as src:
        assert src.crs.to_epsg() == 4326
        assert src.read(1)[0, 0] == 0  # no data
        assert src.colormap(1)[1][3] == 255


def test_mcp_tools(workspace_root):
    from fastmcp import Client

    from specret.mcp_server import build_server
    from specret.workspace import Workspace

    ws = Workspace(root=workspace_root, chapters=None)

    async def go():
        async with Client(build_server(ws)) as c:
            res = (await c.call_tool("list_resources", {})).data
            assert res["sensors"] == ["emit"] and res["library_size"] == 4
            assert res["gas_scenes"] == {}
            names = {t.name for t in await c.list_tools()}
            assert {"detect_plumes", "list_plumes", "explain_plume", "quantify_plume"} <= names
            out = (
                await c.call_tool(
                    "identify_spectrum",
                    {
                        "wavelengths_nm": WL.tolist(),
                        "reflectance": (0.7 * MATERIALS["Calcite"]).tolist(),
                    },
                )
            ).data
            assert out["summary"]["best_group"] == "calcite"
            assert out["matches"][0]["features"][0]["window"] == "mg_oh_co3"
            entry = (await c.call_tool("library_entry", {"spectrum_id": "t:2"})).data
            assert entry["material_name"] == "Calcite"
            exp = (
                await c.call_tool(
                    "explain_match",
                    {
                        "spectrum_id": "t:3",
                        "wavelengths_nm": WL.tolist(),
                        "reflectance": MATERIALS["Calcite"].tolist(),
                    },
                )
            ).data
            assert exp["feature_fit_score"] < 0.5  # dolomite's 2.32 um band misses calcite's 2.34

    asyncio.run(go())


def test_viewer_field_upload(workspace_root):
    from fastapi.testclient import TestClient

    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    ws = Workspace(root=workspace_root, chapters=None)
    _add_field_library(workspace_root)  # the upload path resamples to the field sensor
    client = TestClient(create_app(ws))
    assert "Spectral Match Viewer" in client.get("/").text
    assert client.get("/api/scenes").json() == []
    assert client.get("/api/gas/scenes").json() == []
    assert client.get("/api/gas/20230608T162322_2315911_014/plumes").status_code == 404

    csv = "wavelength,rock\n" + "\n".join(
        f"{w},{v}" for w, v in zip(WL, 0.8 * MATERIALS["Kaolinite"], strict=True)
    )
    res = client.post("/api/field", files={"file": ("rock.csv", csv, "text/csv")})
    assert res.status_code == 200
    body = res.json()[0]
    assert body["summary"]["best_group"] == "kaolin"
    assert len(body["query"]) == len(body["wavelengths_nm"])
    assert (
        client.post(
            "/api/field", files={"file": ("x.exe", b"MZ", "application/octet-stream")}
        ).status_code
        == 400
    )


def _add_field_library(root: Path) -> None:
    """Reuse the toy library under the field sensor's name."""
    lib = SensorLibrary.load(root / "data/resampled/emit/library.npz")
    SensorLibrary(
        Sensor("field_asd", WL, lib.sensor.fwhm_nm, lib.sensor.good_mask),
        lib.spectrum_ids,
        lib.matrix,
    ).save(root / "data/resampled/field_asd/library.npz")


def test_cli_identify_field(workspace_root, monkeypatch, capsys):
    import json

    import pandas as pd

    from specret.cli import main

    _add_field_library(workspace_root)
    field = workspace_root / "field"
    field.mkdir()
    for name in ("Kaolinite", "Calcite"):
        rows = "\n".join(f"{w},{v}" for w, v in zip(WL, 0.8 * MATERIALS[name], strict=True))
        (field / f"{name.lower()}.csv").write_text(f"wavelength,{name}\n{rows}\n")
    monkeypatch.chdir(workspace_root)

    main(["identify", "field", "--chapters", "all", "--csv", "out/report.csv"])
    text = capsys.readouterr().out
    assert "kaolin" in text and "calcite" in text and "wrote 2 rows" in text
    report = pd.read_csv("out/report.csv")
    assert list(report["best_group"]) == ["calcite", "kaolin"]  # files in sorted order

    main(["identify", "field/*.csv", "--chapters", "all", "--json", "-k", "2"])
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 2 and all(len(r["matches"]) == 2 for r in results)

    with pytest.raises(SystemExit, match="no such file"):
        main(["identify", "missing.asd"])


def test_cli_identify_scene_pixel(workspace_root, monkeypatch, capsys):
    import specret.workspace
    from specret.cli import main
    from specret.io.emit import EmitScene

    class GeoScene(FakeScene):
        nearest_pixel = EmitScene.nearest_pixel

    scene_file = workspace_root / "data/scenes/emit/EMIT_L2A_RFL_fake.nc"
    scene_file.parent.mkdir(parents=True)
    scene_file.touch()
    monkeypatch.setattr(specret.workspace, "open_scene", lambda path: GeoScene())
    monkeypatch.chdir(workspace_root)

    # Pixel (0, 2) holds Calcite (see FakeScene); the scene is named by id, not path.
    main(["identify", "EMIT_L2A_RFL_fake", "--row", "0", "--col", "2", "--chapters", "all"])
    text = capsys.readouterr().out
    assert "pixel row 0, col 2" in text and "best match: Calcite" in text

    main(["identify", "EMIT_L2A_RFL_fake", "--lat", "37.599", "--lon", "-117.299", "--unmix"])
    assert "pixel row 1, col 1" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="outside"):
        main(["identify", "EMIT_L2A_RFL_fake", "--row", "9", "--col", "0"])
    with pytest.raises(SystemExit, match="give row and col"):
        main(["identify", "EMIT_L2A_RFL_fake"])


UTM11_MAP_INFO = ["UTM", "1.000", "1.000", "500000.0", "4160000.0", "15.0", "15.0", "11", "North",
                  "WGS-84", "units=Meters"]  # fmt: skip


def write_aviris(path: Path, map_info: list | None = UTM11_MAP_INFO) -> Path:
    """A 3x4 ENVI cube laid out like FakeScene, with -9999 fill at (0, 0)."""
    import spectral.io.envi as envi

    cube = np.stack([FakeScene().cube[r] for r in range(3)]).astype(np.float32)
    cube[0, 0] = -9999
    meta = {
        "wavelength": WL.tolist(),
        "fwhm": [8.0] * WL.size,
        "wavelength units": "Nanometers",
        "data ignore value": -9999,
    }
    if map_info:
        meta["map info"] = map_info
    path.parent.mkdir(parents=True, exist_ok=True)
    envi.save_image(str(path), cube, metadata=meta, force=True)
    return path


def test_aviris_georeference(tmp_path):
    from specret.io.aviris import open_aviris

    scene = open_aviris(write_aviris(tmp_path / "f_geo_img.hdr"))
    assert scene.geotransform == (500000.0, 15.0, 0.0, 4160000.0, 0.0, -15.0)
    assert np.isnan(scene.pixel(0, 0)).all()
    lat, lon = scene.pixel_latlon(1, 2)
    assert 37.5 < lat < 37.6 and -117.01 < lon < -116.99  # UTM 11N, near its central meridian
    assert scene.nearest_pixel(lat, lon) == (1, 2)
    with pytest.raises(ValueError, match="outside"):
        scene.nearest_pixel(lat + 1, lon)

    bare = open_aviris(write_aviris(tmp_path / "f_bare_img.hdr", map_info=None))
    assert bare.geotransform is None and bare.pixel_latlon(0, 1) is None
    with pytest.raises(ValueError, match="no map info"):
        bare.nearest_pixel(lat, lon)


def test_aviris_map_viewer_and_identify(workspace_root, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from specret.cli import main
    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    hdr = write_aviris(workspace_root / "data/scenes/aviris/f_test_img.hdr")
    monkeypatch.chdir(workspace_root)
    main(["map-scene", "data/resampled/emit/library.npz", "--scene", str(hdr), "--chapters", "M"])
    with rasterio.open(workspace_root / "outputs/maps/f_test_img/group.tif") as src:
        assert src.crs.to_epsg() == 32611 and src.transform.c == 500000.0
        assert src.read(1)[0, 0] == 0  # fill pixel -> no data

    client = TestClient(create_app(Workspace(root=workspace_root, chapters=None)))
    assert client.get("/api/scenes").json() == [
        {"scene_id": "f_test_img", "kind": "aviris", "mapped": True, "job": None}
    ]
    assert client.get("/api/map/f_test_img.png").status_code == 200
    body = client.get("/api/pixel/f_test_img", params={"x": 2, "y": 0}).json()
    assert body["summary"]["best_group"] == "calcite"  # (0 + 2) % 4 -> Calcite
    assert 37.5 < body["pixel"]["lat"] < 37.6
    assert client.get("/api/pixel/f_test_img", params={"x": 0, "y": 0}).status_code == 404
    assert client.get("/api/pixel/f_test_img", params={"x": 9, "y": 0}).status_code == 400

    lat, lon = body["pixel"]["lat"], body["pixel"]["lon"]
    main(["identify", "f_test_img", "--lat", str(lat), "--lon", str(lon), "--chapters", "all"])
    assert "pixel row 0, col 2" in capsys.readouterr().out


def test_viewer_maps_scene_in_background(workspace_root):
    import time

    from fastapi.testclient import TestClient

    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    write_aviris(workspace_root / "data/scenes/aviris/f_job_img.hdr")
    client = TestClient(create_app(Workspace(root=workspace_root, chapters=None)))
    assert "plotly" in client.get("/static/plotly.min.js").text[:500].lower()
    assert client.get("/api/scenes").json()[0]["mapped"] is False
    assert client.get("/api/map/f_job_img.png").status_code == 404

    job_id = client.post("/api/map/f_job_img").json()["job_id"]
    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] not in ("queued", "running"):
            break
        time.sleep(0.1)
    assert job["state"] == "done" and job["progress"] == 1.0
    assert client.get("/api/scenes").json()[0]["mapped"] is True
    assert client.get("/api/map/f_job_img.png").status_code == 200

    px = client.get("/api/pixel/f_job_img", params={"x": 2, "y": 1}).json()["pixel"]
    again = client.get("/api/pixel/f_job_img", params={"lat": px["lat"], "lon": px["lon"]})
    assert again.json()["pixel"] == px  # lat/lon lands back on the same pixel
    assert client.get("/api/pixel/f_job_img", params={"lat": 0, "lon": 0}).status_code == 400
    assert client.post("/api/map/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_viewer_pixel_by_lat_lon_on_emit_grid(workspace_root, monkeypatch):
    from fastapi.testclient import TestClient

    import specret.workspace
    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    scene_file = workspace_root / "data/scenes/emit/EMIT_L2A_RFL_fake.nc"
    scene_file.parent.mkdir(parents=True)
    scene_file.touch()
    monkeypatch.setattr(specret.workspace, "open_scene", lambda path: FakeScene())
    client = TestClient(create_app(Workspace(root=workspace_root, chapters=None)))

    # FakeScene's grid starts at (37.6, -117.3) with 0.001 degree cells and an identity GLT.
    body = client.get(
        "/api/pixel/EMIT_L2A_RFL_fake", params={"lat": 37.5995, "lon": -117.2975}
    ).json()
    assert (body["pixel"]["x"], body["pixel"]["y"]) == (2, 0)
    assert body["summary"]["best_group"] == "calcite"  # (0 + 2) % 4 -> Calcite
    assert client.get("/api/pixel/EMIT_L2A_RFL_fake").status_code == 400


def test_viewer_serves_the_page_it_started_with(workspace_root, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import specret.viewer.app
    from specret.workspace import Workspace

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>v1</p>", encoding="utf-8")
    monkeypatch.setattr(specret.viewer.app, "STATIC", static)
    client = TestClient(specret.viewer.app.create_app(Workspace(root=workspace_root)))
    (static / "index.html").write_text("<p>v2</p>", encoding="utf-8")  # code updated on disk
    assert client.get("/").text == "<p>v1</p>"


def test_viewer_serves_the_react_build_and_keeps_the_legacy_page(
    workspace_root, tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    import specret.viewer.app
    from specret.workspace import Workspace

    static = tmp_path / "static"
    (static / "app" / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<p>legacy</p>", encoding="utf-8")
    (static / "app" / "index.html").write_text(
        '<script src="/app/assets/index-x.js"></script>', encoding="utf-8"
    )
    (static / "app" / "assets" / "index-x.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(specret.viewer.app, "STATIC", static)
    client = TestClient(specret.viewer.app.create_app(Workspace(root=workspace_root)))
    assert "/app/assets/index-x.js" in client.get("/").text
    assert client.get("/legacy").text == "<p>legacy</p>"
    assert client.get("/app/assets/index-x.js").text == "console.log(1)"


def test_committed_viewer_build_is_complete():
    """The React build is committed: its index must point at assets that exist."""
    import re

    from specret.viewer.app import STATIC

    index = STATIC / "app" / "index.html"
    assert index.exists(), "run `pnpm run build` (or npm) in src/specret/viewer/web"
    refs = re.findall(r'(?:src|href)="/app/(assets/[^"]+)"', index.read_text(encoding="utf-8"))
    assert refs and all((STATIC / "app" / r).exists() for r in refs)


def test_committed_viewer_build_matches_its_sources():
    """Fails when src/specret/viewer/web changed without rebuilding (pnpm run build)."""
    import importlib.util

    from specret.viewer.app import STATIC

    script = Path(__file__).resolve().parents[1] / "scripts" / "viewer_source_hash.py"
    spec = importlib.util.spec_from_file_location("viewer_source_hash", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    recorded = (STATIC / "app" / "source-hash.txt").read_text(encoding="utf-8").strip()
    assert recorded == mod.source_hash(), "viewer sources changed: rebuild with `pnpm run build`"


def test_viewer_pixel_evidence_matches_explain_match(workspace_root, monkeypatch):
    """V3: the pixel panel's per-window evidence is what MCP explain_match reports, and the
    unmixing decomposition adds up."""
    from fastapi.testclient import TestClient

    import specret.workspace
    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    scene_file = workspace_root / "data/scenes/emit/EMIT_L2A_RFL_fake.nc"
    scene_file.parent.mkdir(parents=True)
    scene_file.touch()
    monkeypatch.setattr(specret.workspace, "open_scene", lambda path: FakeScene())
    ws = Workspace(root=workspace_root, chapters=None)
    client = TestClient(create_app(ws))
    body = client.get("/api/pixel/EMIT_L2A_RFL_fake", params={"x": 2, "y": 0}).json()
    top = body["matches"][0]
    ref = ws.explain(FakeScene().pixel(0, 2), "emit", top["spectrum_id"])
    assert top["features"] == ref["windows"][: len(top["features"])]

    prof = {p["window"]: p for p in body["profiles"]}
    assert prof and all(len(p["matches"]) == min(3, len(body["matches"])) for p in prof.values())
    for f in top["features"]:
        p = prof[f["window"]]
        assert p["depth"][0] == pytest.approx(f["query_depth"], abs=1e-9)
        assert p["center_nm"][1] == pytest.approx(f["cand_center_nm"], abs=1e-9)

    u = body["unmixing"]["spectra"]
    model = np.sum(u["components"], axis=0) + np.asarray(u["baseline"])
    assert model == pytest.approx(np.asarray(u["model"]))
    assert np.asarray(u["model"]) + np.asarray(u["residual"]) == pytest.approx(
        np.asarray(body["query"])
    )
