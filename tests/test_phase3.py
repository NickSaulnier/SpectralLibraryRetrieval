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
    # The upload path resamples to the field sensor; reuse the toy library under that name.
    (workspace_root / "data/resampled/field_asd").mkdir(parents=True)
    lib = SensorLibrary.load(workspace_root / "data/resampled/emit/library.npz")
    SensorLibrary(
        Sensor("field_asd", WL, lib.sensor.fwhm_nm, lib.sensor.good_mask),
        lib.spectrum_ids,
        lib.matrix,
    ).save(workspace_root / "data/resampled/field_asd/library.npz")
    client = TestClient(create_app(ws))
    assert "Spectral Match Viewer" in client.get("/").text
    assert client.get("/api/scenes").json() == []

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
