"""Viewer Phase V4 (PLAN_VIEWER.md): gas across sensors, swath gridding, compare view.

Synthetic data only.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine


def _swath(lines=60, samples=40, step_m=5.0, heading_deg=30.0, lat0=35.4, lon0=-119.0):
    """Pixel centres of a straight flight line on a slant."""
    t = np.radians(heading_deg)
    i, j = np.mgrid[0:lines, 0:samples].astype(float)
    north = step_m * (i * np.cos(t) - j * np.sin(t))
    east = step_m * (i * np.sin(t) + j * np.cos(t))
    lat = lat0 + north / 111_320.0
    lon = lon0 + east / (111_320.0 * np.cos(np.radians(lat0)))
    return lat, lon


def test_swath_grid_takes_the_nearest_pixel():
    from specret.gas.grid import spacing_m, swath_grid

    lat, lon = _swath()
    assert spacing_m(lat, lon) == pytest.approx(5.0, rel=1e-3)
    g = swath_grid(lat, lon)
    values = np.arange(lat.size, dtype=np.float32).reshape(lat.shape)
    out = g.take(values)
    filled = np.isfinite(out)
    # A slanted strip covers well under its bounding box, and every pixel lands somewhere
    assert 0.3 < filled.mean() < 0.9
    assert np.unique(out[filled]).size > 0.8 * lat.size
    # Each filled cell holds a pixel within 1.5 cells of its centre
    rows, cols = np.nonzero(filled)
    cx, cy = g.transform @ (cols + 0.5, rows + 0.5)
    src = out[rows, cols].astype(int)
    d_lat = (lat.ravel()[src] - cy) * 111_320
    d_lon = (lon.ravel()[src] - cx) * 111_320 * np.cos(np.radians(35.4))
    assert np.hypot(d_lat, d_lon).max() <= 1.5 * 5.0 + 1e-6


def test_write_flight_grids_maps_and_outlines_plumes(tmp_path, monkeypatch):
    from specret.gas import lwir

    lat, lon = _swath()
    shape = lat.shape
    alpha = np.zeros(shape, dtype=np.float32)
    alpha[20:30, 10:20] = 300.0
    cands = np.zeros(shape, dtype=np.int32)
    cands[20:30, 10:20] = 2  # the second candidate row becomes plume 1
    cands[40:45, 5:10] = 1  # a candidate without a gas: not a plume
    low = np.zeros(shape, dtype=bool)
    low[:, :3] = True
    maps = lwir.FlightMaps(
        {"NH3": alpha}, np.full(shape, 4.0, dtype=np.float32), low, cands, spectra=[]
    )
    df = pd.DataFrame(
        {
            "detection_gas": ["NH3", "NH3"],
            "candidate_id": [1, 2],
            "segment": [0, 0],
            "gas": [None, "NH3"],
            "gas_z": [1.0, 9.0],
            "lat": [lat[42, 7], lat[25, 15]],
            "lon": [lon[42, 7], lon[25, 15]],
        }
    )
    monkeypatch.setattr(lwir, "HYTES_MODEL", str(tmp_path / "none.json"))
    flight = SimpleNamespace(name="2014-07-08.190512.Test.Line1", geolocation=lambda: (lat, lon))
    folder = lwir.write_flight(flight, df, maps, tmp_path)

    plumes = pd.read_csv(folder / "plumes.csv")
    assert plumes["plume_id"].tolist() == [1] and plumes["gas"].tolist() == ["NH3"]
    assert plumes["source_lat"].iloc[0] == pytest.approx(lat[25, 15])
    gj = json.loads((folder / "plumes.geojson").read_text())
    (feat,) = gj["features"]
    assert feat["geometry"]["type"] == "Polygon" and feat["properties"]["plume_id"] == 1
    with rasterio.open(folder / "plume_mask.tif") as ds:
        lab = ds.read(1)
    with rasterio.open(folder / "nh3_enh.tif") as ds:
        enh = ds.read(1)
        assert ds.crs.to_epsg() == 4326
    assert set(np.unique(lab)) == {0, 1}
    assert np.all(enh[lab == 1] == 300.0)
    with rasterio.open(folder / "low_contrast.tif") as ds:
        mask = ds.read(1)
        assert ds.nodata == 0
    assert set(np.unique(mask)) == {0, 1} and (folder / "delta_t.tif").exists()


def test_gas_sources_cover_all_sensors(tmp_path):
    from specret.viewer.layers import scene_entries, scene_layers
    from specret.workspace import Workspace

    scenes = tmp_path / "data/scenes"
    (scenes / "emit_gas").mkdir(parents=True)
    (scenes / "emit_gas/EMIT_L1B_RAD_001_20230608T162322_2315911_014.nc").touch()
    (scenes / "avirisng/ang20161001t205456").mkdir(parents=True)
    (scenes / "avirisng/ang20161001t205456/ang20161001t205456_rdn_v1n2_img.hdr").touch()
    flight = "2014-07-06.194610.SaltonSea.Line2"
    (scenes / "hytes" / flight).mkdir(parents=True)
    (scenes / "hytes" / flight / "x-level_1a.hdr").touch()
    ws = Workspace(root=tmp_path)
    src = ws.gas_sources()
    assert {k: v.sensor for k, v in src.items()} == {
        "20230608T162322_2315911_014": "EMIT",
        "ang20161001t205456": "AVIRIS-NG",
        flight: "HyTES",
    }
    assert ws.gas_dir(flight) == tmp_path / "outputs/gas/hytes" / flight

    # A HyTES flight's outputs become layers, with the thermal-contrast context
    out = ws.gas_dir(flight)
    out.mkdir(parents=True)
    for name in ("nh3_enh.tif", "delta_t.tif", "low_contrast.tif"):
        with rasterio.open(
            out / name,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=Affine(1e-4, 0, -115.6, 0, -1e-4, 33.2),
        ) as ds:
            ds.write(np.ones((1, 4, 4), dtype=np.float32))
    e = {x.key: x for x in scene_entries(ws)}[flight]
    assert e.sensor == "HyTES" and e.gas_scene == flight and e.bounds is not None
    names = [x.name for x in scene_layers(ws, e)]
    assert {"nh3_enh", "delta_t", "low_contrast"} <= set(names)


def test_compare_view_over_the_visible_area():
    from specret.viewer.compare import Pair, compare_view

    rng = np.random.default_rng(0)
    n = 200
    ref = rng.normal(0, 100, (n, n)).astype(np.float32)
    plume = np.zeros((n, n), dtype=bool)
    plume[50:80, 50:80] = True
    ref[plume] += 1000
    ours = (ref + rng.normal(0, 50, (n, n))).astype(np.float32)
    lat = 35 - np.arange(n)[:, None] * 1e-3 + np.zeros((1, n))
    lon = -119 + np.arange(n)[None, :] * 1e-3 + np.zeros((n, 1))
    pair = Pair(ours, ref, plume, None, lat, lon, "test")
    whole = compare_view(pair, None)
    # r inside the plume: 100 / hypot(100, 50) = 0.894
    assert whole["plume_pixels"]["n"] == 900
    assert whole["plume_pixels"]["r"] == pytest.approx(0.894, abs=0.02)
    assert whole["view_pixels"] == n * n
    assert len(whole["sample"]["plume"]["ref"]) == 900
    # Half the plume in view
    box = (lon[0, 0], lat[n - 1, 0], lon[0, 64] + 1e-4, lat[0, 0])
    half = compare_view(pair, box)
    assert half["plume_pixels"]["n"] == 30 * 15
    assert half["view_pixels"] == n * 65


def test_persistence_lists_the_overpasses_that_covered_a_plume():
    from specret.gas.persistence import SceneResult, persistence

    def scene(sid, plumes, cands):
        return SceneResult(
            sid,
            pd.DataFrame(plumes),
            pd.DataFrame(cands, columns=["gas", "lat", "lon"]),
            (np.array([31.5, 31.6]), np.array([-103.2, -103.3])),
        )

    p = {
        "plume_id": [1],
        "gas": ["CH4"],
        "confidence": [0.9],
        "source_lat": [31.5],
        "source_lon": [-103.2],
    }
    a = scene("20230608T162322_2315911_014", p, [])
    b = scene(
        "20230701T100000_2300000_001",
        {"plume_id": [], "gas": [], "source_lat": [], "source_lon": []},
        [["CH4", 31.5001, -103.2]],
    )
    c = scene(
        "20230801T100000_2300000_001",
        {"plume_id": [], "gas": [], "source_lat": [], "source_lon": []},
        [],
    )
    row = persistence([a, b, c]).iloc[0]
    assert row["overpasses_covering"] == 2 and row["overpasses_seen"] == 1
    assert row["covered_by"] == "20230701T100000_2300000_001;20230801T100000_2300000_001"
    assert row["seen_in"] == "20230701T100000_2300000_001"


def _tif(path, data, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(1e-4, 0, -118.0, 0, -1e-4, 35.0),
        nodata=nodata,
    ) as ds:
        ds.write(data.astype(np.float32), 1)


def test_compare_and_all_plumes_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    line = "ang20161001t205456"
    rng = np.random.default_rng(1)
    pub = rng.normal(0, 300, (120, 120))
    pub[40:70, 40:70] += 3000  # one strong published plume
    ours = 0.9 * pub + rng.normal(0, 100, pub.shape)
    _tif(tmp_path / f"data/scenes/avirisng/{line}/{line}_cmf_pub_ch4.tif", pub)
    (tmp_path / f"data/scenes/avirisng/{line}/{line}_rdn_v1n2_img.hdr").touch()
    out = tmp_path / "outputs/gas" / line
    _tif(out / "ch4_enh.tif", ours)
    pd.DataFrame(
        {
            "plume_id": [1],
            "gas": ["CH4"],
            "confidence": [0.8],
            "detection_gas": ["CH4"],
            "candidate_id": [1],
            "source_lat": [34.995],
            "source_lon": [-117.995],
            "q_kg_h": [250.0],
        }
    ).to_csv(out / "plumes.csv", index=False)
    pd.DataFrame(
        {
            "scene": [line],
            "plume_id": [1],
            "overpasses_covering": [2],
            "overpasses_seen": [1],
            "seen_in": ["ang20171001t000000"],
            "covered_by": ["ang20171001t000000;ang20181001t000000"],
        }
    ).to_csv(tmp_path / "outputs/gas/persistence.csv", index=False)
    (tmp_path / "configs").mkdir()
    client = TestClient(create_app(Workspace(root=tmp_path)))

    layers = {x["name"]: x for x in client.get(f"/api/geo/{line}/layers").json()}
    assert layers["ch4_enh"]["compare_with"] == "published_ch4"
    whole = client.get(f"/api/geo/{line}/compare", params={"layer": "ch4_enh"}).json()
    assert whole["plume_pixels"]["n"] >= 800 and whole["plume_pixels"]["r"] > 0.9
    assert whole["plume_pixels"]["deming_slope"] == pytest.approx(0.9, abs=0.1)
    assert whole["view_pixels"] == 120 * 120
    # A view of the top-left quarter: about a quarter of the pixels, part of the plume
    part = client.get(
        f"/api/geo/{line}/compare", params={"layer": "ch4_enh", "bbox": "-118,34.994,-117.994,35"}
    ).json()
    assert 0 < part["plume_pixels"]["n"] < whole["plume_pixels"]["n"]
    assert part["view_pixels"] == pytest.approx(60 * 60, rel=0.05)
    assert client.get(f"/api/geo/{line}/compare", params={"layer": "nope"}).status_code == 404
    bad = client.get(f"/api/geo/{line}/compare", params={"layer": "ch4_enh", "bbox": "1,2"})
    assert bad.status_code == 400

    (row,) = client.get("/api/gas/plumes").json()
    assert row["scene_id"] == line and row["sensor"] == "AVIRIS-NG"
    assert row["overpasses_seen"] == 1 and row["covered_by"] == [
        "ang20171001t000000",
        "ang20181001t000000",
    ]
    scenes = {s["scene_id"]: s for s in client.get("/api/gas/scenes").json()}
    assert scenes[line] == {"scene_id": line, "sensor": "AVIRIS-NG", "analyzed": True, "job": None}
