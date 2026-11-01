"""Viewer Phase V1 (PLAN_VIEWER.md): scene registry, layers, XYZ tiles, point values.

A synthetic AVIRIS-NG-style scene: a published CH4 map and our CH4 enhancement as small COGs.
"""

import io

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from PIL import Image

LINE = "ang20990101t000000"
W, N, RES = -118.0, 35.0, 0.0005  # 100 x 100 px, about 5.5 km


def _cog(path, data, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="COG",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(RES, 0, W, 0, -RES, N),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data.astype(np.float32), 1)


@pytest.fixture
def client(tmp_path):
    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    (tmp_path / "configs").mkdir()
    a = np.zeros((100, 100))
    a[40:60, 40:60] = 1000.0  # a "plume" in the middle
    a[:5, :5] = -9999.0  # no data corner
    _cog(tmp_path / "data/scenes/avirisng" / LINE / f"{LINE}_cmf_pub_ch4.tif", a)
    _cog(tmp_path / "outputs/gas" / LINE / "ch4_enh.tif", np.where(a == -9999.0, a, 0.8 * a))
    return TestClient(create_app(Workspace(root=tmp_path)))


def test_scenes_and_layers(client):
    scenes = client.get("/api/geo/scenes").json()
    (s,) = scenes
    assert s["key"] == LINE and s["sensor"] == "AVIRIS-NG"
    assert s["bounds"] == pytest.approx([W, N - 100 * RES, W + 100 * RES, N])
    layers = {layer["name"]: layer for layer in client.get(f"/api/geo/{LINE}/layers").json()}
    assert {"ch4_enh", "published_ch4"} <= set(layers)
    assert layers["ch4_enh"]["units"] == "ppm·m" and layers["ch4_enh"]["ready"]
    assert client.get("/api/geo/nope/layers").status_code == 404


def _tile_of(lon, lat, z):
    import morecantile

    t = morecantile.tms.get("WebMercatorQuad").tile(lon, lat, z)
    return t.x, t.y, t.z


def test_tiles_render_inside_and_are_empty_outside(client):
    x, y, z = _tile_of(W + 50 * RES, N - 50 * RES, 14)
    r = client.get(f"/api/tiles/{LINE}.ch4_enh/{z}/{x}/{y}.png?rescale=0,1000&colormap=magma")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    assert img.size == (256, 256)
    alpha = np.asarray(img)[..., 3]
    assert alpha.max() == 255  # the scene is drawn
    far = _tile_of(10.0, 10.0, 14)
    r = client.get(f"/api/tiles/{LINE}.ch4_enh/{far[2]}/{far[0]}/{far[1]}.png")
    assert np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))[..., 3].max() == 0
    assert client.get(f"/api/tiles/{LINE}.ch4_enh/{z}/{x}/{y}.png?colormap=nope").status_code == 400
    assert client.get(f"/api/tiles/{LINE}.ch4_enh/{z}/{x}/{y}.png?rescale=1").status_code == 400
    assert client.get(f"/api/tiles/{LINE}.missing/{z}/{x}/{y}.png").status_code == 404


def test_values_at_a_point(client):
    inside = client.get(f"/api/geo/{LINE}/values?lat={N - 50 * RES}&lon={W + 50 * RES}").json()
    assert inside[f"{LINE}.ch4_enh"] == pytest.approx(800.0)
    assert inside[f"{LINE}.published_ch4"] == pytest.approx(1000.0)
    nodata = client.get(f"/api/geo/{LINE}/values?lat={N - 2 * RES}&lon={W + 2 * RES}").json()
    assert nodata[f"{LINE}.ch4_enh"] is None
    outside = client.get(f"/api/geo/{LINE}/values?lat=0&lon=0").json()
    assert all(v is None for v in outside.values())


def test_plumes_geojson_and_colormaps(client):
    fc = client.get(f"/api/geo/{LINE}/plumes.geojson").json()
    assert fc == {"type": "FeatureCollection", "features": []}
    ramps = client.get("/api/colormaps").json()
    assert set(ramps) >= {"magma", "viridis", "rdylgn"}
    assert len(ramps["magma"]) == 11 and ramps["magma"][0].startswith("#")
