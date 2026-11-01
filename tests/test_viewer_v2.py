"""Viewer Phase V2 (PLAN_VIEWER.md): category filters, area summaries, agreement layer.

Synthetic rasters only.
"""

import io
import json

import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image

W, N, RES = -117.5, 37.6, 0.001  # 50 x 50 px


def _write(path, data, nodata, colormap=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype.name,
        crs="EPSG:4326",
        transform=Affine(RES, 0, W, 0, -RES, N),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
        if colormap:
            ds.write_colormap(1, colormap)


def _tile():
    import morecantile

    t = morecantile.tms.get("WebMercatorQuad").tile(W + 25 * RES, N - 25 * RES, 14)
    return t.z, t.x, t.y


def test_category_filter_and_confidence_mask(tmp_path):
    from specret.viewer import tiles

    groups = np.ones((50, 50), dtype=np.uint8)
    groups[:, 25:] = 2
    conf = np.full((50, 50), 0.9, dtype=np.float32)
    conf[25:, :] = 0.1
    cmap = {1: (255, 0, 0, 255), 2: (0, 0, 255, 255)}
    _write(tmp_path / "group.tif", groups, 0, cmap)
    _write(tmp_path / "confidence.tif", conf, np.nan)
    z, x, y = _tile()

    def colours(**kw):
        png = tiles.render_categories(tmp_path / "group.tif", z, x, y, **kw)
        im = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")).reshape(-1, 4)
        return {tuple(c) for c in im if c[3] > 0}

    plain = colours()
    assert (255, 0, 0, 255) in plain and (0, 0, 255, 255) in plain
    only_red = colours(keep=(1,))
    assert (0, 0, 255, 255) not in only_red and tiles.UNSELECTED in only_red
    masked = colours(confidence=tmp_path / "confidence.tif", min_p=0.5)
    assert tiles.LOW_CONFIDENCE in masked and (255, 0, 0, 255) in masked
    assert tiles.category_fractions(tmp_path / "group.tif") == {1: 0.5, 2: 0.5}


def test_polygon_area():
    from specret.viewer.area import polygon_area_km2

    # 0.1 x 0.1 degree box at 37.55 N: 11.1 km x 8.8 km
    box = [(-117.3, 37.5), (-117.2, 37.5), (-117.2, 37.6), (-117.3, 37.6)]
    assert polygon_area_km2(box) == pytest.approx(
        11.12 * 11.12 * np.cos(np.radians(37.55)), rel=0.01
    )
    assert polygon_area_km2(box[:3]) == pytest.approx(polygon_area_km2(box) / 2, rel=1e-3)


def test_agreement_layer_and_statistics(tmp_path):
    from specret.viewer import layers

    ours = np.ones((50, 50), dtype=np.uint8)  # all "calcite"
    ours[:10] = 2  # "kaolin"
    ours[45:] = 0  # unmapped
    _write(tmp_path / "map" / "group.tif", ours, 0, {1: (1, 1, 1, 255), 2: (2, 2, 2, 255)})
    (tmp_path / "map" / "groups.csv").write_text(
        'code,group,rgba\n1,calcite,"(1,1,1,255)"\n2,kaolin,"(2,2,2,255)"\n'
    )
    cache = tmp_path / "viewer"
    ref = np.ones((50, 50), dtype=np.uint8)  # L2B MIN: calcite everywhere (its own codes)
    ref[:, :5] = 0  # no detection
    _write(cache / "l2bmin_group2.tif", ref, 0, {1: (9, 9, 9, 255)})
    (cache / "l2bmin_group2.json").write_text(
        json.dumps([{"code": 1, "label": "calcite", "rgb": [9, 9, 9]}])
    )
    depth = np.full((50, 50), 0.2, dtype=np.float32)
    depth[:, 45:] = 0.01  # too shallow to count
    _write(cache / "l2bmin_depth2.tif", depth, np.nan)

    out = layers.build_agreement(tmp_path / "map" / "group.tif", None, None, cache)
    with rasterio.open(out) as ds:
        code = ds.read(1)
    stats = json.loads(out.with_suffix(".json").read_text())
    # reference where mapped (rows 0-44), detected (cols 5+) and deep enough (cols < 45)
    with_ref = 45 * 40
    agree = 35 * 40  # rows 10-44 are calcite in both
    assert stats["pixels_with_reference"] == with_ref
    assert stats["agreement"] == pytest.approx(agree / with_ref)
    assert stats["top_confusions"][0] == {"ours": "kaolin", "l2bmin": "calcite", "pixels": 10 * 40}
    assert code[20, 20] == 1 and code[5, 20] == 2 and code[20, 2] == 3 and code[48, 20] == 0


def test_area_summary_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    line = "ang20990101t000000"
    a = np.zeros((50, 50), dtype=np.float32)
    a[10:20, 10:20] = 1000.0
    _write(tmp_path / "data/scenes/avirisng" / line / f"{line}_cmf_pub_ch4.tif", a, -9999.0)
    (tmp_path / "configs").mkdir()
    client = TestClient(create_app(Workspace(root=tmp_path)))
    w, n = W + 10 * RES, N - 10 * RES
    poly = [[w, n], [w + 10 * RES, n], [w + 10 * RES, n - 10 * RES], [w, n - 10 * RES]]
    r = client.post(f"/api/geo/{line}/summary", json={"polygon": poly})
    assert r.status_code == 200
    s = r.json()
    assert s["minerals"] is None and s["plumes"]["count"] == 0
    (layer,) = s["layers"]
    assert layer["pixels"] >= 81 and layer["median"] == pytest.approx(1000.0)
    assert sum(layer["hist_counts"]) == layer["pixels"]
    assert client.post(f"/api/geo/{line}/summary", json={"polygon": poly[:2]}).status_code == 400
    assert client.post(f"/api/geo/{line}/unmix", json={"polygon": poly}).status_code == 400
