"""Phase G5 (PLAN_GAS.md): cross-sectional flux, multi-overpass persistence, AVIRIS-NG geometry.

Synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from specret.gas.persistence import SceneResult, persistence
from specret.gas.quantify import cross_sectional_flux, kg_per_m2_per_ppmm, pixel_area
from specret.io.avirisng import map_geotransform


def grid(rows=120, cols=200, lat0=31.5, lon0=-103.3, step_deg=0.00054):
    r, c = np.mgrid[0:rows, 0:cols]
    return lat0 - r * step_deg, lon0 + c * step_deg


def test_cross_sectional_flux_recovers_a_steady_gaussian_plume():
    lat, lon = grid()
    src = (60, 20)
    area = pixel_area(lat, lon, *src)
    dx = np.sqrt(area)
    q_kg_s, u = 0.5, 3.0  # 1800 kg/h, wind blowing east
    r, c = np.mgrid[0:120, 0:200]
    x = (c - src[1]) * dx  # downwind distance (m)
    y = (r - src[0]) * dx
    sig_y = 30 + 0.2 * np.clip(x, 0, None)
    omega = np.where(
        x > 0, q_kg_s / (u * np.sqrt(2 * np.pi) * sig_y) * np.exp(-0.5 * (y / sig_y) ** 2), 0.0
    )  # kg/m^2: a steady plume carries Q / U kg per metre downwind
    alpha = omega / kg_per_m2_per_ppmm("CH4", 0.0)
    rows, cols = np.nonzero(alpha > 0.0)
    out = cross_sectional_flux(alpha, rows, cols, "CH4", lat, lon, 0.0, src, u, 0.0, u)
    assert out["transects"] > 50
    assert out["q_csf_kg_h"] == pytest.approx(q_kg_s * 3600, rel=0.05)
    assert out["csf_iqr_rel"] < 0.1
    # No downwind transects when the wind blows the other way
    back = cross_sectional_flux(alpha, rows, cols, "CH4", lat, lon, 0.0, src, -u, 0.0, u)
    assert np.isnan(back["q_csf_kg_h"])


def _scene(sid, plumes, cands, lat0=31.5, lon0=-103.3):
    lat, lon = np.meshgrid(
        np.linspace(lat0 - 0.05, lat0 + 0.05, 50), np.linspace(lon0 - 0.05, lon0 + 0.05, 50)
    )
    return SceneResult(sid, pd.DataFrame(plumes), pd.DataFrame(cands), (lat.ravel(), lon.ravel()))


def test_persistence_counts_repeat_overpasses_not_adjacent_scenes():
    p = {
        "plume_id": [1],
        "gas": ["CH4"],
        "confidence": [0.9],
        "source_lat": [31.5],
        "source_lon": [-103.3],
        "q_kg_h": [500.0],
    }
    near = {"gas": ["CH4"], "lat": [31.5005], "lon": [-103.3005]}  # about 70 m away
    far = {"gas": ["CH4"], "lat": [31.52], "lon": [-103.3]}  # about 2 km away
    co2 = {"gas": ["CO2"], "lat": [31.5], "lon": [-103.3]}
    scenes = [
        _scene("20230608T162322_2315911_014", p, far),
        _scene("20230608T162322_2315911_015", {}, near),  # same overpass, adjacent scene
        _scene("20230731T191822_2321213_014", {}, near),  # repeat that sees it
        _scene("20231020T192945_2329313_028", {}, far),  # repeat that does not
        _scene("20240823T173410_2423611_025", {}, co2),  # repeat with another gas only
        _scene("20250101T000000_2500000_001", {}, near, lat0=40.0, lon0=-100.0),  # elsewhere
    ]
    out = persistence(scenes, radius_m=300.0)
    row = out[out["scene"] == "20230608T162322_2315911_014"].iloc[0]
    assert row["overpasses_covering"] == 3
    assert row["overpasses_seen"] == 1
    assert row["seen_in"] == "20230731T191822_2321213_014"
    assert row["persistence"] == pytest.approx(1 / 3)


def test_avirisng_rotated_map_info_matches_published_geotransform():
    info = [
        "UTM",
        "1",
        "1",
        "825491.629085",
        "3935624.00013",
        "2.9",
        "2.9",
        "10",
        "North",
        "WGS-84",
        "units=Meters",
        "rotation= 5.0000000",
    ]
    gt, wkt = map_geotransform(info)
    # The published CH4 map of this line has a = e' = 2.89, b = d = 0.25 (rasterio affine)
    assert gt[1] == pytest.approx(2.889, abs=1e-3) and gt[5] == pytest.approx(-2.889, abs=1e-3)
    assert gt[2] == pytest.approx(0.2528, abs=1e-3) and gt[4] == pytest.approx(0.2528, abs=1e-3)
    assert "32610" in wkt or "UTM zone 10N" in wkt
