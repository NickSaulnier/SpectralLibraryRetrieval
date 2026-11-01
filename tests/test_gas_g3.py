"""Phase G3 (PLAN_GAS.md): quantification, wind helpers and the read-side gas tools.

No network, scenes or HITRAN tables needed.
"""

import numpy as np
import pandas as pd
import pytest

from specret.gas.plumes import Candidate
from specret.gas.quantify import (
    kg_per_m2_per_ppmm,
    pixel_area,
    quantify,
    rate,
    regrow,
    requantify,
    source_pixel,
)
from specret.io.wind import Wind, in_hrrr_domain


def grid(rows=80, cols=80, lat0=31.5, lon0=-103.3, step_deg=0.00054):
    """A north-up lat/lon grid with ~60 m pixels (rows go south, cols go east)."""
    r, c = np.mgrid[0:rows, 0:cols]
    return lat0 - r * step_deg, lon0 + c * step_deg


def test_mass_per_ppmm_and_pixel_area():
    # 1 ppm·m of CH4 at sea level: 2.55e15 molecules/cm^2 -> about 6.8e-7 kg/m^2
    assert kg_per_m2_per_ppmm("CH4", 0.0) == pytest.approx(6.8e-7, rel=0.02)
    assert kg_per_m2_per_ppmm("CO2", 0.0) / kg_per_m2_per_ppmm("CH4", 0.0) == pytest.approx(
        44.01 / 16.04, rel=1e-6
    )
    lat, lon = grid()
    assert pixel_area(lat, lon, 40, 40) == pytest.approx(
        60.0 * 60.0 * np.cos(np.radians(31.48)), rel=0.02
    )


def test_rate_formula():
    # 100 kg over L = 1000 m at U10 = 2 m/s with U_eff = 0.45 U10 + 0.67 = 1.57 m/s
    assert rate(100.0, 1000.0, 2.0, (0.45, 0.67)) == pytest.approx(100 * 1.57 / 1000 * 3600)


def blob(alpha_peak=1000.0, center=(40, 30), shape=(80, 80)):
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]]
    a = alpha_peak * np.exp(-((rr - center[0]) ** 2 + (cc - center[1]) ** 2) / (2 * 4.0**2))
    # stretch it downwind (east) so the upwind end is the west edge
    a += (
        0.6
        * alpha_peak
        * np.exp(-((rr - center[0]) ** 2) / 8.0 - ((cc - center[1] - 12) ** 2) / 60.0)
    )
    return a


def test_source_is_the_upwind_end_of_the_strong_pixels():
    lat, lon = grid()
    a = blob()
    rows, cols = np.nonzero(a > 100)
    cand = Candidate(1, rows, cols, 10.0, float(a[rows, cols].mean()), float(a.max()))
    r, c = source_pixel(a, cand, lat, lon, wind_u=3.0, wind_v=0.0)  # wind blows east
    assert c < 32  # western (upwind) part of the plume
    r2, c2 = source_pixel(a, cand, lat, lon, wind_u=-3.0, wind_v=0.0)  # wind blows west
    assert c2 > c


def test_regrow_changes_mask_size_with_threshold():
    a = blob()
    z = a / 100.0
    rows, cols = np.nonzero(z > 2.5)
    cand = Candidate(1, rows, cols, float(z.max()), 0.0, 0.0)
    small, _ = regrow(z, cand, 3.0)
    big, _ = regrow(z, cand, 2.0)
    assert small.size < rows.size < big.size


def test_quantify_recovers_a_noise_free_plume_mass():
    lat, lon = grid()
    a = blob()
    z = a / 50.0
    rows, cols = np.nonzero(z > 2.5)
    cand = Candidate(1, rows, cols, float(z.max()), 0.0, 0.0)
    wind = Wind(2.0, 0.0, "test", sd=0.4)
    q = quantify(
        a,
        z,
        cand,
        "CH4",
        lat,
        lon,
        0.0,
        wind,
        np.zeros(a.shape, bool),
        (0.45, 0.67),
        2.5,
        retrieval_rel_sd=0.3,
    )
    m1 = kg_per_m2_per_ppmm("CH4", 0.0)
    expected_ime = a[rows, cols].sum() * m1 * q.pixel_area_m2
    assert q.ime_kg == pytest.approx(expected_ime, rel=1e-6)
    assert q.length_m == pytest.approx(np.sqrt(rows.size * q.pixel_area_m2))
    assert q.q_kg_h == pytest.approx(rate(q.ime_kg, q.length_m, 2.0, (0.45, 0.67)))
    assert q.sigma_parts["retrieval"] == 0.3
    assert q.sigma_parts["wind"] == pytest.approx(0.45 * 0.4 / 1.57)


def test_requantify_with_a_new_wind():
    row = {
        "gas": "CH4",
        "ime_kg": 100.0,
        "length_m": 1000.0,
        "u10": 2.0,
        "u_eff": 1.57,
        "wind_source": "HRRR",
        "rel_sigma_background": 0.1,
        "rel_sigma_retrieval": 0.3,
        "rel_sigma_wind": 0.45 * 1.0 / 1.57,
        "rel_sigma_mask": 0.05,
    }
    same = requantify(row, (0.45, 0.67))
    assert same["q_kg_h"] == pytest.approx(565.2)
    assert same["u10_sd"] == pytest.approx(1.0)
    w = Wind(0.0, 4.0, "user", sd=0.5)
    new = requantify(row, (0.45, 0.67), w)
    assert new["u_eff"] == pytest.approx(0.45 * 4 + 0.67)
    assert new["q_kg_h"] > same["q_kg_h"]
    assert new["relative_uncertainty"]["wind"] == pytest.approx(0.45 * 0.5 / 2.47)


def test_wind_helpers():
    w = Wind(3.0, 4.0, "x")
    assert w.speed == 5.0
    assert w.direction_to_deg == pytest.approx(np.degrees(np.arctan2(3, 4)))
    assert in_hrrr_domain(31.5, -103.2)
    assert not in_hrrr_domain(38.5, 54.0)


def test_gas_tools_read_scene_outputs(tmp_path):
    from specret.gas import tools

    folder = tmp_path / "20230608T162322_2315911_014"
    folder.mkdir()
    plumes = pd.DataFrame(
        {
            "plume_id": [1, 2],
            "gas": ["CH4", "CO2"],
            "confidence": [0.9, 0.4],
            "detection_gas": ["CH4", "CO2"],
            "candidate_id": [3, 1],
            "n_px": [50, 200],
            "max_alpha_ppmm": [900.0, 4e4],
            "terms": ["CH4=600;confuser:calcite=0.1", "CO2=2e4"],
            "confusers_fitted": ["confuser:calcite", ""],
            "carbonate_fraction": [0.5, None],
            "source_lat": [31.5, 31.6],
            "source_lon": [-103.2, -103.3],
            "ime_kg": [50.0, 9000.0],
            "length_m": [400.0, 900.0],
            "u10": [2.0, 3.0],
            "u_eff": [1.57, 2.02],
            "wind_source": ["HRRR", "HRRR"],
            "q_kg_h": [700.0, 72000.0],
            "sigma_q_kg_h": [300.0, 40000.0],
            "rel_sigma_background": [0.1, 0.1],
            "rel_sigma_retrieval": [0.3, 0.3],
            "rel_sigma_wind": [0.3, 0.3],
            "rel_sigma_mask": [0.05, 0.05],
        }
    )
    plumes.to_csv(folder / "plumes.csv", index=False)
    pd.DataFrame(
        {
            "detection_gas": ["CH4", "CO2", "CH4"],
            "candidate_id": [3, 1, 4],
            "verdict": ["CH4", "CO2", "confuser:calcite"],
            "mf_z": [9.0, 7.0, 5.5],
            "gas_z": [8.0, 6.0, 0.5],
            "coherence": [0.6, 0.5, 0.2],
            "vs_confusers": [4.0, 3.0, 0.0],
            "consistency": [0.5, 0.4, 2.0],
            "log_px": [3.9, 5.3, 2.0],
        }
    ).to_csv(folder / "candidates.csv", index=False)

    s = tools.scene_gas_summary(folder)
    assert s["plumes"] == 2 and s["CH4"]["plumes"] == 1 and s["candidates"] == 3
    assert s["candidate_verdicts"]["confuser:calcite"] == 1
    assert [p["plume_id"] for p in tools.list_plumes(folder, "CO2")["plumes"]] == [2]
    assert len(tools.list_plumes(folder, min_q=1000)["plumes"]) == 1
    e = tools.explain_plume(folder, 1)
    assert e["fitted_terms"]["confuser:calcite"] == pytest.approx(0.1)
    assert e["evidence"]["gas_z"] == 8.0
    assert set(e["relative_uncertainty"]) == {"background", "retrieval", "wind", "mask"}
    q = tools.quantify_plume(folder, 1, 0.0, 3.0)
    assert q["u10"] == 3.0 and q["u10_sd"] == 1.5
    with pytest.raises(ValueError):
        tools.explain_plume(folder, 9)
