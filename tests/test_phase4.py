import numpy as np
import pandas as pd
import pytest

from specret.openset.calibrate import ConfidenceModel, accuracy_at_coverage, auroc, ece
from specret.unmix.fcls import baseline_columns, bic, nnls_fit
from specret.workspace import confidence_tier

WL = np.arange(1990.0, 2420.0, 7.5)


def spectrum(*features, base=0.5):
    s = base * np.ones_like(WL)
    for c, d in features:
        s = s * (1 - d * np.exp(-0.5 * ((WL - c) / 12.0) ** 2))
    return s


KAOL = spectrum((2165, 0.12), (2207, 0.18))
MUSC = spectrum((2200, 0.2), (2350, 0.06))
CALC = spectrum((2340, 0.15))


# --- constrained unmixing ---


def test_nnls_recovers_fractions():
    A = np.column_stack([KAOL, CALC])
    fit = nnls_fit(A, 0.3 * KAOL + 0.7 * CALC)
    np.testing.assert_allclose(fit.fractions, [0.3, 0.7], atol=1e-6)
    assert fit.rss < 1e-12


def test_baseline_absorbs_offset_and_slope():
    t = (WL - 1500) / 1000
    x = 0.8 * KAOL + 0.05 + 0.02 * t
    plain = nnls_fit(KAOL[:, None], x)
    with_base = nnls_fit(KAOL[:, None], x, baseline=baseline_columns(WL))
    assert with_base.rss < 1e-10 < plain.rss
    assert with_base.coefficients[0] == pytest.approx(0.8)


def test_sum_to_one_constraint():
    fit = nnls_fit(np.column_stack([KAOL, CALC]), 1.5 * (0.5 * KAOL + 0.5 * CALC), sum_to_one=True)
    assert fit.coefficients.sum() == pytest.approx(1.0, abs=1e-3)


def test_bic_penalises_parameters():
    assert bic(1.0, 100, 3) > bic(1.0, 100, 2)


def test_pursuit_finds_two_components():
    from specret.search.hybrid import HybridRetriever
    from specret.sensors.library import SensorLibrary
    from specret.sensors.sensor import Sensor
    from specret.unmix.pursuit import Unmixer

    sensor = Sensor("toy", WL, np.full(WL.size, 8.0), np.ones(WL.size, bool))
    names = ["Kaolinite", "Muscovite", "Calcite", "Dolomite"]
    lib_m = np.vstack([KAOL, MUSC, CALC, spectrum((2318, 0.15))])
    ids = np.array(["k", "m", "c", "d"])
    canonical = pd.DataFrame(
        {
            "spectrum_id": ids,
            "material_name": names,
            "group": ["kaolin", "white_mica", "calcite", "dolomite"],
            "chapter": "M",
        }
    )
    r = HybridRetriever(SensorLibrary(sensor, ids, lib_m), canonical, k_recall=4, n_rerank=4)
    u = Unmixer(r, canonical, pool_size=4, min_improvement=0.3)
    pure = u.unmix(0.9 * CALC)
    assert [u.groups[i] for i in pure.rows] == ["calcite"]
    mixed = u.unmix(0.5 * KAOL + 0.5 * CALC)
    assert {u.groups[i] for i in mixed.rows} == {"kaolin", "calcite"}
    assert sorted(np.round(mixed.fit.fractions, 2)) == [0.5, 0.5]
    desc = u.describe(mixed, canonical)
    assert {c["group"] for c in desc["components"]} == {"kaolin", "calcite"}


# --- calibration ---


def test_confidence_model_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random(n) < 0.5
    df = pd.DataFrame(
        {
            "score": 0.7 + 0.2 * y + rng.normal(0, 0.05, n),
            "margin": 0.1 + 0.1 * y + rng.normal(0, 0.05, n),
            "sff": 0.3 + 0.4 * y + rng.normal(0, 0.1, n),
            "sam_deg": 6 - 2 * y + rng.normal(0, 1, n),
            "cr_cos": 0.85 + 0.1 * y + rng.normal(0, 0.03, n),
            "depth": rng.uniform(0.05, 0.2, n),
            "fit_rms": 0.05 - 0.01 * y + rng.normal(0, 0.005, n),
            "consensus": np.where(y, 0.8, 0.4),
        }
    )
    m = ConfidenceModel.fit(df, y)
    p = m.predict(df)
    assert auroc(p, y) > 0.95
    assert ece(p, y) < 0.1
    assert accuracy_at_coverage(p, y, 0.3) > 0.95
    m.save(tmp_path / "m.json")
    np.testing.assert_allclose(ConfidenceModel.load(tmp_path / "m.json").predict(df), p)
    empty = df.iloc[:1].copy()
    empty[:] = np.nan
    assert m.predict(empty)[0] == 0.0


def test_auroc_edge_cases():
    assert auroc(np.array([0.1, 0.9]), np.array([False, True])) == 1.0
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([True, True])))


def test_confidence_tiers():
    assert confidence_tier(0.7) == "confident"
    assert confidence_tier(0.4) == "uncertain"
    assert confidence_tier(0.1) == "likely unknown"
    assert confidence_tier(None) == "uncalibrated"
