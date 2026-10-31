import numpy as np
import pandas as pd
import pytest

from specret.distort.simulator import RECIPES, atmospheric_residual, linear_mix, photometric
from specret.rerank.features import FeatureConfig, Window, band_center, depth_profiles
from specret.rerank.sff import compute_features, explain, score_candidates
from specret.search.fusion import rrf
from specret.search.hybrid import HybridRetriever
from specret.sensors.library import SensorLibrary
from specret.sensors.sensor import Sensor

WL = np.arange(1990.0, 2420.0, 7.5)


def gauss(center, depth, width=12.0):
    return 1 - depth * np.exp(-0.5 * ((WL - center) / width) ** 2)


def spectrum(*features, base=0.5, slope=0.0002):
    s = (base + slope * (WL - 2000)) * np.ones_like(WL)
    for c, d in features:
        s = s * gauss(c, d)
    return s


SENSOR = Sensor("toy", WL, np.full(WL.size, 8.0), np.ones(WL.size, bool))
CFG = FeatureConfig(
    (Window("al_oh", 2120, 2250), Window("mg_oh_co3", 2280, 2400)),
    center_sigma_nm=8.0,
    min_depth=0.01,
)

KAOLINITE = spectrum((2165, 0.12), (2207, 0.18))
MUSCOVITE = spectrum((2200, 0.2), (2350, 0.06))
CALCITE = spectrum((2340, 0.15))
DOLOMITE = spectrum((2318, 0.15))


# --- fusion ---


def test_rrf_rewards_agreement_across_lists():
    a = np.array([[5, 1, 2]])
    b = np.array([[1, 5, 3]])
    c = np.array([[1, 7, -1]])
    rows, scores = rrf([a, b, c], n_out=4)
    assert rows[0, 0] == 1  # ranks 2, 1, 1 beat 5's ranks 1, 2, absent
    assert rows[0, 1] == 5
    assert (np.diff(scores[0][rows[0] >= 0]) <= 0).all()


def test_rrf_handles_empty_queries():
    rows, scores = rrf([np.full((1, 3), -1)], n_out=2)
    assert (rows == -1).all() and (scores == 0).all()


# --- feature fitting ---


def test_band_center_is_subband_accurate():
    prof = depth_profiles(spectrum((2213, 0.1)), WL, np.arange(WL.size))
    assert band_center(prof, WL)[0] == pytest.approx(2213, abs=1.5)


def test_sff_separates_calcite_from_dolomite_and_kaolinite_from_muscovite():
    lib = np.vstack([KAOLINITE, MUSCOVITE, CALCITE, DOLOMITE])
    lf = compute_features(lib, SENSOR, CFG)
    queries = np.vstack([0.7 * CALCITE, 1.3 * KAOLINITE])  # brightness changes
    qf = compute_features(queries, SENSOR, CFG)
    cand = np.tile(np.arange(4), (2, 1))
    res = score_candidates(qf, lf, cand, CFG)
    assert np.argmax(res.score[0]) == 2  # calcite beats dolomite (2.34 vs 2.32 um)
    assert np.argmax(res.score[1]) == 0  # doublet beats single 2.20 um feature
    assert res.score[0, 2] > res.score[0, 3] + 0.3


def test_sff_penalises_features_the_query_lacks():
    lib = np.vstack([CALCITE, spectrum((2340, 0.15), (2200, 0.15))])
    lf = compute_features(lib, SENSOR, CFG)
    qf = compute_features(CALCITE, SENSOR, CFG)
    res = score_candidates(qf, lf, np.array([[0, 1]]), CFG)
    assert res.score[0, 0] > res.score[0, 1]
    rows = explain(res, 0, 1)
    assert {r["window"] for r in rows} == {"al_oh", "mg_oh_co3"}


# --- hybrid retriever ---


def test_hybrid_returns_features_and_finds_mineral():
    lib_m = np.vstack([KAOLINITE, MUSCOVITE, CALCITE, DOLOMITE])
    ids = np.array(["k", "m", "c", "d"])
    canonical = pd.DataFrame(
        {
            "spectrum_id": ids,
            "material_name": ["Kaolinite", "Muscovite", "Calcite", "Dolomite"],
            "group": ["kaolin", "white_mica", "calcite", "dolomite"],
            "chapter": "M",
        }
    )
    h = HybridRetriever(SensorLibrary(SENSOR, ids, lib_m), canonical, k_recall=4, n_rerank=4)
    h.features = CFG
    h.lib_features = compute_features(lib_m, SENSOR, CFG)
    out = h.query(0.8 * CALCITE + 0.01, k=2, with_explain=True)[0]
    assert out[0]["spectrum_id"] == "c"
    assert out[0]["features"][0]["window"] == "mg_oh_co3"


# --- distortions ---


def test_distortions_are_seeded_and_bounded():
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    a = atmospheric_residual(CALCITE, WL, rng1, strength=0.03)
    b = atmospheric_residual(CALCITE, WL, rng2, strength=0.03)
    np.testing.assert_array_equal(a, b)
    mixed = linear_mix(CALCITE, MUSCOVITE, np.random.default_rng(0), dominant=(0.7, 0.7))
    np.testing.assert_allclose(mixed, 0.7 * CALCITE + 0.3 * MUSCOVITE)
    p = photometric(CALCITE, WL, np.random.default_rng(0), scale=(2.0, 2.0), tilt=0.0)
    np.testing.assert_allclose(p, 2 * CALCITE)
    assert {"clean", "all", "mix"} <= set(RECIPES)


# --- distance baselines ---


def test_sid_is_zero_for_scaled_copies_and_positive_otherwise():
    from specret.search.baselines import sid_matrix

    d = sid_matrix(np.vstack([CALCITE]), np.vstack([2 * CALCITE, MUSCOVITE]))
    assert d[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert d[0, 1] > 0


def test_brute_force_retrievers_find_the_right_material():
    from specret.search.baselines import BruteForceRetriever

    lib_m = np.vstack([KAOLINITE, MUSCOVITE, CALCITE, DOLOMITE])
    ids = np.array(["k", "m", "c", "d"])
    canonical = pd.DataFrame(
        {
            "spectrum_id": ids,
            "material_name": ["Kaolinite", "Muscovite", "Calcite", "Dolomite"],
            "group": ["kaolin", "white_mica", "calcite", "dolomite"],
            "chapter": "M",
        }
    )
    lib = SensorLibrary(SENSOR, ids, lib_m)
    for metric in ("sid", "cr_euclid"):
        r = BruteForceRetriever(lib, canonical, metric=metric)
        assert r.query(DOLOMITE * 1.0001, k=2)[0][0]["spectrum_id"] == "d"
