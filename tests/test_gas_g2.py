"""Phase G2 (PLAN_GAS.md): plume injection, candidates, joint gas + confuser identification.

Synthetic data only: no HITRAN tables, scenes or library needed.
"""

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("torch")

from specret.gas.identify import FitContext, identify  # noqa: E402
from specret.gas.inject import ALPHA_GRID, Plume, apply_gas, inject, plume_shape  # noqa: E402
from specret.gas.plumes import find_candidates  # noqa: E402


def test_plume_shape_is_peak_normalized_and_local():
    p = Plume({"CH4": 1000.0}, row=50, col=60, length_px=6, width_px=1.5, angle_rad=0.3)
    s = plume_shape((120, 140), p)
    assert s.max() == pytest.approx(1.0)
    assert s[50, 60] > 0.1
    assert s[0, 0] == 0 and s[119, 139] == 0
    # Downwind (angle 0.3 rad, mostly +col) holds more of the plume than upwind
    assert s[:, 61:].sum() > 5 * s[:, :60].sum()


def test_apply_gas_follows_the_transmittance_table():
    nb = 4
    table = np.exp(-np.outer(ALPHA_GRID, [1e-4, 2e-4, 0.0, 5e-5]))
    rad = np.ones((3, 3, nb), dtype=np.float32)
    alpha = np.zeros((3, 3))
    alpha[1, 1] = 1000.0
    apply_gas(rad, alpha, table)
    np.testing.assert_allclose(
        rad[1, 1], np.exp(-1000.0 * np.array([1e-4, 2e-4, 0, 5e-5])), rtol=1e-3
    )
    assert np.all(rad[0, 0] == 1.0)
    out = inject(
        np.ones((40, 40, nb), np.float32),
        [Plume({"CH4": 500.0}, 20, 20, 3, 1, 0.0)],
        {"CH4": table},
    )
    assert out.alpha["CH4"].max() == pytest.approx(500.0)
    assert out.labels[20, 20] == 1 and out.radiance[20, 20, 1] < 1


def test_candidates_find_a_blob_but_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 100, (300, 200))
    assert len(find_candidates(a)[0]) <= 1
    assert len(find_candidates(a, z_high=4.0)[0]) > 2  # why the default seed is z = 5
    a[100:110, 50:58] += 600
    cands, _ = find_candidates(a)
    assert len(cands) >= 1
    top = cands[0]
    assert 95 <= top.centroid[0] <= 115 and 45 <= top.centroid[1] <= 63
    assert top.n_px >= 40


def _spectral_features(centers, lines, width=6.0):
    return sum(d * np.exp(-0.5 * ((centers - c) / width) ** 2) for c, d in lines)


def synthetic_context(signal: str, rows=300, cols=300, seed=1):
    """Relative-residual cube with smooth surface texture, plus one candidate signal."""
    rng = np.random.default_rng(seed)
    centers = np.concatenate([np.arange(1500, 1790, 7.4), np.arange(1950, 2450, 7.4)])
    nb = centers.size
    k_ch4 = 1e-6 * _spectral_features(centers, [(1666, 3), (2200, 2), (2320, 4), (2370, 3)])
    k_co2 = 1e-7 * _spectral_features(centers, [(1575, 2), (1605, 2), (2005, 4), (2060, 4)])
    calcite = -_spectral_features(centers, [(2338, 0.3)], width=25)
    # Surface texture: a few smooth spectral modes with spatially smooth abundances
    modes = rng.normal(size=(6, nb)).cumsum(axis=1) / 40
    fields = np.stack([ndimage.gaussian_filter(rng.normal(size=(rows, cols)), 3) for _ in modes])
    y = np.einsum("mrc,mb->rcb", fields, modes) * 0.2 + rng.normal(0, 0.003, (rows, cols, nb))
    mask = np.zeros((rows, cols), dtype=bool)
    mask[140:155, 140:152] = True
    if signal == "CH4":
        y[mask] += -800 * k_ch4
    elif signal == "calcite":
        y[mask] += 0.15 * calcite
    alpha = np.zeros((rows, cols))
    alpha[mask] = 500
    ctx = FitContext(
        y=y.astype(np.float32),
        usable=np.ones((rows, cols), dtype=bool),
        centers_nm=centers,
        windows=[(1500, 1790), (1950, 2450)],
        gas_k={"CH4": k_ch4, "CO2": k_co2},
        confusers={"calcite": calcite},
        alpha=alpha,
        occupied=ndimage.binary_dilation(mask, iterations=3),
    )
    from specret.gas.plumes import Candidate

    rr, cc = np.nonzero(mask)
    return ctx, Candidate(1, rr, cc, peak_z=6.0, mean_alpha=500.0, max_alpha=500.0)


def test_identify_names_methane():
    ctx, cand = synthetic_context("CH4")
    res = identify(ctx, cand, "CH4", n_null=200)
    assert res.gas == "CH4"
    assert res.terms["CH4"] == pytest.approx(800, rel=0.25)
    assert res.features["gas_z"] > 5
    assert res.features["consistency"] < 3


def test_identify_prefers_the_confuser_for_a_carbonate_patch():
    ctx, cand = synthetic_context("calcite")
    res = identify(ctx, cand, "CH4", n_null=200)
    assert res.gas is None
    assert res.verdict == "confuser:calcite"


def test_identify_reports_no_signal_on_plain_background():
    ctx, cand = synthetic_context("none")
    res = identify(ctx, cand, "CH4", n_null=200)
    assert res.gas is None and res.verdict == "no signal"


def test_benchmark_labels_candidates_by_overlap():
    import pandas as pd

    from specret.eval.gas_benchmark import correct, overlap_labels
    from specret.gas.plumes import Candidate

    class R:
        def __init__(self, rows, cols):
            self.candidate = Candidate(1, np.array(rows), np.array(cols), 6.0, 1.0, 1.0)

    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 3
    recs = [R([2, 3, 4], [2, 3, 4]), R([8, 8, 8, 8, 2], [8, 7, 6, 5, 2]), R([0], [0])]
    assert overlap_labels(recs, labels) == [3, 3, 0]  # 1 of 5 pixels is still 20%
    assert overlap_labels(recs, labels, min_overlap=0.5) == [3, 0, 0]
    df = pd.DataFrame(
        {"plume": [3, 3, 0], "gas": ["CH4", "CO2", "CH4"], "true_gases": ["CH4", "CH4", ""]}
    )
    assert correct(df).tolist() == [1, 0, 0]


def test_confidence_model_takes_its_own_features():
    import pandas as pd

    from specret.openset.calibrate import ConfidenceModel

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (df["a"] + 0.3 * rng.normal(size=400) > 0).astype(int).to_numpy()
    m = ConfidenceModel.fit(df, y, features=("a", "b"))
    assert m.features == ("a", "b")
    assert m.coef[1] > 1 and abs(m.coef[2]) < m.coef[1] / 3
