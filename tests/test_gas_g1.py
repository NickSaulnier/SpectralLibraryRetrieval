"""Phase G1 (PLAN_GAS.md): unit absorption spectrum, matched filter, reference rasters.

Synthetic spectroscopy and radiance only: no HITRAN tables or scenes needed.
"""

import numpy as np
import pytest
import rasterio
from affine import Affine

pytest.importorskip("torch")

from specret.gas.library import LineByLine, standard_atmosphere  # noqa: E402
from specret.gas.matched_filter import column_matched_filter  # noqa: E402
from specret.gas.signatures import (  # noqa: E402
    Conditions,
    band_transmittance,
    molecules_per_ppmm,
    unit_absorption,
)
from specret.io.emit_ghg import plume_mask, read_raster  # noqa: E402


def synthetic_lbl() -> LineByLine:
    """Lorentzian lines at 2150-2450 nm; CH4 strong enough to saturate in the background."""
    atm = standard_atmosphere()
    nu = np.arange(4050.0, 4700.0, 0.02)
    rng = np.random.default_rng(0)

    def lines(n, strength):
        centers = rng.uniform(nu[0], nu[-1], n)
        s = rng.lognormal(0, 1, n) * strength
        width = 0.07
        prof = width / np.pi / ((nu[None, :] - centers[:, None]) ** 2 + width**2)
        return np.repeat((s @ prof)[None, :], atm.n_layers, axis=0).astype(np.float32)

    sigma = {"CH4": lines(300, 1e-21), "H2O": lines(100, 2e-24), "CO2": lines(100, 1e-24)}
    return LineByLine(nu, sigma, atm)


def test_standard_atmosphere():
    atm = standard_atmosphere()
    assert atm.air_column.sum() == pytest.approx(2.15e25, rel=0.02)
    assert atm.p_atm[0] == pytest.approx(0.97, abs=0.02)
    assert np.all(np.diff(atm.p_atm) < 0)
    assert atm.h2o_shape.sum() == pytest.approx(1.0)


def test_molecules_per_ppmm():
    assert molecules_per_ppmm(0.0) == pytest.approx(2.55e15, rel=0.01)
    assert molecules_per_ppmm(1.5) < molecules_per_ppmm(0.0)


def test_unit_absorption_is_linear_for_small_alpha():
    lbl = synthetic_lbl()
    centers = np.arange(2200.0, 2400.0, 7.4)
    fwhm = np.full_like(centers, 8.5)
    cond = Conditions(amf=2.2, h2o_gcm2=1.5, surface_km=0.5)
    k = unit_absorption(lbl, centers, fwhm, cond)
    assert np.all(k > 0)
    small = -np.log(band_transmittance(lbl, centers, fwhm, cond, alpha=10.0)) / 10.0
    np.testing.assert_allclose(small, k, rtol=1e-3)
    # Strong plumes saturate: the effective absorption per ppm·m falls below k
    big = -np.log(band_transmittance(lbl, centers, fwhm, cond, alpha=2e5)) / 2e5
    assert np.all(big < k)
    # A longer path through the atmosphere absorbs more along the plume too
    assert unit_absorption(lbl, centers, fwhm, Conditions(3.0, 1.5, 0.5)).sum() > k.sum()


def synthetic_scene(rows=3000, cols=4, nb=24, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(2.0, 6.0, (cols, nb))
    a = rng.normal(size=(nb, nb))
    cov = 0.01 * (a @ a.T / nb + np.eye(nb))
    chol = np.linalg.cholesky(cov)
    x = mu[None] + rng.normal(size=(rows, cols, nb)) @ chol.T
    k = rng.uniform(0, 4e-4, nb)  # per ppm·m
    return x.astype(np.float32), mu, cov, k


def test_matched_filter_recovers_injected_alpha():
    x, mu, cov, k = synthetic_scene()
    truth = np.zeros(x.shape[:2])
    truth[100:160, :] = 500.0  # ppm·m
    x = x * (1 - k * truth[..., None]).astype(np.float32)
    res = column_matched_filter(x, k, device="cpu")
    est = res.alpha[100:160].mean(axis=0)
    np.testing.assert_allclose(est, 500.0, rtol=0.05)
    # Background noise matches theory: 1 / sqrt(t' Sigma^-1 t)
    for c in range(x.shape[1]):
        t = -mu[c] * k
        expected = 1 / np.sqrt(t @ np.linalg.solve(cov, t))
        assert res.sigma[c] == pytest.approx(expected, rel=0.1)
    # The second pass keeps the plume out of the background statistics
    assert res.excluded[100:160].mean() > 0.9
    assert res.excluded[:100].mean() < 0.01


def test_matched_filter_handles_fill_and_bad_columns():
    x, _, _, k = synthetic_scene(rows=500)
    x[10:20, 0, :] = np.nan  # fill pixels
    x[:, 1, :] = np.nan  # dead column
    x[:480, 2, 3] = np.nan  # too few complete pixels left
    res = column_matched_filter(x, k, device="cpu")
    assert np.all(np.isnan(res.alpha[10:20, 0]))
    assert np.isfinite(res.alpha[20:, 0]).all()
    assert np.isnan(res.alpha[:, 1]).all() and np.isnan(res.sigma[1])
    assert np.isnan(res.alpha[:, 2]).all()
    assert np.isfinite(res.alpha[:, 3]).all()
    # Pixels outside `valid` give no statistics but still get an estimate
    valid = np.ones(x.shape[:2], dtype=bool)
    valid[:250, 3] = False
    res2 = column_matched_filter(x, k, valid=valid, device="cpu")
    assert np.isfinite(res2.alpha[:, 3]).all()
    assert res2.n_valid[3] <= 250


def _tif(path, data, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        nodata=-9999.0,
        transform=transform,
        crs="EPSG:4326",
    ) as ds:
        ds.write(data.astype(np.float32), 1)


def test_plume_mask_places_crops_on_scene_grid(tmp_path):
    res = 0.001
    scene = "20230608T162322_2315911_014"
    full = Affine(res, 0, -103.0, 0, -res, 32.0)
    _tif(tmp_path / f"EMIT_L2B_CH4ENH_002_{scene}.tif", np.zeros((50, 60)), full)
    crop = np.full((4, 5), -9999.0)
    crop[1:3, 1:4] = 800.0
    _tif(
        tmp_path / "EMIT_L2B_CH4PLM_002_20230608T162322_000001.tif",
        crop,
        Affine(res, 0, -103.0 + 10 * res, 0, -res, 32.0 - 20 * res),
    )
    ref = read_raster(tmp_path / f"EMIT_L2B_CH4ENH_002_{scene}.tif")
    m = plume_mask(tmp_path, scene, ref)
    assert m.sum() == 6
    assert m[21:23, 11:14].all()


def test_loo_shrinkage_grows_when_pixels_are_scarce():
    import torch

    from specret.gas.matched_filter import loo_shrinkage

    rng = np.random.default_rng(3)
    nb = 40
    a = rng.normal(size=(nb, nb))
    chol = np.linalg.cholesky(a @ a.T / nb + 0.1 * np.eye(nb))
    out = []
    for rows in (60, 4000):
        x = rng.normal(size=(1, rows, nb)) @ chol.T
        d = torch.as_tensor(x - x.mean(axis=1, keepdims=True))
        out.append(float(loo_shrinkage(d, torch.tensor([rows]))[0]))
    assert 0 < out[1] < out[0] < 1


def test_window_absorption_is_zero_outside_line_by_line_ranges():
    from specret.gas.signatures import window_absorption

    lbl = synthetic_lbl()
    centers = np.array([1000.0, 2200.0, 2300.0, 2600.0])
    fwhm = np.full(4, 8.5)
    cond = Conditions(amf=2.2)
    k = window_absorption([((2150.0, 2450.0), lbl)], centers, fwhm, cond)
    assert k[0] == 0 and k[3] == 0
    np.testing.assert_allclose(k[1:3], unit_absorption(lbl, centers[1:3], fwhm[1:3], cond))


def test_deming_slope_corrects_for_noise_in_the_reference():
    from specret.eval.ghg_compare import deming_slope

    rng = np.random.default_rng(5)
    truth = rng.gamma(2.0, 300.0, 20000)
    x = truth + rng.normal(0, 280, truth.size)  # noisier reference
    y = truth + rng.normal(0, 190, truth.size)
    ols = np.polyfit(x, y, 1)[0]
    assert ols < 0.8
    assert deming_slope(x, y, (190 / 280) ** 2) == pytest.approx(1.0, abs=0.03)
