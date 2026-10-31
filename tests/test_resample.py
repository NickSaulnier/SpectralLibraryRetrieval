import numpy as np
import pytest

from specret.sensors.resample import FWHM_TO_SIGMA, resample

CENTER = 2200.0
SIGMA_A = 5.0  # absorption width
DEPTH = 0.2


def absorption(wl, sigma=SIGMA_A, depth=DEPTH):
    return 0.5 - depth * np.exp(-0.5 * ((wl - CENTER) / sigma) ** 2)


def expected_at_center(fwhm, sigma=SIGMA_A, depth=DEPTH):
    """Gaussian absorption convolved with a unit-area Gaussian kernel, at the band centre."""
    sk = fwhm * FWHM_TO_SIGMA
    return 0.5 - depth * sigma / np.hypot(sigma, sk)


def test_matches_analytic_convolution():
    wl = np.arange(2000.0, 2400.0, 0.1)
    out = resample(wl, absorption(wl), np.array([CENTER]), np.array([10.0]))
    assert out[0] == pytest.approx(expected_at_center(10.0), abs=1e-5)


def test_irregular_grid_matches_regular():
    rng = np.random.default_rng(0)
    wl = np.sort(rng.uniform(2000.0, 2400.0, 6000))
    out = resample(wl, absorption(wl), np.array([CENTER]), np.array([10.0]))
    assert out[0] == pytest.approx(expected_at_center(10.0), abs=5e-4)


def test_band_off_the_grid_is_invalid():
    wl = np.arange(2000.0, 2400.0, 1.0)
    out = resample(wl, absorption(wl), np.array([2000.0, 2200.0, 2500.0]), np.full(3, 10.0))
    assert np.isnan(out[0])  # half the kernel falls off the grid
    assert np.isfinite(out[1])
    assert np.isnan(out[2])


def test_nan_gap_under_band_is_invalid_but_far_nan_is_ignored():
    wl = np.arange(2000.0, 2400.0, 1.0)
    refl = absorption(wl)
    gap = refl.copy()
    gap[(wl > 2190) & (wl < 2210)] = np.nan
    far = refl.copy()
    far[wl > 2350] = np.nan
    out = resample(wl, np.vstack([refl, gap, far]), np.array([CENTER]), np.array([10.0]))
    assert np.isfinite(out[0, 0])
    assert np.isnan(out[1, 0])
    assert out[2, 0] == pytest.approx(out[0, 0], abs=1e-9)


def test_library_resolution_is_not_applied_twice():
    lib_fwhm = 6.0
    s1 = np.hypot(SIGMA_A, lib_fwhm * FWHM_TO_SIGMA)
    wl = np.arange(2000.0, 2400.0, 0.1)
    # The library spectrum as a 6 nm instrument would have recorded it.
    blurred = absorption(wl, sigma=s1, depth=DEPTH * SIGMA_A / s1)
    out = resample(
        wl, blurred, np.array([CENTER]), np.array([10.0]), lib_fwhm=np.full(wl.size, lib_fwhm)
    )
    assert out[0] == pytest.approx(expected_at_center(10.0), abs=1e-5)


def test_coarse_library_falls_back_to_interpolation():
    wl = np.arange(2000.0, 2400.0, 10.0)
    refl = np.linspace(0.2, 0.6, wl.size)
    out = resample(wl, refl, np.array([2205.0]), np.array([8.0]), lib_fwhm=np.full(wl.size, 12.0))
    assert out[0] == pytest.approx(np.interp(2205.0, wl, refl))


def test_descending_input_is_sorted():
    wl = np.arange(2000.0, 2400.0, 0.5)
    a = resample(wl, absorption(wl), np.array([CENTER]), np.array([10.0]))
    b = resample(wl[::-1], absorption(wl)[::-1], np.array([CENTER]), np.array([10.0]))
    assert a[0] == pytest.approx(b[0])
