"""Physics-based distortions that turn clean library spectra into image-like spectra.

Used for the synthetic benchmark now and as training augmentation later (Phase 5). Both uses
must draw from separate seeds (PLAN.md, Decision 6).

All operations work on spectra already on a sensor's bands, except `jitter_resample`, which
re-convolves a native-resolution spectrum with perturbed band centres and widths to mimic
spectral calibration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from specret.sensors.resample import resample
from specret.sensors.sensor import Sensor

# Water-vapour and oxygen absorptions whose correction residuals leak into reflectance.
ATM_FEATURES_NM = (760.0, 940.0, 1140.0, 1380.0, 1880.0)


def photometric(x: np.ndarray, wl: np.ndarray, rng, scale=(0.6, 1.4), tilt=0.08) -> np.ndarray:
    """Overall brightness (illumination, shading) and a mild linear slope."""
    s = rng.uniform(*scale)
    t = rng.uniform(-tilt, tilt)
    return x * s * (1.0 + t * (wl - 1500.0) / 1000.0)


def linear_mix(x: np.ndarray, partner: np.ndarray, rng, dominant=(0.6, 0.9)) -> np.ndarray:
    """Areal mixture with the query material dominant."""
    f = rng.uniform(*dominant)
    return f * x + (1.0 - f) * partner


def add_noise(x: np.ndarray, sigma: np.ndarray | float, rng) -> np.ndarray:
    """Band-wise Gaussian noise; `sigma` is absolute reflectance (e.g. EMIT uncertainty)."""
    return x + rng.normal(0.0, 1.0, x.shape) * sigma


def atmospheric_residual(x: np.ndarray, wl: np.ndarray, rng, strength=0.03) -> np.ndarray:
    """Multiplicative residuals at the gas absorptions plus a smooth VNIR aerosol term.

    A parametric stand-in for the radiative-transfer residual bank planned for Phase 5.
    """
    factor = np.ones_like(wl)
    for c in ATM_FEATURES_NM:
        amp = rng.normal(0.0, strength)
        width = rng.uniform(15.0, 40.0)
        factor += amp * np.exp(-0.5 * ((wl - c) / width) ** 2)
    aerosol = rng.normal(0.0, strength / 2)
    factor += aerosol * np.clip((1000.0 - wl) / 600.0, 0.0, 1.0)
    return x * factor


def jitter_resample(
    native_wl: np.ndarray,
    native_refl: np.ndarray,
    native_fwhm: np.ndarray | None,
    sensor: Sensor,
    rng,
    shift_nm: float = 1.0,
    fwhm_frac: float = 0.1,
) -> np.ndarray:
    """Resample with band centres shifted and widths scaled, as if mis-calibrated."""
    centers = sensor.centers_nm + rng.normal(0.0, shift_nm)
    fwhm = sensor.fwhm_nm * (1.0 + rng.uniform(-fwhm_frac, fwhm_frac))
    return resample(native_wl, native_refl, centers, fwhm, lib_fwhm=native_fwhm)


@dataclass(frozen=True)
class Recipe:
    name: str
    photometric: bool = False
    mix: bool = False
    noise: bool = False
    atmosphere: float = 0.0
    jitter: bool = False


RECIPES = {
    r.name: r
    for r in (
        Recipe("clean"),
        Recipe("photometric", photometric=True),
        Recipe("noise", noise=True),
        Recipe("mix", mix=True),
        Recipe("atm-mild", atmosphere=0.02),
        Recipe("atm-strong", atmosphere=0.06),
        Recipe("jitter", jitter=True),
        Recipe("all", photometric=True, mix=True, noise=True, atmosphere=0.03, jitter=True),
    )
}


def load_noise_curve(path: str | Path | None, n_bands: int, default: float = 0.005) -> np.ndarray:
    """Per-band noise from a saved curve, or a flat `default` when there is none."""
    if path and Path(path).exists():
        curve = np.load(path)
        if curve.size != n_bands:
            raise ValueError(f"{path}: {curve.size} bands, sensor has {n_bands}")
        return curve
    return np.full(n_bands, default)


def emit_noise_curve(
    uncertainty_path: str | Path, lat: np.ndarray, lon: np.ndarray, bbox, max_pixels: int = 20000
) -> np.ndarray:
    """Median per-band reflectance uncertainty over pixels in `bbox` of an RFLUNCERT granule."""
    import netCDF4

    west, south, east, north = bbox
    rows, cols = np.nonzero((lon > west) & (lon < east) & (lat > south) & (lat < north))
    rng = np.random.default_rng(0)
    pick = rng.choice(rows.size, min(max_pixels, rows.size), replace=False)
    with netCDF4.Dataset(uncertainty_path) as ds:
        v = ds["reflectance_uncertainty"]
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        block = np.ma.filled(v[r0:r1, c0:c1, :], np.nan).astype(np.float64)
    samples = block[rows[pick] - r0, cols[pick] - c0]
    samples[samples < 0] = np.nan
    return np.nanmedian(samples, axis=0)
