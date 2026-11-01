"""Unit absorption spectra: how a plume changes at-sensor radiance, band by band (VSWIR).

The model (PLAN_GAS.md section 1.1): sunlight crosses the atmosphere twice, so along the path
the background transmittance is T_bg(nu) = exp(-AMF * tau(nu)), where tau is the vertical
optical depth of H2O, CO2 and CH4 above the surface. A plume at the surface with enhancement
alpha (ppm·m) multiplies this by exp(-AMF * sigma_p(nu) * N1 * alpha), where N1 is the number of
molecules per cm^2 in one ppm·m. The radiance in band b is the band response S_b integrated
over both:

    L_b(alpha) / L_b(0) = ∫ S_b T_bg exp(-AMF sigma_p N1 alpha) / ∫ S_b T_bg

The unit absorption spectrum is k_b = -d ln L_b / d alpha at alpha = 0. Weighting by T_bg
matters, because strong background lines are saturated and add little to k. The solar
spectrum is taken as flat within a band, since solar lines are weak at 2-2.5 µm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specret.gas.library import (
    K_BOLTZMANN,
    VMR,
    LineByLine,
    us_standard_pressure,
    us_standard_temperature,
)

H2O_MOLECULES_PER_G = 6.02214076e23 / 18.015


@dataclass
class Conditions:
    """Scene conditions for one target spectrum."""

    amf: float  # two-way air-mass factor, 1/cos(sun zenith) + 1/cos(view zenith)
    h2o_gcm2: float = 1.5  # water vapour column above the surface
    surface_km: float = 0.0  # surface elevation


def layer_fractions(lbl: LineByLine, surface_km: float) -> np.ndarray:
    """Share of each layer above the surface (0 below it, partial for the layer containing it)."""
    e = lbl.atmosphere.edges_km
    return np.clip((e[1:] - surface_km) / (e[1:] - e[:-1]), 0.0, 1.0)


def surface_layer(lbl: LineByLine, surface_km: float) -> int:
    e = lbl.atmosphere.edges_km
    return int(np.clip(np.searchsorted(e, surface_km, side="right") - 1, 0, e.size - 2))


def molecules_per_ppmm(surface_km: float) -> float:
    """Molecules/cm^2 in 1 ppm·m at the surface: 1e-6 * n_air * 100 cm."""
    p_pa = us_standard_pressure(surface_km)
    n_air = p_pa / (K_BOLTZMANN * us_standard_temperature(surface_km)) / 1e6  # per cm^3
    return 1e-6 * float(n_air) * 100.0


def vertical_optical_depth(lbl: LineByLine, cond: Conditions) -> np.ndarray:
    """Optical depth of the background atmosphere above the surface, straight up."""
    atm = lbl.atmosphere
    frac = layer_fractions(lbl, cond.surface_km)
    tau = np.zeros(lbl.nu_cm1.size)
    for gas, sigma in lbl.sigma.items():
        if gas == "H2O":
            # The scene's water column is what lies above the surface; spread it over the layers.
            shape = atm.h2o_shape * frac
            col = cond.h2o_gcm2 * H2O_MOLECULES_PER_G * shape / shape.sum()
        elif gas in VMR:
            col = VMR[gas] * atm.air_column * frac
        else:
            continue
        tau += col.astype(np.float64) @ sigma
    return tau


def band_response(nu_cm1: np.ndarray, centers_nm, fwhm_nm) -> np.ndarray:
    """Gaussian band responses (bands, n_nu), integrating in wavelength, each summing to 1."""
    wl = 1e7 / nu_cm1
    c = np.asarray(centers_nm, dtype=np.float64)[:, None]
    s = np.asarray(fwhm_nm, dtype=np.float64)[:, None] / (2 * np.sqrt(2 * np.log(2)))
    r = np.exp(-0.5 * ((wl[None, :] - c) / s) ** 2) * (1e7 / nu_cm1**2)
    return r / r.sum(axis=1, keepdims=True)


def _plume_terms(lbl: LineByLine, centers_nm, fwhm_nm, cond: Conditions, gas: str):
    if gas not in lbl.sigma:
        raise ValueError(f"no {gas} cross-sections in this window; have {sorted(lbl.sigma)}")
    lo, hi = lbl.wavelength_nm.min(), lbl.wavelength_nm.max()
    c = np.asarray(centers_nm)
    # Band responses are renormalized over the grid, so a band near its edge is only truncated
    if c.min() < lo or c.max() > hi:
        raise ValueError(
            f"bands {c.min():.0f}-{c.max():.0f} nm fall outside the {lo:.0f}-{hi:.0f} grid"
        )
    s = band_response(lbl.nu_cm1, centers_nm, fwhm_nm)
    t_bg = np.exp(-cond.amf * vertical_optical_depth(lbl, cond))
    sigma_p = lbl.sigma[gas][surface_layer(lbl, cond.surface_km)].astype(np.float64)
    tau_p = cond.amf * sigma_p * molecules_per_ppmm(cond.surface_km)  # per ppm·m
    return s, t_bg, tau_p


def band_transmittance(
    lbl: LineByLine, centers_nm, fwhm_nm, cond: Conditions, alpha: float, gas: str = "CH4"
) -> np.ndarray:
    """L_b(alpha) / L_b(0) for a plume of `alpha` ppm·m: the exact (non-linear) forward model."""
    s, t_bg, tau_p = _plume_terms(lbl, centers_nm, fwhm_nm, cond, gas)
    return (s @ (t_bg * np.exp(-tau_p * alpha))) / (s @ t_bg)


def unit_absorption(
    lbl: LineByLine, centers_nm, fwhm_nm, cond: Conditions, gas: str = "CH4"
) -> np.ndarray:
    """k_b = -d ln L_b / d alpha at alpha = 0, per ppm·m."""
    s, t_bg, tau_p = _plume_terms(lbl, centers_nm, fwhm_nm, cond, gas)
    return (s @ (t_bg * tau_p)) / (s @ t_bg)


def window_absorption(
    pieces: list[tuple[tuple[float, float], LineByLine]],
    centers_nm,
    fwhm_nm,
    cond: Conditions,
    gas: str = "CH4",
) -> np.ndarray:
    """Unit absorption for bands spread over several line-by-line ranges; 0 outside them."""
    c = np.asarray(centers_nm, dtype=np.float64)
    f = np.asarray(fwhm_nm, dtype=np.float64)
    k = np.zeros(c.size)
    for (lo, hi), lbl in pieces:
        m = (c >= lo) & (c <= hi)
        if m.any():
            k[m] = unit_absorption(lbl, c[m], f[m], cond, gas)
    return k
