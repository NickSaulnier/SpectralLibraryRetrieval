"""Line-by-line absorption cross-sections from the HITRAN tables in data/raw/hitran.

For each spectral window, every gas gets a cross-section (cm^2/molecule) on a fine wavenumber
grid at the temperature and pressure of each layer of a standard atmosphere. That is enough to
build both pieces of the VSWIR model in gas/signatures.py:
* the background transmittance along the light path (all gases, all layers above the surface);
* the plume cross-section (the layer at the surface).

Computing a window takes a few minutes with HAPI, so the result is cached as
data/derived/gas/lbl_<lo>_<hi>nm.npz. (PLAN_GAS.md names a parquet library; arrays on a
shared grid fit an npz better.)
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from specret.io.hitran import DEFAULT_DIR as HITRAN_DIR
from specret.io.hitran import load_gases
from specret.sensors.sensor import CONFIG_DIR

CACHE_DIR = Path("data/derived/gas")

K_BOLTZMANN = 1.380649e-23  # J/K
G0 = 9.80665  # m/s^2
M_AIR = 28.9647e-3 / 6.02214076e23  # kg per molecule of dry air
PA_PER_ATM = 101325.0

# Layer edges (km). Thin near the ground, where the plume and most of the water vapour are.
LAYER_EDGES_KM = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50.0])
H2O_SCALE_HEIGHT_KM = 2.0
VMR = {"CH4": 1.9e-6, "CO2": 420e-6}  # well-mixed background; H2O is set per scene
# Lines weaker than this (cm^-1/(molecule cm^-2) at 296 K) are dropped. 1e-27 keeps every line
# that matters for a water column of a few g/cm^2.
SW_MIN = 1e-27


def plume_config(config_dir: Path = CONFIG_DIR) -> dict:
    return yaml.safe_load((config_dir / "plumes.yaml").read_text())


def us_standard_temperature(z_km: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 temperature (K) up to 50 km."""
    z = np.asarray(z_km, dtype=np.float64)
    return np.select(
        [z <= 11, z <= 20, z <= 32],
        [288.15 - 6.5 * z, np.full_like(z, 216.65), 216.65 + (z - 20)],
        228.65 + 2.8 * (z - 32),
    )


@dataclass
class Atmosphere:
    edges_km: np.ndarray  # (L+1,)
    t_k: np.ndarray  # (L,) layer temperature
    p_atm: np.ndarray  # (L,) layer pressure
    air_column: np.ndarray  # (L,) molecules/cm^2 of air in each layer
    h2o_shape: np.ndarray  # (L,) relative H2O column per layer (sums to 1)

    @property
    def n_layers(self) -> int:
        return self.t_k.size


def us_standard_pressure(z_km) -> np.ndarray:
    """Hydrostatic pressure (Pa) of the standard atmosphere, from 1 atm at sea level."""
    z = np.linspace(0, 50.0, 5001)
    t = us_standard_temperature(z)
    # dlnP/dz = -g m / (k T), integrated with the trapezoid rule
    dlnp = -G0 * M_AIR / (K_BOLTZMANN * t) * 1e3
    lnp = np.concatenate([[0.0], np.cumsum(0.5 * (dlnp[1:] + dlnp[:-1]) * np.diff(z))])
    return PA_PER_ATM * np.exp(np.interp(z_km, z, lnp))


def standard_atmosphere(edges_km: np.ndarray = LAYER_EDGES_KM) -> Atmosphere:
    """Hydrostatic layers of the US standard atmosphere, from sea level."""
    p_edges = us_standard_pressure(edges_km)
    mid = 0.5 * (edges_km[1:] + edges_km[:-1])
    air = (p_edges[:-1] - p_edges[1:]) / (G0 * M_AIR) / 1e4  # per m^2 -> per cm^2
    h2o = np.exp(-edges_km[:-1] / H2O_SCALE_HEIGHT_KM) - np.exp(-edges_km[1:] / H2O_SCALE_HEIGHT_KM)
    return Atmosphere(
        edges_km=edges_km,
        t_k=us_standard_temperature(mid),
        p_atm=np.sqrt(p_edges[:-1] * p_edges[1:]) / PA_PER_ATM,
        air_column=air,
        h2o_shape=h2o / h2o.sum(),
    )


@dataclass
class LineByLine:
    """Cross-sections for one window: sigma[gas] is (layers, n_nu) in cm^2/molecule."""

    nu_cm1: np.ndarray
    sigma: dict[str, np.ndarray]
    atmosphere: Atmosphere

    @property
    def wavelength_nm(self) -> np.ndarray:
        return 1e7 / self.nu_cm1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        a = self.atmosphere
        np.savez_compressed(
            path,
            nu_cm1=self.nu_cm1,
            gases=np.array(list(self.sigma)),
            **{f"sigma_{g}": s for g, s in self.sigma.items()},
            edges_km=a.edges_km,
            t_k=a.t_k,
            p_atm=a.p_atm,
            air_column=a.air_column,
            h2o_shape=a.h2o_shape,
        )

    @classmethod
    def load(cls, path: Path) -> LineByLine:
        with np.load(path) as z:
            atm = Atmosphere(z["edges_km"], z["t_k"], z["p_atm"], z["air_column"], z["h2o_shape"])
            sigma = {str(g): z[f"sigma_{g}"] for g in z["gases"]}
            return cls(z["nu_cm1"], sigma, atm)


@contextlib.contextmanager
def _quiet():
    """HAPI prints a banner on import and a line per calculation."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def cross_sections(
    gas: str,
    nu_lo: float,
    nu_hi: float,
    atm: Atmosphere,
    step: float = 0.01,
    wing: float = 25.0,
    hitran_dir: Path = HITRAN_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Voigt cross-sections (layers, n_nu) for `gas` from its cached VSWIR HITRAN table."""
    with _quiet():
        import hapi

        hapi.VARIABLES["BACKEND_DATABASE_NAME"] = str(hitran_dir)
        table = f"{gas}_vswir"
        if table not in hapi.LOCAL_TABLE_CACHE:
            hapi.storage2cache(table)
        sub = f"_{gas}_{nu_lo:.0f}_{nu_hi:.0f}"
        hapi.select(
            table,
            Conditions=(
                "AND",
                ("between", "nu", nu_lo - wing, nu_hi + wing),
                (">=", "sw", SW_MIN),
            ),
            DestinationTableName=sub,
            Output=False,
        )
        rows = []
        nu = None
        for t, p in zip(atm.t_k, atm.p_atm, strict=True):
            nu, xs = hapi.absorptionCoefficient_Voigt(
                SourceTables=sub,
                WavenumberRange=[nu_lo, nu_hi],
                WavenumberStep=step,
                WavenumberWing=wing,
                Environment={"T": float(t), "p": float(p)},
                HITRAN_units=True,
            )
            rows.append(xs.astype(np.float32))
        hapi.dropTable(sub)
    return np.asarray(nu, dtype=np.float64), np.stack(rows)


def window_path(lo_nm: float, hi_nm: float, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"lbl_{lo_nm:.0f}_{hi_nm:.0f}nm.npz"


def lbl_ranges(windows: list[list[float]]) -> list[tuple[float, float]]:
    """The parts of matched-filter windows (nm) that the VSWIR HITRAN tables cover."""
    cov_lo, cov_hi = load_gases()["hitran_ranges_cm1"]["vswir"]
    lo_nm, hi_nm = 1e7 / cov_hi, 1e7 / cov_lo
    out = []
    for lo, hi in windows:
        a, b = max(lo, lo_nm), min(hi, hi_nm)
        if a < b:
            out.append((float(a), float(b)))
    return out


def build_window(
    lo_nm: float,
    hi_nm: float,
    gases: list[str] | None = None,
    cache_dir: Path = CACHE_DIR,
    hitran_dir: Path = HITRAN_DIR,
    force: bool = False,
) -> LineByLine:
    """Cross-sections for a window (nm), computed once and cached."""
    path = window_path(lo_nm, hi_nm, cache_dir)
    if path.exists() and not force:
        return LineByLine.load(path)
    cfg = plume_config()["library"]
    gases = gases or cfg["background_gases"]
    known = load_gases()["gases"]
    atm = standard_atmosphere()
    cov_lo, cov_hi = load_gases()["hitran_ranges_cm1"]["vswir"]
    nu_lo = max(1e7 / (hi_nm + cfg["margin_nm"]), cov_lo)
    nu_hi = min(1e7 / (lo_nm - cfg["margin_nm"]), cov_hi)
    sigma = {}
    nu = None
    for g in gases:
        if g not in known:
            raise ValueError(f"unknown gas {g}; see configs/gases.yaml")
        print(f"line-by-line {g}: {nu_lo:.0f}-{nu_hi:.0f} cm-1, {atm.n_layers} layers")
        nu, sigma[g] = cross_sections(
            g, nu_lo, nu_hi, atm, cfg["step_cm1"], cfg["wing_cm1"], hitran_dir
        )
    lbl = LineByLine(nu, sigma, atm)
    lbl.save(path)
    return lbl
