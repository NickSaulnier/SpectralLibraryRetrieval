"""HyTES thermal-infrared flights (ORNL DAAC), for LWIR gas detection (PLAN_GAS.md G4).

A flight folder (data/scenes/hytes/<flight>/) holds:
* <flight>-...-level_1a.dat/.hdr   L1 radiance, ENVI BIP float32, (lines, 512, 256 bands),
                                   W m^-2 sr^-1 µm^-1, band centres 7.5-12 µm
* <flight>-...-geoLat/geoLong/geoAlt.csv   per-pixel latitude, longitude, ground altitude
* <flight>-....L2.hdf5             land surface temperature (lines, 512), emissivity, and
                                   per-band ISAC atmospheric terms: transmittance (tau),
                                   upwelling path radiance (L_up) and downwelling sky radiance
The Chino dairy flight came without geolocation or L2. Its atmosphere then defaults to tau = 1
and L_up = 0 (a low flight over a dry scene; the matched filter is insensitive to a common
offset), and surface temperature comes from brightness temperature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HYTES_ROOT = Path("data/scenes/hytes")
FWHM_NM = 17.8  # HyTES spectral response, about one band spacing

C1 = 1.191042972e8  # 2 h c^2, in W m^-2 sr^-1 µm^4
C2 = 1.4387769e4  # h c / k, in µm K


def planck(wavelength_nm, t_k):
    """Blackbody radiance, W m^-2 sr^-1 µm^-1."""
    lam = np.asarray(wavelength_nm, dtype=np.float64) / 1e3
    return C1 / lam**5 / np.expm1(C2 / (lam * np.asarray(t_k, dtype=np.float64)))


def brightness_temperature(wavelength_nm, radiance):
    lam = np.asarray(wavelength_nm, dtype=np.float64) / 1e3
    r = np.clip(np.asarray(radiance, dtype=np.float64), 1e-6, None)
    return C2 / (lam * np.log1p(C1 / (lam**5 * r)))


@dataclass
class HytesFlight:
    name: str
    folder: Path
    radiance: np.ndarray  # memmap (lines, samples, bands)
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    tau: np.ndarray  # (bands,) ISAC transmittance, 1 without L2
    l_up: np.ndarray  # (bands,) ISAC path radiance, 0 without L2
    l2_path: Path | None
    geo_paths: dict[str, Path]

    @property
    def shape(self) -> tuple[int, int]:
        return self.radiance.shape[:2]

    @property
    def time(self) -> datetime:
        m = re.match(r"(\d{4}-\d{2}-\d{2})\.(\d{6})", self.name)
        return datetime.strptime(" ".join(m.groups()), "%Y-%m-%d %H%M%S").replace(tzinfo=UTC)

    def band_indices(self, lo_nm: float, hi_nm: float) -> np.ndarray:
        return np.flatnonzero((self.centers_nm >= lo_nm) & (self.centers_nm <= hi_nm))

    def read(self, rows: slice, idx: np.ndarray) -> np.ndarray:
        v = np.asarray(self.radiance[rows][:, :, idx], dtype=np.float32)
        v[~np.isfinite(v) | (v <= 0)] = np.nan
        return v

    def geolocation(self, rows: slice | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """(lat, lon) per pixel, or None without geolocation files."""
        if "lat" not in self.geo_paths:
            return None
        import pandas as pd

        def load(p):
            return pd.read_csv(p, header=None, dtype=np.float64).to_numpy()

        lat, lon = load(self.geo_paths["lat"]), load(self.geo_paths["lon"])
        return (lat, lon) if rows is None else (lat[rows], lon[rows])

    def land_surface_temperature(self, rows: slice) -> np.ndarray | None:
        if self.l2_path is None:
            return None
        import netCDF4

        with netCDF4.Dataset(self.l2_path) as ds:
            v = np.ma.filled(ds["l2_land_surface_temperature"][rows], np.nan).astype(np.float32)
        v[v <= 0] = np.nan
        return v


def find_flight(name: str, root: Path = HYTES_ROOT) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    hits = [d for d in sorted(root.iterdir()) if d.is_dir() and name in d.name]
    if not hits:
        raise ValueError(f"no HyTES flight matching '{name}' under {root}")
    return hits[0]


def flights(root: Path = HYTES_ROOT) -> list[str]:
    return [d.name for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []


def open_hytes(folder: str | Path) -> HytesFlight:
    import spectral

    folder = find_flight(str(folder))
    hdr = next(folder.glob("*level_1a.hdr"))
    img = spectral.open_image(str(hdr))
    centers = np.array(img.metadata["wavelength"], dtype=np.float64) * 1e3  # µm -> nm
    l2 = next(folder.glob("*.L2.hdf5"), None)
    tau, l_up = np.ones(centers.size), np.zeros(centers.size)
    if l2 is not None:
        import netCDF4

        with netCDF4.Dataset(l2) as ds:
            tau = np.asarray(ds["isac_trans_scaled"][:], dtype=np.float64)
            l_up = np.asarray(ds["isac_sky_up_scaled"][:], dtype=np.float64)
    geo = {}
    for key, tag in (("lat", "geoLat"), ("lon", "geoLong"), ("alt", "geoAlt")):
        hit = next(folder.glob(f"*{tag}.csv"), None)
        if hit is not None:
            geo[key] = hit
    return HytesFlight(
        name=folder.name,
        folder=folder,
        radiance=img.open_memmap(),
        centers_nm=centers,
        fwhm_nm=np.full(centers.size, FWHM_NM),
        tau=tau,
        l_up=l_up,
        l2_path=l2,
        geo_paths=geo,
    )
