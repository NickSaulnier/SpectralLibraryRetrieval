"""AVIRIS-NG L1B radiance (ORNL DAAC AVIRIS-NG_L1B_radiance_2095), for gas detection (G5).

A flight line folder (data/scenes/avirisng/<ang id>/) holds ENVI files:
* <id>_rdn_v1n2_img   radiance, BIL float32, uW cm^-2 nm^-1 sr^-1 (the same units as EMIT),
                      already resampled onto a rotated UTM map grid (`map info`)
* <id>_rdn_v1n2_glt   for each map pixel, the raw (sample, line) it came from, 1-based; negative
                      values are nearest-neighbour fills, 0 is no data
* <id>_rdn_v1n2_obs   observation geometry in raw (sensor) geometry, EMIT's 11-band layout
* <id>_rdn_v1n2_loc   longitude, latitude, elevation in raw geometry

The matched filter needs detector columns, so the map-grid radiance is scattered back into raw
geometry through the GLT's positive entries. The scene then exposes the same interface as
io/emit_l1b.EmitL1B, and the gas pipeline (gas/detect.py, gas/pipeline.py) runs unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from specret.io.emit import orthorectify

AVIRISNG_ROOT = Path("data/scenes/avirisng")
FILL = -9999.0
OBS_TO_SENSOR_ZENITH = 2
OBS_TO_SUN_ZENITH = 4
LINE_ID = re.compile(r"(ang\d{8}t\d{6})")


def _envi(path: Path):
    from spectral import envi

    data = path.with_suffix(".bin")
    return envi.open(str(path), str(data)) if data.exists() else envi.open(str(path))


def map_geotransform(map_info: list[str]) -> tuple[tuple[float, ...], str]:
    """GDAL geotransform (with rotation) and CRS WKT from an ENVI `map info` (UTM)."""
    from pyproj import CRS

    x0, y0 = float(map_info[3]), float(map_info[4])
    dx, dy = float(map_info[5]), float(map_info[6])
    zone, hemi = int(map_info[7]), map_info[8].strip()
    rot = 0.0
    for item in map_info:
        if "rotation" in item:
            rot = float(item.split("=")[1])
    t = np.radians(rot)
    # ENVI rotates the grid counter-clockwise about the tie point
    gt = (x0, dx * np.cos(t), dy * np.sin(t), y0, dx * np.sin(t), -dy * np.cos(t))
    epsg = (32600 if hemi.lower().startswith("n") else 32700) + zone
    return tuple(float(v) for v in gt), CRS.from_epsg(epsg).to_wkt()


@dataclass
class AvirisNgScene:
    path: Path  # radiance header
    img: object  # spectral image (map grid, BIL memmap)
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    glt_x: np.ndarray  # (map rows, map cols) raw sample, 1-based, 0 = none
    glt_y: np.ndarray
    lat: np.ndarray  # raw geometry
    lon: np.ndarray
    elev_m: np.ndarray
    obs: np.ndarray | None
    geotransform: tuple[float, ...]
    crs_wkt: str
    glt_own: np.ndarray | None = None  # map pixels holding their raw pixel (not a fill)
    _scatter: tuple = field(default=None, repr=False)

    @property
    def scene_id(self) -> str:
        return LINE_ID.search(self.path.name).group(1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape

    @property
    def time(self) -> datetime:
        return datetime.strptime(self.scene_id[3:], "%Y%m%dt%H%M%S").replace(tzinfo=UTC)

    def band_indices(self, lo_nm: float, hi_nm: float) -> np.ndarray:
        return np.flatnonzero((self.centers_nm >= lo_nm) & (self.centers_nm <= hi_nm))

    def _pairs(self):
        """(map rows, map cols, raw lines, raw samples) for the GLT's own (positive) entries."""
        if self._scatter is None:
            ok = (self.glt_x > 0) & (self.glt_y > 0)
            if self.glt_own is not None:
                ok &= self.glt_own
            r, c = np.nonzero(ok)
            self._scatter = (r, c, self.glt_y[ok] - 1, self.glt_x[ok] - 1)
        return self._scatter

    def read_bands(self, idx: np.ndarray) -> np.ndarray:
        """Radiance for the given bands in raw geometry, (lines, samples, len(idx)), NaN fill."""
        mm = self.img.open_memmap(interleave="bip")
        r, c, lr, ls = self._pairs()
        out = np.full(self.shape + (idx.size,), np.nan, dtype=np.float32)
        for i0 in range(0, r.size, 200_000):  # chunked gather keeps memory flat
            sl = slice(i0, i0 + 200_000)
            out[lr[sl], ls[sl]] = mm[r[sl], c[sl]][:, idx]
        out[out <= FILL + 1] = np.nan
        return out

    def amf(self) -> np.ndarray:
        if self.obs is None:
            return np.full(self.shape, 1 + 1 / np.cos(np.radians(30.0)), dtype=np.float32)
        sun = self.sun_zenith_band()
        sz = np.radians(self.obs[:, :, sun])
        vz = np.radians(self.obs[:, :, OBS_TO_SENSOR_ZENITH])
        a = (1 / np.cos(sz) + 1 / np.cos(vz)).astype(np.float32)
        a[(self.obs[:, :, sun] <= FILL + 1) | ~np.isfinite(a)] = np.nan
        return a

    def sun_zenith_band(self) -> int:
        """Index of the to-sun zenith band. Some v1 OBS files hold sun zenith and azimuth in
        each other's place despite their band names, so pick the one within 0-90 degrees."""
        med = [float(np.nanmedian(self.obs[:, :, i])) for i in (OBS_TO_SUN_ZENITH, 3)]
        return OBS_TO_SUN_ZENITH if 0 <= med[0] <= 90 else 3

    def cloud_water(self) -> np.ndarray:
        """No L2A mask comes with the radiance: nothing is screened."""
        return np.zeros(self.shape, dtype=bool)

    def flagged(self) -> np.ndarray:
        return self.cloud_water()

    def h2o_gcm2(self) -> float | None:
        return None

    def orthorectify(self, image: np.ndarray, fill: float = np.nan) -> np.ndarray:
        return orthorectify(image, self.glt_x, self.glt_y, fill)


def find_line(name: str, root: Path = AVIRISNG_ROOT) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    hits = sorted(root.rglob(f"*{name}*_rdn_*_img.hdr"))
    if not hits:
        raise ValueError(f"no AVIRIS-NG radiance for '{name}' under {root}")
    return hits[0]


def open_avirisng(path: str | Path) -> AvirisNgScene:
    path = find_line(str(path))
    img = _envi(path)
    md = img.metadata
    stem = path.name.removesuffix("_img.hdr")
    folder = path.parent
    glt = _envi(folder / f"{stem}_glt.hdr").load().astype(np.int64)
    loc = np.asarray(_envi(folder / f"{stem}_loc.hdr").load(), dtype=np.float64)
    obs_hdr = folder / f"{stem}_obs.hdr"
    obs = np.asarray(_envi(obs_hdr).load(), dtype=np.float32) if obs_hdr.exists() else None
    gt, wkt = map_geotransform(md["map info"])
    return AvirisNgScene(
        path=path,
        img=img,
        centers_nm=np.array(md["wavelength"], dtype=np.float64),
        fwhm_nm=np.array(md["fwhm"], dtype=np.float64),
        glt_x=np.abs(glt[:, :, 0]),
        glt_y=np.abs(glt[:, :, 1]),
        lat=loc[:, :, 1],
        lon=loc[:, :, 0],
        elev_m=loc[:, :, 2],
        obs=obs,
        geotransform=gt,
        crs_wkt=wkt,
        glt_own=(glt[:, :, 0] > 0) & (glt[:, :, 1] > 0),
    )
