"""EMIT L1B at-sensor radiance (EMITL1BRAD) with its observation geometry, for gas detection.

Granule layout (same swath geometry and GLT as L2A, see io/emit.py):
    RAD  /radiance (downtrack, crosstrack, bands) in uW/cm^2/sr/nm, fill -9999
         /sensor_band_parameters/ wavelengths, fwhm   /location/ lat, lon, elev, glt_x, glt_y
    OBS  /obs (downtrack, crosstrack, 11): path length, to-sensor azimuth and zenith, to-sun
         azimuth and zenith, phase, slope, aspect, cos(i), UTC time, Earth-sun distance
The L2A MASK granule of the same scene, when present, adds cloud and water flags and the
retrieved water vapour column. Files are found by scene id in the same folder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from specret.io.emit import FILL_VALUE, orthorectify

OBS_TO_SENSOR_ZENITH = 2
OBS_TO_SUN_ZENITH = 4
MASK_AGGREGATE = 7  # cloud, dilated cloud, cirrus, water or spacecraft flag
MASK_CLOUD_CIRRUS_WATER = (0, 1, 2)
MASK_H2O = 6  # g/cm^2
SCENE_ID = re.compile(r"(\d{8}T\d{6}_\d{7}_\d{3})")


def scene_id(path: str | Path) -> str:
    m = SCENE_ID.search(Path(path).name)
    if not m:
        raise ValueError(f"no EMIT scene id in {path}")
    return m.group(1)


def sibling(path: Path, product: str) -> Path | None:
    """Another granule of the same scene in the same folder, e.g. product 'L1B_OBS'; None if
    there is none (or `path` is not an EMIT granule)."""
    if not SCENE_ID.search(Path(path).name):
        return None
    hits = sorted(path.parent.glob(f"EMIT_{product}_*{scene_id(path)}*.nc"))
    return hits[0] if hits else None


@dataclass
class EmitL1B:
    path: Path
    radiance: xr.DataArray  # lazy (downtrack, crosstrack, bands)
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elev_m: np.ndarray
    glt_x: np.ndarray
    glt_y: np.ndarray
    geotransform: tuple[float, ...]
    crs_wkt: str
    obs: xr.DataArray | None = None  # lazy (downtrack, crosstrack, 11)
    mask: xr.DataArray | None = None  # lazy L2A mask (downtrack, crosstrack, 8)

    @property
    def scene_id(self) -> str:
        return scene_id(self.path)

    @property
    def shape(self) -> tuple[int, int]:
        return self.radiance.shape[:2]

    @property
    def time(self) -> datetime:
        return datetime.strptime(self.scene_id.split("_")[0], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)

    def band_indices(self, lo_nm: float, hi_nm: float) -> np.ndarray:
        return np.flatnonzero((self.centers_nm >= lo_nm) & (self.centers_nm <= hi_nm))

    def read_bands(self, idx: np.ndarray) -> np.ndarray:
        """Radiance for the given bands, (rows, cols, len(idx)) float32, with fill as NaN."""
        lo, hi = int(idx.min()), int(idx.max()) + 1
        v = self.radiance[:, :, lo:hi].values[:, :, idx - lo].astype(np.float32, copy=False)
        v[v <= FILL_VALUE + 1] = np.nan
        return v

    def amf(self) -> np.ndarray:
        """Two-way air-mass factor per pixel. Without OBS, the scene is taken as nadir at 30°."""
        if self.obs is None:
            return np.full(self.shape, 1 + 1 / np.cos(np.radians(30.0)), dtype=np.float32)
        o = self.obs.values
        sz = np.radians(o[:, :, OBS_TO_SUN_ZENITH])
        vz = np.radians(o[:, :, OBS_TO_SENSOR_ZENITH])
        a = (1 / np.cos(sz) + 1 / np.cos(vz)).astype(np.float32)
        a[(o[:, :, OBS_TO_SUN_ZENITH] <= FILL_VALUE + 1) | ~np.isfinite(a)] = np.nan
        return a

    def flagged(self) -> np.ndarray:
        """Pixels the L2A mask flags as cloud, cirrus or water (all False without a mask)."""
        if self.mask is None:
            return np.zeros(self.shape, dtype=bool)
        return self.mask[:, :, MASK_AGGREGATE].values > 0

    def cloud_water(self) -> np.ndarray:
        """Cloud, cirrus or water pixels, without the dilated-cloud margin (all False without
        a mask). This is the screen EMIT's GHG products use."""
        if self.mask is None:
            return np.zeros(self.shape, dtype=bool)
        m = self.mask[:, :, list(MASK_CLOUD_CIRRUS_WATER)].values
        return (m > 0).any(axis=-1)

    def h2o_gcm2(self) -> float | None:
        """Median retrieved water vapour column over the scene, or None without a mask."""
        if self.mask is None:
            return None
        w = self.mask[:, :, MASK_H2O].values
        w = w[(w > 0) & np.isfinite(w)]
        return float(np.median(w)) if w.size else None

    def orthorectify(self, image: np.ndarray, fill: float = np.nan) -> np.ndarray:
        return orthorectify(image, self.glt_x, self.glt_y, fill)


def find_l1b(scene: str | Path, root: Path = Path("data/scenes")) -> Path:
    """The RAD granule for a scene id, or the path itself if it is a file."""
    p = Path(scene)
    if p.is_file():
        return p
    hits = sorted(root.rglob(f"EMIT_L1B_RAD_*{scene}*.nc"))
    if not hits:
        raise ValueError(f"no EMIT L1B radiance for '{scene}' under {root}")
    return hits[0]


def open_emit_l1b(path: str | Path) -> EmitL1B:
    path = Path(path)
    root = xr.open_dataset(path)
    with xr.open_dataset(path, group="sensor_band_parameters") as bp:
        centers = bp["wavelengths"].values.astype(np.float64)
        fwhm = bp["fwhm"].values.astype(np.float64)
    with xr.open_dataset(path, group="location") as loc:
        lat, lon, elev = loc["lat"].values, loc["lon"].values, loc["elev"].values
        glt_x = loc["glt_x"].values.astype(np.int64)
        glt_y = loc["glt_y"].values.astype(np.int64)
    obs_path = sibling(path, "L1B_OBS")
    mask_path = sibling(path, "L2A_MASK")
    return EmitL1B(
        path=path,
        radiance=root["radiance"],
        centers_nm=centers,
        fwhm_nm=fwhm,
        lat=lat,
        lon=lon,
        elev_m=elev,
        glt_x=glt_x,
        glt_y=glt_y,
        geotransform=tuple(np.asarray(root.attrs.get("geotransform", []), dtype=float)),
        crs_wkt=str(root.attrs.get("spatial_ref", "")),
        obs=xr.open_dataset(obs_path)["obs"] if obs_path else None,
        mask=xr.open_dataset(mask_path)["mask"] if mask_path else None,
    )
