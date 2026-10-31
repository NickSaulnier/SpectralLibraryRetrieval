"""EMIT L2A surface reflectance granules (NetCDF4, LP DAAC short name EMITL2ARFL).

Granule layout:
    /reflectance                     (downtrack, crosstrack, bands), fill -9999
    /sensor_band_parameters/         wavelengths, fwhm (nm), good_wavelengths (0/1)
    /location/                       lat, lon, elev, glt_x, glt_y (1-based, 0 = no data)
Per-band uncertainty ships in a separate EMIT_L2A_RFLUNCERT granule (reflectance_uncertainty).

The cube stays in sensor (swath) geometry and is read lazily. Use `orthorectify` on 2-D
outputs such as material maps rather than on the full cube, which would need several GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

FILL_VALUE = -9999.0


def read_band_parameters(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers_nm, fwhm_nm, good_mask) from an L2A granule."""
    with xr.open_dataset(path, group="sensor_band_parameters") as bp:
        centers = bp["wavelengths"].values.astype(np.float64)
        fwhm = bp["fwhm"].values.astype(np.float64)
        good = bp["good_wavelengths"].values.astype(bool)
    return centers, fwhm, good


@dataclass
class EmitScene:
    path: Path
    reflectance: xr.DataArray  # lazy, (downtrack, crosstrack, bands)
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    good_mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    glt_x: np.ndarray
    glt_y: np.ndarray
    geotransform: tuple[float, ...]
    crs_wkt: str
    uncertainty: xr.DataArray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.reflectance.shape[:2]

    def pixel(self, row: int, col: int) -> np.ndarray:
        v = self.reflectance[row, col, :].values.astype(np.float64)
        v[v <= FILL_VALUE + 1] = np.nan
        return v

    def pixel_uncertainty(self, row: int, col: int) -> np.ndarray | None:
        if self.uncertainty is None:
            return None
        v = self.uncertainty[row, col, :].values.astype(np.float64)
        v[v <= FILL_VALUE + 1] = np.nan
        return v

    def block(self, rows: slice, cols: slice) -> np.ndarray:
        v = self.reflectance[rows, cols, :].values.astype(np.float32)
        v[v <= FILL_VALUE + 1] = np.nan
        return v

    def nearest_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        d = (self.lat - lat) ** 2 + ((self.lon - lon) * np.cos(np.radians(lat))) ** 2
        r, c = np.unravel_index(np.nanargmin(d), d.shape)
        return int(r), int(c)

    def orthorectify(self, image: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Map a swath-geometry array (rows, cols[, k]) onto the GLT output grid."""
        return orthorectify(image, self.glt_x, self.glt_y, fill)


def orthorectify(
    image: np.ndarray, glt_x: np.ndarray, glt_y: np.ndarray, fill: float = np.nan
) -> np.ndarray:
    valid = (glt_x > 0) & (glt_y > 0)
    out_shape = glt_x.shape + image.shape[2:]
    out = np.full(out_shape, fill, dtype=np.result_type(image.dtype, np.float32))
    out[valid] = image[glt_y[valid] - 1, glt_x[valid] - 1]
    return out


def open_emit_l2a(path: str | Path, uncertainty_path: str | Path | None = None) -> EmitScene:
    path = Path(path)
    root = xr.open_dataset(path)
    centers, fwhm, good = read_band_parameters(path)
    with xr.open_dataset(path, group="location") as loc:
        lat = loc["lat"].values
        lon = loc["lon"].values
        glt_x = loc["glt_x"].values.astype(np.int64)
        glt_y = loc["glt_y"].values.astype(np.int64)
    unc = None
    if uncertainty_path is not None:
        unc = xr.open_dataset(uncertainty_path)["reflectance_uncertainty"]
    return EmitScene(
        path=path,
        reflectance=root["reflectance"],
        centers_nm=centers,
        fwhm_nm=fwhm,
        good_mask=good,
        lat=lat,
        lon=lon,
        glt_x=glt_x,
        glt_y=glt_y,
        geotransform=tuple(np.asarray(root.attrs.get("geotransform", []), dtype=float)),
        crs_wkt=str(root.attrs.get("spatial_ref", "")),
        uncertainty=unc,
    )


def search_emit(
    bbox: tuple[float, float, float, float],
    temporal: tuple[str, str],
    short_names: tuple[str, ...] = ("EMITL2ARFL",),
    cloud_cover_max: int | None = 20,
):
    """Find EMIT granules over a (west, south, east, north) box. Searching needs no login."""
    import earthaccess

    results = []
    for name in short_names:
        kwargs = dict(short_name=name, bounding_box=bbox, temporal=temporal)
        if cloud_cover_max is not None and name == "EMITL2ARFL":
            kwargs["cloud_cover"] = (0, cloud_cover_max)
        results.extend(earthaccess.search_data(**kwargs))
    return results
