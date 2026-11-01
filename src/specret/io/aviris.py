"""AVIRIS-Classic and AVIRIS-NG reflectance cubes in ENVI format.

Band centres and FWHM come from the ENVI header. Older products store reflectance as
integers scaled by 10,000; the scale is inferred from a pixel sample and applied on read.
Reflectance products are delivered orthorectified, so the cube's grid is already the map
grid; its georeference comes from the header's `map info` (read through GDAL).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spectral.io.envi as envi
from affine import Affine

from specret.preprocess.normalize import infer_scale


def _to_nm(values: list, units: str | None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    u = (units or "").lower()
    if "micro" in u or u in ("um", "µm") or (not u and v.max() < 10):
        v = v * 1000.0
    return v


def read_envi_band_parameters(hdr_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    h = envi.read_envi_header(str(hdr_path))
    units = h.get("wavelength units")
    centers = _to_nm(h["wavelength"], units)
    if "fwhm" in h:
        fwhm = _to_nm(h["fwhm"], units)
    else:  # fall back to band spacing
        fwhm = np.gradient(centers)
    return centers, fwhm


@dataclass
class AvirisScene:
    hdr_path: Path
    cube: object  # spectral SpyFile (memory-mapped)
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    scale: float
    ignore_value: float | None
    map_info: list | None
    geotransform: tuple[float, ...] | None = None  # GDAL order; None without map info
    crs_wkt: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.cube.shape[:2]

    def _clean(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if self.ignore_value is not None:
            v[v == self.ignore_value] = np.nan
        v = v * self.scale
        v[v < -0.5] = np.nan  # fill values such as -50 or -9999 after scaling
        return v

    def pixel(self, row: int, col: int) -> np.ndarray:
        return self._clean(self.cube.read_pixel(row, col))

    def block(self, rows: slice, cols: slice) -> np.ndarray:
        return self._clean(self.cube[rows, cols, :]).astype(np.float32)

    def orthorectify(self, image: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """The cube is already on its map grid, so swath and map geometry coincide."""
        return image

    def _require_georef(self) -> Affine:
        if self.geotransform is None or self.crs_wkt is None:
            raise ValueError(f"{self.hdr_path.name} has no map info; give row and col")
        return Affine.from_gdal(*self.geotransform)

    def nearest_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        from rasterio.warp import transform

        to_pixel = ~self._require_georef()
        xs, ys = transform("EPSG:4326", self.crs_wkt, [lon], [lat])
        col, row = to_pixel @ (xs[0], ys[0])
        row, col = int(np.floor(row)), int(np.floor(col))
        rows, cols = self.shape
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"({lat}, {lon}) is outside {self.hdr_path.name}")
        return row, col

    def pixel_latlon(self, row: int, col: int) -> tuple[float, float] | None:
        """Latitude and longitude of a pixel centre, or None without map info."""
        if self.geotransform is None or self.crs_wkt is None:
            return None
        from rasterio.warp import transform

        x, y = self._require_georef() @ (col + 0.5, row + 0.5)
        lons, lats = transform(self.crs_wkt, "EPSG:4326", [x], [y])
        return float(lats[0]), float(lons[0])


def _georeference(image_path: str) -> tuple[tuple[float, ...] | None, str | None]:
    """GDAL geotransform and CRS from the ENVI header's map info, if it has any."""
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning, RasterioIOError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(image_path) as ds:
                if ds.crs is None or ds.transform.is_identity:
                    return None, None
                return tuple(ds.transform.to_gdal()), ds.crs.to_wkt()
    except RasterioIOError:
        return None, None


def open_aviris(hdr_path: str | Path, image_path: str | Path | None = None) -> AvirisScene:
    hdr_path = Path(hdr_path)
    cube = envi.open(str(hdr_path), None if image_path is None else str(image_path))
    centers, fwhm = read_envi_band_parameters(hdr_path)
    h = cube.metadata
    ignore = h.get("data ignore value")
    ignore = float(ignore) if ignore is not None else None

    # Infer the stored scale from a sparse grid of pixels.
    rows = np.linspace(0, cube.shape[0] - 1, 8).astype(int)
    cols = np.linspace(0, cube.shape[1] - 1, 8).astype(int)
    sample = np.stack([cube.read_pixel(r, c) for r in rows for c in cols]).astype(np.float64)
    if ignore is not None:
        sample[sample == ignore] = np.nan
    sample[sample <= -9000] = np.nan
    scale = float(h.get("reflectance scale factor", 0)) or None
    scale = 1.0 / scale if scale else infer_scale(sample)

    geotransform, crs_wkt = _georeference(cube.filename)
    return AvirisScene(
        hdr_path, cube, centers, fwhm, scale, ignore, h.get("map info"), geotransform, crs_wkt
    )
