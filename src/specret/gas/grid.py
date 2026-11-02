"""Put a swath with per-pixel latitude and longitude onto a regular lat/lon grid (V4).

HyTES flights come with geolocation per pixel, not a map grid. The grid cell is the median
ground spacing between neighbouring pixels, and each cell takes the value of its nearest
pixel within 1.5 cells; cells farther from every pixel stay empty. Only cells within one
cell of a pixel are searched, so a diagonal flight line does not cost its whole bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy import ndimage
from scipy.spatial import cKDTree

M_PER_DEG = 111_320.0


@dataclass
class Grid:
    transform: Affine  # EPSG:4326
    shape: tuple[int, int]
    index: np.ndarray  # (rows, cols) flat swath index of each cell's pixel, -1 = empty

    def take(self, a: np.ndarray, fill=np.nan) -> np.ndarray:
        """A swath array (lines, samples) on the grid."""
        flat = a.reshape(-1)
        out = np.full(self.shape, fill, dtype=np.result_type(a.dtype, np.asarray(fill).dtype))
        ok = self.index >= 0
        out[ok] = flat[self.index[ok]]
        return out


def spacing_m(lat: np.ndarray, lon: np.ndarray) -> float:
    """Median distance between neighbouring pixels, along and across track (m)."""
    k = np.cos(np.radians(np.nanmedian(lat)))
    d = []
    for ax in (0, 1):
        dy = np.diff(lat, axis=ax) * M_PER_DEG
        dx = np.diff(lon, axis=ax) * M_PER_DEG * k
        d.append(np.nanmedian(np.hypot(dx, dy)))
    return float(min(d))


def swath_grid(lat: np.ndarray, lon: np.ndarray, cell_m: float | None = None) -> Grid:
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat0 = float(np.nanmedian(lat))
    k = np.cos(np.radians(lat0))
    cell_m = cell_m or spacing_m(lat, lon)
    dlat = cell_m / M_PER_DEG
    dlon = dlat / k
    west, north = float(np.nanmin(lon)), float(np.nanmax(lat))
    cols = int(np.ceil((np.nanmax(lon) - west) / dlon)) + 1
    rows = int(np.ceil((north - np.nanmin(lat)) / dlat)) + 1
    transform = Affine(dlon, 0, west, 0, -dlat, north)

    # Cells holding a pixel, grown by one: the only cells worth a nearest-pixel search
    pr = ((north - lat[ok]) / dlat).astype(int)
    pc = ((lon[ok] - west) / dlon).astype(int)
    hit = np.zeros((rows, cols), dtype=bool)
    hit[pr, pc] = True
    hit = ndimage.binary_dilation(hit)
    r, c = np.nonzero(hit)
    cy = north - (r + 0.5) * dlat
    cx = west + (c + 0.5) * dlon
    tree = cKDTree(np.column_stack([lon[ok] * k, lat[ok]]))
    dist, j = tree.query(np.column_stack([cx * k, cy]), distance_upper_bound=1.5 * dlat)
    found = np.isfinite(dist)
    index = np.full((rows, cols), -1, dtype=np.int64)
    index[r[found], c[found]] = np.flatnonzero(ok.reshape(-1))[j[found]]
    return Grid(transform, (rows, cols), index)
