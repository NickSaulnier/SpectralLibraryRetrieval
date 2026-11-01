"""XYZ map tiles (Web Mercator, 256 px PNG) from the viewer's COG layers, with rio-tiler.

Resampling is nearest-neighbour, so zooming in shows real pixels. No data and pixels outside a
scene are transparent. Rendered tiles are cached in memory, keyed by file and modification
time, so a rebuilt layer is never served stale.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

TILE = 256
COLORMAPS = ("viridis", "magma", "inferno", "cividis", "rdylgn", "rdbu_r", "greys_r", "turbo")


def empty_tile() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


EMPTY = empty_tile()


UNSELECTED = (40, 40, 40, 140)  # categories filtered out: dimmed, still showing the scene
LOW_CONFIDENCE = (205, 205, 205, 255)  # below the confidence threshold
LOW_CODE = 255


def render(
    path: Path, kind: str, z: int, x: int, y: int, rescale=(0.0, 1.0), colormap="viridis"
) -> bytes:
    """PNG bytes of one tile; a transparent tile outside the layer."""
    return _render(str(path), path.stat().st_mtime_ns, kind, z, x, y, tuple(rescale), colormap)


def render_categories(
    path: Path,
    z: int,
    x: int,
    y: int,
    keep: tuple[int, ...] | None = None,
    confidence: Path | None = None,
    min_p: float = 0.0,
) -> bytes:
    """A categorical tile where only `keep` codes keep their colour (the rest dimmed) and, with
    a confidence raster on the same grid, pixels below `min_p` drawn as LOW_CONFIDENCE."""
    conf = (str(confidence), confidence.stat().st_mtime_ns) if confidence and min_p > 0 else None
    return _render_categories(str(path), path.stat().st_mtime_ns, z, x, y, keep, conf, min_p)


@lru_cache(maxsize=4096)
def _render_categories(path, mtime, z, x, y, keep, conf, min_p) -> bytes:
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    with Reader(path) as src:
        if not src.tile_exists(x, y, z):
            return EMPTY
        try:
            img = src.tile(x, y, z, tilesize=TILE, indexes=1, resampling_method="nearest")
        except TileOutsideBounds:
            return EMPTY
        colors = dict(src.colormap or {})
    codes = img.array[0].filled(0).astype(np.int64)
    if keep is not None:
        for c in list(colors):
            if c and c not in keep:
                colors[c] = UNSELECTED
    if conf is not None:
        with Reader(conf[0]) as csrc:
            p = csrc.tile(x, y, z, tilesize=TILE, indexes=1, resampling_method="nearest")
        low = np.ma.filled(p.array[0].astype(np.float64), np.nan) < min_p
        codes = np.where(low & (codes > 0), LOW_CODE, codes)
        colors[LOW_CODE] = LOW_CONFIDENCE
    img.array = np.ma.MaskedArray(codes.astype(np.uint8)[None], mask=img.array.mask)
    return img.render(img_format="PNG", colormap=colors)


def category_fractions(path: Path) -> dict[int, float]:
    """Share of the valid pixels in each code of a categorical raster (read once, cached)."""
    return dict(_fractions(str(path), path.stat().st_mtime_ns))


@lru_cache(maxsize=64)
def _fractions(path: str, mtime: int) -> tuple:
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1)
        nodata = ds.nodata
    v = a[a != nodata] if nodata is not None else a.ravel()
    v = v[v > 0]
    if not v.size:
        return ()
    codes, counts = np.unique(v, return_counts=True)
    return tuple((int(c), float(n / v.size)) for c, n in zip(codes, counts, strict=True))


@lru_cache(maxsize=4096)
def _render(path: str, mtime: int, kind: str, z: int, x: int, y: int, rescale, colormap) -> bytes:
    from rio_tiler.colormap import cmap
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    with Reader(path) as src:
        if not src.tile_exists(x, y, z):
            return EMPTY
        try:
            if kind == "rgb":
                img = src.tile(
                    x, y, z, tilesize=TILE, indexes=(1, 2, 3), resampling_method="nearest"
                )
                return img.render(img_format="PNG")
            img = src.tile(x, y, z, tilesize=TILE, indexes=1, resampling_method="nearest")
        except TileOutsideBounds:
            return EMPTY
        if kind == "categorical":
            colors = src.colormap or {}
            return img.render(img_format="PNG", colormap=colors)
    lo, hi = rescale
    data = img.data[0].astype(np.float64)
    scaled = np.clip((data - lo) / (hi - lo if hi != lo else 1.0), 0, 1) * 255
    img.array = np.ma.MaskedArray(
        scaled.astype(np.uint8)[None], mask=img.array.mask | ~np.isfinite(data)[None]
    )
    return img.render(img_format="PNG", colormap=cmap.get(colormap))


def point_value(path: Path, lon: float, lat: float) -> float | None:
    """The layer's value at a location, or None outside it or at no data."""
    from rio_tiler.errors import PointOutsideBounds
    from rio_tiler.io import Reader

    try:
        with Reader(str(path)) as src:
            pt = src.point(lon, lat, indexes=1)
    except PointOutsideBounds:
        return None
    v = float(pt.data[0])
    masked = bool(np.ma.getmaskarray(pt.array)[0])
    return None if masked or not np.isfinite(v) else v
