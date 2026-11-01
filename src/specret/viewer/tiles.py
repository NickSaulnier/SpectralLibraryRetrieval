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


def render(
    path: Path, kind: str, z: int, x: int, y: int, rescale=(0.0, 1.0), colormap="viridis"
) -> bytes:
    """PNG bytes of one tile; a transparent tile outside the layer."""
    return _render(str(path), path.stat().st_mtime_ns, kind, z, x, y, tuple(rescale), colormap)


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
