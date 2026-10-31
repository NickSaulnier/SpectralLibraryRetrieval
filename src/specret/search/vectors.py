"""Turn resampled spectra into the named vectors used for search.

Each representation is a band window plus a transform, followed by L2 normalisation, so
cosine similarity between vectors is the spectral angle in that representation:

* `sam`      - raw reflectance, all good bands (the baseline)
* `swir2`    - raw reflectance, good bands in 2.0-2.5 um
* `cr`       - absorption depth (1 - continuum-removed) over all good bands
* `cr_swir2` - absorption depth with a local continuum fitted inside 2.0-2.5 um only

The same functions prepare library rows and queries, so both sides always match.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from specret.preprocess.continuum import band_depth
from specret.preprocess.normalize import l2_normalize
from specret.sensors.library import SensorLibrary

MIN_DEPTH_NORM = 1e-4  # depth vectors flatter than this have no features to compare

Transform = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (wavelengths, spectra) -> values

TRANSFORMS: dict[str, Transform | None] = {
    "sam": None,
    "swir2": None,
    "cr": band_depth,
    "cr_swir2": band_depth,
}


def vector_masks(lib: SensorLibrary) -> dict[str, np.ndarray]:
    good, swir2 = lib.sensor.good_mask, lib.swir2_mask
    return {"sam": good, "swir2": swir2, "cr": good, "cr_swir2": swir2}


def represent(
    spectra: np.ndarray, wavelengths: np.ndarray, mask: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Vectors for `spectra` on `mask` bands. Returns (unit vectors, row_is_valid)."""
    x = np.atleast_2d(np.asarray(spectra, dtype=np.float64))[:, mask]
    ok = np.isfinite(x).all(axis=1)
    transform = TRANSFORMS[kind]
    if transform is not None and ok.any():
        x = x.copy()
        x[ok] = transform(wavelengths[mask], x[ok])
        ok &= np.isfinite(x).all(axis=1)
        floor = MIN_DEPTH_NORM
    else:
        floor = 0.0
    norms = np.linalg.norm(np.where(np.isfinite(x), x, 0.0), axis=1)
    ok &= norms > floor
    vec = np.zeros_like(x)
    vec[ok] = l2_normalize(x[ok])
    return vec, ok


def masked_unit(spectra: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw-reflectance unit vectors on `mask` bands (the `sam` representation)."""
    return represent(spectra, np.zeros(mask.size), mask, "sam")


def library_vectors(lib: SensorLibrary) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (vectors for every library row, row_is_valid)."""
    c = lib.sensor.centers_nm
    return {k: represent(lib.matrix, c, m, k) for k, m in vector_masks(lib).items()}


def query_vectors(
    lib: SensorLibrary, spectra: np.ndarray, kinds: tuple[str, ...] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Queries must already be on the library's sensor bands."""
    c = lib.sensor.centers_nm
    masks = vector_masks(lib)
    return {k: represent(spectra, c, masks[k], k) for k in (kinds or tuple(masks))}
