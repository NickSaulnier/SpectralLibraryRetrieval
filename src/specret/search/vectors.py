"""Turn resampled spectra into the named vectors used for search.

Phase 1 has `sam` (all good bands) and `swir2` (good bands in 2.0-2.5 um). Phase 2 adds
`cr` (continuum-removed) and `deriv`. The same functions prepare library rows and queries,
so both sides always use identical band masks.
"""

from __future__ import annotations

import numpy as np

from specret.preprocess.normalize import l2_normalize
from specret.sensors.library import SensorLibrary


def masked_unit(spectra: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select `mask` bands, L2-normalise each row. Returns (vectors, row_is_valid)."""
    x = np.atleast_2d(np.asarray(spectra, dtype=np.float64))[:, mask]
    ok = np.isfinite(x).all(axis=1) & (np.abs(x).sum(axis=1) > 0)
    vec = np.zeros_like(x)
    vec[ok] = l2_normalize(x[ok])
    return vec, ok


def vector_masks(lib: SensorLibrary) -> dict[str, np.ndarray]:
    return {"sam": lib.sensor.good_mask, "swir2": lib.swir2_mask}


def library_vectors(lib: SensorLibrary) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (vectors for every library row, row_is_valid)."""
    return {name: masked_unit(lib.matrix, m) for name, m in vector_masks(lib).items()}


def query_vectors(
    lib: SensorLibrary, spectra: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Queries must already be on the library's sensor bands."""
    return {name: masked_unit(spectra, m) for name, m in vector_masks(lib).items()}
