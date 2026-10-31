"""Feature windows and band-depth profiles for spectral feature fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from specret.sensors.sensor import CONFIG_DIR, Sensor


@dataclass(frozen=True)
class Window:
    name: str
    lo_nm: float
    hi_nm: float
    note: str = ""


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[Window, ...]
    center_sigma_nm: float
    min_depth: float


def load_features(path: str | Path | None = None) -> FeatureConfig:
    cfg = yaml.safe_load(Path(path or CONFIG_DIR / "features.yaml").read_text())
    windows = tuple(
        Window(name, *spec["range_nm"], spec.get("note", ""))
        for name, spec in cfg["windows"].items()
    )
    return FeatureConfig(windows, float(cfg["center_sigma_nm"]), float(cfg["min_depth"]))


def window_bands(sensor: Sensor, window: Window, min_bands: int = 5) -> np.ndarray | None:
    """Indices of the sensor's good bands inside the window, or None if too few."""
    c = sensor.centers_nm
    idx = np.flatnonzero(sensor.good_mask & (c >= window.lo_nm) & (c <= window.hi_nm))
    return idx if idx.size >= min_bands else None


def depth_profiles(spectra: np.ndarray, wavelengths: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Band depth inside a window against a straight-line continuum.

    The continuum runs through the mean of the two bands at each end of the window, so a
    single noisy endpoint does not tilt it. Returns (m, len(idx)); NaN rows stay NaN.
    """
    S = np.atleast_2d(np.asarray(spectra, dtype=np.float64))[:, idx]
    wl = np.asarray(wavelengths, dtype=np.float64)[idx]
    x0, x1 = wl[:2].mean(), wl[-2:].mean()
    y0, y1 = S[:, :2].mean(axis=1), S[:, -2:].mean(axis=1)
    t = (wl - x0) / (x1 - x0)
    cont = y0[:, None] + (y1 - y0)[:, None] * t[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = 1.0 - S / cont
    depth[~(cont > 0)] = np.nan
    return depth


def band_center(depth: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Centre of the deepest band, refined with a parabola through its neighbours.

    Works on (..., n) arrays and returns (...,) in the units of `wl`.
    """
    d = np.nan_to_num(depth, nan=-np.inf)
    i = np.clip(np.argmax(d, axis=-1), 1, d.shape[-1] - 2)

    def take(offset: int) -> np.ndarray:
        return np.take_along_axis(d, (i + offset)[..., None], axis=-1)[..., 0]

    left, mid, right = take(-1), take(0), take(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = left - 2 * mid + right
        shift = np.where(np.isfinite(denom) & (denom < 0), 0.5 * (left - right) / denom, 0.0)
    shift = np.clip(np.nan_to_num(shift), -1, 1)
    return np.interp(i + shift, np.arange(wl.size), wl)
