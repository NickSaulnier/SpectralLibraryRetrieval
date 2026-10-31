"""Build and store a library resampled to one sensor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from specret.preprocess.masks import window_mask
from specret.sensors.resample import resample
from specret.sensors.sensor import Sensor

SWIR2_WINDOW_NM = (2000.0, 2500.0)


@dataclass
class SensorLibrary:
    sensor: Sensor
    spectrum_ids: np.ndarray  # (m,) str
    matrix: np.ndarray  # (m, n_bands) reflectance, NaN where invalid

    @property
    def full_coverage(self) -> np.ndarray:
        """Spectra valid on every good band: eligible for the full-range vectors."""
        return np.isfinite(self.matrix[:, self.sensor.good_mask]).all(axis=1)

    @property
    def swir2_mask(self) -> np.ndarray:
        return self.sensor.good_mask & window_mask(self.sensor.centers_nm, *SWIR2_WINDOW_NM)

    @property
    def swir2_coverage(self) -> np.ndarray:
        """Spectra valid on every good band in 2.0-2.5 um: eligible for the swir2 vector."""
        return np.isfinite(self.matrix[:, self.swir2_mask]).all(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sensor_name=self.sensor.name,
            centers_nm=self.sensor.centers_nm,
            fwhm_nm=self.sensor.fwhm_nm,
            good_mask=self.sensor.good_mask,
            spectrum_ids=self.spectrum_ids.astype(str),
            matrix=self.matrix,
        )

    @classmethod
    def load(cls, path: str | Path) -> SensorLibrary:
        z = np.load(path, allow_pickle=False)
        sensor = Sensor(str(z["sensor_name"]), z["centers_nm"], z["fwhm_nm"], z["good_mask"])
        return cls(sensor, z["spectrum_ids"], z["matrix"])


def _grid_key(row: pd.Series) -> bytes:
    # ASD standard-res, hi-res and NG share a wavelength grid but not a resolution,
    # so the grouping key includes the FWHM too.
    key = np.round(np.asarray(row["wavelength_nm"], dtype=np.float64), 4).tobytes()
    if row["native_fwhm_nm"] is not None:
        key += np.round(np.asarray(row["native_fwhm_nm"], dtype=np.float64), 4).tobytes()
    return key


def build_sensor_library(
    df: pd.DataFrame, sensor: Sensor, use_library_fwhm: bool = True, min_coverage: float = 0.9
) -> SensorLibrary:
    """Resample every canonical spectrum to `sensor`, batching spectra that share a grid."""
    out = np.full((len(df), sensor.n_bands), np.nan)
    keys = df.apply(_grid_key, axis=1)
    for _, idx in df.groupby(keys).indices.items():
        first = df.iloc[idx[0]]
        stack = np.vstack(df["reflectance"].iloc[idx].to_list())
        lib_fwhm = first["native_fwhm_nm"] if use_library_fwhm else None
        out[idx] = resample(
            first["wavelength_nm"],
            stack,
            sensor.centers_nm,
            sensor.fwhm_nm,
            lib_fwhm=lib_fwhm,
            min_coverage=min_coverage,
        )
    return SensorLibrary(sensor, df["spectrum_id"].to_numpy().astype(str), out)
