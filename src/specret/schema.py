"""Canonical library record and its Parquet storage.

The canonical store keeps every spectrum at its native resolution. Sensor-specific
versions are derived from it and can always be rebuilt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ARRAY_COLUMNS = ("wavelength_nm", "reflectance", "native_fwhm_nm")


@dataclass
class SpectrumRecord:
    spectrum_id: str  # unique per measurement, e.g. "usgs07a:5976"
    sample_id: str  # leakage-control key: shared by all measurements of one physical sample
    source: str  # e.g. "usgs_splib07a", "ecostress"
    source_record: str  # file name or record number in the source
    material_name: str
    wavelength_nm: np.ndarray
    reflectance: np.ndarray  # 0-1 scale, NaN where invalid
    native_fwhm_nm: np.ndarray | None = None
    mineral: str | None = None
    group: str | None = None  # filled from configs/taxonomy.yaml (Phase 2)
    material_class: str | None = None
    chapter: str | None = None  # USGS chapter letter: M, S, C, L, O, A, V
    grain_size: str | None = None
    purity_flag: str | None = None
    purity_note: str | None = None
    instrument: str | None = None
    measurement: str | None = None  # AREF, RREF, RTGC, ...
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=np.float64)
        self.reflectance = np.asarray(self.reflectance, dtype=np.float64)
        if self.wavelength_nm.shape != self.reflectance.shape:
            raise ValueError(
                f"{self.spectrum_id}: wavelength and reflectance lengths differ "
                f"({self.wavelength_nm.size} vs {self.reflectance.size})"
            )
        if self.native_fwhm_nm is not None:
            self.native_fwhm_nm = np.asarray(self.native_fwhm_nm, dtype=np.float64)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.reflectance)

    @property
    def coverage_nm(self) -> tuple[float, float]:
        wl = self.wavelength_nm[self.valid_mask]
        if wl.size == 0:
            return (float("nan"), float("nan"))
        return (float(wl.min()), float(wl.max()))


def records_to_frame(records: list[SpectrumRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = asdict(r)
        row.pop("extra")
        for col in ARRAY_COLUMNS:
            row[col] = None if row[col] is None else np.asarray(row[col]).tolist()
        row["coverage_min_nm"], row["coverage_max_nm"] = r.coverage_nm
        row["n_valid"] = int(r.valid_mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_library(
    records: list[SpectrumRecord], path: str | Path, classify: bool = False
) -> pd.DataFrame:
    """Write the canonical parquet. `classify` fills group/material_class from the taxonomy."""
    df = records_to_frame(records)
    if classify:
        from specret.taxonomy import apply_taxonomy

        df = apply_taxonomy(df)
    if df["spectrum_id"].duplicated().any():
        dupes = df.loc[df["spectrum_id"].duplicated(), "spectrum_id"].tolist()[:5]
        raise ValueError(f"duplicate spectrum_id values, e.g. {dupes}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def read_library(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ARRAY_COLUMNS:
        df[col] = df[col].map(lambda v: None if v is None else np.asarray(v, dtype=np.float64))
    return df
