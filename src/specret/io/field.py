"""Field spectrometer files: ASD binary (.asd) and CSV/ASCII exports.

ASD binary layout (ASD "Indico" file format, versions ASD/as1-as8):
    0     3 bytes   version string ("ASD", "as6", "as7", ...)
    186   uint8     data_type (0 raw, 1 reflectance, 2 radiance, ...)
    191   float32   wavelength of channel 1 (nm)
    195   float32   wavelength step (nm)
    199   uint8     data_format (0 float32, 1 int32, 2 float64)
    204   uint16    channel count
    484   ...       spectrum data
Files from version as6 on append the white-reference spectrum after a short reference
header: 2 flag bytes, two 8-byte timestamps, a 2-byte description length, the description.

This reader follows the published layout but has only been checked against synthetic files.
Validate it on real files (see PLAN.md, section 3.4) before relying on it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_SIZE = 484
DATA_TYPES = {
    0: "raw",
    1: "reflectance",
    2: "radiance",
    3: "no_units",
    4: "irradiance",
    5: "qi",
    6: "transmittance",
    7: "unknown",
    8: "absorbance",
}
DATA_FORMATS = {0: ("f", 4), 1: ("i", 4), 2: ("d", 8)}
VERSIONS_WITH_REFERENCE = {"as6", "as7", "as8"}


@dataclass
class FieldSpectrum:
    name: str
    wavelength_nm: np.ndarray
    values: np.ndarray  # as stored
    data_type: str
    reference: np.ndarray | None = None

    @property
    def reflectance(self) -> np.ndarray:
        """Reflectance: stored directly, or target / white reference for raw files."""
        if self.data_type == "reflectance":
            return self.values
        if self.reference is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = self.values / self.reference
            r[~np.isfinite(r)] = np.nan
            return r
        raise ValueError(f"{self.name}: stored as {self.data_type} with no white reference")


def read_asd(path: str | Path) -> FieldSpectrum:
    path = Path(path)
    buf = path.read_bytes()
    version = buf[0:3].decode("ascii", errors="replace")
    data_type = DATA_TYPES.get(buf[186], "unknown")
    ch1, step = struct.unpack_from("<ff", buf, 191)
    fmt_code = buf[199]
    (channels,) = struct.unpack_from("<H", buf, 204)
    if fmt_code not in DATA_FORMATS:
        raise ValueError(f"{path.name}: unsupported data_format {fmt_code}")
    code, size = DATA_FORMATS[fmt_code]
    nbytes = channels * size

    spec = np.array(struct.unpack_from(f"<{channels}{code}", buf, HEADER_SIZE), dtype=np.float64)
    wl = ch1 + step * np.arange(channels, dtype=np.float64)

    reference = None
    if version in VERSIONS_WITH_REFERENCE:
        off = HEADER_SIZE + nbytes + 2 + 8 + 8
        (desc_len,) = struct.unpack_from("<H", buf, off)
        off += 2 + desc_len
        if off + nbytes <= len(buf):
            reference = np.array(struct.unpack_from(f"<{channels}{code}", buf, off), dtype=float)
    return FieldSpectrum(path.stem, wl, spec, data_type, reference)


def read_field_table(path: str | Path) -> list[FieldSpectrum]:
    """Read a CSV/TSV export.

    Two layouts are recognised:
    * long: first column wavelength, one column per spectrum;
    * wide (EcoSIS style): one row per spectrum, wavelength-named columns plus metadata.
    Values above 2 are treated as percent.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    numeric_cols = []
    for c in df.columns:
        try:
            numeric_cols.append((float(str(c)), c))
        except ValueError:
            pass

    spectra: list[FieldSpectrum] = []
    if len(numeric_cols) > 50:  # wide
        wl = np.array([w for w, _ in numeric_cols])
        cols = [c for _, c in numeric_cols]
        meta_cols = [c for c in df.columns if c not in cols]
        for i, row in df.iterrows():
            name = str(row[meta_cols[0]]) if meta_cols else f"{path.stem}_{i}"
            vals = row[cols].to_numpy(dtype=np.float64)
            spectra.append(FieldSpectrum(name, wl, vals, "reflectance"))
    else:  # long
        wl = df.iloc[:, 0].to_numpy(dtype=np.float64)
        for c in df.columns[1:]:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            spectra.append(FieldSpectrum(str(c), wl, vals, "reflectance"))

    for s in spectra:
        if np.nanmax(s.values) > 2.0:
            s.values = s.values / 100.0
        if np.nanmax(s.wavelength_nm) < 10:
            s.wavelength_nm = s.wavelength_nm * 1000.0
    return spectra


def read_field(path: str | Path) -> list[FieldSpectrum]:
    path = Path(path)
    if path.suffix.lower() == ".asd":
        return [read_asd(path)]
    return read_field_table(path)
