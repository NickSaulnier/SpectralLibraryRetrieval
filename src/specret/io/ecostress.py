"""Parser for ECOSTRESS spectral library text files (successor to the ASTER library).

Each file has a "Key: value" header (Name, Type, Class, Subclass, Particle Size, Sample No.,
Owner, ..., X Units, Y Units, ...) followed by two-column wavelength/value data.
Reflectance is usually stored in percent, and wavelengths in micrometres, sometimes
descending. Both are normalised here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from specret.preprocess.normalize import is_plausible
from specret.schema import SpectrumRecord

KEY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .]*?):\s*(.*)$")


def _parse_data_line(line: str) -> tuple[float, float] | None:
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def read_ecostress_file(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (header, wavelength_nm ascending, reflectance 0-1)."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    header: dict[str, str] = {}
    data: list[tuple[float, float]] = []
    last_key = None
    for line in lines:
        pair = _parse_data_line(line)
        if pair is not None and (data or "X Units" in header or not KEY_RE.match(line)):
            data.append(pair)
            continue
        if data:
            continue  # trailing text after the data block
        m = KEY_RE.match(line)
        if m:
            last_key = m.group(1).strip()
            header[last_key] = m.group(2).strip()
        elif line.strip() and last_key:  # continuation of a multi-line Description
            header[last_key] += " " + line.strip()

    if not data:
        raise ValueError(f"{path}: no data block found")
    x, y = np.array(data).T

    x_units = header.get("X Units", "").lower()
    if "micro" in x_units or x.max() < 100:
        x = x * 1000.0
    y_units = header.get("Y Units", "").lower()
    if "percent" in y_units:
        y = y / 100.0
    order = np.argsort(x)
    return header, x[order], y[order]


def load_ecostress_dir(root: str | Path, max_wavelength_nm: float | None = 3000.0):
    """Parse every *.spectrum.txt under `root` into canonical records."""
    records: list[SpectrumRecord] = []
    for path in sorted(Path(root).rglob("*.txt")):
        if "ancillary" in path.name.lower():
            continue
        try:
            header, wl, refl = read_ecostress_file(path)
        except ValueError:
            continue
        if max_wavelength_nm is not None:
            keep = wl <= max_wavelength_nm
            wl, refl = wl[keep], refl[keep]
        if wl.size < 2:
            continue  # thermal-only spectrum
        sample = header.get("Sample No.", path.stem)
        typ = header.get("Type", "")
        name = header.get("Name", path.stem)
        records.append(
            SpectrumRecord(
                spectrum_id=f"ecostress:{path.stem}",
                sample_id=f"ecostress:{header.get('Owner', '')}:{sample}",
                source="ecostress",
                source_record=path.name,
                material_name=name,
                mineral=name if typ.lower() == "mineral" else None,
                material_class="/".join(
                    v for v in (header.get("Class"), header.get("Subclass")) if v
                )
                or None,
                grain_size=header.get("Particle Size"),
                instrument=header.get("Measurement"),
                measurement="AREF",
                wavelength_nm=wl,
                reflectance=refl,
                purity_flag=None if is_plausible(refl) else "implausible_scale",
            )
        )
    return records
