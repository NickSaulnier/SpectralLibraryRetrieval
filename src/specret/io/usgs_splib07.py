"""Parser for the USGS Spectral Library Version 7 ASCII release (Kokaly et al., 2017).

Layout of an extracted ASCIIdata_<set> folder:

    Chapter<letter>_<name>/<prefix>_<Material>_<Sample>_<INSTR><settings>_<MTYPE>.txt
    <prefix>_Wavelengths_<INSTR>...txt       (one per instrument, or one for convolved sets)
    <prefix>_Bandpass_(FWHM)_<INSTR>...txt   (or <prefix>_Resolution_... for convolved sets)

Every data file has one header line, e.g.
    " splib07a Record=5976: Kaolinite CM9                 BECKb AREF"
followed by one value per line. Deleted channels are stored as -1.23e34.
The letters after the instrument code (usually lowercase) describe spectrometer settings,
not purity.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from specret.preprocess.normalize import is_plausible
from specret.schema import SpectrumRecord
from specret.sensors.sensor import read_usgs_column

INSTRUMENTS = ("ASDFR", "ASDHR", "ASDNG", "BECK", "NIC4", "AVIRIS")
DEFAULT_MEASUREMENTS = ("AREF", "RREF", "RTGC")  # skips transmittance (TRAN)
DELETED_THRESHOLD = -1e30

HEADER_RE = re.compile(
    r"^\s*(?P<prefix>\S+)\s+Record=(?P<record>\d+):\s*(?P<title>.*?)\s+"
    r"(?P<instr>" + "|".join(INSTRUMENTS) + r")(?P<settings>[A-Za-z]*)\s+(?P<mtype>[A-Z]+)\s*$"
)
CHAPTER_RE = re.compile(r"^Chapter(?P<letter>[A-Z])_")
SAMPLE_TOKEN_RE = re.compile(r"(?=.*\d)(?=.*[A-Za-z])")
TEMPERATURE_RE = re.compile(r"^\d+K$")


def parse_header(line: str) -> dict:
    m = HEADER_RE.match(line)
    if not m:
        raise ValueError(f"unrecognised splib07 header: {line!r}")
    return m.groupdict()


def split_title(title: str) -> tuple[str, str | None]:
    """Split 'Kaolinite CM9' into material 'Kaolinite' and sample code 'CM9'.

    The sample code is the first token after the material name that mixes letters and digits
    (CM9, GDS376, BR93-22C, NMNH133746). Temperatures such as '293K' are skipped.
    """
    tokens = title.split()
    for i, tok in enumerate(tokens[1:], start=1):
        if SAMPLE_TOKEN_RE.match(tok) and not TEMPERATURE_RE.match(tok):
            return " ".join(tokens[:i]), tok
    return title.strip(), None


def read_spectrum_file(path: Path) -> tuple[dict, np.ndarray]:
    lines = path.read_text(encoding="latin-1").splitlines()
    header = parse_header(lines[0])
    values = np.array([float(x) for x in lines[1:] if x.strip()], dtype=np.float64)
    values[values < DELETED_THRESHOLD] = np.nan
    return header, values


class _BandFiles:
    """Finds the wavelength and bandpass file that goes with each instrument."""

    def __init__(self, root: Path):
        self.root = root
        files = [p for p in root.glob("*.txt")]
        self.wavelength_files = [p for p in files if "_Wavelengths_" in p.name]
        self.fwhm_files = [p for p in files if "_Bandpass_" in p.name or "_Resolution_" in p.name]
        # Convolved sets ship a single wavelength/resolution pair used by every spectrum.
        self.single = len(self.wavelength_files) == 1
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _pick(self, kind: str, instr: str) -> Path | None:
        files = self.wavelength_files if kind == "wl" else self.fwhm_files
        if self.single:
            return files[0] if files else None
        token = "ASD" if (kind == "wl" and instr.startswith("ASD")) else instr
        hits = [p for p in files if f"_{token}_" in p.name or f"_{token} " in p.name]
        return hits[0] if hits else None

    def get(self, kind: str, instr: str) -> np.ndarray | None:
        key = (kind, "*" if self.single else instr)
        if key not in self._cache:
            path = self._pick(kind, instr)
            self._cache[key] = None if path is None else read_usgs_column(path) * 1000.0
        return self._cache[key]


def load_splib07_dir(
    root: str | Path,
    measurements: tuple[str, ...] = DEFAULT_MEASUREMENTS,
    max_wavelength_nm: float | None = 3000.0,
) -> list[SpectrumRecord]:
    """Parse one extracted ASCIIdata_<set> folder into canonical records.

    Spectra are trimmed to `max_wavelength_nm` (VSWIR scope) and dropped if nothing valid is
    left. Records failing the reflectance scale check are kept but flagged in `purity_flag`.
    """
    root = Path(root)
    bands = _BandFiles(root)
    records: list[SpectrumRecord] = []

    for chapter_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = CHAPTER_RE.match(chapter_dir.name)
        if not m:
            continue  # e.g. errorbars/
        chapter = m.group("letter")
        for path in sorted(chapter_dir.glob("*.txt")):
            header, values = read_spectrum_file(path)
            if header["mtype"] not in measurements:
                continue
            instr = header["instr"]
            wl = bands.get("wl", instr)
            if wl is None or wl.size != values.size:
                raise ValueError(f"{path.name}: no wavelength file with {values.size} channels")
            fwhm = bands.get("fwhm", instr)
            if fwhm is not None and fwhm.size != wl.size:
                fwhm = None

            if max_wavelength_nm is not None:
                keep = wl <= max_wavelength_nm
                wl, values = wl[keep], values[keep]
                fwhm = None if fwhm is None else fwhm[keep]
            if not np.isfinite(values).any():
                continue

            material, sample = split_title(header["title"])
            prefix = header["prefix"]
            records.append(
                SpectrumRecord(
                    spectrum_id=f"{prefix}:{header['record']}",
                    sample_id=f"usgs:{sample}" if sample else f"usgs:{header['title'].strip()}",
                    source=f"usgs_{prefix.lower()}",
                    source_record=path.stem.removeprefix(f"{prefix}_"),
                    material_name=material,
                    mineral=material if chapter == "M" else None,
                    chapter=chapter,
                    instrument=f"{instr}{header['settings']}",
                    measurement=header["mtype"],
                    wavelength_nm=wl,
                    reflectance=values,
                    native_fwhm_nm=fwhm,
                    purity_flag=None if is_plausible(values) else "implausible_scale",
                )
            )
    return records
