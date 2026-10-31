"""Sensor band definitions (centres, FWHM, bad-band mask) loaded from configs/sensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from specret.preprocess.masks import range_mask

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class Sensor:
    name: str
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    good_mask: np.ndarray  # True for bands used in matching

    def __post_init__(self) -> None:
        self.centers_nm = np.asarray(self.centers_nm, dtype=np.float64)
        self.fwhm_nm = np.asarray(self.fwhm_nm, dtype=np.float64)
        self.good_mask = np.asarray(self.good_mask, dtype=bool)
        n = self.centers_nm.size
        if self.fwhm_nm.size != n or self.good_mask.size != n:
            raise ValueError(f"{self.name}: centres, FWHM and mask lengths differ")

    @property
    def n_bands(self) -> int:
        return self.centers_nm.size

    @property
    def n_good(self) -> int:
        return int(self.good_mask.sum())

    def with_mask(self, extra_good: np.ndarray) -> Sensor:
        """Return a copy whose mask also requires `extra_good` (e.g. EMIT good_wavelengths)."""
        return Sensor(self.name, self.centers_nm, self.fwhm_nm, self.good_mask & extra_good)


def read_usgs_column(path: str | Path) -> np.ndarray:
    """Read a USGS splib07 single-column file (one header line, one value per line)."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    return np.array([float(x) for x in lines[1:] if x.strip()], dtype=np.float64)


def bad_band_ranges(set_names: list[str], config_dir: Path = CONFIG_DIR) -> list[list[float]]:
    sets = yaml.safe_load((config_dir / "bad_bands.yaml").read_text())
    ranges: list[list[float]] = []
    for name in set_names:
        ranges.extend(sets[name])
    return ranges


def _fwhm_from_pieces(centers: np.ndarray, pieces: list[list[float]]) -> np.ndarray:
    fwhm = np.full(centers.shape, np.nan)
    for lo, hi, value in pieces:
        fwhm[(centers >= lo) & (centers <= hi)] = value
    if np.isnan(fwhm).any():
        raise ValueError("fwhm_nm pieces do not cover the whole grid")
    return fwhm


def load_sensor(
    name_or_path: str | Path,
    band_file: str | Path | None = None,
    config_dir: Path = CONFIG_DIR,
) -> Sensor:
    """Build a Sensor from a YAML config.

    `name_or_path` is a config path or a file stem in configs/sensors (e.g. "emit").
    `band_file` supplies band centres for configs whose band_source is a data product
    (an EMIT L2A granule, or an ENVI header for AVIRIS/PRISMA).
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = config_dir / "sensors" / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(path.read_text())
    source = cfg["band_source"]

    if source == "usgs_files":
        scale = 1000.0 if cfg.get("units", "um") == "um" else 1.0
        centers = read_usgs_column(REPO_ROOT / cfg["centers_file"]) * scale
        fwhm = read_usgs_column(REPO_ROOT / cfg["fwhm_file"]) * scale
    elif source == "grid":
        start, stop, step = cfg["grid_nm"]
        centers = np.arange(start, stop + step / 2, step, dtype=np.float64)
        fwhm = _fwhm_from_pieces(centers, cfg["fwhm_nm"])
    elif source in ("emit_l2a", "envi"):
        if band_file is None:
            raise ValueError(f"sensor '{cfg['name']}' reads its bands from a file: pass band_file")
        if source == "emit_l2a":
            from specret.io.emit import read_band_parameters

            centers, fwhm, good = read_band_parameters(band_file)
        else:
            from specret.io.aviris import read_envi_band_parameters

            centers, fwhm = read_envi_band_parameters(band_file)
            good = np.ones(centers.size, dtype=bool)
        mask = ~range_mask(centers, bad_band_ranges(cfg["bad_band_sets"], config_dir)) & good
        return Sensor(cfg["name"], centers, fwhm, mask)
    else:
        raise ValueError(f"unknown band_source '{source}' in {path}")

    mask = ~range_mask(centers, bad_band_ranges(cfg["bad_band_sets"], config_dir))
    return Sensor(cfg["name"], centers, fwhm, mask)
