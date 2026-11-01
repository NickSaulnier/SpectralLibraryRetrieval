"""Shared state for the MCP server and the viewer: libraries, retrievers, scenes and maps.

Everything is discovered from the project's data folders and loaded lazily, then cached.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from specret.preprocess.splice import splice_correct
from specret.rerank.sff import compute_features, explain, score_candidates
from specret.schema import read_library
from specret.search.hybrid import HybridRetriever
from specret.sensors.library import SensorLibrary
from specret.sensors.resample import resample

DEFAULT_CHAPTERS = ("M", "S", "C", "V")  # minerals, mixtures, coatings, vegetation


def jsonable(v):
    """Make numpy scalars and NaN JSON friendly."""
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [jsonable(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


@dataclass
class SceneInfo:
    scene_id: str
    path: Path
    kind: str  # "emit" or "aviris"


@dataclass
class Workspace:
    root: Path = Path(".")
    chapters: tuple[str, ...] | None = DEFAULT_CHAPTERS
    _canonical: pd.DataFrame | None = None
    _libs: dict[str, SensorLibrary] = field(default_factory=dict)
    _retrievers: dict[str, HybridRetriever] = field(default_factory=dict)
    _scenes: dict[str, object] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # --- discovery -------------------------------------------------------------------

    @property
    def canonical(self) -> pd.DataFrame:
        if self._canonical is None:
            self._canonical = read_library(self.root / "data/canonical/library.parquet")
        return self._canonical

    def sensors(self) -> list[str]:
        return sorted(p.parent.name for p in (self.root / "data/resampled").glob("*/library.npz"))

    def scenes(self) -> dict[str, SceneInfo]:
        out = {}
        for p in sorted((self.root / "data/scenes").rglob("EMIT_L2A_RFL_*.nc")):
            out[p.stem] = SceneInfo(p.stem, p, "emit")
        for p in sorted((self.root / "data/scenes").rglob("*.hdr")):
            out[p.stem] = SceneInfo(p.stem, p, "aviris")
        return out

    def map_dir(self, scene_id: str) -> Path:
        return self.root / "outputs/maps" / scene_id

    # --- loading ---------------------------------------------------------------------

    def library(self, sensor: str) -> SensorLibrary:
        if sensor not in self._libs:
            path = self.root / "data/resampled" / sensor / "library.npz"
            if not path.exists():
                raise ValueError(f"no library for sensor '{sensor}'; have {self.sensors()}")
            self._libs[sensor] = SensorLibrary.load(path)
        return self._libs[sensor]

    def retriever(self, sensor: str) -> HybridRetriever:
        with self._lock:
            if sensor not in self._retrievers:
                self._retrievers[sensor] = HybridRetriever(
                    self.library(sensor), self.canonical, chapters=self.chapters
                )
            return self._retrievers[sensor]

    def scene(self, scene_id: str):
        """A scene by id (file stem under data/scenes) or by path to its .nc / .hdr file."""
        if scene_id not in self._scenes:
            info = self.scenes().get(scene_id)
            if info is not None:
                path = info.path
            elif Path(scene_id).is_file():
                path = Path(scene_id)
            else:
                raise ValueError(f"unknown scene '{scene_id}'; have {sorted(self.scenes())}")
            self._scenes[scene_id] = open_scene(path)
        return self._scenes[scene_id]

    def sensor_for_scene(self, scene) -> str:
        """The resampled library whose bands match this scene."""
        for name in self.sensors():
            c = self.library(name).sensor.centers_nm
            sc = np.asarray(scene.centers_nm)
            if c.shape == sc.shape and np.allclose(c, sc, atol=0.5):
                return name
        path = getattr(scene, "path", None) or getattr(scene, "hdr_path", "<scene file>")
        cfg = "emit" if str(path).endswith(".nc") else "aviris_classic"
        raise ValueError(
            "no resampled library matches this scene's bands; run "
            f"`specret resample --sensor {cfg} --band-file {path}`"
        )

    @staticmethod
    def locate(scene, row=None, col=None, lat=None, lon=None) -> tuple[int, int]:
        """Swath (row, col) from either a row/col pair or a lat/lon pair."""
        if lat is not None and lon is not None:
            if not hasattr(scene, "nearest_pixel"):
                raise ValueError("lat/lon lookup needs an EMIT scene; give row and col")
            return scene.nearest_pixel(lat, lon)
        if row is None or col is None:
            raise ValueError("give row and col, or lat and lon")
        rows, cols = scene.shape
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"pixel ({row}, {col}) is outside the {rows} x {cols} scene")
        return row, col

    @staticmethod
    def pixel_info(scene, row: int, col: int) -> dict:
        info = {"row": int(row), "col": int(col)}
        if hasattr(scene, "lat"):
            info.update(lat=float(scene.lat[row, col]), lon=float(scene.lon[row, col]))
        return info

    # --- operations ------------------------------------------------------------------

    def to_sensor(
        self, wavelengths_nm, reflectance, sensor: str, native_fwhm_nm=None
    ) -> np.ndarray:
        """Put a spectrum on a sensor's bands (passes through if it already is)."""
        wl = np.asarray(wavelengths_nm, dtype=np.float64)
        refl = np.asarray(reflectance, dtype=np.float64)
        s = self.library(sensor).sensor
        if wl.shape == s.centers_nm.shape and np.allclose(wl, s.centers_nm, atol=0.5):
            return refl
        return resample(wl, refl, s.centers_nm, s.fwhm_nm, lib_fwhm=native_fwhm_nm)

    def confidence_model(self, sensor: str):
        """Calibrated model from configs/calibration/<sensor>.json, or None."""
        from specret.openset.calibrate import ConfidenceModel

        path = self.root / "configs/calibration" / f"{sensor}.json"
        return ConfidenceModel.load(path) if path.exists() else None

    def noise_curve(self, sensor: str) -> np.ndarray | None:
        path = self.root / "data/derived" / f"{sensor}_noise.npy"
        if path.exists():
            curve = np.load(path)
            if curve.size == self.library(sensor).sensor.n_bands:
                return curve
        return None

    def unmixer(self, sensor: str):
        from specret.unmix.pursuit import Unmixer

        key = f"unmix:{sensor}"
        with self._lock:
            if key not in self._retrievers:
                self._retrievers[key] = Unmixer(self.retriever(sensor), self.canonical)
            return self._retrievers[key]

    def identify(self, spectrum: np.ndarray, sensor: str, k: int = 5, unmix: bool = False) -> dict:
        r = self.retriever(sensor)
        matches = r.query(spectrum, k=k, with_explain=True)[0]
        for m in matches:
            m["features"] = m.get("features", [])[:4]
        p = None
        model = self.confidence_model(sensor)
        if model is not None and matches:
            from specret.openset.calibrate import confidence_features

            _, feats = confidence_features(r, spectrum)
            p = float(model.predict(feats)[0])
        out = {"sensor": sensor, "summary": _summary(matches, p), "matches": matches}
        if unmix:
            out["unmixing"] = self.unmix(spectrum, sensor)
        return jsonable(out)

    def unmix(self, spectrum: np.ndarray, sensor: str) -> dict | None:
        u = self.unmixer(sensor)
        result = u.unmix(np.asarray(spectrum, dtype=np.float64), self.noise_curve(sensor))
        return None if result is None else jsonable(u.describe(result, self.canonical))

    def explain(self, spectrum: np.ndarray, sensor: str, spectrum_id: str) -> dict:
        r = self.retriever(sensor)
        ids = list(r.lib.spectrum_ids)
        if spectrum_id not in ids:
            raise ValueError(f"'{spectrum_id}' is not in the {sensor} library")
        row = ids.index(spectrum_id)
        qf = compute_features(np.atleast_2d(spectrum), r.lib.sensor, r.features)
        res = score_candidates(qf, r.lib_features, np.array([[row]]), r.features)
        meta = self.canonical.set_index("spectrum_id").loc[spectrum_id]
        return jsonable(
            {
                "spectrum_id": spectrum_id,
                "material_name": meta["material_name"],
                "group": meta["group"],
                "purity_grade": meta.get("purity_flag"),
                "feature_fit_score": float(res.score[0, 0]),
                "windows": explain(res, 0, 0),
                "sample_note": meta.get("purity_note"),
                "window_definitions": {
                    w.name: {"range_nm": [w.lo_nm, w.hi_nm], "note": w.note}
                    for w in r.features.windows
                },
            }
        )

    def identify_field(self, path: str | Path, k: int = 5, unmix: bool = False) -> list[dict]:
        from specret.io.field import read_field
        from specret.sensors import load_sensor

        sensor = "field_asd"
        field_cfg = load_sensor("asd_fieldspec")
        out = []
        for spec in read_field(path):
            refl = splice_correct(spec.wavelength_nm, spec.reflectance)
            native = np.interp(spec.wavelength_nm, field_cfg.centers_nm, field_cfg.fwhm_nm)
            x = self.to_sensor(spec.wavelength_nm, refl, sensor, native_fwhm_nm=native)
            out.append({"name": spec.name, **self.identify(x, sensor, k, unmix=unmix)})
        return out


def open_scene(path: str | Path):
    """EMIT L2A granule (.nc) or AVIRIS ENVI header (.hdr)."""
    if str(path).endswith(".nc"):
        from specret.io.emit import open_emit_l2a

        return open_emit_l2a(path)
    from specret.io.aviris import open_aviris

    return open_aviris(path)


CONFIDENT = 0.5  # calibrated P(best group is right) at or above which the answer stands
UNKNOWN_BELOW = 0.3  # below this, report "likely not in the library"


def confidence_tier(p: float | None) -> str:
    if p is None:
        return "uncalibrated"
    if p >= CONFIDENT:
        return "confident"
    if p >= UNKNOWN_BELOW:
        return "uncertain"
    return "likely unknown"


def _summary(matches: list[dict], p_correct: float | None = None) -> dict:
    """Headline for the best match, with the calibrated probability that its group is right."""
    if not matches:
        return {"verdict": "no match", "reason": "too few usable bands or no absorption features"}
    top = matches[0]
    others = [m for m in matches[1:] if m.get("group") != top.get("group")]
    tier = confidence_tier(p_correct)
    return {
        "best_material": top.get("material_name"),
        "best_group": top.get("group"),
        "score": top["score"],
        "margin_to_other_group": top["score"] - (others[0]["score"] if others else 0.0),
        "p_group_correct": p_correct,
        "confidence": tier,
        "verdict": (
            "likely a material not in the library, or misidentified"
            if tier == "likely unknown"
            else f"{top.get('group')} ({tier})"
        ),
    }
