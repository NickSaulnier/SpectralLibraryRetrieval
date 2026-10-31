"""Phase 1 retrieval: exact cosine (SAM-equivalent) search against a sensor library."""

from __future__ import annotations

import numpy as np
import pandas as pd

from specret.preprocess.normalize import sam_degrees
from specret.preprocess.splice import splice_correct
from specret.search.gpu_exact import ExactIndex
from specret.search.vectors import query_vectors, represent, vector_masks
from specret.sensors.library import SensorLibrary
from specret.sensors.resample import resample

RESULT_FIELDS = (
    "material_name",
    "mineral",
    "group",
    "material_class",
    "chapter",
    "source",
    "instrument",
    "sample_id",
)
MIN_QUERY_BANDS = 20  # below this a partial-coverage query is not attempted


class Retriever:
    def __init__(
        self,
        lib: SensorLibrary,
        canonical: pd.DataFrame,
        vector: str = "sam",
        device: str | None = None,
        chapters: tuple[str, ...] | None = None,
    ):
        """`chapters` restricts the searchable library, e.g. ("M", "S", "C") for minerals,
        mixtures and coatings only."""
        self.meta = canonical.set_index("spectrum_id")
        if chapters is not None:
            keep = self.meta.loc[lib.spectrum_ids, "chapter"].isin(chapters).to_numpy()
            lib = SensorLibrary(lib.sensor, lib.spectrum_ids[keep], lib.matrix[keep])
        self.lib = lib
        self.vector = vector
        self.mask = vector_masks(lib)[vector]
        vecs, ok = represent(lib.matrix, lib.sensor.centers_nm, self.mask, vector)
        self.rows = np.flatnonzero(ok)  # index position -> row of self.lib
        self.index = ExactIndex(vecs[ok], lib.spectrum_ids[ok], device=device)

    def search_rows(self, spectra: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k as (cosine, library row) arrays of shape (nq, k); row -1 for invalid queries."""
        spectra = np.atleast_2d(spectra)
        qv, ok = query_vectors(self.lib, spectra, kinds=(self.vector,))[self.vector]
        k = min(k, len(self.index))
        scores = np.full((len(spectra), k), np.nan, dtype=np.float32)
        rows = np.full((len(spectra), k), -1, dtype=np.int64)
        if ok.any():
            s, i = self.index.search(qv[ok], k)
            scores[ok], rows[ok] = s, self.rows[i]
        return scores, rows

    def check_bands(self, centers_nm: np.ndarray) -> None:
        c = self.lib.sensor.centers_nm
        if centers_nm.shape != c.shape or not np.allclose(centers_nm, c, atol=0.5):
            raise ValueError(
                f"query bands do not match library sensor '{self.lib.sensor.name}'. "
                "Resample the library to the scene's bands first "
                "(specret resample --sensor <cfg> --band-file <scene file>)."
            )

    def _match(self, sid: str, cosine: float) -> dict:
        m = self.meta.loc[sid]
        return {
            "spectrum_id": sid,
            "cosine": float(cosine),
            "sam_deg": float(sam_degrees(cosine)),
            **{f: m.get(f) for f in RESULT_FIELDS},
        }

    def _partial_query(self, spectrum: np.ndarray, k: int) -> list[dict]:
        """Search on the bands this query actually has, against library rows valid there."""
        mask = self.mask & np.isfinite(spectrum)
        if mask.sum() < MIN_QUERY_BANDS:
            return []
        c = self.lib.sensor.centers_nm
        q, q_ok = represent(spectrum, c, mask, self.vector)
        lib, lib_ok = represent(self.lib.matrix, c, mask, self.vector)
        if not (q_ok[0] and lib_ok.any()):
            return []
        sim = lib[lib_ok] @ q[0]
        ids = self.lib.spectrum_ids[lib_ok]
        top = np.argsort(-sim)[:k]
        return [self._match(str(ids[i]), sim[i]) for i in top]

    def query(self, spectra: np.ndarray, k: int = 5) -> list[list[dict]]:
        """Top-k matches for each spectrum (rows on the library's sensor bands).

        Queries missing some good bands fall back to a search restricted to the bands they
        have. A query with fewer than MIN_QUERY_BANDS usable bands returns no matches, and so
        does a featureless query in a continuum-removed representation.
        """
        spectra = np.atleast_2d(spectra)
        qv, ok = query_vectors(self.lib, spectra, kinds=(self.vector,))[self.vector]
        out: list[list[dict]] = [[] for _ in range(len(spectra))]
        if ok.any():
            scores, idx = self.index.search(qv[ok], k)
            ids = self.index.ids
            for qi, s_row, i_row in zip(np.flatnonzero(ok), scores, idx, strict=True):
                out[qi] = [self._match(str(ids[i]), s) for s, i in zip(s_row, i_row, strict=True)]
        missing = ~np.isfinite(spectra[:, self.mask]).all(axis=1)
        for qi in np.flatnonzero(~ok & missing):
            out[qi] = self._partial_query(spectra[qi], k)
        return out

    def query_field(
        self,
        wavelength_nm: np.ndarray,
        reflectance: np.ndarray,
        k: int = 5,
        splice: bool = True,
        native_fwhm_nm: np.ndarray | None = None,
    ) -> list[dict]:
        """Resample a high-resolution field spectrum to the library sensor, then query.

        `native_fwhm_nm` is the field instrument's resolution on its own wavelength grid.
        It defaults to the asd_fieldspec config, so an ASD spectrum is not blurred twice.
        """
        r = splice_correct(wavelength_nm, reflectance) if splice else reflectance
        if native_fwhm_nm is None:
            from specret.sensors import load_sensor

            field = load_sensor("asd_fieldspec")
            native_fwhm_nm = np.interp(wavelength_nm, field.centers_nm, field.fwhm_nm)
        s = self.lib.sensor
        on_bands = resample(wavelength_nm, r, s.centers_nm, s.fwhm_nm, lib_fwhm=native_fwhm_nm)
        return self.query(on_bands, k)[0]


def format_matches(matches: list[dict]) -> str:
    if not matches:
        return "no matches: too few valid bands in the query"
    lines = [f"{'rank':>4}  {'SAM deg':>7}  {'material':<40} {'instrument':<10} spectrum_id"]
    for rank, m in enumerate(matches, start=1):
        lines.append(
            f"{rank:>4}  {m['sam_deg']:7.2f}  {str(m['material_name'])[:40]:<40} "
            f"{str(m['instrument'])[:10]:<10} {m['spectrum_id']}"
        )
    return "\n".join(lines)
