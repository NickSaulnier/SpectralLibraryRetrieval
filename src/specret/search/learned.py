"""Retrieval with the learned encoder: same interface as search.retrieve.Retriever."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from specret.search.gpu_exact import ExactIndex
from specret.search.retrieve import RESULT_FIELDS, filter_library
from specret.sensors.library import SensorLibrary

DEFAULT_ENCODER = Path("models/encoder_v1.pt")


@lru_cache(maxsize=2)
def _load(path: str, device: str):
    from specret.learn.model import load_encoder

    return load_encoder(path, device)


class LearnedRetriever:
    def __init__(
        self,
        lib: SensorLibrary,
        canonical: pd.DataFrame,
        encoder_path: str | Path = DEFAULT_ENCODER,
        device: str | None = None,
        chapters: tuple[str, ...] | None = None,
        purity: tuple[str, ...] | None = None,
    ):
        import torch

        from specret.learn.model import embed

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load(str(encoder_path), self.device)
        self.meta = canonical.set_index("spectrum_id")
        self.lib = filter_library(lib, self.meta, chapters, purity)
        self.vector = "learned"
        s = self.lib.sensor
        self.mask = s.good_mask
        vecs, ok = embed(self.model, self.lib.matrix, s.centers_nm, s.fwhm_nm, s.good_mask)
        self.rows = np.flatnonzero(ok)
        self.index = ExactIndex(vecs[ok], self.lib.spectrum_ids[ok], device=self.device)

    def check_bands(self, centers_nm: np.ndarray) -> None:
        c = self.lib.sensor.centers_nm
        if centers_nm.shape != c.shape or not np.allclose(centers_nm, c, atol=0.5):
            raise ValueError(f"query bands do not match library sensor '{self.lib.sensor.name}'")

    def embed_queries(self, spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from specret.learn.model import embed

        s = self.lib.sensor
        return embed(self.model, np.atleast_2d(spectra), s.centers_nm, s.fwhm_nm, s.good_mask)

    def search_rows(self, spectra: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q, ok = self.embed_queries(spectra)
        k = min(k, len(self.index))
        scores = np.full((len(q), k), np.nan, dtype=np.float32)
        rows = np.full((len(q), k), -1, dtype=np.int64)
        if ok.any():
            s, i = self.index.search(q[ok], k)
            scores[ok], rows[ok] = s, self.rows[i]
        return scores, rows

    def query(self, spectra: np.ndarray, k: int = 5) -> list[list[dict]]:
        scores, rows = self.search_rows(spectra, k)
        out = []
        for s_row, r_row in zip(scores, rows, strict=True):
            matches = []
            for s, r in zip(s_row, r_row, strict=True):
                if r < 0:
                    break
                sid = str(self.lib.spectrum_ids[r])
                m = self.meta.loc[sid]
                matches.append(
                    {
                        "spectrum_id": sid,
                        "cosine": float(s),
                        "sam_deg": float("nan"),
                        **{f: m.get(f) for f in RESULT_FIELDS},
                    }
                )
            out.append(matches)
        return out
