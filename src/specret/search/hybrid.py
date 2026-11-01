"""Hybrid retrieval: rank fusion over several vectors, then feature-fit reranking.

1. Recall: exact top-K in each of `vectors` (default sam, cr, cr_swir2).
2. Fusion: reciprocal rank fusion keeps the best `n_rerank` candidates.
3. Rerank: spectral feature fitting on diagnostic windows (rerank/sff.py); ties fall back to
   the fused rank. With rerank=False the fused order is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from specret.preprocess.normalize import sam_degrees
from specret.rerank.features import load_features
from specret.rerank.sff import SFFResult, compute_features, explain, score_candidates
from specret.search.fusion import rrf
from specret.search.retrieve import RESULT_FIELDS, Retriever
from specret.search.vectors import represent
from specret.sensors.library import SensorLibrary

DEFAULT_VECTORS = ("sam", "cr", "cr_swir2")


@dataclass
class RankResult:
    rows: np.ndarray  # (nq, n_rerank) library rows, best first; -1 = none
    fused: np.ndarray  # reciprocal-rank-fusion score
    combined: np.ndarray  # final ranking score in [0, 1]; -inf for padding
    sff: SFFResult | None  # feature-fit details, None when rerank is off


def _sub(
    lib: SensorLibrary,
    canonical: pd.DataFrame,
    vector: str,
    device: str | None,
    encoder_path: str | None = None,
):
    """A sub-retriever: a physics vector, or "learned" (default models/encoder_v1.pt)."""
    if vector == "learned":
        from specret.search.learned import DEFAULT_ENCODER, LearnedRetriever

        return LearnedRetriever(
            lib, canonical, encoder_path=encoder_path or DEFAULT_ENCODER, device=device
        )
    return Retriever(lib, canonical, vector, device=device)


class HybridRetriever:
    def __init__(
        self,
        lib: SensorLibrary,
        canonical: pd.DataFrame,
        vectors: tuple[str, ...] = DEFAULT_VECTORS,
        chapters: tuple[str, ...] | None = None,
        purity: tuple[str, ...] | None = None,
        k_recall: int = 50,
        n_rerank: int = 20,
        rerank: bool = True,
        sff_weight: float = 0.25,
        device: str | None = None,
        encoder_path: str | None = None,
    ):
        """`sff_weight` blends the rerank score with the fused rank score (both scaled to
        0-1 per query): 1.0 orders by feature fit alone, 0.0 keeps the fused order. The
        default 0.25 was chosen on the validation split (results/phase2bc_hybrid.md)."""
        first = Retriever(
            lib, canonical, vectors[0], device=device, chapters=chapters, purity=purity
        )
        self.lib = first.lib  # chapter-filtered
        self.meta = first.meta
        self.subs = [first] + [
            _sub(self.lib, canonical, v, device, encoder_path) for v in vectors[1:]
        ]
        self.k_recall, self.n_rerank, self.rerank = k_recall, n_rerank, rerank
        self.sff_weight = sff_weight
        self.features = load_features()
        self.lib_features = compute_features(self.lib.matrix, self.lib.sensor, self.features)
        good = self.lib.sensor.good_mask
        self._sam_lib, _ = represent(self.lib.matrix, self.lib.sensor.centers_nm, good, "sam")

    def check_bands(self, centers_nm: np.ndarray) -> None:
        self.subs[0].check_bands(centers_nm)

    def rank(self, spectra: np.ndarray) -> RankResult:
        """Candidate rows (nq, n_rerank) in final order with their scores."""
        spectra = np.atleast_2d(spectra)
        lists = [sub.search_rows(spectra, self.k_recall)[1] for sub in self.subs]
        rows, fused = rrf(lists, self.n_rerank)
        top = fused.max(axis=1, keepdims=True)
        fused_norm = np.divide(fused, top, out=np.zeros_like(fused), where=top > 0)
        if not self.rerank:
            return RankResult(rows, fused, np.where(rows >= 0, fused_norm, -np.inf), None)
        qf = compute_features(spectra, self.lib.sensor, self.features)
        res = score_candidates(qf, self.lib_features, rows, self.features)
        # Blend feature fit with the fused score, then break ties on the fused score.
        sff = np.where(np.isfinite(res.score), res.score, 0.0)
        combined = self.sff_weight * sff + (1.0 - self.sff_weight) * fused_norm
        combined[rows < 0] = -np.inf
        order = np.lexsort((-fused, -combined), axis=1)

        def take(a: np.ndarray) -> np.ndarray:
            return np.take_along_axis(a, order, axis=1)

        res.score = take(res.score)
        for d in res.per_window.values():
            for key in d:
                d[key] = take(np.asarray(d[key]))
        return RankResult(take(rows), take(fused), take(combined), res)

    def query(
        self, spectra: np.ndarray, k: int = 5, batch: int = 4096, with_explain: bool = False
    ) -> list[list[dict]]:
        spectra = np.atleast_2d(spectra)
        out: list[list[dict]] = []
        good = self.lib.sensor.good_mask
        for start in range(0, len(spectra), batch):
            chunk = spectra[start : start + batch]
            ranked = self.rank(chunk)
            rows, fused, res = ranked.rows, ranked.fused, ranked.sff
            qsam, _ = represent(chunk, self.lib.sensor.centers_nm, good, "sam")
            for qi in range(len(chunk)):
                matches = []
                for ci in range(min(k, rows.shape[1])):
                    r = rows[qi, ci]
                    if r < 0:
                        break
                    sid = str(self.lib.spectrum_ids[r])
                    m = self.meta.loc[sid]
                    cos = float(self._sam_lib[r] @ qsam[qi])
                    item = {
                        "spectrum_id": sid,
                        "score": float(ranked.combined[qi, ci]),
                        "fused": float(fused[qi, ci]),
                        "sff": None if res is None else float(res.score[qi, ci]),
                        "cosine": cos,
                        "sam_deg": float(sam_degrees(cos)),
                        **{f: m.get(f) for f in RESULT_FIELDS},
                    }
                    if with_explain and res is not None:
                        item["features"] = explain(res, qi, ci)
                    matches.append(item)
                out.append(matches)
        return out
