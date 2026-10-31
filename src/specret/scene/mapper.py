"""Scene-scale mapping: run the hybrid retriever on every pixel.

The plan anticipated clustering pixels and reranking only centroids, but with numba
continuum removal, GPU exact search and vectorised feature fitting, the full hybrid
pipeline runs on every pixel of an EMIT scene in minutes, so no clustering is needed.

Results stay in sensor (swath) geometry. `scene.outputs` orthorectifies them for GeoTIFFs,
and the viewer and MCP tools read the swath arrays directly for per-pixel lookups.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from specret.search.hybrid import HybridRetriever

NO_DATA = -1


@dataclass
class SceneMap:
    scene_path: str
    library_ids: np.ndarray  # (n_lib,) spectrum ids; top_rows index into this
    library_groups: np.ndarray  # (n_lib,) taxonomy group per library row
    top_rows: np.ndarray  # (rows, cols, k) int32, -1 = no data
    scores: np.ndarray  # (rows, cols, k) float32 combined ranking score in [0, 1]
    margin: np.ndarray  # (rows, cols) float32: top-1 score minus best other-group score
    seconds: float = 0.0
    p_correct: np.ndarray | None = None  # (rows, cols) calibrated P(top-1 group is right)

    @property
    def shape(self) -> tuple[int, int]:
        return self.top_rows.shape[:2]

    @property
    def valid(self) -> np.ndarray:
        return self.top_rows[..., 0] >= 0

    def top1_group(self) -> np.ndarray:
        """(rows, cols) group name of the best match; "" where there is no data."""
        out = np.full(self.shape, "", dtype=object)
        v = self.valid
        out[v] = self.library_groups[self.top_rows[..., 0][v]]
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene_path=self.scene_path,
            library_ids=self.library_ids.astype(str),
            library_groups=self.library_groups.astype(str),
            top_rows=self.top_rows,
            scores=self.scores,
            margin=self.margin,
            seconds=self.seconds,
            p_correct=self.p_correct if self.p_correct is not None else np.array([]),
        )

    @classmethod
    def load(cls, path: str | Path) -> SceneMap:
        z = np.load(path, allow_pickle=False)
        return cls(
            str(z["scene_path"]),
            z["library_ids"],
            z["library_groups"],
            z["top_rows"],
            z["scores"],
            z["margin"],
            float(z["seconds"]),
            z["p_correct"] if "p_correct" in z and z["p_correct"].size else None,
        )


def _margin(rows: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Top-1 score minus the best score of a candidate from a different group."""
    out = np.full(rows.shape[0], np.nan, dtype=np.float32)
    valid = rows[:, 0] >= 0
    g = np.where(rows >= 0, groups[np.maximum(rows, 0)], "")
    other = (g != g[:, :1]) & (rows >= 0)
    best_other = np.where(other, scores, -np.inf).max(axis=1)
    m = scores[:, 0] - best_other
    m[~np.isfinite(m)] = scores[~np.isfinite(m), 0]  # no other group in the candidates
    out[valid] = m[valid]
    return out


def map_scene(
    scene,
    retriever: HybridRetriever,
    canonical: pd.DataFrame,
    k: int = 5,
    block_rows: int = 64,
    progress: bool = True,
    confidence=None,
) -> SceneMap:
    """Rank every pixel of `scene` (EmitScene or AvirisScene) against the library.

    With a `confidence` model (openset.calibrate.ConfidenceModel) each pixel also gets a
    calibrated P(top-1 group is right).
    """
    retriever.check_bands(np.asarray(scene.centers_nm))
    lib_ids = retriever.lib.spectrum_ids
    groups = canonical.set_index("spectrum_id").loc[lib_ids, "group"].fillna("").to_numpy()
    n_rows, n_cols = scene.shape
    top_rows = np.full((n_rows, n_cols, k), NO_DATA, dtype=np.int32)
    scores = np.full((n_rows, n_cols, k), np.nan, dtype=np.float32)
    margin = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    p_correct = np.full((n_rows, n_cols), np.nan, dtype=np.float32) if confidence else None

    t0 = time.perf_counter()
    starts = range(0, n_rows, block_rows)
    for r0 in tqdm(starts, desc="mapping", unit="block", disable=not progress):
        r1 = min(r0 + block_rows, n_rows)
        block = scene.block(slice(r0, r1), slice(0, n_cols)).reshape(-1, len(scene.centers_nm))
        has_data = np.isfinite(block).any(axis=1)
        if not has_data.any():
            continue
        spectra = block[has_data].astype(np.float64)
        ranked = retriever.rank(spectra)
        rows = ranked.rows[:, :k]
        sc = ranked.combined[:, :k]
        sc = np.where(np.isfinite(sc), sc, np.nan)

        flat_rows = np.full((block.shape[0], k), NO_DATA, dtype=np.int32)
        flat_scores = np.full((block.shape[0], k), np.nan, dtype=np.float32)
        flat_margin = np.full(block.shape[0], np.nan, dtype=np.float32)
        flat_rows[has_data] = rows
        flat_scores[has_data] = sc
        flat_margin[has_data] = _margin(ranked.rows, ranked.combined, groups)

        top_rows[r0:r1] = flat_rows.reshape(r1 - r0, n_cols, k)
        scores[r0:r1] = flat_scores.reshape(r1 - r0, n_cols, k)
        margin[r0:r1] = flat_margin.reshape(r1 - r0, n_cols)
        if confidence is not None:
            from specret.openset.calibrate import confidence_features

            _, feats = confidence_features(retriever, spectra, ranked)
            flat_p = np.full(block.shape[0], np.nan, dtype=np.float32)
            flat_p[has_data] = confidence.predict(feats)
            p_correct[r0:r1] = flat_p.reshape(r1 - r0, n_cols)

    return SceneMap(
        str(getattr(scene, "path", getattr(scene, "hdr_path", ""))),
        lib_ids,
        groups,
        top_rows,
        scores,
        margin,
        time.perf_counter() - t0,
        p_correct,
    )


def group_fractions(
    smap: SceneMap,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    min_p: float = 0.0,
) -> pd.DataFrame:
    """Share of mapped pixels per top-1 group, optionally inside a (W, S, E, N) box.

    Pixels whose calibrated P(correct) is below `min_p` count as "likely_unknown". Maps
    without a confidence layer fall back to the raw top-1 score.
    """
    sel = smap.valid.copy()
    if bbox is not None:
        if lat is None or lon is None:
            raise ValueError("bbox needs the scene's lat/lon arrays")
        west, south, east, north = bbox
        sel &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    groups = smap.top1_group()[sel].astype(str)
    conf = smap.p_correct if smap.p_correct is not None else smap.scores[..., 0]
    c = conf[sel]
    groups[c < min_p] = "likely_unknown"
    counts = pd.Series(groups).value_counts()
    return pd.DataFrame(
        {
            "pixels": counts,
            "fraction": counts / max(int(sel.sum()), 1),
            "median_p": [float(np.nanmedian(c[groups == g])) for g in counts.index],
        }
    )
