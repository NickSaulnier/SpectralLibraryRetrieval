"""Synthetic retrieval benchmark on the frozen split (PLAN.md, Phase 2b).

Queries are test-split spectra on a sensor's bands, distorted by each recipe in
distort/simulator.py. The reference library holds only train and val samples, so a query's
own physical sample is never in the library. Distortions use their own fixed seed.

Synthetic numbers are a sanity check. Real-scene agreement decides what ships.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from specret.distort.simulator import (
    RECIPES,
    add_noise,
    atmospheric_residual,
    jitter_resample,
    linear_mix,
    photometric,
)
from specret.sensors.library import SensorLibrary

BENCHMARK_SEED = 7_2026  # distinct from any training seed
METHODS = ("sam", "sid", "swir2", "cr", "cr_euclid", "cr_swir2", "rrf", "hybrid")


def _subset(lib: SensorLibrary, ids: set[str]) -> SensorLibrary:
    keep = np.array([s in ids for s in lib.spectrum_ids])
    return SensorLibrary(lib.sensor, lib.spectrum_ids[keep], lib.matrix[keep])


def make_queries(
    lib: SensorLibrary,
    canonical: pd.DataFrame,
    splits: pd.DataFrame,
    recipe_name: str,
    noise_sigma: np.ndarray,
    split: str = "test",
    ref_splits: tuple[str, ...] = ("train", "val"),
) -> tuple[np.ndarray, pd.DataFrame, SensorLibrary]:
    """Distorted queries, their truth table, and the reference library.

    For tuning, use split="val" with ref_splits=("train",); report on split="test".
    """
    recipe = RECIPES[recipe_name]
    rng = np.random.default_rng(BENCHMARK_SEED + zlib.crc32(recipe_name.encode()))
    ref = _subset(lib, set(splits.loc[splits["split"].isin(ref_splits), "spectrum_id"]))
    test = _subset(lib, set(splits.loc[splits["split"] == split, "spectrum_id"]))
    test = SensorLibrary(
        test.sensor, test.spectrum_ids[test.full_coverage], test.matrix[test.full_coverage]
    )

    meta = canonical.set_index("spectrum_id")
    wl, good = lib.sensor.centers_nm, lib.sensor.good_mask
    ref_groups = meta.loc[ref.spectrum_ids, "group"].to_numpy()
    ref_full = ref.full_coverage

    queries = np.empty_like(test.matrix)
    for i, sid in enumerate(test.spectrum_ids):
        x = test.matrix[i].copy()
        row = meta.loc[sid]
        if recipe.jitter:
            j = jitter_resample(
                row["wavelength_nm"], row["reflectance"], row["native_fwhm_nm"], lib.sensor, rng
            )
            if np.isfinite(j[good]).all():
                x = j
        if recipe.photometric:
            x = photometric(x, wl, rng)
        if recipe.mix:
            pool = np.flatnonzero(ref_full & (ref_groups != row["group"]))
            x = linear_mix(x, ref.matrix[rng.choice(pool)], rng)
        if recipe.atmosphere:
            x = atmospheric_residual(x, wl, rng, strength=recipe.atmosphere)
        if recipe.noise:
            x = add_noise(x, noise_sigma, rng)
        queries[i] = x

    truth = meta.loc[test.spectrum_ids, ["sample_id", "group", "material_name"]].reset_index()
    return queries, truth, ref


def _metrics(matches: list[list[dict]], truth: pd.DataFrame, k: int) -> dict:
    g_top1, g_rk, m_top1, rr = [], [], [], []
    for m, (_, t) in zip(matches, truth.iterrows(), strict=True):
        groups = [x.get("group") for x in m[:k]]
        names = [str(x.get("material_name", "")).lower() for x in m[:1]]
        g_top1.append(bool(groups) and groups[0] == t["group"])
        g_rk.append(t["group"] in groups)
        m_top1.append(bool(names) and names[0] == str(t["material_name"]).lower())
        rr.append(1.0 / (groups.index(t["group"]) + 1) if t["group"] in groups else 0.0)
    return {
        "group_top1": np.mean(g_top1),
        f"group_recall@{k}": np.mean(g_rk),
        "group_mrr": np.mean(rr),
        "material_top1": np.mean(m_top1),
    }


def build_method(
    name: str,
    ref: SensorLibrary,
    canonical: pd.DataFrame,
    chapters: tuple[str, ...] | None = None,
    purity: tuple[str, ...] | None = None,
    encoder_path: str | None = None,
):
    """Retriever for a method name: a vector, `rrf`, `hybrid`, `sid`, `cr_euclid`,
    `learned`, `rrf+learned` or `hybrid+learned`."""
    if name == "learned":
        from specret.search.learned import DEFAULT_ENCODER, LearnedRetriever

        return LearnedRetriever(
            ref,
            canonical,
            encoder_path=encoder_path or DEFAULT_ENCODER,
            chapters=chapters,
            purity=purity,
        )
    if name in ("rrf+learned", "hybrid+learned"):
        from specret.search.hybrid import DEFAULT_VECTORS, HybridRetriever

        return HybridRetriever(
            ref,
            canonical,
            vectors=(*DEFAULT_VECTORS, "learned"),
            chapters=chapters,
            purity=purity,
            rerank=name.startswith("hybrid"),
            encoder_path=encoder_path,
        )
    if name in ("sid", "cr_euclid"):
        from specret.search.baselines import BruteForceRetriever

        return BruteForceRetriever(ref, canonical, metric=name, chapters=chapters, purity=purity)
    if name in ("rrf", "hybrid"):
        from specret.search.hybrid import HybridRetriever

        return HybridRetriever(
            ref, canonical, chapters=chapters, purity=purity, rerank=name == "hybrid"
        )
    from specret.search.retrieve import Retriever

    return Retriever(ref, canonical, vector=name, chapters=chapters, purity=purity)


def run_benchmark(
    lib: SensorLibrary,
    canonical: pd.DataFrame,
    splits: pd.DataFrame,
    recipes: list[str],
    methods: list[str],
    noise_sigma: np.ndarray,
    k: int = 5,
    purity: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows = []
    retrievers: dict[str, object] = {}
    for recipe in recipes:
        queries, truth, ref = make_queries(lib, canonical, splits, recipe, noise_sigma)
        for method in methods:
            if method not in retrievers:
                retrievers[method] = build_method(method, ref, canonical, purity=purity)
            matches = retrievers[method].query(queries, k=max(k, 10))
            rows.append(
                {"recipe": recipe, "method": method, "n": len(truth), **_metrics(matches, truth, k)}
            )
    return pd.DataFrame(rows)
