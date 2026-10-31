"""Calibrate and evaluate the confidence model on the frozen split (PLAN.md, Phase 4).

Reference library: train samples only. Known queries: val (for fitting) and test (for
reporting). Unknown queries: the open-set groups, which are absent from the library. Each
query appears under several distortion recipes.

Evaluation is cross-validated over the open-set groups: each fold fits on val plus four
unknown groups and scores test plus the fifth, so every reported unknown group was unseen
during fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from specret.eval.benchmark import make_queries
from specret.openset.calibrate import (
    ConfidenceModel,
    accuracy_at_coverage,
    auroc,
    confidence_features,
    ece,
)
from specret.search.hybrid import HybridRetriever
from specret.sensors.library import SensorLibrary

RECIPES = ("clean", "noise", "mix", "atm-mild", "all")


def build_table(
    lib: SensorLibrary,
    canonical: pd.DataFrame,
    splits: pd.DataFrame,
    noise: np.ndarray,
    recipes: tuple[str, ...] = RECIPES,
) -> tuple[pd.DataFrame, HybridRetriever]:
    """One row per (query, recipe): features, true group, top-1 group, correctness."""
    parts, retriever = [], None
    for split in ("val", "test", "openset"):
        for recipe in recipes:
            q, truth, ref = make_queries(
                lib, canonical, splits, recipe, noise, split=split, ref_splits=("train",)
            )
            if retriever is None:
                retriever = HybridRetriever(ref, canonical)
            ranked, feats = confidence_features(retriever, q)
            groups = retriever.meta.loc[retriever.lib.spectrum_ids, "group"].to_numpy()
            top = ranked.rows[:, 0]
            feats["true_group"] = truth["group"].to_numpy()
            feats["top_group"] = np.where(top >= 0, groups[np.maximum(top, 0)], "")
            feats["split"], feats["recipe"] = split, recipe
            feats["known"] = split != "openset"
            feats["correct"] = feats["known"] & (feats["top_group"] == feats["true_group"])
            parts.append(feats)
    return pd.concat(parts, ignore_index=True), retriever


def cross_validate(table: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold P(correct) for every test-known and open-set row.

    Test-known rows are scored in every fold; their predictions are averaged.
    """
    table = table.reset_index(drop=True).assign(qid=lambda d: d.index)
    unknown_groups = sorted(table.loc[~table["known"], "true_group"].unique())
    preds = []
    for g in unknown_groups:
        fit_rows = (table["split"] == "val") | (
            (table["split"] == "openset") & (table["true_group"] != g)
        )
        model = ConfidenceModel.fit(table[fit_rows], table.loc[fit_rows, "correct"].to_numpy())
        eval_rows = (table["split"] == "test") | (
            (table["split"] == "openset") & (table["true_group"] == g)
        )
        part = table.loc[eval_rows, ["qid"]].copy()
        part["p_correct"] = model.predict(table[eval_rows])
        preds.append(part)
    p = pd.concat(preds).groupby("qid")["p_correct"].mean()
    out = table.loc[p.index].copy()
    out["p_correct"] = p.to_numpy()
    return out


def report(df: pd.DataFrame, baseline: str = "score") -> pd.DataFrame:
    """Metrics for the calibrated probability and for the raw top-1 score."""
    rows = []
    for name in ("p_correct", baseline):
        s = df[name].to_numpy(dtype=float)
        s = np.where(np.isfinite(s), s, 0.0)
        y = df["correct"].to_numpy()
        rows.append(
            {
                "signal": name,
                "auroc_known_vs_unknown": auroc(s, df["known"].to_numpy()),
                "auroc_correct_vs_not": auroc(s, y),
                "acc@100%": accuracy_at_coverage(s, y, 1.0),
                "acc@80%": accuracy_at_coverage(s, y, 0.8),
                "acc@60%": accuracy_at_coverage(s, y, 0.6),
                "ece": ece(s, y) if name == "p_correct" else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("signal")


def fit_final(table: pd.DataFrame, threshold: float = 0.5) -> ConfidenceModel:
    """Model for the tools: fitted on val and every open-set group (never on test)."""
    rows = table["split"].isin(["val", "openset"])
    model = ConfidenceModel.fit(table[rows], table.loc[rows, "correct"].to_numpy())
    model.threshold = threshold
    return model


def coverage_at_threshold(df: pd.DataFrame, threshold: float) -> dict:
    """Share of queries answered (p >= threshold) and the accuracy of those answers."""
    answered = df["p_correct"] >= threshold
    return {
        "threshold": threshold,
        "answered": float(answered.mean()),
        "accuracy_answered": float(df.loc[answered, "correct"].mean()) if answered.any() else 0.0,
        "unknown_flagged": float((~answered[~df["known"]]).mean()),
        "known_correct_kept": float(answered[df["correct"]].mean()),
    }
