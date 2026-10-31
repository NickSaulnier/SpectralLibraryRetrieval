"""Sample-level train/val/test/open-set splits.

Library spectra of one physical sample (different instruments, grain-size splits) share a
`sample_id`, so the split is made over samples. Splitting over spectra would put near-copies
on both sides and inflate every metric.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from specret.sensors.sensor import CONFIG_DIR

SPLITS_DIR = CONFIG_DIR / "splits"


def load_split_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "splits.yaml").read_text())


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign each eligible spectrum to train/val/test/openset. `df` needs `group`."""
    eligible = df[df["chapter"].isin(cfg["chapters"]) & df["measurement"].isin(cfg["measurements"])]
    eligible = eligible[eligible["group"] != "unclassified"]
    rng = np.random.default_rng(cfg["seed"])
    fr = cfg["fractions"]
    openset = set(cfg["openset_groups"])

    # A sample can carry spectra from several groups in rare naming cases; use its majority group.
    sample_group = eligible.groupby("sample_id")["group"].agg(lambda s: s.value_counts().index[0])
    assignment: dict[str, str] = {}
    for group, samples in sample_group.groupby(sample_group):
        ids = np.array(sorted(samples.index))
        if group in openset:
            assignment.update(dict.fromkeys(ids, "openset"))
            continue
        if len(ids) < cfg["min_samples"]:
            assignment.update(dict.fromkeys(ids, "train"))
            continue
        rng.shuffle(ids)
        n_val = max(1, round(fr["val"] * len(ids)))
        n_test = max(1, round(fr["test"] * len(ids)))
        for i, sid in enumerate(ids):
            assignment[sid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    out = eligible[["spectrum_id", "sample_id", "group"]].copy()
    out["split"] = out["sample_id"].map(assignment)
    return out.sort_values("spectrum_id").reset_index(drop=True)


def split_hash(splits: pd.DataFrame) -> str:
    payload = splits[["spectrum_id", "split"]].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def write_splits(splits: pd.DataFrame, version: str, force: bool = False) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{version}.csv"
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: splits are frozen, bump the version instead")
    splits.to_csv(path, index=False, lineterminator="\n")
    (SPLITS_DIR / f"{version}.sha256").write_text(split_hash(splits) + "\n")
    return path


def load_splits(version: str | None = None) -> pd.DataFrame:
    """Load a frozen split and verify it against its recorded hash."""
    version = version or load_split_config()["version"]
    splits = pd.read_csv(SPLITS_DIR / f"{version}.csv")
    expected = (SPLITS_DIR / f"{version}.sha256").read_text().strip()
    if split_hash(splits) != expected:
        raise ValueError(f"{version}: contents do not match the recorded hash")
    return splits
