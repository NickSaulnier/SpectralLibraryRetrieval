"""Material name -> group -> class, from configs/taxonomy.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from specret.sensors.sensor import CONFIG_DIR

UNKNOWN = "unclassified"


@dataclass(frozen=True)
class Taxonomy:
    rules: tuple[tuple[re.Pattern, str, str], ...]  # (pattern, group, class) in file order
    chapter_groups: dict[str, tuple[str, str]]

    def classify(self, name: str | None, chapter: str | None = None) -> tuple[str, str]:
        """Return (group, class). Chapter-level groups win for A/O/V/L spectra."""
        if chapter in self.chapter_groups:
            return self.chapter_groups[chapter]
        text = (name or "").strip()
        for rx, group, cls in self.rules:
            if rx.match(text):
                return group, cls
        return UNKNOWN, UNKNOWN

    def group_of(self, name: str | None, chapter: str | None = None) -> str:
        return self.classify(name, chapter)[0]

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(g for _, g, _ in self.rules))


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    path = Path(path) if path else CONFIG_DIR / "taxonomy.yaml"
    cfg = yaml.safe_load(path.read_text())
    rules = []
    for group, spec in cfg["groups"].items():
        for pat in spec["patterns"]:
            rules.append((re.compile(pat, re.IGNORECASE), group, spec["class"]))
    chapters = {ch: (v["group"], v["class"]) for ch, v in cfg.get("chapter_groups", {}).items()}
    return Taxonomy(tuple(rules), chapters)


def apply_taxonomy(df: pd.DataFrame, taxonomy: Taxonomy | None = None) -> pd.DataFrame:
    """Fill the `group` and `material_class` columns of a canonical library frame."""
    tax = taxonomy or load_taxonomy()
    pairs = [tax.classify(n, c) for n, c in zip(df["material_name"], df["chapter"], strict=True)]
    out = df.copy()
    out["group"] = [g for g, _ in pairs]
    out["material_class"] = [c for _, c in pairs]
    return out
