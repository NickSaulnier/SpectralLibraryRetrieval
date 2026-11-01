"""HITRAN line lists for the gases in configs/gases.yaml, fetched with HAPI (no login needed).

Each gas and range is cached as HAPI's own table files, data/raw/hitran/<GAS>_<range>.data
and .header, with every isotopologue of the molecule included. HITRAN asks users to cite
the HITRAN database paper (Gordon et al., 2022, JQSRT 277, 107949) in published work.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from specret.sensors.sensor import CONFIG_DIR

DEFAULT_DIR = Path("data/raw/hitran")


def load_gases(config_dir: Path = CONFIG_DIR) -> dict:
    return yaml.safe_load((config_dir / "gases.yaml").read_text())


def isotopologue_ids(molecule: int) -> list[int]:
    """HITRAN global isotopologue ids for every isotopologue of `molecule`."""
    import hapi

    return sorted(v[0] for (m, _), v in hapi.ISO.items() if m == molecule)


def fetch_lines(
    gases: list[str] | None = None,
    ranges: list[str] | None = None,
    out_dir: Path = DEFAULT_DIR,
) -> list[Path]:
    """Download line lists into `out_dir`; tables already on disk are skipped.

    A gas is fetched for a range only if it has diagnostic windows there (configs/gases.yaml).
    HITRAN has no lines for some gas/range pairs (C2H6 at 1.5-2.5 µm is cross-section only);
    those are reported and skipped.
    """
    import hapi

    cfg = load_gases()
    names = gases or [g for g, v in cfg["gases"].items() if v["hitran_id"] is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    hapi.db_begin(str(out_dir))
    written = []
    for name in names:
        g = cfg["gases"][name]
        for rng in ranges or list(cfg["hitran_ranges_cm1"]):
            if not g[rng]:
                continue
            table = f"{name}_{rng}"
            path = out_dir / f"{table}.data"
            if not path.exists():
                lo, hi = cfg["hitran_ranges_cm1"][rng]
                try:
                    hapi.fetch_by_ids(table, isotopologue_ids(g["hitran_id"]), lo, hi)
                except Exception as e:  # HAPI raises a bare Exception when there are no lines
                    print(f"skipped {table}: no HITRAN lines in {lo}-{hi} cm-1 ({e})")
                    continue
            written.append(path)
    return written
