"""Project status (`specret doctor`) and one-command setup (`specret setup`).

`inspect()` looks at what the pipeline has produced so far and returns one Item per
requirement, each with the `specret` commands that would fix it. `setup()` runs those
commands in pipeline order, re-inspecting after each item, until everything required and
recommended is in place. Paths are relative to the working directory, as for every command.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from specret.io.download import CUPRITE_BBOX

USGS = Path("data/raw/usgs_splib07/extracted")
USGS_SETS = ("splib07a", "splib07b_cvAVIRISc2014")
USGS_HTML = Path("data/raw/usgs_splib07_html/pages")
CANONICAL = Path("data/canonical/library.parquet")
RESAMPLED = Path("data/resampled")
CALIBRATION = Path("configs/calibration")
SPLITS = Path("configs/splits/splits_v1.csv")
DERIVED = Path("data/derived")
FIXED_SENSORS = {"avirisc": "aviris_classic", "field_asd": "asd_fieldspec"}  # folder -> config
LEVELS = ("required", "recommended", "optional")


@dataclass
class Item:
    key: str
    level: str  # "required", "recommended" or "optional"
    name: str
    ok: bool
    detail: str
    steps: list[list[str]] = field(default_factory=list)  # `specret` argv that fix it
    hint: str | None = None  # what to do by hand, or a note on the steps


def _rel(p: Path) -> str:
    return p.as_posix()


def _library_centers() -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(RESAMPLED.glob("*/library.npz")):
        with np.load(p, allow_pickle=False) as z:
            out[p.parent.name] = z["centers_nm"]
    return out


def _scene_centers(info) -> np.ndarray:
    if info.kind == "emit":
        from specret.io.emit import read_band_parameters

        return read_band_parameters(info.path)[0]
    from specret.io.aviris import read_envi_band_parameters

    return read_envi_band_parameters(info.path)[0]


def _matches(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(a, b, atol=0.5)


def _scene_library(info, centers: np.ndarray, taken: set[str]) -> tuple[str, str]:
    """Sensor config and output folder for a scene whose bands no library matches."""
    if info.kind == "emit":
        cfg, base = "emit", "emit"
    elif centers.size > 300:
        cfg, base = "aviris_ng", "avirisng"
    else:
        cfg, base = "aviris_classic_scene", "avirisc_scene"
    return cfg, base if base not in taken else f"{base}_{info.scene_id}"


def _noise_step(scenes: dict) -> list[str] | None:
    """noise-curve over Cuprite when an RFL/RFLUNCERT pair covers it, else a whole scene."""
    import xarray as xr

    pairs = []
    for info in scenes.values():
        unc = info.path.with_name(info.path.name.replace("_RFL_", "_RFLUNCERT_"))
        if info.kind == "emit" and unc.is_file():
            pairs.append((info.path, unc))
    if not pairs:
        return None
    w, s, e, n = CUPRITE_BBOX
    for rfl, unc in pairs:
        with xr.open_dataset(rfl, group="location") as loc:
            lat, lon = loc["lat"].values, loc["lon"].values
        if ((lon > w) & (lon < e) & (lat > s) & (lat < n)).sum() >= 1000:
            return ["noise-curve", "--rfl", _rel(rfl), "--uncert", _rel(unc)]
    rfl, unc = pairs[0]
    with xr.open_dataset(rfl, group="location") as loc:
        lat, lon = loc["lat"].values, loc["lon"].values
    bbox = [np.nanmin(lon), np.nanmin(lat), np.nanmax(lon), np.nanmax(lat)]
    return ["noise-curve", "--rfl", _rel(rfl), "--uncert", _rel(unc), "--bbox"] + [
        f"{v:.4f}" for v in bbox
    ]


def inspect() -> list[Item]:
    """What is in place, what is missing, and the commands that would fix it."""
    if not Path("configs/sensors").is_dir():
        return [
            Item(
                "project",
                "required",
                "Project folder",
                False,
                f"{Path.cwd()} is not the project folder",
                hint="run specret from the folder that holds pyproject.toml",
            )
        ]
    from specret.workspace import Workspace

    items = []
    missing = [s for s in USGS_SETS if not (USGS / f"ASCIIdata_{s}").is_dir()]
    items.append(
        Item(
            "usgs",
            "required",
            "USGS spectral library",
            not missing,
            f"missing {', '.join(missing)}" if missing else ", ".join(USGS_SETS),
            [["download", "usgs"]],
        )
    )

    if CANONICAL.is_file():
        import pyarrow.parquet as pq

        detail = f"{pq.read_metadata(CANONICAL).num_rows:,} spectra"
    else:
        detail = f"no {_rel(CANONICAL)}"
    ok = CANONICAL.is_file()
    items.append(Item("canonical", "required", "Canonical library", ok, detail, [["ingest"]]))

    libs = _library_centers()
    fixed_missing = [d for d in FIXED_SENSORS if d not in libs]
    items.append(
        Item(
            "sensors",
            "required",
            "Sensor libraries",
            not fixed_missing,
            f"have {', '.join(libs) or 'none'}"
            + (f"; missing {', '.join(fixed_missing)}" if fixed_missing else ""),
            [["resample", "--sensor", FIXED_SENSORS[d]] for d in fixed_missing],
        )
    )

    ws = Workspace()
    scenes = ws.scenes()
    steps, planned, unmatched = [], [], []
    taken = set(libs)
    for info in scenes.values():
        centers = _scene_centers(info)
        if any(_matches(centers, c) for c in libs.values()):
            continue
        unmatched.append(info.scene_id)
        if any(_matches(centers, c) for c in planned):
            continue  # another scene with the same bands already resamples for it
        cfg, out = _scene_library(info, centers, taken)
        taken.add(out)
        planned.append(centers)
        steps.append(
            [
                "resample",
                "--sensor",
                cfg,
                "--band-file",
                _rel(info.path),
                "--out",
                _rel(RESAMPLED / out / "library.npz"),
            ]  # fmt: skip
        )
    kinds = sorted({i.kind.upper() for i in scenes.values()})
    if not scenes:
        detail = "no scenes in data/scenes (optional: specret download emit)"
    elif unmatched:
        detail = f"{len(unmatched)} of {len(scenes)} scenes have no library for their bands"
    else:
        detail = f"{len(scenes)} {'/'.join(kinds)} scenes, all with a library"
    items.append(Item("scene_libs", "required", "Scene libraries", not unmatched, detail, steps))

    items.append(
        Item(
            "splits",
            "recommended",
            "Benchmark split",
            SPLITS.is_file(),
            _rel(SPLITS) if SPLITS.is_file() else f"no {_rel(SPLITS)}",
            [["make-splits"]],
            hint="needed by calibrate and benchmark",
        )
    )

    has_emit = "emit" in libs or any(i.kind == "emit" for i in scenes.values())
    if has_emit:
        noise = DERIVED / "emit_noise.npy"
        ok = noise.is_file()
        step = None if ok else _noise_step(scenes)
        items.append(
            Item(
                "noise",
                "recommended",
                "EMIT noise curve",
                ok,
                _rel(noise) if ok else "none: calibration and unmixing assume flat noise",
                [step] if step else [],
                hint=None
                if ok or step
                else "needs an EMIT RFLUNCERT granule (specret download emit fetches it)",
            )
        )

    uncal = [d for d in libs if not (CALIBRATION / f"{d}.json").is_file()]
    items.append(
        Item(
            "calibration",
            "recommended",
            "Confidence calibration",
            not uncal,
            f"missing for {', '.join(uncal)}" if uncal else ", ".join(libs) or "no libraries yet",
            [
                [
                    "calibrate",
                    _rel(RESAMPLED / d / "library.npz"),
                    "--noise",
                    _rel(DERIVED / f"{d}_noise.npy"),
                    "--out",
                    _rel(CALIBRATION / f"{d}.json"),
                ]
                for d in uncal
            ],  # fmt: skip
            hint=None if not uncal else "without it, answers are reported as uncalibrated",
        )
    )

    unmapped = [s for s in scenes if not (ws.map_dir(s) / "scene_map.npz").is_file()]
    map_steps = []
    for sid in unmapped:
        centers = _scene_centers(scenes[sid])
        lib = next((d for d, c in libs.items() if _matches(centers, c)), None)
        if lib is not None:
            npz = _rel(RESAMPLED / lib / "library.npz")
            map_steps.append(["map-scene", npz, "--scene", _rel(scenes[sid].path)])
    items.append(
        Item(
            "maps",
            "optional",
            "Scene maps (viewer)",
            not unmapped,
            f"{len(scenes) - len(unmapped)} of {len(scenes)} scenes mapped",
            map_steps,
            hint="specret setup --map runs these, about 2 min per EMIT scene",
        )
    )
    items.extend(_gas(ws))
    items.extend(_environment())
    return items


def _gas(ws) -> list[Item]:
    """Gas plume prerequisites (PLAN_GAS.md): HITRAN tables, line-by-line cache, L1B scenes,
    calibrated confidence and analyzed scenes. All optional for mineral work."""
    from specret.gas.library import lbl_ranges, plume_config, window_path

    hitran = Path("data/raw/hitran")
    tables = sorted(p.stem for p in hitran.glob("*_vswir.data"))
    have = {"CH4_vswir", "CO2_vswir", "H2O_vswir"} <= set(tables)
    items = [
        Item(
            "hitran",
            "optional",
            "HITRAN line lists (gases)",
            have,
            f"{len(tables)} VSWIR tables" if tables else "not downloaded",
            [["download", "hitran"]],
        )
    ]
    cfg = plume_config()
    ranges = lbl_ranges(cfg["matched_filter"]["windows"]["CH4"])
    want = {*cfg["library"]["background_gases"], *cfg["library"]["plume_gases"]}
    cached = []
    for lo, hi in ranges:
        path = window_path(lo, hi)
        if path.exists():
            with np.load(path) as z:
                cached.append(want <= set(map(str, z["gases"])))
        else:
            cached.append(False)
    items.append(
        Item(
            "gas_library",
            "optional",
            "Gas line-by-line cache",
            all(cached),
            f"{sum(cached)} of {len(ranges)} ranges built",
            [] if all(cached) or not have else [["gas-library"]],
            hint="about 5 min per range, once",
        )
    )
    l1b = ws.gas_scenes()
    done = [s for s in l1b if (ws.gas_dir(s) / "plumes.csv").exists()]
    items.append(
        Item(
            "gas_scenes",
            "optional",
            "EMIT radiance scenes (gas)",
            bool(l1b),
            f"{len(l1b)} L1B scenes, {len(done)} analyzed" if l1b else "none",
            [] if l1b else [["download", "emit-gas"]],
            hint="specret detect-plumes <scene id> analyzes one (about 1-2 min)",
        )
    )
    model = CALIBRATION / "gas_emit.json"
    items.append(
        Item(
            "gas_confidence",
            "optional",
            "Gas plume confidence model",
            model.is_file(),
            "calibrated" if model.is_file() else "missing",
            [] if model.is_file() or not l1b else [["gas-benchmark"]],
            hint="specret gas-benchmark fits it from injected plumes (about 15 min)",
        )
    )
    return items


def _environment() -> list[Item]:
    """Optional extras: sample notes, Earthdata login, Qdrant, GPU."""
    items = [
        Item(
            "notes",
            "optional",
            "USGS sample notes",
            USGS_HTML.is_dir(),
            "loaded into explanations" if USGS_HTML.is_dir() else "not downloaded",
            hint="fetched by hand, see data/raw/MANIFEST.md; then re-run specret ingest",
        )
    ]

    home = Path.home()
    netrc = next((p for p in (home / ".netrc", home / "_netrc") if p.is_file()), None)
    has_login = bool(
        (netrc and "urs.earthdata.nasa.gov" in netrc.read_text(errors="ignore"))
        or (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))
    )
    items.append(
        Item(
            "earthdata",
            "optional",
            "Earthdata login",
            has_login,
            "saved" if has_login else "not saved",
            hint="uv run python scripts/earthdata_login.py (needed for specret download emit)",
        )
    )

    import requests

    try:
        up = requests.get("http://localhost:6333/readyz", timeout=1).ok
    except requests.RequestException:
        up = False
    items.append(
        Item(
            "qdrant",
            "optional",
            "Qdrant server",
            up,
            "running on localhost:6333" if up else "not running",
            hint="docker compose up -d (only specret index uses it)",
        )
    )

    import importlib.util

    gpu = None
    if importlib.util.find_spec("torch") is not None:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    items.append(
        Item(
            "gpu",
            "optional",
            "GPU search",
            gpu is not None,
            f"CUDA on {gpu}" if gpu else "CPU search",
            hint="uv sync --extra gpu for CUDA PyTorch; CPU works, just slower for maps",
        )
    )
    return items


def report(items: list[Item]) -> str:
    lines = []
    for level in LEVELS:
        rows = [it for it in items if it.level == level]
        if rows:
            lines.append(f"\n{level.capitalize()}")
        for it in rows:
            mark = "ok" if it.ok else ("!!" if level == "required" else "--")
            lines.append(f"  [{mark}] {it.name:<24} {it.detail}")
            if it.ok:
                continue
            shown = it.steps[:3] if level != "optional" else []
            for argv in shown:
                lines.append(f"{'':7}-> specret {' '.join(argv)}")
            if shown and len(it.steps) > 3:
                lines.append(f"{'':7}   ...and {len(it.steps) - 3} more")
            if it.hint:
                lines.append(f"{'':7}{it.hint}")
    if any(it.key == "project" for it in items):
        return "\n".join(lines)
    if any(not it.ok and it.steps and it.level != "optional" for it in items):
        lines.append("\nRun `specret setup` to do the steps above.")
    else:
        lines.append("\nEverything required is in place. Try `specret identify` or `specret view`.")
    return "\n".join(lines)


def setup(
    run: Callable[[list[str]], None], map_scenes: bool = False, log: Callable = print
) -> list[Item]:
    """Run missing steps in pipeline order until nothing runnable is left; returns the status."""
    attempted = set()
    while True:
        items = inspect()
        if any(it.key == "project" for it in items):
            return items
        wanted = {"maps"} if map_scenes else set()
        todo = [
            it
            for it in items
            if not it.ok and it.steps and (it.level != "optional" or it.key in wanted)
        ]
        if not todo:
            return items
        item = todo[0]
        signature = (item.key, tuple(tuple(a) for a in item.steps))
        if signature in attempted:
            raise RuntimeError(f"{item.name} is still incomplete after: {item.steps}")
        attempted.add(signature)
        log(f"\n=== {item.name}: {item.detail}")
        for argv in item.steps:
            log(f"--> specret {' '.join(argv)}")
            run(argv)
