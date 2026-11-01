"""Phase G2 benchmark: detection, identification, false plumes and confidence calibration.

Data (configs/gas_scenes.yaml):
* no-plume scenes (Mountain Pass, a carbonatite with calcite and dolomite, and Cuprite): their
  own candidates are false plumes; synthetic plumes injected into them (gas/inject.py) are the
  labelled positives;
* plume scenes: candidates are checked against the EMIT L2B CH4PLM/CO2PLM catalogue (recall).

The confidence model is a logistic regression on the identification features predicting "a
real plume of the named gas". Calibration and evaluation never share a scene: the model is
fitted on the injected draws of one no-plume scene and scored on the other, both ways
(cross-fitting). The deployed model is fitted on both and used for the catalogue scenes.

"Plain" means every matched-filter candidate; "filtered" means candidates for which the
identifier names a gas with confidence at or above plumes.yaml identify.confidence_threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from specret.gas.detect import prepare
from specret.gas.inject import inject, random_plumes, transmittance_table
from specret.gas.library import plume_config
from specret.gas.pipeline import analyze
from specret.io.emit_ghg import plume_paths, read_raster
from specret.io.emit_l1b import EmitL1B, find_l1b, open_emit_l1b
from specret.openset.calibrate import ConfidenceModel, auroc, ece

FEATURES = ("mf_z", "gas_z", "coherence", "vs_confusers", "consistency", "log_px")
GAS_SETS = [
    {"CH4": (150.0, 3000.0)},
    {"CO2": (1.5e4, 3e5)},
    {"CH4": (150.0, 3000.0), "CO2": (1.5e4, 3e5)},
]
OUT = Path("outputs/gas/g2")
MODEL_PATH = Path("configs/calibration/gas_emit.json")


def scenes(path: Path = Path("configs/gas_scenes.yaml")) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["scenes"]


def records_frame(records, scene_id: str, seed: int | None = None) -> pd.DataFrame:
    return pd.DataFrame([{"scene": scene_id, "seed": seed, **r.row()} for r in records])


def overlap_labels(records, labels: np.ndarray, min_overlap: float = 0.2) -> list[int]:
    """For each candidate, the label (>0) covering at least `min_overlap` of its pixels."""
    out = []
    for r in records:
        lab = labels[r.candidate.rows, r.candidate.cols]
        hit = lab[lab > 0]
        ok = hit.size > 0 and hit.size >= min_overlap * lab.size
        out.append(int(np.bincount(hit).argmax()) if ok else 0)
    return out


def catalogue_labels(scene: EmitL1B, gas: str) -> tuple[np.ndarray, int]:
    """Swath-geometry label image of the EMIT plume complexes (1..n), via the GLT."""
    gt = scene.geotransform
    labels = np.zeros(scene.shape, dtype=np.int32)
    paths = plume_paths(scene.path.parent, scene.scene_id, gas)
    for i, p in enumerate(paths, start=1):
        r = read_raster(p)
        c0 = int(round((r.transform.c - gt[0]) / gt[1]))
        r0 = int(round((r.transform.f - gt[3]) / gt[5]))
        h, w = r.data.shape
        sub_y, sub_x = np.nonzero(np.isfinite(r.data) & (r.data > 0))
        oy, ox = sub_y + r0, sub_x + c0
        ok = (oy >= 0) & (ox >= 0) & (oy < scene.glt_x.shape[0]) & (ox < scene.glt_x.shape[1])
        gx, gy = scene.glt_x[oy[ok], ox[ok]], scene.glt_y[oy[ok], ox[ok]]
        v = (gx > 0) & (gy > 0)
        labels[gy[v] - 1, gx[v] - 1] = i
    return labels, len(paths)


def run_real(scene_cfg: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    scene = open_emit_l1b(find_l1b(scene_cfg["id"]))
    _, recs = analyze(prepare(scene))
    df = records_frame(recs, scene.scene_id)
    n_cat = {}
    df["catalogue"] = 0
    for gas in scene_cfg.get("gases") or []:
        labels, n = catalogue_labels(scene, gas)
        n_cat[gas] = n
        hit = np.array(overlap_labels(recs, labels, min_overlap=0.0))
        sel = (df["detection_gas"] == gas).to_numpy() & (hit > 0)
        df.loc[sel, "catalogue"] = hit[sel]
    return df, n_cat


def run_injected(
    scene_cfg: dict, seeds: list[int], n_plumes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene = open_emit_l1b(find_l1b(scene_cfg["id"]))
    prep = prepare(scene)
    gases = sorted({g for s in GAS_SETS for g in s})
    tables = {
        g: transmittance_table(prep.pieces, prep.centers_nm, prep.fwhm_nm, prep.conditions, g)
        for g in gases
    }
    frames, truth = [], []
    for seed in seeds:
        plumes = random_plumes(scene.shape, n_plumes, GAS_SETS, ~prep.screen, seed)
        inj = inject(prep.radiance, plumes, tables)
        _, recs = analyze(prep, radiance=inj.radiance, seed=seed)
        df = records_frame(recs, scene.scene_id, seed)
        df["plume"] = overlap_labels(recs, inj.labels)
        df["true_gases"] = [
            ",".join(sorted(plumes[p - 1].gases)) if p > 0 else "" for p in df["plume"]
        ]
        frames.append(df)
        for i, p in enumerate(plumes, start=1):
            truth.append(
                {
                    "scene": scene.scene_id,
                    "seed": seed,
                    "plume": i,
                    "gases": ",".join(sorted(p.gases)),
                    "peak_CH4": p.gases.get("CH4", 0.0),
                    "peak_CO2": p.gases.get("CO2", 0.0),
                    "n_px": int((inj.labels == i).sum()),
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth)


def run_all(out: Path = OUT, seeds: int = 4, n_plumes: int = 25, log=print) -> None:
    """Run every scene (real) and every no-plume scene (injected); save to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    catalogue = {}
    for s in scenes():
        df, n_cat = run_real(s)
        df.to_parquet(out / f"real_{s['id']}.parquet")
        catalogue[s["id"]] = n_cat
        log(f"{s['id']}: {len(df)} candidates, catalogue {n_cat}")
        if s["role"] == "no_plume":
            inj, truth = run_injected(s, list(range(100, 100 + seeds)), n_plumes)
            inj.to_parquet(out / f"injected_{s['id']}.parquet")
            truth.to_parquet(out / f"truth_{s['id']}.parquet")
            log(f"{s['id']}: {len(truth)} injected plumes, {len(inj)} candidates")
    (out / "catalogue.json").write_text(json.dumps(catalogue, indent=1), encoding="utf-8")


def correct(df: pd.DataFrame) -> np.ndarray:
    """1 when a candidate is on an injected plume and names one of its gases."""
    return np.array(
        [
            int(p > 0 and isinstance(g, str) and g in t.split(","))
            for p, g, t in zip(df["plume"], df["gas"], df["true_gases"], strict=True)
        ]
    )


def keep(df: pd.DataFrame, p: np.ndarray, threshold: float | None = None) -> np.ndarray:
    if threshold is None:
        threshold = plume_config()["identify"]["confidence_threshold"]
    return (p >= threshold) & df["gas"].notna().to_numpy()


def evaluate(out: Path = OUT, features=FEATURES) -> dict:
    """Metrics from the saved runs; also fits and saves the deployed confidence model."""
    cfg = {s["id"]: s for s in scenes()}
    catalogue = json.loads((out / "catalogue.json").read_text(encoding="utf-8"))
    neg = [i for i, s in cfg.items() if s["role"] == "no_plume"]
    inj = pd.concat(
        [pd.read_parquet(out / f"injected_{i}.parquet") for i in neg], ignore_index=True
    )
    truth = pd.concat([pd.read_parquet(out / f"truth_{i}.parquet") for i in neg], ignore_index=True)
    inj["y"] = correct(inj)

    m = {"n_injected_candidates": len(inj), "positives": int(inj["y"].sum())}
    # Cross-fitted confidence: each no-plume scene scored by a model fitted on the other
    p_cf = np.full(len(inj), np.nan)
    models = {}
    for s in neg:
        test = (inj["scene"] == s).to_numpy()
        models[s] = ConfidenceModel.fit(
            inj[~test], inj.loc[~test, "y"].to_numpy(), features=features
        )
        p_cf[test] = models[s].predict(inj[test])
    m["ece"] = ece(p_cf, inj["y"].to_numpy())
    m["auroc"] = auroc(p_cf, inj["y"].to_numpy().astype(bool))
    kept = keep(inj, p_cf)

    # False plumes on the real no-plume scenes, plain vs filtered (out-of-scene model)
    for s in neg:
        real = pd.read_parquet(out / f"real_{s}.parquet")
        p = models[s].predict(real) if len(real) else np.zeros(0)
        m[f"false_plain:{s}"] = len(real)
        m[f"false_filtered:{s}"] = int(keep(real, p).sum()) if len(real) else 0
        m[f"verdicts:{s}"] = real["verdict"].value_counts().to_dict() if len(real) else {}

    # Detection and identification of injected plumes
    key = ["scene", "seed", "plume"]
    on = inj[inj["plume"] > 0]
    found = set(map(tuple, on[key].to_numpy()))
    found_f = set(map(tuple, on.loc[kept[inj["plume"].to_numpy() > 0], key].to_numpy()))
    tk = list(map(tuple, truth[key].to_numpy()))
    truth["found_plain"] = [k in found for k in tk]
    truth["found_filtered"] = [k in found_f for k in tk]
    m["truth"] = truth
    best = on.sort_values("mf_z", ascending=False).drop_duplicates(key)
    m["id_accuracy"] = float(best["y"].mean()) if len(best) else float("nan")
    m["id_confusion"] = pd.crosstab(best["true_gases"], best["verdict"]).to_dict()

    # Deployed model on the catalogue scenes
    model = ConfidenceModel.fit(inj, inj["y"].to_numpy(), features=features)
    model.save(MODEL_PATH)
    for s, c in cfg.items():
        if c["role"] != "plume":
            continue
        real = pd.read_parquet(out / f"real_{s}.parquet")
        p = model.predict(real) if len(real) else np.zeros(0)
        k = keep(real, p)
        for gas, n in catalogue[s].items():
            cat = (real["detection_gas"] == gas) & (real["catalogue"] > 0)
            plain = real.loc[cat, "catalogue"].nunique()
            filt = real.loc[cat & k & (real["gas"] == gas), "catalogue"].nunique()
            m[f"recall:{s}:{gas}"] = (int(plain), int(filt), int(n))
        m[f"candidates:{s}"] = (len(real), int(k.sum()))
    m["model"] = {"features": list(features), "coef": model.coef.tolist()}
    return m
