"""Phase G3 checks on quantification (no controlled-release data is available yet).

1. Rates against EMIT L2B CH4PLM: the catalogue's emission-rate estimates (kg/h, with their
   own HRRR or ERA5 wind) for the plume complexes that have one. Our rate is computed with our
   wind and, to separate method from wind, with EMIT's wind speed.
2. IME recovery on injected plumes of known mass (gas/inject.py): measured IME against the
   true mass inside the detected mask and in the whole plume, and how often the error lies
   within the stated background + mask uncertainty. This checks the retrieval-to-mass chain;
   it does not check U_eff, which needs real releases or LES plumes.
3. Large, diffuse plumes: are they detected and named, or called "no signal" like the three
   weak Turkmenistan catalogue plumes in G2?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from specret.eval.gas_benchmark import catalogue_labels, overlap_labels
from specret.gas.detect import prepare
from specret.gas.inject import Plume, inject, random_plumes, transmittance_table
from specret.gas.pipeline import analyze, build_plumes
from specret.gas.quantify import kg_per_m2_per_ppmm, pixel_area, rate
from specret.io.emit_ghg import plume_paths
from specret.io.emit_l1b import find_l1b, open_emit_l1b
from specret.io.wind import Wind

OUT = Path("outputs/gas/g3")


def emit_rates(scene_ids: list[str], log=print) -> pd.DataFrame:
    """Our plumes matched to EMIT CH4PLM complexes that carry a rate."""
    from specret.gas.library import plume_config
    from specret.gas.outputs import write_scene

    coef = tuple(plume_config()["quantify"]["u_eff"])
    rows = []
    for sid in scene_ids:
        scene = open_emit_l1b(find_l1b(sid))
        prep = prepare(scene)
        enh, recs = analyze(prep)
        plumes = build_plumes(prep, enh, recs)
        write_scene(prep, enh, recs, plumes)
        labels, _ = catalogue_labels(scene, "CH4")
        metas = sorted(scene.path.parent.glob(f"EMIT_L2B_CH4PLMMETA_*_{sid.split('_')[0]}_*.json"))
        paths = plume_paths(scene.path.parent, sid, "CH4")
        hit = overlap_labels([p.record for p in plumes], labels, min_overlap=0.0)
        for i, (meta, _) in enumerate(zip(metas, paths, strict=True), start=1):
            props = json.loads(meta.read_text(encoding="utf-8"))["features"][0]["properties"]
            q_emit = props.get("Emissions Rate Estimate (kg/hr)")
            if q_emit in (None, "NA"):
                continue
            ours = [
                p
                for p, h in zip(plumes, hit, strict=True)
                if h == i and p.record.ident.gas == "CH4"
            ]
            row = {
                "scene": sid,
                "plume": props["Plume ID"],
                "q_emit": float(q_emit),
                "sigma_emit": float(props["Emissions Rate Estimate Uncertainty (kg/hr)"]),
                "wind_emit": float(props["Wind Speed (m/s)"]),
                "wind_source_emit": props["Wind Speed Source"],
                "fetch_emit_m": float(props["Fetch Length (m)"]),
                "matched": len(ours),
            }
            if ours:
                p = max(ours, key=lambda x: x.quant.ime_kg if x.quant else 0)
                q = p.quant
                row.update(
                    q_ours=q.q_kg_h,
                    sigma_ours=q.sigma_q_kg_h,
                    wind_ours=q.u10,
                    wind_source_ours=q.wind_source,
                    ime_kg=q.ime_kg,
                    length_m=q.length_m,
                    q_ours_emit_wind=rate(q.ime_kg, q.length_m, row["wind_emit"], coef),
                    confidence=p.confidence,
                )
            rows.append(row)
        log(f"{sid}: {len(plumes)} plumes")
    return pd.DataFrame(rows)


def ime_recovery(
    scene_ids: list[str], seeds: list[int], n_plumes: int = 20, diffuse: bool = False, log=print
) -> pd.DataFrame:
    """One row per injected plume: true mass, detection, verdict, measured IME and sigma."""
    rows = []
    wind = Wind(2.0, 0.0, "fixed", sd=1.0)  # IME does not depend on wind
    for sid in scene_ids:
        scene = open_emit_l1b(find_l1b(sid))
        prep = prepare(scene)
        tables = {
            "CH4": transmittance_table(
                prep.pieces, prep.centers_nm, prep.fwhm_nm, prep.conditions, "CH4"
            )
        }
        m1 = kg_per_m2_per_ppmm("CH4", prep.conditions.surface_km)
        for seed in seeds:
            sets = [{"CH4": (300.0, 800.0)}] if diffuse else [{"CH4": (300.0, 3000.0)}]
            plumes = random_plumes(scene.shape, n_plumes, sets, ~prep.screen, seed)
            if diffuse:  # long and wide, weak per pixel
                rng = np.random.default_rng(seed)
                plumes = [
                    Plume(
                        p.gases,
                        p.row,
                        p.col,
                        float(rng.uniform(15, 30)),
                        float(rng.uniform(3, 6)),
                        p.angle_rad,
                        p.seed,
                    )
                    for p in plumes
                ]
            inj = inject(prep.radiance, plumes, tables)
            enh, recs = analyze(prep, radiance=inj.radiance, seed=seed)
            found = build_plumes(prep, enh, recs, wind=wind)
            hit_rec = overlap_labels(recs, inj.labels)
            hit_pl = overlap_labels([p.record for p in found], inj.labels)
            for i, pl in enumerate(plumes, start=1):
                lab = inj.labels == i
                r0, c0 = pl.row, pl.col
                area = pixel_area(scene.lat, scene.lon, r0, c0)
                true_total = float(inj.alpha["CH4"][lab].sum() * m1 * area)
                cands = [r for r, h in zip(recs, hit_rec, strict=True) if h == i]
                mine = [p for p, h in zip(found, hit_pl, strict=True) if h == i]
                row = {
                    "scene": sid,
                    "seed": seed,
                    "plume": i,
                    "peak_ppmm": pl.gases["CH4"],
                    "n_px": int(lab.sum()),
                    "true_kg": true_total,
                    "detected": bool(cands),
                    "verdict": cands[0].ident.verdict if cands else "",
                    "reported": bool(mine),
                }
                if mine:
                    p = max(mine, key=lambda x: x.quant.ime_kg)
                    c = p.record.candidate
                    true_in_mask = float(
                        inj.alpha["CH4"][c.rows, c.cols].sum() * m1 * p.quant.pixel_area_m2
                    )
                    parts = p.quant.sigma_parts
                    rel = float(
                        np.sqrt(
                            np.nansum([parts[k] ** 2 for k in ("background", "mask", "retrieval")])
                        )
                    )
                    row.update(
                        ime_kg=p.quant.ime_kg,
                        true_in_mask_kg=true_in_mask,
                        sigma_ime_kg=abs(p.quant.ime_kg) * rel,
                        confidence=p.confidence,
                    )
                rows.append(row)
            log(f"{sid} seed {seed}: {len(found)} plumes reported")
    return pd.DataFrame(rows)
