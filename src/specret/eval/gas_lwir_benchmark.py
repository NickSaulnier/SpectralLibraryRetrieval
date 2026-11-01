"""Phase G4 benchmark: LWIR identification on injected plumes in real HyTES radiance.

* Identification: single-gas plumes of CH4, N2O, NH3, SO2 and C2H4 whose peak enhancement is
  drawn so that the plume SNR (peak / the gas map's background sigma) spans about 2-40.
  Accuracy is measured on detected plumes with SNR >= 5.
* Open set: plumes of a gas that is left out of the library. A correct answer is "not a
  library gas" (no named gas, or low confidence), never another gas with high confidence.
* Thermal contrast: NH3 plumes at plume temperatures from 10 K below to 10 K above the
  surface. Pixels within 1 K are masked as undetectable by construction.
Natural candidates in the same segments are labelled negative for calibration. Some may be
real emitters, so the false-alarm figures here are upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from specret.gas.inject import random_plumes
from specret.gas.lwir import (
    analyze_lwir,
    inject_lwir,
    lwir_enhancement,
    plume_transmittance_table,
    prepare_lwir,
)
from specret.io.hytes import open_hytes
from specret.openset.calibrate import ConfidenceModel, auroc, ece

FEATURES = ("mf_z", "gas_z", "coherence", "vs_confusers", "consistency", "log_px")
GASES = ["CH4", "N2O", "NH3", "SO2", "C2H4"]
SEGMENTS = {"SaltonSea": slice(2000, 3875), "KernRiver": slice(6000, 8000)}


def background_sigma(prep, gases) -> dict[str, float]:
    out = {}
    for g in gases:
        a = lwir_enhancement(prep, g).alpha
        a = a[np.isfinite(a)]
        out[g] = float(1.4826 * np.median(np.abs(a - np.median(a))))
    return out


def label(recs, inj) -> pd.DataFrame:
    rows = []
    for r in recs:
        lab = inj.labels[r.candidate.rows, r.candidate.cols]
        hit = lab[lab > 0]
        pid = int(np.bincount(hit).argmax()) if hit.size >= 0.2 * lab.size and hit.size else 0
        true = ",".join(sorted(inj.plumes[pid - 1].gases)) if pid else ""
        rows.append({**r.row(), "plume": pid, "true_gases": true})
    df = pd.DataFrame(rows)
    if len(df):
        df["y"] = [
            int(p > 0 and isinstance(g, str) and g in t.split(","))
            for p, g, t in zip(df["plume"], df["gas"], df["true_gases"], strict=True)
        ]
    return df


def run_draw(prep, tables, plumes, library=None, t_plume=None, seed=0):
    inj = inject_lwir(prep, plumes, tables, t_plume)
    if t_plume is not None:
        prep.t_plume, saved = t_plume, prep.t_plume
    try:
        _, recs = analyze_lwir(prep, inj.radiance, library=library, seed=seed)
    finally:
        if t_plume is not None:
            prep.t_plume = saved
    return label(recs, inj), inj


def identification(flight: str, seeds, n_plumes=25, log=print):
    f = open_hytes(flight)
    prep = prepare_lwir(f, SEGMENTS[flight])
    sig = background_sigma(prep, GASES)
    tables = {g: plume_transmittance_table(prep, g) for g in GASES}
    frames, truth = [], []
    for seed in seeds:
        sets = [{g: (2 * sig[g], 40 * sig[g])} for g in GASES]
        plumes = random_plumes(prep.radiance.shape[:2], n_plumes, sets, ~prep.screen, seed)
        df, _ = run_draw(prep, tables, plumes, seed=seed)
        df["flight"], df["seed"], df["test"] = flight, seed, "identification"
        frames.append(df)
        for i, p in enumerate(plumes, start=1):
            ((g, peak),) = p.gases.items()
            truth.append(
                {
                    "flight": flight,
                    "seed": seed,
                    "plume": i,
                    "gas": g,
                    "peak": peak,
                    "snr": peak / sig[g],
                }
            )
        log(f"{flight} seed {seed}: {len(df)} candidates")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth), sig


def open_set(flight: str, held_out: list[str], seed: int, n_plumes=20, log=print):
    f = open_hytes(flight)
    prep = prepare_lwir(f, SEGMENTS[flight])
    sig = background_sigma(prep, held_out)
    tables = {g: plume_transmittance_table(prep, g) for g in held_out}
    frames = []
    for g in held_out:
        sets = [{g: (8 * sig[g], 40 * sig[g])}]
        plumes = random_plumes(prep.radiance.shape[:2], n_plumes, sets, ~prep.screen, seed)
        library = [x for x in GASES if x != g]
        df, _ = run_draw(prep, tables, plumes, library=library, seed=seed)
        df["flight"], df["held_out"], df["test"] = flight, g, "open_set"
        frames.append(df)
        log(f"{flight} open set without {g}: {len(df)} candidates")
    return pd.concat(frames, ignore_index=True)


def contrast_sweep(flight: str, deltas, seed: int, n_plumes=20, log=print):
    f = open_hytes(flight)
    prep = prepare_lwir(f, SEGMENTS[flight])
    t_surface = float(np.nanmedian(prep.surface_bt()))
    sig = background_sigma(prep, ["NH3"])["NH3"]
    table = {"NH3": plume_transmittance_table(prep, "NH3")}
    rows = []
    for d in deltas:
        plumes = random_plumes(
            prep.radiance.shape[:2], n_plumes, [{"NH3": (15 * sig, 15 * sig)}], ~prep.screen, seed
        )
        df, inj = run_draw(prep, table, plumes, t_plume=t_surface + d, seed=seed)
        on = df[df["plume"] > 0] if len(df) else df
        found = on["plume"].nunique() if len(on) else 0
        named = on[on["gas"] == "NH3"]["plume"].nunique() if len(on) else 0
        rows.append(
            {
                "flight": flight,
                "delta_t": d,
                "plumes": len(plumes),
                "detected": found,
                "named_NH3": named,
            }
        )
        log(f"{flight} dT {d:+}: detected {found}/{len(plumes)}, named NH3 {named}")
    return pd.DataFrame(rows)


def cross_fit(df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    p = np.full(len(df), np.nan)
    for fl in df["flight"].unique():
        test = (df["flight"] == fl).to_numpy()
        m = ConfidenceModel.fit(df[~test], df.loc[~test, "y"].to_numpy(), features=features)
        p[test] = m.predict(df[test])
    return p


def summarize(ident: pd.DataFrame, truth: pd.DataFrame, opens: pd.DataFrame, threshold: float):
    out = {}
    p = cross_fit(ident)
    out["ece"] = ece(p, ident["y"].to_numpy())
    out["auroc"] = auroc(p, ident["y"].to_numpy().astype(bool))
    ident = ident.assign(p=p)
    on = ident[ident["plume"] > 0]
    best = on.sort_values("mf_z", ascending=False).drop_duplicates(["flight", "seed", "plume"])
    t = truth.merge(
        best[["flight", "seed", "plume", "gas", "y", "p"]].rename(columns={"gas": "named"}),
        on=["flight", "seed", "plume"],
        how="left",
    )
    t["detected"] = t["y"].notna()
    out["truth"] = t
    hi = t[t["snr"] >= 5]
    out["detected_snr5"] = float(hi["detected"].mean())
    out["id_accuracy_snr5"] = float(hi.loc[hi["detected"], "y"].mean())
    out["id_by_gas_snr5"] = hi[hi["detected"]].groupby("gas")["y"].mean().round(3).to_dict()
    out["confusion_snr5"] = pd.crosstab(
        hi.loc[hi["detected"], "gas"], hi.loc[hi["detected"], "named"].fillna("none")
    ).to_dict()
    # Natural candidates (not on injected plumes) reported at the threshold, per draw
    nat = ident[ident["plume"] == 0]
    out["natural_reported_per_draw"] = float(
        ((nat["p"] >= threshold) & nat["gas"].notna())
        .groupby([nat["flight"], nat["seed"]])
        .sum()
        .mean()
    )
    # Open set: plumes of held-out gases, scored by the model fitted on all identification draws
    model = ConfidenceModel.fit(ident, ident["y"].to_numpy(), features=FEATURES)
    o = opens[opens["plume"] > 0].copy()
    o["p"] = model.predict(o)
    ob = o.sort_values("mf_z", ascending=False).drop_duplicates(["flight", "held_out", "plume"])
    out["open_set"] = {
        g: {
            "plumes_detected": int(len(sub)),
            "named_other_gas_confident": int(((sub["p"] >= threshold) & sub["gas"].notna()).sum()),
            "named_other_gas_any": int(sub["gas"].notna().sum()),
            "named_as": sub["gas"].fillna("none").value_counts().to_dict(),
        }
        for g, sub in ob.groupby("held_out")
    }
    return out, model
