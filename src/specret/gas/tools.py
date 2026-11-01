"""Read-side helpers for gas results, shared by the MCP server and the viewer.

They read what `specret detect-plumes` (or the gas job) wrote to outputs/gas/<scene id>/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from specret.workspace import jsonable

SCENE_ID = re.compile(r"(\d{8}T\d{6}_\d{7}_\d{3})")


def _plumes(folder: Path) -> pd.DataFrame:
    path = folder / "plumes.csv"
    if not path.exists():
        raise ValueError(f"no plume results in {folder}; run detect_plumes first")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _summary_row(r: pd.Series) -> dict:
    keys = (
        "plume_id",
        "gas",
        "confidence",
        "source_lat",
        "source_lon",
        "n_px",
        "max_alpha_ppmm",
        "q_kg_h",
        "sigma_q_kg_h",
        "u10",
        "wind_source",
        "confusers_fitted",
        "carbonate_fraction",
    )
    return {k: r.get(k) for k in keys if k in r}


def scene_gas_summary(folder: Path) -> dict:
    df = _plumes(folder)
    out = {"scene_id": folder.name, "plumes": int(len(df))}
    if len(df):
        for g, sub in df.groupby("gas"):
            q = sub["q_kg_h"].dropna() if "q_kg_h" in sub else pd.Series(dtype=float)
            out[g] = {
                "plumes": int(len(sub)),
                "total_q_kg_h": float(q.sum()) if len(q) else None,
                "total_sigma_q_kg_h": float(np.sqrt((sub["sigma_q_kg_h"].dropna() ** 2).sum()))
                if len(q)
                else None,
                "largest_q_kg_h": float(q.max()) if len(q) else None,
            }
    cand = folder / "candidates.csv"
    if cand.exists():
        c = pd.read_csv(cand)
        out["candidates"] = int(len(c))
        out["candidate_verdicts"] = c["verdict"].value_counts().to_dict()
    return jsonable(out)


def list_plumes(folder: Path, gas: str | None = None, min_q: float | None = None) -> dict:
    df = _plumes(folder)
    if len(df) and gas:
        df = df[df["gas"] == gas]
    if len(df) and min_q is not None and "q_kg_h" in df:
        df = df[df["q_kg_h"] >= min_q]
    return jsonable(
        {"scene_id": folder.name, "plumes": [_summary_row(r) for _, r in df.iterrows()]}
    )


def fit_spectra(folder: Path, detection_gas: str, candidate_id: int) -> dict | None:
    """A candidate's stored fit (gas/outputs.write_spectra) in log-radiance units, or None
    for scenes detected before the spectra were stored."""
    path = folder / "plume_spectra.npz"
    key = f"{detection_gas}_{int(candidate_id)}/"
    if not path.exists():
        return None
    with np.load(path) as z:
        if key + "obs" not in z:
            return None
        obs, base = z[key + "obs"], z[key + "baseline"]
        terms = dict(zip(z[key + "term_names"].tolist(), z[key + "terms"], strict=True))
        model = base + sum(terms.values(), np.zeros_like(obs))
        return {
            "units": "ln(radiance / column background), minus the shifted-mask null mean",
            "wavelengths_nm": z["centers_nm"].tolist(),
            "windows": z["windows"].tolist(),
            "observed": obs.tolist(),
            "null_sd": z[key + "null_sd"].tolist(),
            "baseline": base.tolist(),
            "terms": {t: v.tolist() for t, v in terms.items()},
            "model": model.tolist(),
            "remainder": (obs - model).tolist(),
            "signatures_per_ppmm": dict(
                zip(z["signature_names"].tolist(), z["signatures"].tolist(), strict=True)
            ),
        }


def explain_plume(folder: Path, plume_id: int, spectra: bool = True) -> dict:
    df = _plumes(folder)
    row = df[df["plume_id"] == plume_id]
    if row.empty:
        raise ValueError(f"no plume {plume_id} in {folder.name}")
    r = row.iloc[0]
    cand = pd.read_csv(folder / "candidates.csv")
    c = cand[
        (cand["candidate_id"] == r["candidate_id"]) & (cand["detection_gas"] == r["detection_gas"])
    ].iloc[0]
    features = ("mf_z", "gas_z", "coherence", "vs_confusers", "consistency", "log_px")
    terms = dict(t.split("=") for t in str(r["terms"]).split(";") if "=" in t)
    return jsonable(
        {
            "plume_id": plume_id,
            "gas": r["gas"],
            "confidence": r["confidence"],
            "verdict": c["verdict"],
            "fitted_terms": {k: float(v) for k, v in terms.items()},
            "evidence": {f: c[f] for f in features if f in c},
            "evidence_meaning": {
                "mf_z": "peak z-score in the matched-filter map",
                "gas_z": "z-score of the gas in the joint gas + confuser fit",
                "coherence": "spatial autocorrelation of the enhancement (plume-like > 0.4)",
                "vs_confusers": "log(1 + chi-square gained over the best confuser-only fit)",
                "consistency": "|z| of the gas amount's difference between 1.65 and 2.3 µm",
                "log_px": "log of the plume's size in pixels",
            },
            "carbonate_fraction": r.get("carbonate_fraction"),
            "rate": {
                k: r.get(k)
                for k in (
                    "ime_kg",
                    "length_m",
                    "u10",
                    "u_eff",
                    "wind_source",
                    "q_kg_h",
                    "sigma_q_kg_h",
                )
            },
            "relative_uncertainty": {
                k.removeprefix("rel_sigma_"): r[k] for k in r.index if k.startswith("rel_sigma_")
            },
            "cross_sectional_flux": {
                k: r.get(k) for k in ("q_csf_kg_h", "transects", "csf_iqr_rel") if k in r
            },
            "source": {k: r.get(k) for k in ("source_lat", "source_lon")},
            "wind": {k: r.get(k) for k in ("wind_u", "wind_v") if k in r},
            "spectra": fit_spectra(folder, r["detection_gas"], r["candidate_id"])
            if spectra
            else None,
        }
    )


def quantify_plume(folder: Path, plume_id: int, u=None, v=None, sd=None) -> dict:
    from specret.gas.library import plume_config
    from specret.gas.quantify import requantify
    from specret.io.wind import Wind

    df = _plumes(folder)
    row = df[df["plume_id"] == plume_id]
    if row.empty:
        raise ValueError(f"no plume {plume_id} in {folder.name}")
    wind = None
    if u is not None and v is not None:
        wind = Wind(float(u), float(v), "user")
        wind.sd = float(sd) if sd is not None else 0.5 * wind.speed
    coef = tuple(plume_config()["quantify"]["u_eff"])
    return jsonable({"plume_id": plume_id, **requantify(row.iloc[0].to_dict(), coef, wind)})


def gas_signature(gas: str = "CH4", amf: float = 2.2, h2o_gcm2: float = 1.5) -> dict:
    from specret.gas.library import build_window, lbl_ranges, plume_config
    from specret.gas.pipeline import LIBRARY_NPZ
    from specret.gas.signatures import Conditions, window_absorption
    from specret.sensors.library import SensorLibrary

    windows = plume_config()["matched_filter"]["windows"]["CH4"]
    sensor = SensorLibrary.load(LIBRARY_NPZ).sensor  # EMIT bands
    c, f = sensor.centers_nm, sensor.fwhm_nm
    pieces = [(r, build_window(*r)) for r in lbl_ranges(windows)]
    inside = np.any([(c >= lo) & (c <= hi) for (lo, hi), _ in pieces], axis=0)
    k = window_absorption(pieces, c[inside], f[inside], Conditions(amf, h2o_gcm2), gas)
    return jsonable(
        {
            "gas": gas,
            "units": "d ln(radiance) per ppm·m",
            "conditions": {"amf": amf, "h2o_gcm2": h2o_gcm2},
            "wavelength_nm": np.round(c[inside], 2).tolist(),
            "k": k.tolist(),
            "strongest_bands_nm": np.round(c[inside][np.argsort(-k)[:5]], 1).tolist(),
        }
    )


def gas_section(ws, scene_id: str) -> dict | None:
    """Gas summary for a scene id or any granule name containing it, if it was analyzed."""
    m = SCENE_ID.search(scene_id)
    if not m:
        return None
    folder = ws.gas_dir(m.group(1))
    return scene_gas_summary(folder) if (folder / "plumes.csv").exists() else None
