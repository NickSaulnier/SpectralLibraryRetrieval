"""LWIR gas plumes in HyTES radiance (PLAN_GAS.md 1.2 and Phase G4).

A plume layer at temperature T_p with band transmittance tau_p sits over a surface whose
surface-leaving radiance is L_s. The sensor sees, with the atmosphere's band transmittance tau
and path radiance L_up (HyTES ISAC terms),

    x = tau [tau_p L_s + (1 - tau_p) B(T_p)] + L_up

so the change a thin plume makes, with L_s = (x0 - L_up) / tau, is

    dx ≈ (tau B(T_p) + L_up - x0) * sigma_b * alpha

where sigma_b is the gas's band-averaged absorption per ppm·m (line-by-line at the surface,
gas/library.build_lwir_window). The target therefore depends on the thermal contrast: a plume
cooler than the ground absorbs, a warmer one emits, and one at ground temperature is invisible.

Detection uses the column matched filter with t = sigma * (tau B(T_p) + L_up) - sigma * mu
(matched_filter offset). Each pixel's contrast differs from its column's, so the estimate is
rescaled by the ratio of the two contrasts, and pixels whose effective contrast
|T_p - BT(L_s)| is below `low_contrast_k` are masked as "cannot detect". T_p is the MERRA-2
2 m air temperature at the overpass unless given.

Flights are processed in along-track segments (`segment_rows`), each with its own column
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from specret.gas.identify import FitContext, identify, relative_residual
from specret.gas.library import build_lwir_window, plume_config
from specret.gas.matched_filter import MatchedFilterResult, column_matched_filter
from specret.gas.plumes import Candidate, find_candidates
from specret.gas.signatures import band_response, molecules_per_ppmm
from specret.io.hytes import HytesFlight, brightness_temperature, planck

CONTRAST_NM = 10800.0  # clean window band for the surface's brightness temperature
BASELINE_WINDOWS = [(7550, 8600), (8600, 10000), (10000, 11950)]


@dataclass
class LwirPrepared:
    flight: HytesFlight
    rows: slice
    idx: np.ndarray
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    radiance: np.ndarray  # (R, C, B)
    tau: np.ndarray  # (B,)
    l_up: np.ndarray  # (B,)
    t_plume: float
    sigma: dict[str, np.ndarray]  # gas -> (B,) band absorption per ppm·m
    plume_gases: dict[str, np.ndarray] = field(default_factory=dict)  # line-by-line, for injection
    screen: np.ndarray | None = None  # (R, C) unusable pixels
    lbl: object = None

    def surface_bt(self, radiance: np.ndarray | None = None) -> np.ndarray:
        """Brightness temperature of the surface-leaving radiance at CONTRAST_NM (R, C)."""
        x = self.radiance if radiance is None else radiance
        j = int(np.argmin(np.abs(self.centers_nm - CONTRAST_NM)))
        ls = (x[:, :, j] - self.l_up[j]) / self.tau[j]
        return brightness_temperature(self.centers_nm[j], ls).astype(np.float32)

    @property
    def emitted(self) -> np.ndarray:
        """tau B(T_p) + L_up per band: what a fully opaque plume would look like."""
        return self.tau * planck(self.centers_nm, self.t_plume) + self.l_up


def band_sigma(lbl, centers_nm, fwhm_nm, gas: str, surface_km: float = 0.0) -> np.ndarray:
    s = band_response(lbl.nu_cm1, centers_nm, fwhm_nm)
    return s @ lbl.sigma[gas][0].astype(np.float64) * molecules_per_ppmm(surface_km)


def plume_temperature(flight: HytesFlight, rows: slice) -> float:
    from specret.io.wind import air_temperature

    cfg = plume_config()["lwir"]
    geo = flight.geolocation(rows)
    if geo is not None:
        lat, lon = float(np.nanmean(geo[0])), float(np.nanmean(geo[1]))
    else:
        site = next((v for k, v in cfg.get("sites", {}).items() if k in flight.name), None)
        if site is None:
            raise ValueError(f"{flight.name} has no geolocation; give the plume temperature")
        lat, lon = site
    return air_temperature(lat, lon, flight.time)


def edge_screen(rad: np.ndarray, edge: int) -> np.ndarray:
    """Unusable pixels: any non-finite band, or within `edge` columns of the detector's ends
    (HyTES edge columns are noisy and produce spurious candidates)."""
    screen = ~np.isfinite(rad).all(axis=-1)
    if edge:
        screen[:, :edge] = True
        screen[:, -edge:] = True
    return screen


def prepare_lwir(
    flight: HytesFlight, rows: slice, t_plume: float | None = None, gases=None
) -> LwirPrepared:
    cfg = plume_config()["lwir"]
    lo, hi = cfg["window_nm"]
    idx = flight.band_indices(lo, hi)
    for a, b in cfg.get("exclude_nm", []):
        idx = idx[(flight.centers_nm[idx] < a) | (flight.centers_nm[idx] > b)]
    lbl = build_lwir_window(lo, hi)
    centers, fwhm = flight.centers_nm[idx], flight.fwhm_nm[idx]
    rad = flight.read(rows, idx)
    gases = gases or [*cfg["gases"], *cfg["nuisance"]]
    sigma = {g: band_sigma(lbl, centers, fwhm, g) for g in gases if g in lbl.sigma}
    return LwirPrepared(
        flight=flight,
        rows=rows,
        idx=idx,
        centers_nm=centers,
        fwhm_nm=fwhm,
        radiance=rad,
        tau=flight.tau[idx],
        l_up=flight.l_up[idx],
        t_plume=float(t_plume if t_plume is not None else plume_temperature(flight, rows)),
        sigma=sigma,
        screen=edge_screen(rad, cfg.get("edge_columns", 0)),
        lbl=lbl,
    )


@dataclass
class LwirEnhancement:
    gas: str
    alpha: np.ndarray  # (R, C) ppm·m, contrast-corrected; NaN where masked
    result: MatchedFilterResult
    delta_t: np.ndarray  # (R, C) T_plume - BT(surface), K
    low_contrast: np.ndarray  # (R, C) masked as undetectable


def lwir_enhancement(
    prep: LwirPrepared, gas: str, radiance: np.ndarray | None = None, device: str | None = None
) -> LwirEnhancement:
    cfg = plume_config()
    mf = {**cfg["matched_filter"], **cfg["lwir"].get("matched_filter", {})}
    x = prep.radiance if radiance is None else radiance
    s = prep.sigma[gas]
    e = prep.emitted
    res = column_matched_filter(
        x,
        s,
        valid=~prep.screen,
        offset=s * e,
        n_pass=mf["n_pass"],
        shrinkage=mf["shrinkage"],
        exclude_sigma=mf["exclude_sigma"],
        min_pixels_factor=mf["min_pixels_factor"],
        device=device,
    )
    # Rescale from the column's contrast to each pixel's: sum_b s^2 (e - x) / sum_b s^2 (e - mu)
    w = s**2
    c_pix = np.tensordot(e[None, None, :] - np.nan_to_num(x), w, axes=([2], [0]))
    c_col = (e[None, :] - res.mu) @ w  # (C,)
    ratio = c_pix / np.where(np.abs(c_col) > 0, c_col, np.nan)[None, :]
    delta_t = prep.t_plume - prep.surface_bt(x)
    low = np.abs(delta_t) < cfg["lwir"]["low_contrast_k"]
    alpha = res.alpha / np.where(np.abs(ratio) > 0.2, ratio, np.nan)
    alpha[low | prep.screen] = np.nan
    return LwirEnhancement(gas, alpha.astype(np.float32), res, delta_t, low)


def merge_candidates(found: dict[str, list[Candidate]], enh) -> dict[str, list[Candidate]]:
    """One candidate per connected region of all gases' candidates. Overlapping gases share
    pixels (a plume shows in several maps), so each region is identified once, under the gas
    whose map peaks highest there."""
    from scipy import ndimage

    shape = next(iter(enh.values())).alpha.shape
    union = np.zeros(shape, dtype=bool)
    for cands in found.values():
        for c in cands:
            union[c.rows, c.cols] = True
    lab, n = ndimage.label(union, structure=np.ones((3, 3)))
    best: dict[int, tuple[float, str]] = {}
    for g, cands in found.items():
        for c in cands:
            region = int(lab[c.rows[0], c.cols[0]])
            if region not in best or c.peak_z > best[region][0]:
                best[region] = (c.peak_z, g)
    out: dict[str, list[Candidate]] = {g: [] for g in found}
    objects = ndimage.find_objects(lab)
    for region, (peak, g) in sorted(best.items(), key=lambda t: -t[1][0]):
        sl = objects[region - 1]
        rr, cc = np.nonzero(lab[sl] == region)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        a = enh[g].alpha[rr, cc]
        out[g].append(
            Candidate(len(out[g]) + 1, rr, cc, peak, float(np.nanmean(a)), float(np.nanmax(a)))
        )
    return out


def gas_responses(prep: LwirPrepared, mu_bar: np.ndarray, gases) -> dict[str, np.ndarray]:
    """d ln(x) / d alpha per band for each gas, at a background radiance mu_bar (B,)."""
    return {g: prep.sigma[g] * (prep.emitted - mu_bar) / mu_bar for g in gases}


@dataclass
class LwirRecord:
    detection_gas: str
    candidate: Candidate
    ident: object
    delta_t: float

    def row(self) -> dict:
        c, i = self.candidate, self.ident
        return {
            "detection_gas": self.detection_gas,
            "candidate_id": c.id,
            "n_px": c.n_px,
            "row": c.centroid[0],
            "col": c.centroid[1],
            "mean_alpha": c.mean_alpha,
            "max_alpha": c.max_alpha,
            "delta_t": self.delta_t,
            "verdict": i.verdict,
            "gas": i.gas,
            "terms": ";".join(f"{k}={v:.3g}" for k, v in i.terms.items()),
            **i.features,
        }


def analyze_lwir(
    prep: LwirPrepared,
    radiance: np.ndarray | None = None,
    library: list[str] | None = None,
    seed: int = 0,
    device: str | None = None,
) -> tuple[dict[str, LwirEnhancement], list[LwirRecord]]:
    """Matched filter per searched gas, candidates, and identification against `library`
    (default: every searched gas; leaving one out tests open-set behaviour)."""
    cfg = plume_config()
    lcfg, icfg = cfg["lwir"], cfg["identify"]
    x = prep.radiance if radiance is None else radiance
    searched = [g for g in lcfg["gases"] if g in prep.sigma]
    library = [g for g in (library or searched) if g in prep.sigma]
    nuisance = [g for g in lcfg["nuisance"] if g in prep.sigma]
    enh = {g: lwir_enhancement(prep, g, x, device) for g in library}
    ccfg = {**cfg["candidates"], **lcfg.get("candidates", {})}
    found = merge_candidates({g: find_candidates(e.alpha, **ccfg)[0] for g, e in enh.items()}, enh)

    first = enh[library[0]]
    usable = ~prep.screen & ~first.low_contrast
    y = relative_residual(x, first.result.mu, usable)
    occupied = np.zeros(usable.shape, dtype=bool)
    for cands in found.values():
        for c in cands:
            occupied[c.rows, c.cols] = True
    from scipy import ndimage

    occupied = ndimage.binary_dilation(occupied, iterations=3)
    named = tuple(g for g in library if g in lcfg["named"])
    records = []
    for g, cands in found.items():
        for c in cands:
            mu_bar = first.result.mu[np.unique(c.cols)].mean(axis=0)
            resp = gas_responses(prep, mu_bar, [*library, *nuisance])
            ctx = FitContext(
                y=y,
                usable=usable,
                centers_nm=prep.centers_nm,
                windows=BASELINE_WINDOWS,
                gas_k={k: -v for k, v in resp.items()},  # identify() fits W @ (-k)
                confusers={},
                alpha=enh[g].alpha,
                occupied=occupied,
            )
            ident = identify(
                ctx,
                c,
                g,
                icfg["max_terms"],
                icfg["null_samples"],
                seed,
                named,
                icfg["flag_penalty"],
                tuple(nuisance),
                lcfg["retrieve_top_n"],
            )
            dt = float(np.nanmedian(enh[g].delta_t[c.rows, c.cols]))
            records.append(LwirRecord(g, c, ident, dt))
    return enh, records


def plume_transmittance_table(prep: LwirPrepared, gas: str) -> np.ndarray:
    """Band transmittance of a surface plume layer (len(ALPHA_GRID), B), line by line."""
    from specret.gas.inject import ALPHA_GRID

    s = band_response(prep.lbl.nu_cm1, prep.centers_nm, prep.fwhm_nm)
    tau1 = prep.lbl.sigma[gas][0].astype(np.float64) * molecules_per_ppmm(0.0)
    return np.stack([s @ np.exp(-tau1 * a) for a in ALPHA_GRID])


def inject_lwir(
    prep: LwirPrepared, plumes, tables: dict[str, np.ndarray], t_plume: float | None = None
):
    """Radiance with thermal plumes: x' = x + (1 - tau_p) (tau B(T_p) + L_up - x), where
    tau_p is the product of each gas's plume-layer transmittance (gas/inject.py shapes)."""
    from specret.gas.inject import ALPHA_GRID, Injection, plume_shape

    x = prep.radiance
    shape = x.shape[:2]
    alpha: dict[str, np.ndarray] = {}
    labels = np.zeros(shape, dtype=np.int32)
    for i, p in enumerate(plumes):
        sh = plume_shape(shape, p)
        labels[(sh > 0.05) & (labels == 0)] = i + 1
        for g, peak in p.gases.items():
            alpha.setdefault(g, np.zeros(shape))
            alpha[g] += peak * sh
    tau_p = np.ones(x.shape, dtype=np.float32)
    for g, a in alpha.items():
        r, c = np.nonzero(a > 0)
        v = a[r, c]
        j = np.clip(np.searchsorted(ALPHA_GRID, v) - 1, 0, ALPHA_GRID.size - 2)
        w = ((v - ALPHA_GRID[j]) / (ALPHA_GRID[j + 1] - ALPHA_GRID[j]))[:, None]
        tau_p[r, c] *= ((1 - w) * tables[g][j] + w * tables[g][j + 1]).astype(np.float32)
    tp = prep.t_plume if t_plume is None else t_plume
    emitted = (prep.tau * planck(prep.centers_nm, tp) + prep.l_up).astype(np.float32)
    out = x + (1 - tau_p) * (emitted[None, None, :] - x)
    return Injection(out.astype(np.float32), alpha, list(plumes), labels)


def analyze_flight(
    flight: HytesFlight,
    t_plume: float | None = None,
    rows: slice | None = None,
    log=print,
):
    """Every segment of a flight: (candidates DataFrame with flight rows and lat/lon,
    {gas: alpha (lines, samples) float32})."""
    import pandas as pd

    seg = plume_config()["lwir"]["segment_rows"]
    lines = flight.shape[0]
    start, stop = (0, lines) if rows is None else (rows.start or 0, rows.stop or lines)
    geo = flight.geolocation()
    frames, maps = [], {}
    for r0 in range(start, stop, seg):
        r1 = min(r0 + seg, stop)
        if r1 - r0 < 200:  # too few lines for column statistics: fold into the last segment
            continue
        prep = prepare_lwir(flight, slice(r0, r1), t_plume)
        enh, recs = analyze_lwir(prep)
        for g, e in enh.items():
            maps.setdefault(g, np.full(flight.shape, np.nan, dtype=np.float32))[r0:r1] = e.alpha
        df = pd.DataFrame([r.row() for r in recs])
        if len(df):
            df["row"] += r0
            df["segment"] = r0
            df["t_plume"] = prep.t_plume
            if geo is not None:
                ri, ci = df["row"].round().astype(int), df["col"].round().astype(int)
                df["lat"], df["lon"] = geo[0][ri, ci], geo[1][ri, ci]
            frames.append(df)
        log(f"{flight.name} rows {r0}-{r1}: {len(recs)} candidates")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, maps


HYTES_MODEL = "configs/calibration/gas_hytes.json"


def write_flight(flight: HytesFlight, df, maps: dict, out_dir="outputs/gas/hytes") -> Path:
    """candidates.csv, plumes.csv and plumes.geojson (source points), and the enhancement maps
    (enhancement.npz, ppm·m per gas in flight geometry) for one flight."""
    import json

    import pandas as pd

    from specret.openset.calibrate import ConfidenceModel
    from specret.workspace import jsonable

    folder = Path(out_dir) / flight.name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "enhancement.npz", **{g: a.astype(np.float16) for g, a in maps.items()}
    )
    df.to_csv(folder / "candidates.csv", index=False)
    model_path = Path(HYTES_MODEL)
    thr = plume_config()["identify"]["confidence_threshold"]
    plumes = pd.DataFrame()
    if len(df):
        p = (
            ConfidenceModel.load(model_path).predict(df)
            if model_path.exists()
            else np.full(len(df), np.nan)
        )
        df = df.assign(confidence=p)
        keep = df["gas"].notna() & ((df["confidence"] >= thr) | df["confidence"].isna())
        plumes = (
            df[keep].sort_values(["confidence", "gas_z"], ascending=False).reset_index(drop=True)
        )
        plumes.insert(0, "plume_id", np.arange(1, len(plumes) + 1))
    plumes.to_csv(folder / "plumes.csv", index=False)
    feats = []
    if "lat" in plumes:
        for _, r in plumes.iterrows():
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                    "properties": jsonable(r.to_dict()),
                }
            )
    (folder / "plumes.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "flight": flight.name, "features": feats}),
        encoding="utf-8",
    )
    return folder
