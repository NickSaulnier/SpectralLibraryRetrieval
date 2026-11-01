"""Emission rates from plumes: IME, effective wind and an uncertainty budget (PLAN_GAS.md 1.3).

    IME = sum over plume pixels of alpha * m1 * A_px                [kg]
    Q   = IME * U_eff / L,   L = sqrt(plume area),  U_eff = a * U10 + b      [kg/s]

m1 is the gas mass per m^2 in one ppm·m, from the same surface air density that defines our
ppm·m (gas/signatures.molecules_per_ppmm). U_eff is the effective wind of the IME method; its
coefficients are calibrated on large-eddy simulations per sensor resolution (configs/plumes.yaml;
EMIT: 0.45 U10 + 0.67, Ferrari et al. 2025).

Uncertainty, summed in quadrature as relative errors:
* background: the spread of the same mask's IME when shifted to plume-free parts of the scene
  (correlated noise and surface texture, measured rather than assumed);
* wind: the wind's standard deviation (io/wind.py; at least half the speed by default);
* mask: half the change in Q when the mask's lower threshold moves by +/- 0.5 sigma;
* retrieval: the spread of IME against the true mass inside the mask on injected plumes
  (saturation, the filter partly absorbing the plume), a fixed relative term from
  configs/plumes.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from specret.gas.plumes import Candidate
from specret.gas.signatures import molecules_per_ppmm
from specret.io.hitran import load_gases

N_A = 6.02214076e23
EARTH_RADIUS_M = 6371008.8


@dataclass
class Quantification:
    gas: str
    n_px: int
    pixel_area_m2: float
    ime_kg: float
    length_m: float
    u10: float
    u_eff: float
    wind_source: str
    q_kg_h: float
    sigma_q_kg_h: float
    sigma_parts: dict  # relative uncertainty by source
    source_row: int
    source_col: int
    source_lat: float
    source_lon: float
    wind_u: float = float("nan")  # m/s toward east, at the source
    wind_v: float = float("nan")  # m/s toward north

    def as_dict(self) -> dict:
        return asdict(self)


def kg_per_m2_per_ppmm(gas: str, surface_km: float) -> float:
    molar_mass = load_gases()["gases"][gas]["molar_mass"]  # g/mol
    return molecules_per_ppmm(surface_km) * 1e4 / N_A * molar_mass / 1e3


def local_metres(lat: np.ndarray, lon: np.ndarray, row: int, col: int):
    """Ground vectors (east, north) in metres of one step down-track and one step cross-track."""
    r0, r1 = max(row - 1, 0), min(row + 1, lat.shape[0] - 1)
    c0, c1 = max(col - 1, 0), min(col + 1, lat.shape[1] - 1)
    k = np.radians(lat[row, col])

    def vec(dlat, dlon, steps):
        return np.array([np.radians(dlon) * np.cos(k), np.radians(dlat)]) * EARTH_RADIUS_M / steps

    down = vec(lat[r1, col] - lat[r0, col], lon[r1, col] - lon[r0, col], r1 - r0)
    cross = vec(lat[row, c1] - lat[row, c0], lon[row, c1] - lon[row, c0], c1 - c0)
    return down, cross


def pixel_area(lat, lon, row, col) -> float:
    down, cross = local_metres(lat, lon, row, col)
    return float(abs(down[0] * cross[1] - down[1] * cross[0]))


def source_pixel(alpha, cand: Candidate, lat, lon, wind_u: float, wind_v: float) -> tuple[int, int]:
    """The most upwind of the plume's strong pixels (at least half the plume's high value)."""
    a = np.nan_to_num(alpha[cand.rows, cand.cols], nan=-np.inf)
    top = a >= 0.5 * np.percentile(a[np.isfinite(a)], 95)
    if not np.hypot(wind_u, wind_v) > 0 or not top.any():
        i = int(np.argmax(a))
        return int(cand.rows[i]), int(cand.cols[i])
    k = np.cos(np.radians(lat[cand.rows, cand.cols]))
    east = np.radians(lon[cand.rows, cand.cols]) * k * EARTH_RADIUS_M
    north = np.radians(lat[cand.rows, cand.cols]) * EARTH_RADIUS_M
    downwind = (east * wind_u + north * wind_v) / np.hypot(wind_u, wind_v)
    i = np.flatnonzero(top)[np.argmin(downwind[top])]
    return int(cand.rows[i]), int(cand.cols[i])


def regrow(z: np.ndarray, cand: Candidate, z_low: float) -> tuple[np.ndarray, np.ndarray]:
    """The candidate's region re-grown at another lower threshold (connected to its peak)."""
    r0, r1 = max(cand.rows.min() - 30, 0), min(cand.rows.max() + 31, z.shape[0])
    c0, c1 = max(cand.cols.min() - 30, 0), min(cand.cols.max() + 31, z.shape[1])
    sub = np.nan_to_num(z[r0:r1, c0:c1], nan=-np.inf)
    lab, _ = ndimage.label(sub > z_low, structure=np.ones((3, 3)))
    i = int(np.argmax(sub[cand.rows - r0, cand.cols - c0]))
    keep = lab == lab[cand.rows[i] - r0, cand.cols[i] - c0]
    rr, cc = np.nonzero(keep)
    return rr + r0, cc + c0


def background_sd(alpha, cand: Candidate, occupied, n: int = 200, seed: int = 0) -> float:
    """Standard deviation of sum(alpha) over the candidate's mask shifted to plume-free places."""
    rng = np.random.default_rng(seed + cand.id)
    rows, cols = alpha.shape
    sums = []
    for _ in range(20 * n):
        if len(sums) >= n:
            break
        dr, dc = rng.integers(-250, 251, size=2)
        r, c = cand.rows + dr, cand.cols + dc
        if r.min() < 0 or c.min() < 0 or r.max() >= rows or c.max() >= cols or occupied[r, c].any():
            continue
        v = alpha[r, c]
        if np.isfinite(v).mean() < 0.9:
            continue
        sums.append(np.nanmean(v) * v.size)
    return float(np.std(sums)) if len(sums) >= 20 else float("nan")


def rate(ime_kg: float, length_m: float, u10: float, coef: tuple[float, float]) -> float:
    u_eff = coef[0] * u10 + coef[1]
    return ime_kg * u_eff / length_m * 3600.0


def quantify(
    alpha: np.ndarray,
    z: np.ndarray,
    cand: Candidate,
    gas: str,
    lat: np.ndarray,
    lon: np.ndarray,
    surface_km: float,
    wind,
    occupied: np.ndarray,
    u_eff_coef: tuple[float, float],
    z_low: float,
    mask_step: float = 0.5,
    retrieval_rel_sd: float = 0.0,
) -> Quantification:
    """Emission rate of one plume (swath geometry). `wind` is an io.wind.Wind."""
    r, c = int(np.median(cand.rows)), int(np.median(cand.cols))
    area = pixel_area(lat, lon, r, c)
    m1 = kg_per_m2_per_ppmm(gas, surface_km)

    def ime_len(rr, cc):
        return float(np.nansum(alpha[rr, cc]) * m1 * area), float(np.sqrt(rr.size * area))

    ime, length = ime_len(cand.rows, cand.cols)
    q = rate(ime, length, wind.speed, u_eff_coef)
    # Mask sensitivity: re-grow at z_low +/- mask_step
    qs = []
    for zl in (z_low - mask_step, z_low + mask_step):
        rr, cc = regrow(z, cand, zl)
        if rr.size:
            qs.append(rate(*ime_len(rr, cc), wind.speed, u_eff_coef))
    rel_mask = (max(qs) - min(qs)) / 2 / abs(q) if len(qs) == 2 and q else np.nan
    bg = background_sd(alpha, cand, occupied) * m1 * area
    rel_bg = bg / abs(ime) if ime else np.nan
    u_eff = u_eff_coef[0] * wind.speed + u_eff_coef[1]
    rel_wind = u_eff_coef[0] * wind.sd / u_eff if np.isfinite(wind.sd) else np.nan
    parts = {
        "background": rel_bg,
        "retrieval": retrieval_rel_sd,
        "wind": rel_wind,
        "mask": rel_mask,
    }
    rel = float(np.sqrt(np.nansum(np.square(list(parts.values())))))
    sr, sc = source_pixel(alpha, cand, lat, lon, wind.u, wind.v)
    return Quantification(
        gas=gas,
        n_px=cand.n_px,
        pixel_area_m2=area,
        ime_kg=ime,
        length_m=length,
        u10=wind.speed,
        u_eff=u_eff,
        wind_source=wind.source,
        q_kg_h=q,
        sigma_q_kg_h=abs(q) * rel,
        sigma_parts={k: float(v) for k, v in parts.items()},
        source_row=sr,
        source_col=sc,
        source_lat=float(lat[sr, sc]),
        source_lon=float(lon[sr, sc]),
        wind_u=float(wind.u),
        wind_v=float(wind.v),
    )


def requantify(row: dict, u_eff_coef: tuple[float, float], wind=None) -> dict:
    """A plume's rate from its plumes.csv row, with a new wind (io.wind.Wind) or the stored one."""
    a, b = u_eff_coef
    if wind is not None:
        u10, sd, src = wind.speed, wind.sd, wind.source
    else:
        u10, src = float(row["u10"]), row["wind_source"]
        sd = float(row["rel_sigma_wind"]) * float(row["u_eff"]) / a
    u_eff = a * u10 + b
    q = rate(float(row["ime_kg"]), float(row["length_m"]), u10, u_eff_coef)
    parts = {
        "background": float(row["rel_sigma_background"]),
        "retrieval": float(row.get("rel_sigma_retrieval", 0.0) or 0.0),
        "wind": a * sd / u_eff,
        "mask": float(row["rel_sigma_mask"]),
    }
    rel = float(np.sqrt(np.nansum(np.square(list(parts.values())))))
    return {
        "gas": row["gas"],
        "ime_kg": float(row["ime_kg"]),
        "length_m": float(row["length_m"]),
        "u10": u10,
        "u10_sd": sd,
        "wind_source": src,
        "u_eff": u_eff,
        "q_kg_h": q,
        "sigma_q_kg_h": abs(q) * rel,
        "relative_uncertainty": {k: float(v) for k, v in parts.items()},
    }


def cross_sectional_flux(
    alpha: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    gas: str,
    lat: np.ndarray,
    lon: np.ndarray,
    surface_km: float,
    source: tuple[int, int],
    wind_u: float,
    wind_v: float,
    u10: float,
    min_transects: int = 3,
) -> dict:
    """Second rate estimate (PLAN_GAS.md open question 3): mass flux through transects across
    the wind, Q = U10 * median over downwind transects of the integrated column across the
    plume. Pixels are binned by their downwind distance from the source, one pixel wide; each
    bin's mass divided by its width is the line density (kg/m) crossing that distance.

    Returns q_kg_h (NaN with fewer than `min_transects` downwind bins), the number of
    transects and their spread (interquartile range relative to the median)."""
    speed = float(np.hypot(wind_u, wind_v))
    if not speed > 0:
        return {"q_csf_kg_h": float("nan"), "transects": 0, "csf_iqr_rel": float("nan")}
    sr, sc = source
    area = pixel_area(lat, lon, sr, sc)
    m1 = kg_per_m2_per_ppmm(gas, surface_km)
    k = np.cos(np.radians(lat[sr, sc]))
    east = np.radians(lon[rows, cols] - lon[sr, sc]) * k * EARTH_RADIUS_M
    north = np.radians(lat[rows, cols] - lat[sr, sc]) * EARTH_RADIUS_M
    down = (east * wind_u + north * wind_v) / speed
    width = float(np.sqrt(area))
    b = np.floor(down / width).astype(int)
    mass = np.nan_to_num(alpha[rows, cols]) * m1 * area
    keep = b >= 0
    if not keep.any():
        return {"q_csf_kg_h": float("nan"), "transects": 0, "csf_iqr_rel": float("nan")}
    bins = np.bincount(b[keep], weights=mass[keep])
    counts = np.bincount(b[keep])
    line = bins[counts > 0] / width  # kg per m of downwind distance
    if line.size < min_transects:
        return {
            "q_csf_kg_h": float("nan"),
            "transects": int(line.size),
            "csf_iqr_rel": float("nan"),
        }
    med = float(np.median(line))
    q75, q25 = np.percentile(line, [75, 25])
    return {
        "q_csf_kg_h": med * u10 * 3600.0,
        "transects": int(line.size),
        "csf_iqr_rel": float((q75 - q25) / abs(med)) if med else float("nan"),
    }
