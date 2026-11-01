"""Synthetic plumes injected into real L1B radiance: the G2 benchmark data (PLAN_GAS.md).

A plume is an enhancement field alpha(row, col) in ppm·m for one or more gases. Its shape is a
wind-elongated Gaussian puff from a point source, broken up by a smoothed random field so it
looks turbulent rather than smooth. Radiance is multiplied by the exact band transmittance of
each gas (the line-by-line forward model of gas/signatures.band_transmittance, tabulated
over alpha), so the
injected signal is non-linear in alpha exactly as a real plume's is. Several gases in one plume
multiply; overlap between their lines within a band is ignored.

Injection draws and calibration draws use separate seeds (see eval/gas_benchmark.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from specret.gas.signatures import Conditions, _plume_terms

ALPHA_GRID = np.concatenate([[0.0], np.geomspace(1.0, 2e6, 200)])  # ppm·m


@dataclass
class Plume:
    gases: dict[str, float]  # gas -> peak enhancement (ppm·m)
    row: int  # source pixel
    col: int
    length_px: float  # downwind extent (1/e)
    width_px: float
    angle_rad: float  # downwind direction
    seed: int = 0


@dataclass
class Injection:
    radiance: np.ndarray  # (rows, cols, bands) with plumes
    alpha: dict[str, np.ndarray] = field(default_factory=dict)  # gas -> (rows, cols) ppm·m
    plumes: list[Plume] = field(default_factory=list)
    labels: np.ndarray | None = None  # (rows, cols) plume index + 1 where shape > 5% of peak


def plume_shape(shape: tuple[int, int], p: Plume) -> np.ndarray:
    """Peak-normalized enhancement shape (rows, cols), 0 far from the plume."""
    rows, cols = shape
    rng = np.random.default_rng(p.seed)
    half = int(4 * max(p.length_px, p.width_px)) + 3
    r0, r1 = max(p.row - half, 0), min(p.row + half + 1, rows)
    c0, c1 = max(p.col - half, 0), min(p.col + half + 1, cols)
    rr, cc = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    dr, dc = rr - p.row, cc - p.col
    along = dr * np.sin(p.angle_rad) + dc * np.cos(p.angle_rad)
    across = -dr * np.cos(p.angle_rad) + dc * np.sin(p.angle_rad)
    # Downwind the plume widens; upwind it stops at the source
    width = p.width_px * (1 + np.clip(along, 0, None) / max(p.length_px, 1))
    g = np.exp(-np.clip(along, 0, None) / p.length_px - 0.5 * (across / width) ** 2)
    g *= np.exp(-0.5 * (np.clip(-along, 0, None) / 0.7) ** 2)
    texture = ndimage.gaussian_filter(rng.normal(size=g.shape), 1.5)
    g *= np.clip(1 + 1.5 * texture / (texture.std() + 1e-12), 0.1, None)
    out = np.zeros(shape)
    out[r0:r1, c0:c1] = g / g.max()
    return out


def transmittance_table(pieces, centers_nm, fwhm_nm, cond: Conditions, gas: str) -> np.ndarray:
    """Band transmittance (len(ALPHA_GRID), bands); 1 for bands outside line-by-line ranges."""
    c = np.asarray(centers_nm, dtype=np.float64)
    f = np.asarray(fwhm_nm, dtype=np.float64)
    table = np.ones((ALPHA_GRID.size, c.size))
    for (lo, hi), lbl in pieces:
        m = (c >= lo) & (c <= hi)
        if m.any() and gas in lbl.sigma:
            s, t_bg, tau_p = _plume_terms(lbl, c[m], f[m], cond, gas)
            norm = s @ t_bg
            for i, a in enumerate(ALPHA_GRID):
                table[i, m] = (s @ (t_bg * np.exp(-tau_p * a))) / norm
    return table


def apply_gas(radiance: np.ndarray, alpha: np.ndarray, table: np.ndarray) -> None:
    """Multiply radiance in place by the tabulated transmittance at each pixel's alpha."""
    idx = np.nonzero(alpha > 0)
    a = alpha[idx]
    j = np.clip(np.searchsorted(ALPHA_GRID, a) - 1, 0, ALPHA_GRID.size - 2)
    w = ((a - ALPHA_GRID[j]) / (ALPHA_GRID[j + 1] - ALPHA_GRID[j]))[:, None]
    t = (1 - w) * table[j] + w * table[j + 1]
    radiance[idx] *= t.astype(radiance.dtype)


def inject(
    radiance: np.ndarray,
    plumes: list[Plume],
    tables: dict[str, np.ndarray],
    label_floor: float = 0.05,
) -> Injection:
    """Radiance with `plumes` added; `tables` from transmittance_table, one per gas."""
    out = radiance.copy()
    shape = radiance.shape[:2]
    alpha: dict[str, np.ndarray] = {}
    labels = np.zeros(shape, dtype=np.int32)
    for i, p in enumerate(plumes):
        s = plume_shape(shape, p)
        labels[(s > label_floor) & (labels == 0)] = i + 1
        for gas, peak in p.gases.items():
            alpha.setdefault(gas, np.zeros(shape))
            alpha[gas] += peak * s
    for gas, a in alpha.items():
        apply_gas(out, a, tables[gas])
    return Injection(out, alpha, list(plumes), labels)


def random_plumes(
    shape: tuple[int, int],
    n: int,
    gas_sets: list[dict[str, tuple[float, float]]],
    allowed: np.ndarray,
    seed: int,
    margin: int = 30,
) -> list[Plume]:
    """n plumes at random allowed pixels, spaced apart. gas_sets gives, per plume type, each
    gas's (lo, hi) peak range in ppm·m, drawn log-uniformly; the type is chosen uniformly."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    ok = allowed.copy()
    ok[:margin], ok[-margin:], ok[:, :margin], ok[:, -margin:] = False, False, False, False
    taken = np.zeros(shape, dtype=bool)
    out = []
    for _ in range(50 * n):
        if len(out) == n:
            break
        r, c = int(rng.integers(margin, rows - margin)), int(rng.integers(margin, cols - margin))
        if not ok[r, c] or taken[r, c]:
            continue
        kind = gas_sets[int(rng.integers(len(gas_sets)))]
        gases = {
            g: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for g, (lo, hi) in kind.items()
        }
        length = float(rng.uniform(2.0, 12.0))
        p = Plume(
            gases=gases,
            row=r,
            col=c,
            length_px=length,
            width_px=float(rng.uniform(0.8, 2.5)),
            angle_rad=float(rng.uniform(0, 2 * np.pi)),
            seed=int(rng.integers(1 << 30)),
        )
        out.append(p)
        rad = int(6 * length)
        taken[max(r - rad, 0) : r + rad, max(c - rad, 0) : c + rad] = True
    return out
