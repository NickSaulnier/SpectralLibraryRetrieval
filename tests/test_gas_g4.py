"""Phase G4 (PLAN_GAS.md): LWIR thermal-contrast model on synthetic HyTES-like radiance.

No HyTES files or HITRAN tables needed.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from specret.gas.inject import ALPHA_GRID, Plume  # noqa: E402
from specret.gas.lwir import LwirPrepared, inject_lwir, lwir_enhancement  # noqa: E402
from specret.io.hytes import brightness_temperature, planck  # noqa: E402

CENTERS = np.arange(7600.0, 11900.0, 17.8)


def test_planck_and_brightness_temperature_invert():
    t = np.array([280.0, 300.0, 330.0])
    for wl in (8000.0, 10800.0, 11900.0):
        np.testing.assert_allclose(brightness_temperature(wl, planck(wl, t)), t, rtol=1e-9)
    # About 9.9 W m^-2 sr^-1 µm^-1 at 10 µm and 300 K
    assert planck(10000.0, 300.0) == pytest.approx(9.92, rel=0.01)


def thermal_scene(rows=1500, cols=6, t_plume=305.0, seed=0):
    """Surface 320-335 K with emissivity texture, a mild atmosphere, and an NH3-like band."""
    rng = np.random.default_rng(seed)
    nb = CENTERS.size
    t_s = rng.uniform(320, 335, (rows, cols))[..., None]
    emis = 0.96 + 0.02 * rng.normal(size=(rows, cols, 1)) * np.sin(CENTERS / 400.0)[None, None, :]
    tau = np.full(nb, 0.9)
    l_up = np.full(nb, 0.6)
    surf = emis * planck(CENTERS, t_s)
    x = (tau * surf + l_up + rng.normal(0, 0.01, (rows, cols, nb))).astype(np.float32)
    sigma = 4e-6 * np.exp(-0.5 * ((CENTERS - 10500) / 60) ** 2) + 2e-6 * np.exp(
        -0.5 * ((CENTERS - 10750) / 40) ** 2
    )
    prep = LwirPrepared(
        flight=None,
        rows=slice(0, rows),
        idx=np.arange(nb),
        centers_nm=CENTERS,
        fwhm_nm=np.full(nb, 17.8),
        radiance=x,
        tau=tau,
        l_up=l_up,
        t_plume=t_plume,
        sigma={"NH3": sigma},
        screen=np.zeros((rows, cols), dtype=bool),
    )
    table = np.exp(-np.outer(ALPHA_GRID, sigma))
    return prep, table


def test_lwir_matched_filter_recovers_alpha_with_contrast_correction():
    prep, table = thermal_scene()
    plume = Plume({"NH3": 3000.0}, row=700, col=3, length_px=60, width_px=40, angle_rad=1.57)
    inj = inject_lwir(prep, [plume], {"NH3": table})
    assert np.all(inj.radiance[700, 3] <= prep.radiance[700, 3] + 1e-4)  # cool plume absorbs
    enh = lwir_enhancement(prep, "NH3", inj.radiance, device="cpu")
    core = inj.alpha["NH3"] > 1500
    est = np.nanmedian(enh.alpha[core] / inj.alpha["NH3"][core])
    assert est == pytest.approx(1.0, abs=0.15)
    bg = enh.alpha[inj.alpha["NH3"] == 0]
    assert abs(np.nanmedian(bg)) < 0.2 * 3000
    assert np.nanmedian(enh.delta_t) < -10  # plume 305 K over a 320-335 K surface


def test_warm_plume_emits_and_low_contrast_is_masked():
    prep, table = thermal_scene(t_plume=350.0)
    plume = Plume({"NH3": 3000.0}, row=700, col=3, length_px=60, width_px=40, angle_rad=1.57)
    inj = inject_lwir(prep, [plume], {"NH3": table})
    j = int(np.argmin(np.abs(CENTERS - 10500)))
    assert inj.radiance[700, 3, j] > prep.radiance[700, 3, j]  # warm plume emits
    enh = lwir_enhancement(prep, "NH3", inj.radiance, device="cpu")
    core = inj.alpha["NH3"] > 1500
    assert np.nanmedian(enh.alpha[core]) > 1500  # still a positive amount

    # Plume temperature equal to the surface's: nothing to see, and it is masked
    prep_eq, _ = thermal_scene(t_plume=327.0)
    enh_eq = lwir_enhancement(prep_eq, "NH3", device="cpu")
    assert enh_eq.low_contrast.any()
    assert np.isnan(enh_eq.alpha[enh_eq.low_contrast]).all()


def test_merge_candidates_identifies_each_region_once():
    from types import SimpleNamespace

    from specret.gas.lwir import merge_candidates
    from specret.gas.plumes import Candidate

    shape = (40, 40)
    a = {g: np.zeros(shape, dtype=np.float32) for g in ("NH3", "C2H4")}
    enh = {g: SimpleNamespace(alpha=v) for g, v in a.items()}

    def cand(r0, c0, z):
        rr, cc = np.mgrid[r0 : r0 + 3, c0 : c0 + 3]
        return Candidate(1, rr.ravel(), cc.ravel(), z, 1.0, 1.0)

    found = {
        "NH3": [cand(5, 5, 9.0), cand(30, 30, 12.0)],
        "C2H4": [cand(6, 6, 11.0)],  # overlaps NH3's first region, with a higher peak
    }
    out = merge_candidates(found, enh)
    assert len(out["NH3"]) == 1 and len(out["C2H4"]) == 1
    merged = out["C2H4"][0]
    assert merged.peak_z == 11.0 and merged.n_px == 9 + 9 - 4  # union of the overlapping masks
    assert out["NH3"][0].peak_z == 12.0
