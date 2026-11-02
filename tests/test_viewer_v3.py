"""Viewer Phase V3 (PLAN_VIEWER.md): stored plume fit spectra and the evidence they carry.

Synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from test_gas_g2 import synthetic_context  # noqa: E402

from specret.gas.identify import identify  # noqa: E402


def test_fit_spectra_add_up_to_the_observed_residual():
    ctx, cand = synthetic_context("CH4")
    res = identify(ctx, cand, "CH4", n_null=200)
    f = res.spectra
    assert f is not None and set(f.terms) == set(res.terms)
    # Each term's contribution is its coefficient times its unwhitened signature
    assert f.terms["CH4"] == pytest.approx(-res.terms["CH4"] * ctx.gas_k["CH4"])
    model = f.baseline + sum(f.terms.values())
    assert np.allclose(f.obs, model + f.remainder)
    # The fit explains the injected methane: the remainder is within the null spread
    sig = np.abs(f.terms["CH4"])
    assert np.abs(f.remainder).mean() < np.abs(f.obs).mean()
    assert np.abs(f.remainder[sig > 0.5 * sig.max()]).mean() < 2 * f.null_sd.mean()


def test_spectra_round_trip_through_explain_plume(tmp_path):
    from specret.gas import tools
    from specret.gas.outputs import write_spectra

    ctx, cand = synthetic_context("CH4")
    res = identify(ctx, cand, "CH4", n_null=200)
    folder = tmp_path / "20230608T162322_2315911_014"
    folder.mkdir()
    write_spectra(folder / "plume_spectra.npz", [(f"CH4_{cand.id}", res.spectra)])
    pd.DataFrame(
        {
            "plume_id": [1],
            "gas": ["CH4"],
            "confidence": [0.9],
            "detection_gas": ["CH4"],
            "candidate_id": [cand.id],
            "terms": [";".join(f"{k}={v:.3g}" for k, v in res.terms.items())],
            "source_lat": [31.5],
            "source_lon": [-103.2],
            "q_kg_h": [700.0],
            "sigma_q_kg_h": [300.0],
            "rel_sigma_wind": [0.3],
            "q_csf_kg_h": [450.0],
            "transects": [6],
            "csf_iqr_rel": [0.4],
            "wind_u": [1.0],
            "wind_v": [-2.0],
        }
    ).to_csv(folder / "plumes.csv", index=False)
    pd.DataFrame(
        {"detection_gas": ["CH4"], "candidate_id": [cand.id], "verdict": ["CH4"], "mf_z": [6.0]}
    ).to_csv(folder / "candidates.csv", index=False)

    e = tools.explain_plume(folder, 1)
    s = e["spectra"]
    assert s["wavelengths_nm"] == pytest.approx(ctx.centers_nm.tolist(), abs=1e-3)
    assert s["observed"] == pytest.approx(res.spectra.obs.tolist(), abs=1e-6)
    assert s["terms"]["CH4"] == pytest.approx(res.spectra.terms["CH4"].tolist(), rel=1e-5)
    assert np.allclose(np.add(s["model"], s["remainder"]), s["observed"], atol=1e-6)
    assert set(s["signatures_per_ppmm"]) == {"CH4", "CO2"}
    assert e["cross_sectional_flux"]["q_csf_kg_h"] == 450.0
    assert e["wind"] == {"wind_u": 1.0, "wind_v": -2.0}
    assert tools.explain_plume(folder, 1, spectra=False)["spectra"] is None
    # Scenes detected before V3 have no stored spectra
    (folder / "plume_spectra.npz").unlink()
    assert tools.explain_plume(folder, 1)["spectra"] is None
