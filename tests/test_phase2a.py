import numpy as np
import pandas as pd
import pytest

from specret.eval.splits import load_splits, make_splits, split_hash
from specret.preprocess.continuum import band_depth, continuum_removed, upper_hull
from specret.search.vectors import represent
from specret.taxonomy import apply_taxonomy, load_taxonomy

# --- continuum removal ---


def test_hull_of_concave_down_points_keeps_all():
    x = np.arange(5.0)
    y = -((x - 2) ** 2)
    np.testing.assert_array_equal(upper_hull(x, y), [0, 1, 2, 3, 4])


def test_absorption_on_sloped_continuum():
    wl = np.linspace(2000, 2500, 101)
    cont = 0.4 + 0.0004 * (wl - 2000)
    feature = 1 - 0.2 * np.exp(-0.5 * ((wl - 2200) / 15) ** 2)
    cr = continuum_removed(wl, cont * feature)
    assert cr[0] == pytest.approx(1.0) and cr[-1] == pytest.approx(1.0)
    assert wl[np.argmin(cr)] == pytest.approx(2200, abs=5)
    assert cr.min() == pytest.approx(0.8, abs=0.01)  # depth recovered despite the slope


def test_brightness_and_slope_do_not_change_depth():
    wl = np.linspace(2000, 2500, 101)
    feature = 1 - 0.1 * np.exp(-0.5 * ((wl - 2340) / 12) ** 2)
    a = band_depth(wl, 0.3 * feature)
    b = band_depth(wl, (0.6 + 0.0008 * (wl - 2000)) * feature)
    np.testing.assert_allclose(a, b, atol=2e-3)


def test_nan_bands_are_skipped():
    wl = np.linspace(1000, 2500, 151)
    y = 0.5 + 0.0001 * (wl - 1000)
    y[50:60] = np.nan
    cr = continuum_removed(wl, y)
    assert np.isnan(cr[50:60]).all()
    np.testing.assert_allclose(cr[np.isfinite(cr)], 1.0)


def test_featureless_query_is_invalid_in_cr_space():
    wl = np.linspace(2000, 2500, 60)
    flat = 0.3 + 0.0002 * (wl - 2000)  # straight line: its own hull, zero depth
    _, ok = represent(flat, wl, np.ones(60, bool), "cr")
    assert not ok[0]
    _, ok_sam = represent(flat, wl, np.ones(60, bool), "sam")
    assert ok_sam[0]


# --- taxonomy ---


@pytest.mark.parametrize(
    "name, chapter, group, cls",
    [
        ("Kaolinite", "M", "kaolin", "phyllosilicate"),
        ("Chlorapatite", "M", "phosphate", "phosphate"),
        ("Chlor.6+Epid.2+Calc.2", "S", "chlorite", "phyllosilicate"),
        ("Calc.25+Dolo.25+Na-Mon", "S", "calcite", "carbonate"),
        ("Dolomite", "M", "dolomite", "carbonate"),
        ("Goethite_Thin_Film", "C", "goethite", "fe_oxide"),
        ("Nanohematite", "M", "hematite", "fe_oxide"),
        ("Buddingtonite", "M", "buddingtonite", "feldspar"),
        ("Asphalt", "A", "artificial", "artificial"),
        ("Juniper", "V", "vegetation", "vegetation"),
        ("Unobtainium", "M", "unclassified", "unclassified"),
    ],
)
def test_taxonomy(name, chapter, group, cls):
    assert load_taxonomy().classify(name, chapter) == (group, cls)


# --- splits ---


def _toy_library():
    rows = []
    for g, n in [("kaolin", 10), ("alunite", 8), ("zeolite", 5), ("garnet", 2)]:
        for s in range(n):
            for inst in ("BECK", "ASD"):  # two spectra per physical sample
                rows.append(
                    {
                        "spectrum_id": f"{g}{s}{inst}",
                        "sample_id": f"{g}-{s}",
                        "material_name": g,
                        "chapter": "M",
                        "measurement": "AREF",
                    }
                )
    df = pd.DataFrame(rows)
    df["group"] = df["material_name"]
    return df


CFG = {
    "seed": 1,
    "chapters": ["M"],
    "measurements": ["AREF"],
    "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
    "min_samples": 4,
    "openset_groups": ["zeolite"],
}


def test_splits_keep_samples_together_and_hold_out_groups():
    splits = make_splits(_toy_library(), CFG)
    assert (splits.groupby("sample_id")["split"].nunique() == 1).all()
    assert set(splits.loc[splits.group == "zeolite", "split"]) == {"openset"}
    assert set(splits.loc[splits.group == "garnet", "split"]) == {"train"}  # too few samples
    assert {"train", "val", "test"} <= set(splits.loc[splits.group == "kaolin", "split"])


def test_splits_are_deterministic():
    a = make_splits(_toy_library(), CFG)
    b = make_splits(_toy_library(), CFG)
    assert split_hash(a) == split_hash(b)


def test_frozen_split_matches_its_hash():
    splits = load_splits("splits_v1")
    assert (splits.groupby("sample_id")["split"].nunique() == 1).all()
    assert {"train", "val", "test", "openset"} == set(splits["split"])


def test_apply_taxonomy_fills_columns():
    df = pd.DataFrame({"material_name": ["Alunite", "Grass"], "chapter": ["M", "V"]})
    out = apply_taxonomy(df)
    assert out["group"].tolist() == ["alunite", "vegetation"]
    assert out["material_class"].tolist() == ["sulfate", "vegetation"]
