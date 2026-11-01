"""Phase G0 (PLAN_GAS.md): gas scene list, gas registry and download helpers. No network."""

import yaml

from specret.io.download import EMIT_PRODUCTS, _scene_window, _source_scenes
from specret.io.hitran import load_gases
from specret.sensors.sensor import CONFIG_DIR


def test_scene_window():
    assert _scene_window("20230608T162322_2315911_014") == (
        "2023-06-08T16:22:22Z",
        "2023-06-08T16:25:22Z",
    )


def test_source_scenes():
    g = {
        "umm": {
            "AdditionalAttributes": [
                {"Name": "PLUME_ID", "Values": ["524"]},
                {
                    "Name": "SOURCE_SCENES",
                    "Values": ["['EMIT_L2B_CH4ENH_V002_20220826T174642_2223812_024']"],
                },
            ]
        }
    }
    assert "20220826T174642_2223812_024" in _source_scenes(g)
    assert _source_scenes({"umm": {}}) == ""


def test_gas_scenes_config():
    cfg = yaml.safe_load((CONFIG_DIR / "gas_scenes.yaml").read_text())
    ids = [s["id"] for s in cfg["scenes"]]
    assert len(ids) == len(set(ids))
    for s in cfg["scenes"]:
        assert set(s["products"]) <= set(EMIT_PRODUCTS)
        assert s["role"] in ("plume", "no_plume")
        if s["role"] == "plume":
            assert any(p.endswith("plm") for p in s["products"]) and s["gases"]
        else:
            assert not s["gases"]


def test_gas_registry():
    cfg = load_gases()
    ranges = cfg["hitran_ranges_cm1"]
    for name, g in cfg["gases"].items():
        assert g["molar_mass"] > 0, name
        assert set(g["quantify"]) <= set(ranges), name
        for rng in g["quantify"]:
            assert g[rng], f"{name} quantified in {rng} without a window there"
        for rng in ranges:
            # Windows (nm) must sit inside the fetched wavenumber range (cm^-1).
            lo_cm, hi_cm = ranges[rng]
            for lo, hi in g[rng]:
                assert 1e7 / hi_cm <= lo < hi <= 1e7 / lo_cm, (name, rng, lo, hi)
