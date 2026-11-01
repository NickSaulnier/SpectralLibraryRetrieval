from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from specret.sensors.library import SensorLibrary
from specret.sensors.sensor import Sensor

WL = np.arange(1990.0, 2420.0, 7.5)
WL_FLIGHT = WL + 3.0  # an AVIRIS flight whose bands match no library


def save_library(path: Path, name: str, centers: np.ndarray) -> None:
    sensor = Sensor(name, centers, np.full(centers.size, 8.0), np.ones(centers.size, bool))
    SensorLibrary(sensor, np.array(["t:0"]), np.full((1, centers.size), 0.5)).save(path)


def write_envi(path: Path, centers: np.ndarray) -> None:
    import spectral.io.envi as envi

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"wavelength": centers.tolist(), "fwhm": [8.0] * centers.size}
    envi.save_image(str(path), np.full((2, 2, centers.size), 0.5, np.float32), metadata=meta)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project with an EMIT library and two AVIRIS flights that share unmatched bands."""
    import specret.project

    (tmp_path / "configs/sensors").mkdir(parents=True)
    (tmp_path / "configs/splits").mkdir()
    (tmp_path / "configs/splits/splits_v1.csv").touch()  # committed in the real repo
    save_library(tmp_path / "data/resampled/emit/library.npz", "emit", WL)
    write_envi(tmp_path / "data/scenes/aviris/f1_img.hdr", WL_FLIGHT)
    write_envi(tmp_path / "data/scenes/aviris/f2_img.hdr", WL_FLIGHT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(specret.project, "_environment", lambda: [])
    return tmp_path


def fake_run(calls: list):
    """Stand-in for the CLI that records each command and creates its output."""

    def run(argv: list[str]) -> None:
        calls.append(argv)
        cmd = argv[0]
        opt = {a: argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith("--")}
        if cmd == "download":
            for s in ("splib07a", "splib07b_cvAVIRISc2014"):
                Path(f"data/raw/usgs_splib07/extracted/ASCIIdata_{s}").mkdir(parents=True)
        elif cmd == "ingest":
            Path("data/canonical").mkdir(parents=True)
            pd.DataFrame({"spectrum_id": ["t:0"]}).to_parquet("data/canonical/library.parquet")
        elif cmd == "resample":
            name = {"aviris_classic": "avirisc", "asd_fieldspec": "field_asd"}.get(
                opt["--sensor"], "scene"
            )
            out = opt.get("--out", f"data/resampled/{name}/library.npz")
            save_library(Path(out), name, WL_FLIGHT if "--band-file" in opt else WL)
        elif cmd in ("calibrate", "map-scene"):
            out = opt.get("--out") or f"outputs/maps/{Path(opt['--scene']).stem}/scene_map.npz"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()

    return run


def test_inspect_plans_missing_steps(project):
    from specret.project import inspect, report

    items = {it.key: it for it in inspect()}
    assert not items["usgs"].ok and items["usgs"].steps == [["download", "usgs"]]
    assert items["canonical"].steps == [["ingest"]]
    assert items["sensors"].steps == [
        ["resample", "--sensor", "aviris_classic"],
        ["resample", "--sensor", "asd_fieldspec"],
    ]
    # Both flights share bands, so one library serves them.
    assert items["scene_libs"].detail == "2 of 2 scenes have no library for their bands"
    assert items["scene_libs"].steps == [
        ["resample", "--sensor", "aviris_classic_scene", "--band-file",
         "data/scenes/aviris/f1_img.hdr", "--out", "data/resampled/avirisc_scene/library.npz"]
    ]  # fmt: skip
    assert items["calibration"].steps[0][:2] == ["calibrate", "data/resampled/emit/library.npz"]
    assert not items["noise"].steps and "RFLUNCERT" in items["noise"].hint
    assert not items["maps"].steps  # no library for these scenes yet
    text = report(list(items.values()))
    assert "[!!] USGS spectral library" in text and "Run `specret setup`" in text


def test_setup_runs_steps_in_order_until_done(project):
    from specret.project import setup

    calls = []
    items = setup(fake_run(calls), log=lambda *_: None)
    assert [c[0] for c in calls] == ["download", "ingest"] + ["resample"] * 3 + ["calibrate"] * 4
    assert [c[1] for c in calls if c[0] == "calibrate"] == [
        f"data/resampled/{d}/library.npz" for d in ("avirisc", "avirisc_scene", "emit", "field_asd")
    ]
    assert all(it.ok for it in items if it.level == "required")

    calls.clear()
    items = setup(fake_run(calls), map_scenes=True, log=lambda *_: None)
    assert [c[0] for c in calls] == ["map-scene", "map-scene"]
    assert {it.key: it for it in items}["maps"].ok

    calls.clear()
    setup(fake_run(calls), log=lambda *_: None)
    assert calls == []  # nothing left to do


def test_setup_stops_when_a_step_does_not_help(project):
    from specret.project import setup

    with pytest.raises(RuntimeError, match="USGS spectral library is still incomplete"):
        setup(lambda argv: None, log=lambda *_: None)


def test_doctor_cli(project, capsys, tmp_path_factory, monkeypatch):
    from specret.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["doctor"])
    assert exc.value.code == 1 and "-> specret download usgs" in capsys.readouterr().out

    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    with pytest.raises(SystemExit):
        main(["doctor"])
    assert "is not the project folder" in capsys.readouterr().out
