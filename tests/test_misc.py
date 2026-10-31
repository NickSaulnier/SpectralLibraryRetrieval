import numpy as np
import pytest

from specret.preprocess.normalize import infer_scale, is_plausible, sam_degrees
from specret.schema import SpectrumRecord, read_library, write_library
from specret.sensors import load_sensor


@pytest.mark.parametrize(
    "values, scale",
    [
        (np.array([0.1, 0.5, 0.9]), 1.0),
        (np.array([10.0, 50.0, 90.0]), 0.01),
        (np.array([1000, 5000, 9000]), 1e-4),
    ],
)
def test_infer_scale(values, scale):
    assert infer_scale(values) == scale


def test_plausibility_and_sam():
    assert is_plausible(np.array([0.2, 0.4]))
    assert not is_plausible(np.array([40.0, 60.0]))
    assert sam_degrees(np.array([1.0]))[0] == pytest.approx(0.0)
    assert sam_degrees(np.array([0.0]))[0] == pytest.approx(90.0)


def test_parquet_round_trip(tmp_path):
    rec = SpectrumRecord(
        spectrum_id="x:1",
        sample_id="s",
        source="t",
        source_record="f",
        material_name="m",
        wavelength_nm=[400.0, 500.0, 600.0],
        reflectance=[0.1, np.nan, 0.3],
        native_fwhm_nm=[10.0, 10.0, 10.0],
    )
    write_library([rec], tmp_path / "lib.parquet")
    df = read_library(tmp_path / "lib.parquet")
    assert df.loc[0, "coverage_min_nm"] == 400.0
    assert df.loc[0, "n_valid"] == 2
    assert np.isnan(df.loc[0, "reflectance"][1])


def test_duplicate_ids_rejected(tmp_path):
    rec = SpectrumRecord("x:1", "s", "t", "f", "m", [1.0, 2.0], [0.1, 0.2])
    with pytest.raises(ValueError):
        write_library([rec, rec], tmp_path / "lib.parquet")


def test_field_sensor_grid():
    s = load_sensor("asd_fieldspec")
    assert s.n_bands == 2151
    assert s.fwhm_nm[0] == 3.0 and s.fwhm_nm[-1] == 10.0
    assert not s.good_mask[s.centers_nm == 1400.0][0]
    assert s.good_mask[s.centers_nm == 2200.0][0]
