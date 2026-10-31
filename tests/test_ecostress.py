import numpy as np
import pytest

from specret.io.ecostress import load_ecostress_dir, read_ecostress_file

SAMPLE = """Name: Kaolinite KGa-1 (well-ordered)
Type: Mineral
Class: Silicate
Subclass: Phyllosilicate
Particle Size: 0-125 um
Sample No.: KGa-1
Owner: JHU
Wavelength Range: All
Origin: Source Clay Repository
Collection Date: N/A
Description: Well crystallized kaolinite.
  Second line of the description.
Measurement: Directional (10 Degree) Hemispherical Reflectance
First Column: X
Second Column: Y
X Units: Wavelength (micrometers)
Y Units: Reflectance (percent)
First X Value: 2.5
Last X Value: 0.4
Number of X Values: 4
Additional Information: none

2.5\t40.0
2.0\t60.0
1.0\t70.0
0.4\t50.0
"""


def test_parse_ecostress_file(tmp_path):
    p = tmp_path / "mineral.silicate.phyllosilicate.fine.vswir.kga-1.jhu.becknic.spectrum.txt"
    p.write_text(SAMPLE)
    header, wl, refl = read_ecostress_file(p)
    assert header["Sample No."] == "KGa-1"
    assert "Second line" in header["Description"]
    np.testing.assert_allclose(wl, [400, 1000, 2000, 2500])
    np.testing.assert_allclose(refl, [0.5, 0.7, 0.6, 0.4])

    (rec,) = load_ecostress_dir(tmp_path)
    assert rec.sample_id == "ecostress:JHU:KGa-1"
    assert rec.mineral == "Kaolinite KGa-1 (well-ordered)"
    assert rec.material_class == "Silicate/Phyllosilicate"
    assert rec.purity_flag is None


def test_missing_data_block_raises(tmp_path):
    p = tmp_path / "empty.spectrum.txt"
    p.write_text("Name: nothing\nType: Mineral\n")
    with pytest.raises(ValueError):
        read_ecostress_file(p)
