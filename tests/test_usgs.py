import numpy as np
import pytest

from specret.eval.resampler_check import compare
from specret.io.usgs_splib07 import load_splib07_dir, parse_header, split_title
from specret.sensors import load_sensor


def _column(path, header, values):
    path.write_text(header + "\n" + "\n".join(f" {v:.7e}" for v in values) + "\n")


@pytest.fixture
def fake_splib(tmp_path):
    root = tmp_path / "ASCIIdata_splib07a"
    chapter = root / "ChapterM_Minerals"
    chapter.mkdir(parents=True)
    wl_um = np.linspace(0.4, 2.5, 50)
    _column(
        root / "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
        " splib07a Record=18: Wavelengths BECK Beckman 0.2-3.0 microns",
        wl_um,
    )
    _column(
        root / "splib07a_Bandpass_(FWHM)_BECK_Beckman_in_microns.txt",
        " splib07a Record=20: Bandpass (FWHM) BECK Beckman  in microns",
        np.full(50, 0.01),
    )
    refl = np.linspace(0.3, 0.8, 50)
    refl[:3] = -1.23e34
    _column(
        chapter / "splib07a_Kaolinite_CM9_BECKb_AREF.txt",
        " splib07a Record=5976: Kaolinite CM9                 BECKb AREF",
        refl,
    )
    _column(
        chapter / "splib07a_Kaolinite_CM9_BECKb_TRAN.txt",
        " splib07a Record=5977: Kaolinite CM9                 BECKb TRAN",
        refl,
    )
    (root / "errorbars").mkdir()
    return root


def test_parse_fake_library(fake_splib):
    recs = load_splib07_dir(fake_splib)
    assert len(recs) == 1  # TRAN skipped
    r = recs[0]
    assert r.spectrum_id == "splib07a:5976"
    assert r.sample_id == "usgs:CM9"
    assert r.material_name == r.mineral == "Kaolinite"
    assert r.chapter == "M"
    assert r.instrument == "BECKb"
    assert r.source_record == "Kaolinite_CM9_BECKb_AREF"
    assert np.isnan(r.reflectance[:3]).all() and np.isfinite(r.reflectance[3:]).all()
    assert r.wavelength_nm[0] == pytest.approx(400.0)
    assert r.native_fwhm_nm[0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "line, title, instr, settings, mtype",
    [
        (
            " splib07a Record=5976: Kaolinite CM9                 BECKb AREF",
            "Kaolinite CM9",
            "BECK",
            "b",
            "AREF",
        ),
        (
            " splib07a Record=17358: BHVO-2F+4% 1-Decene 87K      NIC4AA RREF",
            "BHVO-2F+4% 1-Decene 87K",
            "NIC4",
            "AA",
            "RREF",
        ),
        (
            "s07_AV14 Record=1792: Kaolinite CM9                 BECKb AREF",
            "Kaolinite CM9",
            "BECK",
            "b",
            "AREF",
        ),
        (
            " splib07a Record=99: Acmite NMNH133746 Pyroxene    BECKa AREF",
            "Acmite NMNH133746 Pyroxene",
            "BECK",
            "a",
            "AREF",
        ),
    ],
)
def test_parse_header(line, title, instr, settings, mtype):
    h = parse_header(line)
    assert (h["title"], h["instr"], h["settings"], h["mtype"]) == (title, instr, settings, mtype)


@pytest.mark.parametrize(
    "title, material, sample",
    [
        ("Kaolinite CM9", "Kaolinite", "CM9"),
        ("Acmite NMNH133746 Pyroxene", "Acmite", "NMNH133746"),
        ("Actinolite-Tremolit BR93-22C", "Actinolite-Tremolit", "BR93-22C"),
        ("Aluminum brushed 293K", "Aluminum brushed 293K", None),
    ],
)
def test_split_title(title, material, sample):
    assert split_title(title) == (material, sample)


@pytest.mark.data
def test_resampler_matches_usgs_convolution(usgs_dirs):
    native_dir, av14_dir = usgs_dirs
    report = compare(
        load_splib07_dir(native_dir), load_splib07_dir(av14_dir), load_sensor("aviris_classic")
    )
    assert len(report.per_spectrum) > 2000
    assert report.median_rel_error < 0.01
