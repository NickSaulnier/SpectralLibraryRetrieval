import struct

import numpy as np
import pytest

from specret.io.field import HEADER_SIZE, read_asd, read_field_table
from specret.preprocess.splice import splice_correct


def write_asd(path, version, data_type, values, reference=None, ch1=350.0, step=1.0):
    header = bytearray(HEADER_SIZE)
    header[0:3] = version.encode()
    header[186] = data_type
    struct.pack_into("<ff", header, 191, ch1, step)
    header[199] = 0  # float32
    struct.pack_into("<H", header, 204, len(values))
    body = struct.pack(f"<{len(values)}f", *values)
    blob = bytes(header) + body
    if reference is not None:
        desc = b"white panel"
        blob += b"\x01\x00" + b"\x00" * 16 + struct.pack("<H", len(desc)) + desc
        blob += struct.pack(f"<{len(reference)}f", *reference)
    path.write_bytes(blob)


def test_read_raw_asd_with_reference(tmp_path):
    target = np.linspace(1000, 3000, 2151)
    white = np.full(2151, 4000.0)
    p = tmp_path / "rock00001.asd"
    write_asd(p, "as7", 0, target, white)
    s = read_asd(p)
    assert s.data_type == "raw"
    np.testing.assert_allclose(s.wavelength_nm[[0, -1]], [350, 2500])
    np.testing.assert_allclose(s.reflectance, target / white, rtol=1e-6)


def test_read_old_reflectance_asd(tmp_path):
    refl = np.linspace(0.1, 0.5, 2151)
    p = tmp_path / "old.asd"
    write_asd(p, "ASD", 1, refl)
    s = read_asd(p)
    assert s.reference is None
    np.testing.assert_allclose(s.reflectance, refl, rtol=1e-6)


def test_raw_without_reference_raises(tmp_path):
    p = tmp_path / "raw.asd"
    write_asd(p, "ASD", 0, np.ones(10))
    with pytest.raises(ValueError):
        _ = read_asd(p).reflectance


def test_csv_long_and_wide(tmp_path):
    wl = np.arange(350, 2501)
    long = tmp_path / "long.csv"
    long.write_text("wavelength,a,b\n" + "\n".join(f"{w},{50.0},{0.3}" for w in wl))
    a, b = read_field_table(long)
    assert a.values[0] == pytest.approx(0.5)  # percent converted
    assert b.values[0] == pytest.approx(0.3)

    wide = tmp_path / "wide.csv"
    wide.write_text(
        "sample," + ",".join(map(str, wl)) + "\n" + "rock1," + ",".join(["0.25"] * wl.size)
    )
    (s,) = read_field_table(wide)
    assert s.name == "rock1"
    assert s.wavelength_nm.size == wl.size


def test_splice_correction_removes_detector_steps():
    wl = np.arange(350.0, 2501.0)
    smooth = 0.3 + 0.0001 * (wl - 350)
    stepped = smooth.copy()
    stepped[wl <= 1000] *= 1.05
    stepped[wl > 1800] *= 0.96
    fixed = splice_correct(wl, stepped)
    np.testing.assert_allclose(fixed, smooth, rtol=1e-3)
