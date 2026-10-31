import netCDF4
import numpy as np
import pytest

from specret.io.emit import open_emit_l2a
from specret.sensors import load_sensor


@pytest.fixture
def fake_granule(tmp_path):
    """A tiny file with the EMIT L2A group layout."""
    path = tmp_path / "EMIT_L2A_RFL_fake.nc"
    rows, cols, bands = 4, 3, 6
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("downtrack", rows)
        ds.createDimension("crosstrack", cols)
        ds.createDimension("bands", bands)
        r = ds.createVariable("reflectance", "f4", ("downtrack", "crosstrack", "bands"))
        data = np.arange(rows * cols * bands, dtype=np.float32).reshape(rows, cols, bands) / 100
        data[0, 0, :] = -9999
        r[:] = data
        ds.setncattr("geotransform", [-117.3, 0.0005, 0, 37.6, 0, -0.0005])

        bp = ds.createGroup("sensor_band_parameters")
        bp.createDimension("bands", bands)
        bp.createVariable("wavelengths", "f4", ("bands",))[:] = [500, 1000, 1400, 1900, 2200, 2480]
        bp.createVariable("fwhm", "f4", ("bands",))[:] = np.full(bands, 8.5)
        bp.createVariable("good_wavelengths", "f4", ("bands",))[:] = [1, 1, 0, 1, 1, 1]

        loc = ds.createGroup("location")
        loc.createDimension("downtrack", rows)
        loc.createDimension("crosstrack", cols)
        loc.createDimension("ortho_y", 2)
        loc.createDimension("ortho_x", 2)
        lat = 37.5 + np.arange(rows)[:, None] * 0.001 + np.zeros((1, cols))
        lon = -117.2 + np.arange(cols)[None, :] * 0.001 + np.zeros((rows, 1))
        loc.createVariable("lat", "f8", ("downtrack", "crosstrack"))[:] = lat
        loc.createVariable("lon", "f8", ("downtrack", "crosstrack"))[:] = lon
        # GLT indices are 1-based; 0 means no data.
        loc.createVariable("glt_x", "i4", ("ortho_y", "ortho_x"))[:] = [[1, 3], [0, 2]]
        loc.createVariable("glt_y", "i4", ("ortho_y", "ortho_x"))[:] = [[1, 4], [0, 2]]
    return path, data


def test_open_and_read_pixels(fake_granule):
    path, data = fake_granule
    scene = open_emit_l2a(path)
    assert scene.shape == (4, 3)
    assert np.isnan(scene.pixel(0, 0)).all()  # fill value -> NaN
    np.testing.assert_allclose(scene.pixel(2, 1), data[2, 1], rtol=1e-6)
    assert scene.nearest_pixel(37.502, -117.199) == (2, 1)


def test_orthorectify(fake_granule):
    path, _ = fake_granule
    scene = open_emit_l2a(path)
    swath = np.arange(12, dtype=float).reshape(4, 3)
    ortho = scene.orthorectify(swath)
    assert ortho[0, 0] == swath[0, 0]
    assert ortho[0, 1] == swath[3, 2]
    assert np.isnan(ortho[1, 0])
    assert ortho[1, 1] == swath[1, 1]


def test_sensor_from_granule_combines_masks(fake_granule):
    path, _ = fake_granule
    s = load_sensor("emit", band_file=path)
    # 1400 nm: water band and flagged bad; 1900 nm: water band; 2480 nm: edge.
    np.testing.assert_array_equal(s.good_mask, [True, True, False, False, True, False])
