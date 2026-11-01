import netCDF4
import numpy as np
import torch

from specret.learn.model import BandDecoder, EncoderConfig, SpectralEncoder, make_tokens
from specret.learn.pixels import clear_mask, load_pool, sample_pixels, save_pool
from specret.learn.pretrain import hide_mask, masked_loss

SMALL = EncoderConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, embed_dim=16, n_freq=4)


def _fake_granules(tmp_path, rows=40, cols=30, bands=20):
    rfl = tmp_path / "EMIT_L2A_RFL_fake.nc"
    with netCDF4.Dataset(rfl, "w") as ds:
        ds.createDimension("downtrack", rows)
        ds.createDimension("crosstrack", cols)
        ds.createDimension("bands", bands)
        r = ds.createVariable("reflectance", "f4", ("downtrack", "crosstrack", "bands"))
        data = np.random.default_rng(0).uniform(0.1, 0.5, (rows, cols, bands)).astype("f4")
        data[:5, :, :] = -9999  # fill
        r[:] = data
        bp = ds.createGroup("sensor_band_parameters")
        bp.createDimension("bands", bands)
        bp.createVariable("wavelengths", "f4", ("bands",))[:] = np.linspace(400, 2500, bands)
        bp.createVariable("fwhm", "f4", ("bands",))[:] = np.full(bands, 8.5)
        bp.createVariable("good_wavelengths", "f4", ("bands",))[:] = np.ones(bands)
    mask = tmp_path / "EMIT_L2A_MASK_fake.nc"
    names = [
        "Cloud flag",
        "Cirrus flag",
        "Water flag",
        "Spacecraft Flag",
        "Dilated Cloud Flag",
        "AOD550",
        "H2O (g cm-2)",
        "Aggregate Flag",
    ]
    with netCDF4.Dataset(mask, "w") as ds:
        ds.createDimension("downtrack", rows)
        ds.createDimension("crosstrack", cols)
        ds.createDimension("bands", len(names))
        m = ds.createVariable("mask", "f4", ("downtrack", "crosstrack", "bands"))
        arr = np.zeros((rows, cols, len(names)), "f4")
        arr[:, :10, -1] = 1  # left third flagged
        m[:] = arr
        bp = ds.createGroup("sensor_band_parameters")
        bp.createDimension("bands", len(names))
        v = bp.createVariable("mask_bands", str, ("bands",))
        for i, n in enumerate(names):
            v[i] = n
    return rfl, mask


def test_clear_mask_and_sampling(tmp_path):
    rfl, mask = _fake_granules(tmp_path)
    ok = clear_mask(mask)
    assert not ok[:, :10].any() and ok[:, 10:].all()
    px = sample_pixels(rfl, mask, n=200, tile=10)
    assert 0 < len(px) <= 200 and px.dtype == np.float16
    assert np.isfinite(px.astype(np.float32)).all()  # fill rows dropped
    save_pool(tmp_path / "pool.npz", px, rfl)
    pixels, centers, fwhm, good = load_pool([tmp_path / "pool.npz"])
    assert pixels.shape == px.shape and centers.size == 20 and good.all()


def test_hide_mask_ratio_and_validity():
    rng = np.random.default_rng(0)
    valid = np.ones((8, 100), bool)
    valid[:, 40:50] = False
    hidden = hide_mask(valid, 0.4, rng)
    assert not hidden[~valid].any()
    frac = hidden.sum(1) / valid.sum(1)
    assert (frac > 0.3).all() and (frac < 0.5).all()


def test_masked_loss_backpropagates():
    wl = np.linspace(400, 2500, 60)
    x = np.random.default_rng(1).uniform(0.1, 0.5, (4, 60))
    tok, pad = make_tokens(x, wl, np.full(60, 10.0))
    hidden = hide_mask(~pad, 0.4, np.random.default_rng(2))
    model, dec = SpectralEncoder(SMALL), BandDecoder(SMALL.d_model)
    loss = masked_loss(
        model, dec, torch.as_tensor(tok), torch.as_tensor(pad), torch.as_tensor(hidden)
    )
    loss.backward()
    assert torch.isfinite(loss) and model.mask_token.grad is not None
