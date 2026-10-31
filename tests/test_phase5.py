import numpy as np
import pandas as pd
import pytest
import torch

from specret.learn.model import (
    EncoderConfig,
    SpectralEncoder,
    embed,
    load_encoder,
    make_tokens,
    save_encoder,
)
from specret.learn.train import augment, supcon

WL = np.arange(400.0, 2500.0, 10.0)
FWHM = np.full(WL.size, 10.0)
SMALL = EncoderConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, embed_dim=16, n_freq=4)


def test_tokens_mark_invalid_bands_as_padding():
    x = np.linspace(0.2, 0.6, WL.size)
    x[10:20] = np.nan
    valid = np.ones(WL.size, bool)
    valid[-5:] = False
    tok, pad = make_tokens(x, WL, FWHM, valid)
    assert tok.shape == (1, WL.size, 4) and pad.shape == (1, WL.size)
    assert pad[0, 10:20].all() and pad[0, -5:].all() and not pad[0, :10].any()
    assert tok[0, 0, 2] == pytest.approx(0.4)  # wavelength in um
    # Brightness-free first channel: scaling the spectrum leaves tokens unchanged.
    tok2, _ = make_tokens(3 * x, WL, FWHM, valid)
    np.testing.assert_allclose(tok[..., :2], tok2[..., :2], atol=1e-5)


def test_encoder_is_unit_norm_and_handles_any_band_set(tmp_path):
    torch.manual_seed(0)
    model = SpectralEncoder(SMALL).eval()
    x = np.random.default_rng(0).uniform(0.1, 0.6, (3, WL.size))
    z, ok = embed(model, x, WL, FWHM)
    assert ok.all() and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    # A different sensor (every other band) still embeds.
    z2, ok2 = embed(model, x[:, ::2], WL[::2], FWHM[::2])
    assert ok2.all() and z2.shape == z.shape
    save_encoder(model, tmp_path / "m.pt")
    z3, _ = embed(load_encoder(tmp_path / "m.pt"), x, WL, FWHM)
    np.testing.assert_allclose(z, z3, atol=1e-5)


def test_too_few_bands_is_invalid():
    model = SpectralEncoder(SMALL).eval()
    x = np.full((1, WL.size), np.nan)
    x[0, :5] = 0.3
    _, ok = embed(model, x, WL, FWHM)
    assert not ok[0]


def test_supcon_prefers_grouped_embeddings():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0], [1, 0.1], [0, 1], [0.1, 1]]), dim=1
    )
    bad = good[[0, 2, 1, 3]]
    assert supcon(good, labels, 0.1) < supcon(bad, labels, 0.1)


def test_augment_shapes_and_mask():
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0.2, 0.5, WL.size), (8, 1))
    good = np.ones(WL.size, bool)
    good[:3] = False
    y, keep = augment(x, WL, good, x[::-1], None, rng)
    assert y.shape == x.shape and keep.shape == x.shape
    assert not keep[:, :3].any()
    assert np.isfinite(y).all()


def test_learned_retriever_finds_identical_spectrum(tmp_path):
    from specret.search.learned import LearnedRetriever
    from specret.sensors.library import SensorLibrary
    from specret.sensors.sensor import Sensor

    torch.manual_seed(0)
    path = tmp_path / "enc.pt"
    save_encoder(SpectralEncoder(SMALL).eval(), path)
    rng = np.random.default_rng(1)
    mat = rng.uniform(0.1, 0.6, (6, WL.size))
    ids = np.array([f"s{i}" for i in range(6)])
    lib = SensorLibrary(Sensor("toy", WL, FWHM, np.ones(WL.size, bool)), ids, mat)
    canonical = pd.DataFrame(
        {"spectrum_id": ids, "material_name": ids, "group": "g", "chapter": "M"}
    )
    r = LearnedRetriever(lib, canonical, encoder_path=path, device="cpu")
    assert r.query(mat[3], k=1)[0][0]["spectrum_id"] == "s3"
