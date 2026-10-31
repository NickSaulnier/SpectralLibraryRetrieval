"""Sensor-agnostic spectral encoder (PLAN.md, Phase 5).

Each valid band becomes one token built from
* its reflectance divided by the spectrum's mean (brightness-free),
* its continuum-removed value (absorption depth, the physics prior),
* a sinusoidal encoding of its centre wavelength, and
* its FWHM,
so one model embeds EMIT, AVIRIS or field-spectrometer spectra without retraining. Missing
bands (water bands, dropouts, other sensors' gaps) are simply absent tokens.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from specret.preprocess.continuum import continuum_removed


@dataclass
class EncoderConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    embed_dim: int = 128
    n_freq: int = 16  # wavelength encoding frequencies


def wavelength_encoding(wl_um: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Sinusoids of wavelength (um) at geometric frequencies; (..., 2 * n_freq)."""
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(400.0), n_freq, device=wl_um.device))
    ang = wl_um[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class SpectralEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or EncoderConfig()
        self.value_proj = nn.Linear(2, cfg.d_model)
        self.pos_proj = nn.Sequential(
            nn.Linear(2 * cfg.n_freq + 1, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.embed_dim)
        )

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """tokens (B, T, 4) = [rel. reflectance, CR value, wavelength um, fwhm um]; pad (B, T)
        True where a token is absent. Returns unit-length embeddings (B, embed_dim)."""
        val = self.value_proj(tokens[..., :2])
        pos = self.pos_proj(
            torch.cat([wavelength_encoding(tokens[..., 2], self.cfg.n_freq), tokens[..., 3:4]], -1)
        )
        x = torch.cat([self.cls.expand(len(tokens), -1, -1), val + pos], dim=1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)
        h = self.encoder(x, src_key_padding_mask=mask)
        z = self.head(self.norm(h[:, 0]))
        return nn.functional.normalize(z, dim=-1)


def make_tokens(
    spectra: np.ndarray,
    centers_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Token arrays for spectra on one band set. Invalid (NaN or masked) bands become padding.

    Returns tokens (B, T, 4) float32 and pad (B, T) bool.
    """
    S = np.atleast_2d(np.asarray(spectra, dtype=np.float64)).copy()
    ok = np.isfinite(S) & (S > -0.5)
    if valid is not None:
        ok &= valid
    S[~ok] = np.nan
    mean = np.nanmean(np.where(ok, S, np.nan), axis=1, keepdims=True)
    rel = S / np.where(np.isfinite(mean) & (mean > 1e-6), mean, 1.0)
    cr = continuum_removed(centers_nm, S)
    ok &= np.isfinite(rel) & np.isfinite(cr)
    B, T = S.shape
    tokens = np.zeros((B, T, 4), dtype=np.float32)
    tokens[..., 0] = np.where(ok, rel, 0.0)
    tokens[..., 1] = np.where(ok, cr, 0.0)
    tokens[..., 2] = centers_nm / 1000.0
    tokens[..., 3] = fwhm_nm / 1000.0
    return tokens, ~ok


def save_encoder(model: SpectralEncoder, path, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state": model.state_dict(), **(extra or {})}, path)


def load_encoder(path, device: str = "cpu") -> SpectralEncoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SpectralEncoder(EncoderConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()


@torch.no_grad()
def embed(
    model: SpectralEncoder,
    spectra: np.ndarray,
    centers_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    valid: np.ndarray | None = None,
    batch: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings (n, embed_dim) and a validity flag (spectra with < 10 usable bands fail)."""
    device = next(model.parameters()).device
    out = np.zeros((len(spectra), model.cfg.embed_dim), dtype=np.float32)
    ok = np.zeros(len(spectra), dtype=bool)
    for start in range(0, len(spectra), batch):
        tok, pad = make_tokens(spectra[start : start + batch], centers_nm, fwhm_nm, valid)
        good = (~pad).sum(axis=1) >= 10
        if not good.any():
            continue
        z = model(
            torch.as_tensor(tok[good], device=device), torch.as_tensor(pad[good], device=device)
        )
        idx = np.flatnonzero(good) + start
        out[idx] = z.float().cpu().numpy()
        ok[idx] = True
    return out, ok
