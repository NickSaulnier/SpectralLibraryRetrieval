"""Masked-band pretraining on real pixels (PLAN.md, Phase 5b).

About 40% of each pixel's tokens are hidden (a few contiguous chunks plus scattered bands).
Hidden tokens keep their wavelength and FWHM but lose their values; a small decoder predicts
the hidden [relative reflectance, continuum-removed value] from the encoder's token states.
The loss is mean squared error on hidden tokens only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from specret.learn.model import BandDecoder, EncoderConfig, SpectralEncoder, make_tokens

PRETRAIN_SEED = 13_2026


def hide_mask(valid: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """(B, T) True for tokens to hide: 2-4 contiguous chunks, topped up with random bands."""
    B, T = valid.shape
    hidden = np.zeros((B, T), dtype=bool)
    for b in range(B):
        idx = np.flatnonzero(valid[b])
        if idx.size < 10:
            continue
        target = int(ratio * idx.size)
        n_chunks = rng.integers(2, 5)
        for _ in range(n_chunks):
            w = max(1, target // (2 * n_chunks))
            s = rng.integers(0, max(idx.size - w, 1))
            hidden[b, idx[s : s + w]] = True
        short = target - hidden[b].sum()
        if short > 0:
            rest = idx[~hidden[b, idx]]
            hidden[b, rng.choice(rest, min(short, rest.size), replace=False)] = True
    return hidden


def masked_loss(
    model: SpectralEncoder,
    decoder: BandDecoder,
    tokens: torch.Tensor,
    pad: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    h = model.encode(tokens, pad, hidden)[:, 1:]
    pred = decoder(h).float()
    target = tokens[..., :2].float()
    err = ((pred - target) ** 2).sum(-1)
    return err[hidden].mean()


def pixel_batch(
    pixels: np.ndarray,
    centers: np.ndarray,
    fwhm: np.ndarray,
    good: np.ndarray,
    batch: int,
    ratio: float,
    rng: np.random.Generator,
    device: str,
):
    """Random real pixels -> (tokens, pad, hidden) tensors, with mild brightness jitter."""
    x = pixels[rng.integers(len(pixels), size=batch)].astype(np.float64)
    x = x * rng.uniform(0.8, 1.2, (batch, 1))
    tok, pad = make_tokens(x, centers, fwhm, good)
    hidden = hide_mask(~pad, ratio, rng)
    to = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return to(tok), to(pad), to(hidden)


def pretrain(
    pixels: np.ndarray,
    centers: np.ndarray,
    fwhm: np.ndarray,
    good: np.ndarray,
    out_path: str | Path,
    steps: int = 20000,
    batch: int = 512,
    ratio: float = 0.4,
    lr: float = 5e-4,
    cfg: EncoderConfig | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log=print,
) -> dict:
    rng = np.random.default_rng(PRETRAIN_SEED)
    torch.manual_seed(PRETRAIN_SEED)
    held = rng.random(len(pixels)) < 0.02  # held-out pixels for a reconstruction check
    train_px, val_px = pixels[~held], pixels[held]
    model = SpectralEncoder(cfg or EncoderConfig()).to(device)
    decoder = BandDecoder(model.cfg.d_model).to(device)
    params = [*model.parameters(), *decoder.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        model.train()
        decoder.train()
        tok, pad, hidden = pixel_batch(train_px, centers, fwhm, good, batch, ratio, rng, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = masked_loss(model, decoder, tok, pad, hidden)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            model.eval()
            decoder.eval()
            with torch.no_grad():
                vt, vp, vh = pixel_batch(val_px, centers, fwhm, good, 2048, ratio, rng, device)
                val = float(masked_loss(model, decoder, vt, vp, vh))
            history.append({"step": step, "train": float(loss), "val": val})
            log(
                f"step {step:6d} masked MSE train {float(loss):.4f} val {val:.4f} "
                f"({time.time() - t0:.0f} s)"
            )
    torch.save(
        {
            "config": model.cfg.__dict__,
            "state": model.state_dict(),
            "decoder": decoder.state_dict(),
            "pretrain": {"steps": steps, "ratio": ratio, "n_pixels": int(len(pixels))},
        },
        out_path,
    )
    return {"history": history, "n_pixels": int(len(pixels))}
