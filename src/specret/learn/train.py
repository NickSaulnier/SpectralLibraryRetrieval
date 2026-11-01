"""Supervised contrastive training of the spectral encoder (PLAN.md, Phase 5).

* Data: frozen-split **train** spectra, already resampled to one or more sensors. Spectra with
  partial coverage are kept; their missing bands are simply absent tokens.
* Views: each anchor gets two independently distorted views, each on a randomly chosen
  sensor, so the same material on EMIT and AVIRIS bands is a positive pair.
* Distortions: brightness and slope, areal mixing with another group (dominant 0.6-0.95),
  parametric atmospheric residuals, sensor-like noise, and band dropout. Seeds are separate
  from the benchmark's.
* Loss: SupCon over groups plus 0.5 x SupCon over material names, so two kaolinite samples sit
  closer together than kaolinite and dickite, which sit closer than kaolinite and calcite.
* Model selection: group top-1 on the val split (library = train) under "clean" and "all"
  recipes. The test split is never touched.

Phase 5b options (all off by default):
* `init`: start from a masked-band pretrained checkpoint (learn/pretrain.py).
* `real_pixels` + `real_mix`: with probability `real_mix`, an EMIT view's mixing partner is a
  real scene pixel instead of a library spectrum, so the embedding learns to look past real
  soil and vegetation backgrounds.
* `aux_weight`: keep the masked-band loss on real pixels as an auxiliary term.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from specret.distort.simulator import ATM_FEATURES_NM
from specret.learn.model import (
    BandDecoder,
    EncoderConfig,
    SpectralEncoder,
    embed,
    load_checkpoint,
    make_tokens,
    save_encoder,
)
from specret.sensors.library import SensorLibrary

TRAIN_SEED = 11_2026  # distinct from the benchmark and mixture seeds


@dataclass
class TrainConfig:
    sensors: tuple[str, ...] = ("emit", "avirisc")
    steps: int = 3000
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    temperature: float = 0.1
    material_weight: float = 0.5
    eval_every: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = TRAIN_SEED
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    init: str | None = None
    real_pixels: np.ndarray | None = field(default=None, repr=False)  # on EMIT bands
    real_mix: float = 0.0
    aux_weight: float = 0.0
    aux_batch: int = 256


@dataclass
class SensorData:
    lib: SensorLibrary
    row_of: dict[str, int]
    noise: np.ndarray | None


def _sensor_data(lib: SensorLibrary, ids: list[str], noise: np.ndarray | None) -> SensorData:
    row_of = {sid: i for i, sid in enumerate(lib.spectrum_ids)}
    keep = [row_of[s] for s in ids]
    sub = SensorLibrary(lib.sensor, lib.spectrum_ids[keep], lib.matrix[keep])
    return SensorData(sub, {sid: i for i, sid in enumerate(sub.spectrum_ids)}, noise)


def augment(
    x: np.ndarray,
    wl: np.ndarray,
    good: np.ndarray,
    partners: np.ndarray,
    noise: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Distort a batch (B, n_bands) on one sensor; returns spectra and a per-band keep mask."""
    B, n = x.shape
    t = (wl - 1500.0) / 1000.0
    x = x * rng.uniform(0.5, 1.5, (B, 1)) * (1 + rng.uniform(-0.1, 0.1, (B, 1)) * t)
    mix = rng.random(B) < 0.5
    f = rng.uniform(0.6, 0.95, (B, 1))
    p = np.where(np.isfinite(partners), partners, x)
    x = np.where(mix[:, None], f * x + (1 - f) * p, x)
    factor = np.ones((B, n))
    strength = rng.uniform(0, 0.05, (B, 1))
    for c in ATM_FEATURES_NM:
        amp = rng.normal(0, 1, (B, 1)) * strength
        width = rng.uniform(15, 40, (B, 1))
        factor += amp * np.exp(-0.5 * ((wl - c) / width) ** 2)
    x = x * factor
    sigma = noise if noise is not None else 0.005
    x = x + rng.normal(0, 1, (B, n)) * sigma * rng.uniform(0.5, 3.0, (B, 1))
    keep = np.tile(good, (B, 1))
    for b in np.flatnonzero(rng.random(B) < 0.5):  # contiguous dropout window
        w = int(n * rng.uniform(0.05, 0.25))
        s = rng.integers(0, max(n - w, 1))
        keep[b, s : s + w] = False
    keep &= rng.random((B, n)) > rng.uniform(0, 0.1, (B, 1))
    return x, keep


def supcon(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) over a batch of unit embeddings."""
    sim = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    n_pos = pos.sum(1)
    loss = -(log_prob * pos).sum(1) / n_pos.clamp(min=1)
    return loss[n_pos > 0].mean()


def group_top1(
    model: SpectralEncoder,
    ref: SensorLibrary,
    ref_groups: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
) -> float:
    s = ref.sensor
    zr, okr = embed(model, ref.matrix, s.centers_nm, s.fwhm_nm, s.good_mask)
    zq, okq = embed(model, queries, s.centers_nm, s.fwhm_nm, s.good_mask)
    sim = zq @ zr[okr].T
    pred = ref_groups[okr][np.argmax(sim, axis=1)]
    return float(np.mean((pred == truth) & okq))


def train(
    libs: dict[str, SensorLibrary],
    canonical: pd.DataFrame,
    splits: pd.DataFrame,
    cfg: TrainConfig,
    out_path: str | Path,
    noise_curves: dict[str, np.ndarray] | None = None,
    log=print,
) -> dict:
    """Train, select on val, save the best checkpoint; returns the training log."""
    from specret.eval.benchmark import make_queries

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    meta = canonical.set_index("spectrum_id")
    train_ids = splits.loc[splits["split"] == "train", "spectrum_id"].tolist()
    # Keep spectra with at least half their good bands on every training sensor.
    usable = []
    for sid in train_ids:
        ok = True
        for name in cfg.sensors:
            lib = libs[name]
            row = np.flatnonzero(lib.spectrum_ids == sid)
            if row.size == 0 or np.isfinite(lib.matrix[row[0], lib.sensor.good_mask]).mean() < 0.5:
                ok = False
        if ok:
            usable.append(sid)
    data = {n: _sensor_data(libs[n], usable, (noise_curves or {}).get(n)) for n in cfg.sensors}
    groups = meta.loc[usable, "group"].to_numpy()
    materials = meta.loc[usable, "material_name"].str.lower().str.split().str[0].to_numpy()
    g_codes = pd.factorize(groups)[0]
    m_codes = pd.factorize(materials)[0]
    by_group = [np.flatnonzero(g_codes == g) for g in range(g_codes.max() + 1)]

    # Validation: val queries against a train-only library, on the first sensor's bands.
    val_sensor = cfg.sensors[0]
    val_sets = {}
    for recipe in ("clean", "all"):
        q, truth, ref = make_queries(
            libs[val_sensor],
            canonical,
            splits,
            recipe,
            data[val_sensor].noise
            if data[val_sensor].noise is not None
            else np.full(libs[val_sensor].sensor.n_bands, 0.005),
            split="val",
            ref_splits=("train",),
        )
        val_sets[recipe] = (q, truth["group"].to_numpy())
    ref_groups = meta.loc[ref.spectrum_ids, "group"].to_numpy()

    model = SpectralEncoder(cfg.encoder).to(cfg.device)
    decoder = None
    if cfg.init:
        ckpt = load_checkpoint(cfg.init, cfg.device)
        model = SpectralEncoder(EncoderConfig(**ckpt["config"])).to(cfg.device)
        model.load_state_dict(ckpt["state"])
        if cfg.aux_weight > 0 and "decoder" in ckpt:
            decoder = BandDecoder(model.cfg.d_model).to(cfg.device)
            decoder.load_state_dict(ckpt["decoder"])
    if cfg.aux_weight > 0 and decoder is None:
        decoder = BandDecoder(model.cfg.d_model).to(cfg.device)
    params = [*model.parameters(), *(decoder.parameters() if decoder is not None else [])]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    real = cfg.real_pixels.astype(np.float64) if cfg.real_pixels is not None else None
    if real is not None and "emit" in data:
        if real.shape[1] != data["emit"].lib.sensor.n_bands:
            raise ValueError("real pixels must be on the EMIT library's bands")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    history, best, t0 = [], -1.0, time.time()

    def view(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        name = cfg.sensors[rng.integers(len(cfg.sensors))]
        d = data[name]
        s = d.lib.sensor
        x = d.lib.matrix[idx]
        partner_idx = np.array([rng.choice(np.flatnonzero(g_codes != g_codes[i])) for i in idx])
        partners = d.lib.matrix[partner_idx]
        if real is not None and name == "emit" and cfg.real_mix > 0:
            use_real = rng.random(len(idx)) < cfg.real_mix
            partners = partners.copy()
            partners[use_real] = real[rng.integers(len(real), size=int(use_real.sum()))]
        x, keep = augment(x, s.centers_nm, s.good_mask, partners, d.noise, rng)
        tok, pad = make_tokens(x, s.centers_nm, s.fwhm_nm, keep)
        return torch.as_tensor(tok, device=cfg.device), torch.as_tensor(pad, device=cfg.device)

    for step in range(1, cfg.steps + 1):
        model.train()
        gsel = rng.integers(len(by_group), size=cfg.batch)  # group-balanced anchors
        idx = np.array([rng.choice(by_group[g]) for g in gsel])
        (t1, p1), (t2, p2) = view(idx), view(idx)
        # Different sensors have different band counts: pad to the longer view.
        T = max(t1.shape[1], t2.shape[1])
        t1, t2 = (torch.nn.functional.pad(t, (0, 0, 0, T - t.shape[1])) for t in (t1, t2))
        p1, p2 = (torch.nn.functional.pad(p, (0, T - p.shape[1]), value=True) for p in (p1, p2))
        tokens = torch.cat([t1, t2])
        pad = torch.cat([p1, p2])
        with torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.device == "cuda"):
            z = model(tokens, pad)
        z = torch.nn.functional.normalize(z.float(), dim=-1)
        gl = torch.as_tensor(np.tile(g_codes[idx], 2), device=cfg.device)
        ml = torch.as_tensor(np.tile(m_codes[idx], 2), device=cfg.device)
        loss = supcon(z, gl, cfg.temperature) + cfg.material_weight * supcon(z, ml, cfg.temperature)
        if decoder is not None and real is not None:
            from specret.learn.pretrain import masked_loss, pixel_batch

            es = data["emit"].lib.sensor
            pt, pp, ph = pixel_batch(
                real, es.centers_nm, es.fwhm_nm, es.good_mask, cfg.aux_batch, 0.4, rng, cfg.device
            )
            with torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.device == "cuda"):
                aux = masked_loss(model, decoder, pt, pp, ph)
            loss = loss + cfg.aux_weight * aux.float()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            scores = {r: group_top1(model, ref, ref_groups, q, t) for r, (q, t) in val_sets.items()}
            mean = float(np.mean(list(scores.values())))
            history.append(
                {"step": step, "loss": float(loss), **scores, "seconds": time.time() - t0}
            )
            log(
                f"step {step:5d} loss {float(loss):.3f} val clean {scores['clean']:.3f} "
                f"all {scores['all']:.3f} ({time.time() - t0:.0f} s)"
            )
            if mean > best:
                best = mean
                save_encoder(
                    model,
                    out_path,
                    {
                        "train_config": {
                            "sensors": list(cfg.sensors),
                            "steps": cfg.steps,
                            "best_step": step,
                            "val": scores,
                        }
                    },
                )
    return {"history": history, "best_val_mean": best, "n_train_spectra": len(usable)}
