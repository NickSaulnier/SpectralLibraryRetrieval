# Phase 5b · Real-pixel pretraining for the encoder (not adopted)

**Date:** 2026-10-31
**Question:** does exposing the Phase 5 encoder to real EMIT pixels close its synthetic-to-real gap, enough for `hybrid+learned` to beat the hybrid retriever on real scenes?
**Answer:** no. Each step helps a little, but the best variant still trails the hybrid by about 12 points on the validation scene. No variant passed validation, so the Cuprite test scene was not used. The default retriever stays `hybrid`.

## Data

| Role | Scene | Use |
|---|---|---|
| Pretraining | Pilbara (2024-07-16), Kalahari (2023-05-24), Sar Cheshmeh (2023-06-26) | 599,500 clear pixels (EMIT aggregate mask), spatially stratified (`specret sample-pixels`) |
| Validation | Mountain Pass, California (2024-04-16) | 20,000 random pixels with an L2B MIN ~2 µm detection (depth ≥ 0.05) across the whole scene |
| Test | Cuprite | Not used: no variant passed validation |

EMIT band centres are identical in all five granules, so the existing EMIT-resampled library serves every scene.

## A tokenizer bug found on the way

The first pretraining run had erratic validation loss (0.0014 → 0.24 → 0.13). The cause was real pixels with slightly negative or near-zero reflectance at noisy bands: continuum removal divided by ~0 and produced token values up to ±464. Library spectra almost never hit this, so it also affected Phase 5 inference on real pixels.

**The fix (`learn/model.make_tokens`):** bands at or below 0.002 reflectance are dropped, relative reflectance is clipped to [0, 5], and continuum-removed values to [0, 1]. After the fix, pretraining converged smoothly to a masked-band MSE of 0.0001 on held-out pixels (20,000 steps, 23 min). The Phase 5 baseline was retrained under the fixed tokenizer (`v1b`), so pretraining's effect could be separated from the fix's.

## Results on the validation scene (Mountain Pass)

Library: minerals, mixtures and coatings. Group / class top-1 agreement with EMIT L2B MIN.

| Method | Group top-1 | Class top-1 | Group top-5 | Synthetic val (mean of clean, all) |
|---|---:|---:|---:|---:|
| `sam` | 4.3% | 19.5% | 29.2% | — |
| **`hybrid`** | **39.8%** | **53.0%** | **63.8%** | — |
| `learned` v1 (Phase 5) | 1.7% | 18.3% | 3.7% | 0.707 |
| `hybrid+learned` v1 | 25.1% | 46.5% | 52.6% | — |
| `learned` v1b (fixed tokenizer) | 2.8% | 27.8% | 12.9% | 0.694 |
| `hybrid+learned` v1b | 22.7% | 47.2% | 47.6% | — |
| `learned`, + pretraining | 3.5% | 22.0% | 4.9% | 0.716 |
| `hybrid+learned`, + pretraining | 24.4% | 48.1% | 50.3% | — |
| `learned`, + pretraining + real-pixel mixing | 5.5% | 20.7% | 15.1% | 0.716 |
| `hybrid+learned`, + pretraining + real-pixel mixing | **28.1%** | 44.8% | 51.9% | — |
| + auxiliary masked loss | not finished | | | |

- **Each step helps a little:** pretraining adds about 2 points to `hybrid+learned`, and real-pixel mixing about 4 more.
- **None comes close to the hybrid alone.** The encoder on its own still collapses real pixels onto generic rock/soil spectra.
- **Synthetic validation barely moves** (0.69–0.72). It can't see the real-data gap at all, which confirms the plan's rule that real scenes decide.
- **The auxiliary-loss variant was stopped.** It stalled after its first checkpoint once GPU memory filled (23.8 of 24.5 GB) and made no progress for ~45 minutes. With a 12-point gap to close, it was not retried.

## Why it still fails

Exposure to real pixels changes the representation, but not what the encoder is trained to output. Its labels come from pure library spectra. Real pixels are mixtures of minerals, soil, vegetation and residual atmosphere, so the nearest pure spectrum in embedding space is a generic rock/soil sample. The physics pipeline avoids this by scoring specific absorption features, which survive mixing. Closing the gap probably means training against real mixtures directly:
- **Weak labels from real scenes,** for example L2B MIN or field-checked maps. L2B MIN would make Cuprite agreement circular.
- **A mixture-aware objective,** for example predicting abundances rather than a single identity.

Either is a redesign beyond this phase.

## Kept from this phase

- **Tokenizer fix:** it applies to any future learned model.
- **Real-pixel tooling:** `specret sample-pixels` (EMIT aggregate-mask filtering and stratified sampling) and `specret pretrain-encoder` (masked-band pretraining).
- **Fine-tuning options:** `train-encoder --init`, `--real-pixels`, `--real-mix` and `--aux-weight`.
- **Whole-scene validation:** `compare-l2bmin --whole-scene --max-pixels --encoder`.
- **The ablation runner:** `scripts/run_phase5b_ablation.py`.

## Data note

`scripts/download_phase5b.py` originally matched folder names in the download links, so it also fetched the four reflectance-uncertainty granules and an extra mask: 7.4 GB beyond the planned ~7.6 GB. The filter now matches file names. The extra files are in `data/scenes/emit_extra/` and can provide per-scene noise curves.

## Reproduce

```bash
uv run python scripts/download_phase5b.py
uv run specret sample-pixels --rfl <RFL .nc> --mask <MASK .nc> --out data/derived/pretrain/<site>.npz
uv run specret pretrain-encoder data/derived/pretrain/*.npz --out models/pretrained_v1.pt
uv run specret train-encoder --out models/encoder_v1b.pt
uv run python scripts/run_phase5b_ablation.py
```
