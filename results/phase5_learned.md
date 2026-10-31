# Phase 5 · Learned sensor-agnostic encoder (not adopted)

**Date:** 2026-10-31
**Ship rule (PLAN.md):** adopt the encoder only if it improves real-scene results over the Phase 2 pipeline. Synthetic gains alone don't count. **It did not, so the default retriever stays `hybrid`.**

## What was built (`learn/`, `search/learned.py`)

- **Tokens:** one per valid band:
  - reflectance divided by the spectrum's mean, so brightness drops out;
  - its continuum-removed value;
  - a sinusoidal encoding of the band's centre wavelength;
  - its FWHM.

  Missing bands are absent tokens, so one model takes EMIT (233 good bands), AVIRIS (184) or field spectra.
- **Model:** a 4-layer transformer (d = 128, 4 heads) with a CLS token and a 128-d unit embedding, about 0.6M parameters.
- **Training** (`specret train-encoder`):
  - data: frozen-split train spectra only;
  - two views per anchor, each on a randomly chosen sensor (EMIT or AVIRIS-Classic);
  - augmentations: brightness and slope, areal mixing with another group, parametric atmospheric residuals, sensor-like noise, band dropout;
  - loss: SupCon over groups plus 0.5 × SupCon over material names;
  - group-balanced batches of 256; 4,000 steps in about 5 min on the RTX 4090;
  - checkpoint chosen on the val split. The test split was never used.
- **Integration:** `learned` works as a retriever on its own, or as a fourth vector in fusion: `rrf+learned` and `hybrid+learned` in `benchmark` and `compare-l2bmin`.

## Results

### Synthetic benchmark (test split, EMIT bands, n = 131), group top-1

| Recipe | `sam` | `hybrid` | **`learned`** | `hybrid+learned` |
|---|---:|---:|---:|---:|
| clean | 0.618 | 0.740 | **0.794** | 0.763 |
| noise | 0.618 | 0.695 | **0.771** | 0.748 |
| mix | 0.412 | 0.634 | 0.603 | **0.679** |
| atm-strong | 0.611 | 0.740 | 0.756 | **0.771** |
| jitter | 0.618 | 0.748 | **0.802** | 0.779 |
| all | 0.389 | 0.580 | 0.626 | **0.641** |

### Cross-sensor (the plan's test): train on AVIRIS-Classic bands only, test on EMIT

| Encoder | EMIT clean | EMIT all | AVIRIS clean | AVIRIS all |
|---|---:|---:|---:|---:|
| Trained on AVIRIS only | 0.779 | 0.634 | 0.786 | 0.595 |
| Trained on EMIT + AVIRIS | 0.794 | 0.626 | 0.786 | 0.603 |

There is no accuracy drop when changing sensors. The wavelength-tokenized design is sensor-agnostic as intended.

### Real scene: agreement with EMIT L2B MIN over Cuprite (34,366 pixels, chapters M/S/C)

| Method | Group top-1 | Group top-5 | Class top-1 | Kaolin | Alunite | Calcite |
|---|---:|---:|---:|---:|---:|---:|
| `sam` | 19.1% | 44.7% | 28.3% | 48% | 20% | 0% |
| **`hybrid`** | **55.2%** | 78.9% | **66.8%** | 11% | **98%** | **44%** |
| `learned` | 14.1% | 21.4% | 16.7% | 15% | 32% | 1% |
| `hybrid+learned` | 49.5% | **81.5%** | 52.7% | **32%** | 55% | 36% |

## Findings

1. **A large synthetic-to-real gap.** The encoder is the best method on the synthetic test split, but the worst on the real scene: 14% agreement, below plain SAM. On real pixels it collapses towards generic rock/soil spectra, its most common top-1 for calcite, white mica, kaolin and smectite pixels. This is the failure PLAN.md Decision 6 warned about: the simulator is too clean and too unlike real EMIT pixels.
2. **Fusing it into the hybrid helps kaolin (11% → 32%) and top-5 recall (78.9% → 81.5%),** but costs alunite (98% → 55%) and overall top-1 (55.2% → 49.5%).
3. **Sensor transfer is solved; domain transfer is not.** The failure is synthetic library spectra against real pixels, not one sensor against another.

## What would close the gap (not done)

- **Self-supervised pretraining on real EMIT pixels** (masked-band reconstruction), as planned. It needs scenes other than the Cuprite evaluation scene, to avoid pretraining on the test scene. That means downloading a few more EMIT L2A granules (~1.8 GB each).
- **The radiative-transfer residual bank** (ISOFIT/sRTMnet) instead of parametric atmospheric bumps.
- **Training on mixtures with real-pixel endmembers**, such as vegetation and soil spectra drawn from scenes.

## Reproduce

```bash
uv run specret train-encoder --steps 4000 --out models/encoder_v1.pt
uv run specret benchmark data/resampled/emit/library.npz --methods sam hybrid learned hybrid+learned
uv run specret compare-l2bmin data/resampled/emit/library.npz --rfl <...> --min <...> \
  --chapters M S C --vectors hybrid learned hybrid+learned
```

Model weights are not committed (`models/` is git-ignored); training takes about 5 minutes.
