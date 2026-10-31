# Phase 2b/2c · Rank fusion, feature fitting, and the synthetic benchmark

**Date:** 2026-10-31
**Sensor:** EMIT bands from the 2024-07-27 Cuprite scene (285 bands, 233 good).
**Library:** USGS splib07a.

## What was added

- **Rank fusion** (`search/fusion.py`): reciprocal rank fusion (k = 60) of the top 50 from `sam`, `cr` and `cr_swir2`. The best 20 go to reranking.
- **Spectral feature fitting** (`rerank/sff.py`, windows in `configs/features.yaml`): five generic windows (NH₄, Al-OH, Fe-OH, Mg-OH/CO₃, Fe³⁺). Each has a straight-line local continuum, Pearson shape correlation and a sub-band centre match (σ = 8 nm). A feature present on only one side scores zero in that window. Per-window results are kept, so a match can be explained, for example "Mg-OH/CO₃ window: query centre 2338 nm, depth 0.06; candidate 2340 nm; shape r 0.97".
- **Hybrid retriever** (`search/hybrid.py`): fusion, then a blend of feature-fit score and fused score, with `sff_weight` = 0.25 chosen on the validation split (below).
- **Distortion simulator** (`distort/simulator.py`): brightness and slope, areal mixing (dominant fraction 0.6–0.9 with a different-group partner), band noise from EMIT's own uncertainty, parametric atmospheric residuals at the 0.76/0.94/1.14/1.38/1.88 µm gas features plus an aerosol term, and band-centre/FWHM jitter by re-convolving native spectra.
  - The noise curve comes from this scene's RFLUNCERT granule over Cuprite: median 0.0030 reflectance on good bands (`specret noise-curve`).
- **Benchmark** (`specret benchmark`): queries are frozen-split **test** spectra with full EMIT coverage (n = 131). The reference library is train + val only, so a query's own sample is never present. The benchmark seed is fixed and separate from any future training seed.

## Choosing the blend weight (validation split only)

Queries: val spectra (n = 116). Reference: train only. Mean group top-1 over all eight recipes:

| `sff_weight` | 0.0 (fusion only) | **0.25** | 0.5 | 0.75 | 1.0 (feature fit only) |
|---|---:|---:|---:|---:|---:|
| Mean group top-1 | 0.643 | **0.676** | 0.637 | 0.608 | 0.561 |

## Synthetic benchmark (test split, n = 131), group top-1

| Recipe | `sam` | `swir2` | `cr` | `cr_swir2` | `rrf` | **`hybrid`** |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.618 | 0.557 | **0.763** | 0.656 | 0.748 | 0.740 |
| photometric | 0.588 | 0.557 | **0.779** | 0.656 | 0.740 | 0.740 |
| noise | 0.618 | 0.519 | 0.649 | 0.557 | 0.664 | **0.695** |
| mix | 0.412 | 0.405 | 0.595 | 0.458 | 0.595 | **0.634** |
| atm-mild | 0.603 | 0.557 | 0.748 | 0.656 | **0.756** | 0.733 |
| atm-strong | 0.611 | 0.550 | 0.718 | 0.656 | **0.740** | **0.740** |
| jitter | 0.618 | 0.573 | **0.756** | 0.649 | **0.756** | 0.748 |
| all | 0.389 | 0.359 | 0.542 | 0.435 | 0.565 | **0.580** |

Group recall@5 for `hybrid` is 0.73–0.90 (SAM: 0.66–0.82). Exact-material top-1 for `hybrid` is 0.37–0.57 (SAM: 0.21–0.37). The full tables are printed by `specret benchmark`.

## Real scene: agreement with EMIT L2B MIN over Cuprite (34,366 pixels)

Library restricted to minerals, mixtures and coatings.

| Method | Group top-1 | Group top-5 | Class top-1 |
|---|---:|---:|---:|
| `sam` | 19.1% | 44.7% | 28.3% |
| `cr_swir2` | 47.0% | 73.4% | 63.4% |
| `rrf` | 50.1% | 75.2% | 62.6% |
| **`hybrid` (0.25)** | **55.2%** | **78.9%** | **66.8%** |
| `hybrid` (0.5) | 59.9% | – | 69.9% |
| `hybrid` (0.75) | 66.7% | – | 73.1% |
| `hybrid` (1.0) | 65.3% | 84.8% | 69.7% |

By L2B group, `hybrid` (0.25) top-1: alunite 98%, white mica 97%, silica 87%, calcite 44%, smectite 43%, kaolin 11%. Buddingtonite is 86%, but on very few pixels.

## Findings

1. **The hybrid beats SAM everywhere.** Synthetic group top-1 goes from 0.39–0.62 to 0.58–0.75. Real-scene agreement goes from 19% to 55%.
2. **Feature fitting helps most where it should.** It does best on mixtures and noise (+3 to +4 points over fusion alone) and on the Cuprite alteration minerals. On clean spectra of the whole library (pyroxenes, feldspars, garnets), its SWIR-centred windows add little.
3. **The two evaluations disagree on the weight.** The validation benchmark prefers 0.25. Agreement with L2B MIN keeps rising to 0.75, mainly through calcite (44% → 71%). L2B MIN is itself made by Tetracorder's feature fitting, so it rewards a feature-heavy ranking for partly circular reasons (PLAN.md, Decision 7). The default is the validation-tuned 0.25. A higher weight is a reasonable setting for alteration-mineral mapping specifically.
4. **Kaolin is still the weak group** (11% at 0.25, 25% at 0.75). These pixels are kaolinite–muscovite mixtures, and single-spectrum retrieval answers with whichever end member's features dominate. Phase 4 unmixing is the fix.
5. **Synthetic results remain a sanity check.** The parametric atmosphere is gentle: CR-based methods barely notice it. The Phase 5 radiative-transfer residual bank should be harsher.

## Commands

```bash
uv run specret noise-curve --rfl <L2A RFL .nc> --uncert <L2A RFLUNCERT .nc>
uv run specret benchmark data/resampled/emit/library.npz
uv run specret compare-l2bmin data/resampled/emit/library.npz --rfl <...> --min <...> \
  --chapters M S C --vectors sam cr_swir2 rrf hybrid
```

## Still open in Phase 2

- Baselines not yet in the harness: SID and Euclidean distance on continuum-removed spectra.
- Ingesting splib07 purity notes (needs the HTML metadata release).
- The same benchmark on AVIRIS-Classic bands.
