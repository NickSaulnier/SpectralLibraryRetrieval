# Phase 2a · Taxonomy, frozen splits, and continuum removal on EMIT Cuprite

**Date:** 2026-10-31
**Scene and region:** as in [phase1_emit_cuprite.md](phase1_emit_cuprite.md): EMIT 2024-07-27, Cuprite box, 34,366 pixels where EMIT L2B MIN reports a ~2 µm mineral with band depth ≥ 0.05.

## What changed

- **Taxonomy** (`configs/taxonomy.yaml`): 43 groups in 23 classes. Every splib07a spectrum is classified, and so is every EMIT L2B mineral name except one atmospheric transmission spectrum. Calcite/dolomite and hematite/goethite are separate groups (plan hard pairs). Groups are now finer than the Phase 1 stopgap, which lumped all carbonates together, so the SAM row moves slightly.
- **Frozen split** (`configs/splits/splits_v1.csv` plus its hash): 1,497 mineral, mixture and coating spectra from 942 physical samples. Split by sample: 595 train, 128 val, 128 test, and 91 open-set samples (zeolite, borate, serpentine, scapolite, other sulfate). No sample spans two splits.
- **Continuum removal** (`preprocess/continuum.py`): convex-hull continuum over valid bands. The matching vector is band depth (1 − continuum-removed) with L2 normalisation, so cosine compares absorption shape and position, not brightness or slope. Two new vectors:
  - `cr`: hull over all good bands;
  - `cr_swir2`: local hull fitted inside 2.0–2.5 µm.

## Results (agreement with EMIT L2B MIN, ~2 µm layer)

Library restricted to minerals, mixtures and coatings (`--chapters M S C`). Using the full library costs 1–2 points on every row.

| Vector | Group top-1 | Group top-5 | Class top-1 |
|---|---:|---:|---:|
| `sam` (baseline) | 19.1% | 44.7% | 28.3% |
| `swir2` | 20.2% | 59.8% | 20.3% |
| `cr` | 36.7% | 65.2% | 42.3% |
| **`cr_swir2`** | **47.0%** | **73.4%** | **63.4%** |

By L2B group, top-1 group agreement:

| L2B group | Pixels | `sam` | `cr` | `cr_swir2` |
|---|---:|---:|---:|---:|
| calcite | 14,780 | 0.0% | 10.9% | 44.2% |
| white mica | 7,561 | 35.4% | 90.6% | 73.3% |
| kaolin | 6,692 | 48.4% | 0.7% | 4.3% |
| alunite | 3,013 | 20.4% | 99.0% | 98.4% |
| silica | 1,483 | 0.0% | 56.2% | 52.2% |
| smectite | 530 | 4.9% | 50.6% | 0.0% |

L2B MIN is Tetracorder output, so these numbers measure agreement with Tetracorder, not accuracy.

```bash
uv run specret compare-l2bmin data/resampled/emit/library.npz \
  --rfl data/scenes/emit/EMIT_L2A_RFL_001_20240727T213005_2420914_007.nc \
  --min data/scenes/emit/EMIT_L2B_MIN_001_20240727T213005_2420914_007.nc --chapters M S C
```

## Findings

1. **Continuum removal is the big gain.** It takes group agreement from 19% to 47%, and alunite from 20% to 98%. Brightness and slope no longer drive the match.
2. **The local 2.0–2.5 µm continuum works best overall**, mostly through calcite (0% → 44%). Full-range `cr` is better for white mica and smectite. Fusing the two rankings (RRF, Phase 2c) is the obvious next step.
3. **Kaolin collapses under continuum removal (48% → 4%).** Most L2B "kaolin" pixels are kaolinite–muscovite mixtures (`Kaol+Musc_intimate`, `Kaol_Wxl+0.5Musc`). Our top matches are `Alunite.5+Musc`, lepidolite and illite: mica-dominated spectra whose 2.2 µm features overlap. That's a mixture problem. The 2.165/2.205 µm doublet fitting in SFF (Phase 2c) and the unmixing in Phase 4 are the fixes.
4. **Calcite pixels often match bytownite** (4,636 pixels). A plagioclase has no strong 2.34 µm feature, so that USGS bytownite sample probably carries a carbonate or alteration impurity. splib07's purity notes (HTML, not yet ingested) would let us flag such samples.
5. Restricting to minerals, mixtures and coatings helps a little (+1–2 points). It removes vegetation and man-made matches.
6. Smectite is small (530 pixels) and swings between vectors. Treat it as noise for now.

## Next (Phase 2b/2c)

- Rank fusion (RRF) of `cr` and `cr_swir2`, plus `sam`.
- SFF reranking of the top-k on diagnostic features (kaolinite doublet, 2.165/2.32 µm alunite, 2.20 µm Al-OH, 2.34 µm carbonate).
- Ingest the splib07 purity notes and flag impure samples.
- The synthetic distortion benchmark on the frozen test split.
