# Phase 1 · SAM retrieval vs. the EMIT mineral map over Cuprite

**Date:** 2026-10-31
**Scene:** `EMIT_L2A_RFL_001_20240727T213005_2420914_007` (3% cloud) and the matching `EMIT_L2B_MIN_001_…` granule
**Library:** USGS splib07a (2,456 spectra, all chapters) resampled to this scene's 285 bands, 233 of them good; 1,397 spectra have full coverage
**Region:** Cuprite box (W −117.30, S 37.45, E −117.10, N 37.62)
**Pixels:** 34,366 where L2B MIN reports a ~2 µm mineral with band depth ≥ 0.05
**Comparison:** the coarse mineral group of our top-1 (and top-5) match against the group of L2B MIN's group-2 mineral. Groups come from `specret.eval.emit_l2bmin.COARSE_GROUPS`, a stopgap until the Phase 2 taxonomy.

```bash
uv run specret compare-l2bmin data/resampled/emit/library.npz \
  --rfl data/scenes/emit/EMIT_L2A_RFL_001_20240727T213005_2420914_007.nc \
  --min data/scenes/emit/EMIT_L2B_MIN_001_20240727T213005_2420914_007.nc
```

## Results

L2B MIN is Tetracorder output on splib06, so these numbers measure agreement with Tetracorder, not accuracy.

| L2B group | Pixels | SAM top-1 | SAM top-5 | SWIR2 top-1 | SWIR2 top-5 |
|---|---:|---:|---:|---:|---:|
| carbonate | 14,780 | 0.0% | 0.0% | 0.9% | 13.4% |
| white mica | 7,560 | 34.8% | 78.2% | 5.3% | 51.1% |
| kaolin | 6,692 | 44.0% | 89.2% | 7.9% | 40.8% |
| alunite | 2,259 | 14.4% | 33.6% | 24.2% | 42.6% |
| silica | 1,483 | 0.0% | 0.1% | 64.0% | 93.4% |
| smectite | 530 | 4.3% | 83.8% | 0.0% | 0.8% |
| **all** | 34,366 | **18.1%** | **41.0%** | **8.3%** | **33.3%** |

## Failure modes

1. **Plain SAM matches overall shape, not diagnostic features.** The most frequent top-1 matches for carbonate and white-mica pixels are richterite, andradite, a kaolin–smectite mixture and juniper (*J. roemeri*). Their broad continuum and Fe-oxide slope resemble the pixel's, but their 2 µm absorptions don't. The 2.34 µm carbonate feature is a few percent deep and contributes almost nothing to the angle.
2. **Restricting to 2.0–2.5 µm (`swir2`) on raw reflectance doesn't fix this.** Within the window the continuum slope still dominates. Spectra with similar slopes but the wrong features win: albite, pectolite, cummingtonite, rangeland. Silica is the exception (64% top-1, with chert matching), because its broad 2.2 µm Si-OH shape is itself a slope-like feature.
3. **Candidates are often right in the top 5 but not ranked first** (kaolin 89%, white mica 78%, smectite 84% top-5 with full-range SAM). A reranker over the top-k can recover these.
4. **Library composition matters.** Vegetation and soil/mixture spectra (chapters V and S) compete with minerals. A payload filter (minerals only, or minerals plus mixtures) is a cheap test.

## What this means for Phase 2

- Continuum removal before matching is the expected first big gain, because it isolates absorption depth and shape. Apply it to the `swir2` window first.
- Feature fitting on the top-k (SFF) is supported by result 3.
- Re-run this command after each Phase 2 change. Keep the SAM row in every table.

Per-pixel results are written with `--out outputs/phase1` (not committed).
