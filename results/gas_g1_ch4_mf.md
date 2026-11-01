# Gas Phase G1 · Methane matched filter on EMIT L1B radiance

**Date:** 2026-11-01
**Question:** does our column-wise matched filter, built from HITRAN alone, reproduce the EMIT L2B CH4ENH v002 enhancement maps?
**Answer:** yes. Over the published plume complexes, r = 0.93–0.97 and the noise-corrected slope is 0.95–1.03. Exit criteria: r ≥ 0.85 and slope within ±15%. Our background noise is about 30% lower than EMIT's, and a full scene takes 11–14 s.

## Method

- **Radiance:** EMIT L1B RAD with OBS geometry. The L2A mask supplies cloud, cirrus and water flags and the water vapour column (`io/emit_l1b.py`).
- **Unit absorption spectrum** (`gas/library.py`, `gas/signatures.py`):
  - HITRAN line-by-line Voigt cross-sections for CH4, H2O and CO2, computed in 16 layers of the US standard atmosphere at 0.01 cm⁻¹. Each range is cached once, taking about 5 min per range.
  - The background transmittance along the two-way path weights the plume cross-section within each Gaussian band response: k_b = AMF·N₁·∫S·T_bg·σ / ∫S·T_bg, in ln(radiance) per ppm·m. This captures the saturation of strong background lines without a radiative-transfer code.
  - Per scene, the inputs are the mean AMF from OBS, the median water vapour from the L2A mask (1.5 g/cm² without a mask) and the median surface elevation. Per column, k is interpolated between five AMFs.
- **Matched filter** (`gas/matched_filter.py`):
  - runs per detector column, on the GPU;
  - covariance shrinks toward its diagonal, with the shrinkage chosen per column by leave-one-out likelihood (Theiler 2012);
  - a second pass drops first-pass pixels more than 3σ above the median from the background statistics.
- **Bands:** the ranges EMIT GHG v002 uses: 500–1340, 1500–1790 and 1950–2450 nm (219 bands). k is computed where HITRAN tables are fetched (1493–2532 nm) and set to 0 below that. CH4 absorption there is negligible, but those bands still describe the background.
- **Screening:** pixels flagged as cloud, cirrus or water by L2A, and gas flares (radiance at 2389 nm above 1.5, dilated by 2 px), are excluded from the statistics and from the output. These match EMIT GHG's screens.
- **Comparison** (`eval/ghg_compare.py`): both maps are on the GLT grid. "Plume pixels" means the union of the scene's L2B CH4PLM complexes. The slope is a Deming regression using the ratio of the two maps' background variances, since OLS is biased low when the reference is the noisier map. OLS is reported alongside it.

Run with:

```
specret gas-library
specret detect-plumes <scene ids>
specret compare-ghg <scene ids>
```

## Results (5 scenes)

| Scene | Plume px | r (plumes) | Slope, Deming (OLS) | Mean diff, ppm·m | σ background, ours / EMIT | r (scene) |
|---|---:|---:|---:|---:|---:|---:|
| Permian 20230608T162322 | 3,332 | **0.929** | **0.996** (0.904) | −85 | 193 / 275 | 0.941 |
| Permian 20230608T162334 | 11,030 | **0.925** | **0.952** (0.861) | −100 | 198 / 284 | 0.946 |
| Turkmenistan 20230620T084414 | 7,830 | **0.968** | **1.033** (0.989) | −46 | 237 / 327 | 0.879 |
| Mountain Pass 20240416 (no plume) | — | — | — | — | 209 / 299 | 0.948 |
| Cuprite 20240727 (no plume) | — | — | — | — | 293 / 409 | 0.946 |

- Background medians are within ±4 ppm·m of zero in both maps.
- The negative mean difference over plumes is selection bias: the plume masks are drawn from EMIT's own map, so they favour pixels where EMIT's noise happens to be positive.
- The two CO2 scenes (Ningxia, Madhya Pradesh) are for G2.

## What closed the gap

The first version used 2122–2488 nm, a fixed ridge, and excluded everything in the L2A aggregate flag. It gave plume r of 0.76–0.88 and a background correlation with EMIT of only about 0.6. EMIT's GHG code ([emit-sds/emit-ghg](https://github.com/emit-sds/emit-ghg), v0.3.3 `parallel_mf.py`/`ghg_process.py`, the v002 era) showed three differences. One scene (Permian 20230608T162334), in order of effect:

| Change | Plume r | Background r with EMIT |
|---|---:|---:|
| First version | 0.737 | 0.58 |
| Screen only cloud, cirrus, water and flares (not dilated cloud) | 0.730 | 0.65 |
| + leave-one-out shrinkage | 0.740 | 0.67 |
| + EMIT's wide band ranges, including the 1.65 µm CH4 band | **0.925** | **0.95** |

The wide band set is what matters. Bands without CH4 absorption let the covariance model the surface and atmosphere well enough to separate the plume.

## Shrinkage: deliberately different from EMIT

Our leave-one-out shrinkage matches EMIT's formula on centred data. On one real column it picks 1e-10 for the full band set and 1.6e-5 for the narrow one, matching a reference implementation exactly. EMIT's code feeds the criterion uncentred radiance, which pushes its choice to about 3.5e-3. That shrinkage is why EMIT's map is noisier.

To decide which to use, we injected 1000 ppm·m plumes, using the exact non-linear forward model, into 2% of the pixels of the Permian 20230608T162334 radiance. We then measured recovery and background noise.

| Shrinkage | Passes | Recovered | σ, ppm·m | SNR |
|---|---:|---:|---:|---:|
| **LOO (≈1e-10)** | **2** | **951** | **198** | **4.80** |
| LOO | 1 | 799 | 199 | 4.01 |
| 1e-4 | 2 | 969 | 205 | 4.74 |
| 1e-3 | 2 | 974 | 238 | 4.10 |
| 3.5e-3 (EMIT-like) | 2 | 973 | 285 | 3.41 |
| 1e-2 | 2 | 969 | 372 | 2.60 |

- **Shrinkage:** LOO with two passes has the best SNR, about 40% above EMIT-like shrinkage. The 3.5e-3 row reproduces EMIT's background σ (284), which confirms that the noise difference comes from shrinkage.
- **Second pass:** it keeps plumes out of their own background statistics and raises recovery from 80% to 95%.
- **The remaining 5%:** this is the non-linearity of a 1000 ppm·m plume, against a linear filter. Correcting it is quantification work, in G3.

Defaults in `configs/plumes.yaml`: `shrinkage: loo`, `n_pass: 2`.

## Caveats

- **ppm·m convention:** N₁ uses air density at the scene's surface elevation. It agrees with EMIT within the slope range above, but EMIT's MODTRAN convention was not checked directly.
- **Unmodelled terms:** the solar spectrum is treated as flat within each band, and water vapour is one scene-wide value. Neither shows up as a slope error at this precision.
- **Visual check:** quicklooks of the Turkmenistan and Permian scenes beside EMIT's maps show the same plumes and the same spatial structure, and ours is visibly less noisy. Both maps also show broad, faint surface-correlated enhancement in the Turkmenistan scene. That is the false-positive problem G2 addresses with confuser terms and the mineral-map screen.
- **Validation sample:** the comparison is against EMIT's product, not ground truth. Controlled-release validation is still open (PLAN_GAS.md G0).
