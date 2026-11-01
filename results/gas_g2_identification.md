# Gas Phase G2 · Multi-gas identification and false-positive control (VSWIR, EMIT)

**Date:** 2026-11-01
**Question:** can a joint gas + confuser fit with a calibrated confidence remove the matched filter's false plumes without losing real ones?
**Answer:** mostly. Exit criteria:

- **False plumes: pass.** On the two no-plume scenes they fall by 97%, from 309 to 8.
- **Calibration: pass.** ECE is 0.027 against a target of ≤ 0.1, with AUROC 0.95.
- **Catalogue recall: fails by one plume.** Recall drops 6.1 points against a 5-point limit, which is 3 of 49 EMIT catalogue plumes. All three are weak Turkmenistan plumes where the fit finds no methane above the surface texture. The loss does not depend on the confidence threshold.

CO2 maps were also added. They match EMIT CO2ENH spatially (r = 0.97–0.98), and the comparison exposed a single-pass bias in EMIT's product for large plumes.

## Pipeline

`specret gas-benchmark` runs all of it. `gas/pipeline.analyze` runs a scene:

1. **Matched filters** for CH4 and CO2 (G1). CO2 uses the same bands and code, with its own target.
2. **Candidates** (`gas/plumes.py`):
   - each map is smoothed by 1 px and turned into a z-score per column;
   - hysteresis keeps regions above z = 2.5 that contain a pixel above z = 5, with at least 5 px.
   - Gaussian noise alone gives about 10 seeds above z = 4 per 60,000 px, but about 0.2 above z = 5. Real scenes still give 150–500 candidates each.
3. **Identification** (`gas/identify.py`), per candidate:
   - **Averaged signal:** the mean log radiance ratio ln(x/μ) over the candidate's pixels, in 1500–1790 and 1950–2450 nm.
   - **Background covariance:** estimated from 300 shifted copies of the candidate's own mask placed where no candidate is. That captures spatially correlated surface texture at the candidate's scale, which per-pixel noise does not.
   - **Fit:** after whitening, a non-negative fit with these terms:
     - gases: CH4, CO2, H2O, N2O, CO, NH3 and H2S;
     - five confuser classes;
     - an offset and slope for each window.
   - **Term selection:** greedy BIC, at most 3 terms.
     - Only CH4 and CO2 can be named.
     - N2O, CO, NH3 and H2S are flag-only. They need 3× the BIC penalty to enter, and appear as "consistent with X".
     - H2O is a nuisance term and is never named.
4. **Confusers** (`gas/confusers.py`): calcite, dolomite, other carbonates, hydrocarbons and plastics. Each class's signature is the mean continuum-removed log reflectance of its USGS spectra on EMIT bands. The L2B MIN carbonate fraction of each candidate is recorded but not used as a filter, for the reason below.
5. **Confidence:** a logistic model over six features, predicting "a real plume, and the named gas is right":
   - matched-filter z;
   - the gas's z in the joint fit;
   - spatial coherence;
   - χ² gained over the best confuser-only model;
   - between-window consistency;
   - log size.

   The deployed model is `configs/calibration/gas_emit.json`.

## Benchmark data

- **Injected plumes** (`gas/inject.py`):
  - 25 plumes per draw, 4 draws on each no-plume scene (Mountain Pass, a carbonatite, and Cuprite), 200 plumes in all.
  - Each is wind-elongated with a turbulent texture.
  - Gases are CH4 at 150–3000 ppm·m, CO2 at 15,000–300,000 ppm·m, or both, drawn log-uniformly.
  - Radiance is multiplied by the exact band transmittance from the line-by-line model.
- **Cross-fitting:** the confidence model is fitted on one scene's draws and scored on the other's, both ways, so calibration and evaluation never share a scene.
- **Labels:**
  - a candidate is positive when at least 20% of its pixels lie on an injected plume and its named gas is one of that plume's gases;
  - each scene's own candidates are negatives;
  - in all, 1,386 candidates, 142 of them positive.
- **Catalogue:** 49 EMIT L2B plume complexes on the 5 plume scenes (CH4 7 + 7 + 17, CO2 11 + 7). They are mapped to swath geometry through the GLT.

## Results

### False plumes and recall (confidence threshold 0.2)

| | Plain matched filter | Filtered |
|---|---:|---:|
| False plumes, Mountain Pass | 149 | **2** |
| False plumes, Cuprite | 160 | **6** |
| False plumes per 1000 km² (scene ≈ 5,700 km²) | 26–28 | **0.35–1.05** |
| Catalogue plumes found (of 49) | 47 (95.9%) | **44 (89.8%)** |
| Candidates kept on the plume scenes | 200–511 per scene | 33–53 per scene |

How recall varies with the threshold, scored with the out-of-scene model:

| Threshold | False plumes (MP, Cuprite) | Drop | Catalogue recall | Loss, points |
|---:|---|---:|---:|---:|
| none (named gas only) | 63, 59 | 61% | 44/49 | 6.1 |
| 0.1 | 11, 19 | 90% | 44/49 | 6.1 |
| **0.2** | **2, 6** | **97%** | **44/49** | **6.1** |
| 0.3 | 1, 3 | 99% | 41/49 | 12.2 |
| 0.5 | 0, 1 | 100% | 32/49 | 30.6 |

- **Choosing the threshold:** 0.2 was chosen for about 1 false plume per 1000 km² on the no-plume scenes, not from the catalogue.
- **Above 0.2, recall falls fast.** Real catalogue plumes are larger and weaker per pixel than most injected ones, so their confidences sit lower. A probability calibrated on the injection mix is not a probability for an arbitrary scene's mix.
- **The three plumes always lost:** Turkmenistan complexes 15, 16 and 17. Their smoothed matched-filter z is 6.6–8.6, but in the joint fit methane reaches only z = 1.0–2.1 against the shifted-mask background, so the verdict is "no signal".

  Either the surface texture at their scale really is as strong as their signal, or the shifted-mask null is too conservative for diffuse plumes of more than 150 px. G3's injection of large, diffuse plumes can tell these apart. Lowering the BIC entry bar until they pass would tune on the evaluation set, so it was not done.

### Detection and identification of injected plumes

Fraction of injected plumes found, by peak enhancement (mixed CH4 + CO2 plumes are counted under both gases):

| CH4 peak, ppm·m | n | Plain | Filtered |
|---|---:|---:|---:|
| 150–300 | 28 | 0.39 | 0.29 |
| 300–600 | 34 | 0.29 | 0.12 |
| 600–1200 | 48 | 0.65 | 0.44 |
| 1200–3000 | 42 | 1.00 | 0.98 |

| CO2 peak, ppm·m | n | Plain | Filtered |
|---|---:|---:|---:|
| 15k–30k | 26 | 0.42 | 0.23 |
| 30k–60k | 34 | 0.59 | 0.24 |
| 60k–120k | 32 | 0.84 | 0.66 |
| 120k–300k | 38 | 1.00 | 0.92 |

The low bins are noisy at n ≈ 30. Mixed plumes in them are often found through their other gas.

**Identification:** 88% of detected injected plumes have the correct named gas, counting the best candidate per plume.
- **No CH4/CO2 confusion:** CH4 was never named CO2, and CO2 never CH4.
- **Errors:** 10 were "no signal" (weak) and 4 went to a confuser, all on mixed plumes. One CO2 plume was "consistent with H2S".

On the catalogue scenes, 44 of the 47 detected catalogue plumes were named with the right gas.

### CO2 enhancement against EMIT L2B CO2ENH

| Scene | r (plumes) | Deming slope | σ background, ours / EMIT (ppm·m) |
|---|---:|---:|---:|
| Ningxia 20230610 | 0.966 | 1.39 | 12,314 / 12,901 |
| Madhya Pradesh 20240402 | 0.984 | 1.12 | 12,332 / 12,828 |

- **Single pass reproduces EMIT:** our single-pass filter matches EMIT's CO2ENH almost exactly (r = 0.996 and 0.994, slope 0.96 and 0.97). The higher slope comes entirely from our second pass, which keeps plume pixels out of the background.
- **Which is right:** CO2 injected at 100,000 ppm·m into Mountain Pass decides it.

  | Injected plumes | One pass | Two passes |
  |---|---:|---:|
  | Small (637 px) | 0.77 | 0.91 |
  | Large (10,500 px) | 0.68 | 0.82 |

- **Conclusion:** EMIT's single-pass product understates large CO2 plumes. Ningxia's power-plant complexes cover about 18,000 px, which fits the larger gap there. The remaining 10–18% shortfall in two-pass recovery comes from saturation at 100,000 ppm·m and from plume tails below the 3σ exclusion. Correcting both is G3 quantification work.

## The mineral screen is a flag, not a filter

The Permian Basin is limestone country. True catalogue plumes there sit on pixels whose L2B MIN ~2 µm mineral is 55–99% carbonate, so a hard carbonate screen would remove real plumes. The fraction is reported per candidate. The confuser terms in the fit do the discriminating: on Mountain Pass, 50 of the 149 candidates are explained by carbonate, plastic or hydrocarbon confusers.

## Caveats

- Only two no-plume scenes: the cross-fit is 2-fold, and both scenes are arid and bright.
- Injected plumes are simpler than real ones: one source, and no along-track smearing.
- The catalogue is EMIT's own reviewed product, not ground truth, and it misses some plumes. Many of the strongest uncatalogued Permian candidates are named CH4 with high confidence and may be real emitters.
- C2H6 is not fitted: HITRAN has no VSWIR lines for it, and the cross-section files are not fetched.
