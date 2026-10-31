# Phase 4 · Mixed pixels and unknown materials

**Date:** 2026-10-31
**Sensor:** EMIT bands (2024-07-27 Cuprite scene). Calibration was also fitted on AVIRIS-Classic and field-instrument bands.
**Split:** the frozen `splits_v1`. Everything was tuned on **val** and reported on **test** (or cross-validated over held-out groups).

## Retrieve-and-unmix (`unmix/`)

- **Model:** non-negative least squares on good bands, plus a free offset and linear slope. The baseline absorbs continuum differences between library samples and the pixel. Fractions are relative and the brightness scale is reported separately.
- **Pursuit:**
  1. Seed with the hybrid retriever's best match.
  2. Try the retriever's other candidates, plus the 5 library spectra best correlated with the residual.
  3. Keep the one that lowers the residual most.
  4. Accept it only if its fraction is ≥ 0.1, the BIC improves, and the residual sum of squares drops by ≥ 60%.
  5. Allow at most one endmember per group and 3 in total.
- **Why the 60% rule:** the library never holds the pixel's exact sample, so any extra endmember lowers the residual a little, and the BIC penalty alone added ~2.5 components even to pure spectra. The threshold and pool size were chosen on val by F1. Restricting the pool mostly to retrieval candidates mattered as much as the threshold.

### Synthetic mixtures (test split, 300 queries; `specret eval-unmix`)

Each query is 1–3 test spectra from different groups, with Dirichlet fractions ≥ 0.15, a brightness change and EMIT noise. The library holds train + val only.

| Method | Recall | Precision | F1 | Abundance RMSE |
|---|---:|---:|---:|---:|
| Best single match | 0.466 | 0.670 | 0.549 | 0.425 |
| **Retrieve-and-unmix** | **0.578** | 0.605 | **0.591** | **0.386** |

| True components | Best match: recall | Unmixing: recall | Unmixing: components found |
|---|---:|---:|---:|
| 1 | 0.697 | 0.765 | 1.36 |
| 2 | 0.330 | 0.519 | 1.66 |
| 3 | 0.210 | 0.292 | 1.68 |

Unmixing finds more of the true components, and its abundance errors are lower. It still adds a spurious second component to about a third of pure spectra.

### EMIT Cuprite (150 random pixels per L2B ~2 µm group)

| L2B group | Best match agrees | Any unmixed component agrees | Mean components |
|---|---:|---:|---:|
| alunite | 100% | 100% | 1.7 |
| white mica | 93% | 93% | 2.0 |
| silica | 90% | 90% | 1.6 |
| calcite | 48% | 49% | 1.9 |
| smectite | 44% | 45% | 1.8 |
| **kaolin** | **21%** | **52%** | 1.8 |

The kaolin problem from Phases 2–3 is a mixture problem, as suspected. Typical unmixings are kaolinite + alunite and kaolinite + white mica, the same associations Tetracorder's mixture spectra describe. These numbers measure agreement with Tetracorder, not accuracy.

## Calibrated confidence and unknown materials (`openset/`)

- **The model:** a logistic model predicts P(best match's group is right) from eight cheap signals: ranking score, margin over the best other-group candidate, feature-fit score, SAM angle, continuum-removed cosine in 2.0–2.5 µm, strongest absorption depth, linear-fit residual, and top-5 group consensus.
- **Labels:** queries of materials absent from the library are labelled "not right".
- **Data:** the library is train only. Queries come from val, test and the five held-out open-set groups, each under five distortion recipes. Results are cross-validated by open-set group, so every unknown group is unseen when it is scored.
- **Saved models:** `configs/calibration/<sensor>.json`, fitted on val plus all open-set groups, never on test (`specret calibrate`).

| EMIT bands | Calibrated P | Raw top-1 score |
|---|---:|---:|
| AUROC, right vs wrong answers | **0.825** | 0.784 |
| AUROC, known vs unknown material | 0.722 | 0.710 |
| Accuracy of the 60% most confident answers | **0.568** | 0.538 |
| Expected calibration error | 0.059 | n/a |

AVIRIS-Classic bands give 0.804 / 0.740 / 0.576 / 0.052; field-instrument bands give 0.811 / 0.707 / 0.556 / 0.052.

**Per held-out group** (known vs unknown AUROC): other sulfate 0.83, zeolite 0.80, borate 0.68–0.78, scapolite 0.75–0.77, serpentine 0.52–0.53. Serpentine is nearly indistinguishable from the chlorite and talc in the library. That is a fair reflection of how alike their Mg-OH spectra are, not a bug.

**The target was missed.** The plan aimed for a known-vs-unknown AUROC ≥ 0.85; this reaches 0.72. Adding the fit-residual and consensus signals didn't move it. What the model does well is rank right answers above wrong ones (0.83), and its probabilities are honest (ECE 0.06).

### Tiers used by the tools

| Tier | Rule | Answers given (EMIT, CV) | Accuracy of answers | Unknowns flagged |
|---|---|---:|---:|---:|
| confident | P ≥ 0.5 | 38% | 71% | 82% |
| confident or uncertain | P ≥ 0.3 | 61% | 56% | 57% |
| everything | — | 100% | 40% | 0% |

These accuracies are low in absolute terms because the queries include distorted spectra, unknown materials, and a reduced train-only library. They show the relative effect of the tiers.

### On the real EMIT scene

`map-scene` with the EMIT calibration took 180 s for 1.59M pixels. Across the whole scene, 18.5% of pixels are confident (P ≥ 0.5) and 57.9% fall below 0.3.

Checked against EMIT L2B MIN on the 34,366 Cuprite pixels with a ~2 µm detection:

| Calibrated P | Share of pixels | Agreement with L2B MIN | Mean P |
|---|---:|---:|---:|
| < 0.3 | 47.0% | 36.6% | 0.21 |
| 0.3–0.5 | 38.9% | 62.6% | 0.39 |
| ≥ 0.5 | 14.1% | **88.7%** | 0.60 |

- **The ranking transfers to real data.** AUROC for predicting agreement is 0.737, against 0.690 for the raw score.
- **The probabilities are too pessimistic on real pixels.** Each bin agrees with L2B MIN more often than its mean P suggests. Real pixels are mixed and have shallower features than the library-derived calibration queries.
- **On real scenes, "likely unknown" (grey "low confidence" on the viewer map) means low confidence.** It doesn't reliably mean "absent from the library". Recalibrating on real labelled pixels (field-checked sites) is the fix. L2B MIN itself can't serve as labels, because it shares the library and method family.

## Integration

- **MCP:**
  - every answer now carries `p_group_correct` and a tier (confident / uncertain / likely unknown);
  - new tools `unmix_pixel` and `unmix_spectrum`;
  - `scene_summary` reports `likely_unknown` pixels by calibrated P.
- **Maps:** `map-scene` adds a per-pixel `confidence.tif` when a calibration file exists for the sensor.
- **Viewer:** pixels with P < 0.3 are grey ("low confidence") on the map. A pixel's verdict shows its tier and P, plus the unmixed components.
- **Bug fixed along the way:** a non-reentrant lock deadlocked the first unmix call. The workspace now uses an `RLock`.

## Caveats

- **Calibration distribution shift:** the model was calibrated with a train-only library of minerals, mixtures and coatings, and on library-derived queries. The tools search a larger library that includes vegetation, and real pixels. Expect some shift.
- **Unmixing has no scene layer yet.** At ~0.01–0.1 s per pixel, a full EMIT scene would take about an hour in this implementation. It runs per pixel through the tools and viewer.
- **SUnSAL and MESMA comparisons** (optional in the plan) were not done.
