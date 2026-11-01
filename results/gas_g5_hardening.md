# Gas Phase G5 · Hardening: second sensor, persistence, flux, bias check

**Date:** 2026-11-01
**Scope:** the plan's G5 items that need no new accounts, plus open question 3:
1. the ISOFIT / sRTMnet decision;
2. AVIRIS-NG radiance;
3. multi-overpass persistence;
4. a cross-sectional flux estimate.

PRISMA and EnMAP loaders were left out. Their radiance needs Italian and German space-agency accounts, and the project avoids new accounts after the Copernicus registration failed.

## 1. Is a full radiative-transfer model needed? No

PLAN_GAS.md G5 says to replace the line-by-line + standard-atmosphere signatures with ISOFIT or sRTMnet only if G1–G3 residuals show a systematic bias with water vapour or elevation. The test re-scored the five G1 scenes against EMIT L2B CH4ENH, which EMIT computes from MODTRAN with the scene's own water vapour and elevation.

| Scene | H2O, g/cm² | Elevation, km | AMF | σ ours / EMIT | Deming slope (plumes) |
|---|---:|---:|---:|---:|---:|
| Permian 20230608T162322 | 1.59 | 0.80 | 2.22 | 0.701 | 0.997 |
| Permian 20230608T162334 | 1.58 | 0.90 | 2.21 | 0.695 | 0.952 |
| Turkmenistan 20230620 | 2.14 | 0.00 | 2.05 | 0.726 | 1.033 |
| Mountain Pass 20240416 | 1.5* | 0.93 | 2.18 | 0.699 | — |
| Cuprite 20240727 | 1.5* | 1.65 | 2.14 | 0.715 | — |

\* No L2A mask was downloaded, so the default water column was used.

- **No trend:** the scale ratio is stable within ±2% across 0–1.65 km elevation and 1.5–2.1 g/cm² water, and the plume slopes show no trend.
- **Decision:** no systematic bias at measurable precision, so ISOFIT and sRTMnet are not needed.
- **Caveat:** humid scenes (above about 2.5 g/cm²) are untested.

## 2. AVIRIS-NG radiance (`io/avirisng.py`)

- **Data:**
  - L1B flight line `ang20161001t205456` (ORNL `AVIRIS-NG_L1B_radiance_2095`), 425 bands at 5 nm and 2.9 m pixels;
  - its published CH4 matched-filter map from the California point-source survey (Duren et al. 2019; ORNL `CH4_Plume_AVIRIS-NG_1727`).
- **Loader:**
  - The v1 radiance is already resampled onto a 5°-rotated UTM grid. The GLT's own (positive) entries scatter it back into the 598-sample × 1750-line raw swath, so the matched filter has real detector columns.
  - OBS and LOC files are already in raw geometry.
  - The rotated-grid geotransform matches the published map to 0.01 m.
- **Data quirk found:** this line's OBS file holds sun zenith and sun azimuth in each other's place despite its band names. A sun azimuth of 206° and zenith of 42° fit a 13:54 PDT October overpass. Read as labelled, the AMF came out at −0.09 and the map's sign flipped. The loader now picks the band whose values lie within 0–90°.
- **Pipeline:** EMIT-specific steps (the L2B MIN lookup, the time from the scene id) are now optional, so the gas pipeline runs unchanged on AVIRIS-NG. `specret detect-plumes ang20161001t205456` works.

Against the published CH4 map (plume pixels: published-map smoothed z > 4, regions of at least 30 px):

| Our settings | r (scene) | r (plumes) | Deming slope | σ, ours / published (ppm·m) |
|---|---:|---:|---:|---:|
| Default: wide bands (EMIT v002 ranges), 2 passes | 0.64 | 0.56 | 0.93 | 142 / 306 |
| 2122–2488 nm only, 2 passes | 0.90 | 0.88 | 0.78 | 201 / 306 |
| 2122–2488 nm only, 1 pass | 0.90 | 0.90 | 0.67 | 201 / 306 |
| Wide bands, 1 pass | 0.66 | 0.60 | 0.60 | 144 / 306 |

- **The pattern matches the published map's method.** Restricting to its 2.1–2.5 µm window gives r = 0.90.
- **Our default is still better for detection.** On injected 1000 ppm·m plumes it gives SNR 5.1 against 3.7 for the narrow window, at equal recovery (0.73), so it is kept.
- **Single-pass statistics are biased low:** slopes of 0.60–0.67. At 2.9 m a plume fills many pixels of its column, so it leaks into the background, as large CO2 plumes did in G2.
- **Recovery of injected plumes is 0.73,** against 0.95 on EMIT. AVIRIS-NG has fewer pixels per column relative to its band count (1750 for 325 bands, against 1280 for 219), which fits a sample-covariance shortfall. A sensor-specific gain, or stronger shrinkage, would be the next step. It has not been applied.
- **Not done for AVIRIS-NG:** the confidence model and U_eff are EMIT's, so plume confidence and emission rates are not calibrated for this sensor.

## 3. Multi-overpass persistence (`gas/persistence.py`, `specret gas-persistence`)

For each plume, the tool counts the other overpasses that imaged its source location and those that saw a plume, or a candidate named with the same gas, within 300 m. Adjacent scenes of one orbit don't count as repeats.

**Data:** five Permian scenes from four overpasses: 2023-06-08 (two adjacent scenes), 2023-07-31, 2023-10-20 and 2024-08-23. Each was run through `detect-plumes`.

| | Plumes | In EMIT's catalogue |
|---|---:|---:|
| All plumes | 144 | 15% |
| Covered by another overpass | 57 | — |
| …and seen again | 9 | 0% (median confidence 0.26) |
| …and not seen again | 48 | 10% (median confidence 0.34) |

**Inconclusive.** Footprints overlap only partly, and only nine plumes recur. Those nine are low-confidence and none is catalogued, so this sample can't show that persistence raises precision. The table is produced for every set of scenes. Using it in the confidence model needs many more repeats over the same sources, and a label source other than EMIT's conservative catalogue.

## 4. Cross-sectional flux (`quantify.cross_sectional_flux`)

The flux through one-pixel-wide transects across the wind, downwind of the source, is Q = U10 × median(line density). It is added to every plume record as `q_csf_kg_h`, with its transect count and spread.

- **Check:** on a synthetic steady Gaussian plume it recovers the true rate to within 5%.
- **On the five Permian scenes:** 123 of 144 plumes have at least 3 downwind transects (median 6).

| | CSF ÷ IME rate |
|---|---:|
| All plumes (geometric mean) | 0.62 (IQR 0.40–1.10); correlation 0.95 in log |
| Plumes of 100 px or more | 0.47 (IQR 0.38–0.70) |

The two estimates track each other closely, but CSF is lower, especially for large plumes. It sits between our IME/U_eff rates and EMIT's fetch-limited catalogue rates (G3: ours are about 2.6× EMIT's). That hints the IME rates run high for large plumes. Controlled releases remain the only way to settle it.

## Not done

- **PRISMA and EnMAP loaders:** they need agency accounts.
- **For AVIRIS-NG:** a gain calibration, plus a confidence model and U_eff of its own.
- **Persistence as a confidence feature:** this needs more overlapping repeats and better labels.
- **From earlier phases:**
  - enhancement-weighted identification of diffuse plumes (G3, G4);
  - TIR emissivity confusers (G4);
  - an anomaly detector for gases missing from the library (G4).
