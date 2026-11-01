# Gas Phase G4 · LWIR gas plumes in HyTES thermal-infrared radiance

**Date:** 2026-11-01
**Question:** with a thermal-contrast target, can the same matched filter and joint-fit identifier find and name gases in HyTES flights?
**Answer:** for injected plumes, yes. On real flights it works in part. Exit criteria:

- **Identification at plume SNR ≥ 5: pass.** 93.6% of detected injected plumes are named correctly, against a target of 80%, with no confusion between the five gases.
- **Low-contrast mask: pass.** Detection collapses near zero contrast, and pixels within 1 K are masked by construction.
- **Published gases top-1: 2 of 3 flights.**
  - Salton Sea NH3: yes, at the mud pots.
  - Chino dairies NH3: the top three candidates are all NH3, but at confidence too low to report.
  - Kern River CH4: no. CH4 is found, but other gases outrank it.

## Method

### Data: `io/hytes.py`
- L1 radiance: ENVI BIP, 256 bands, 7.5–12 µm, in W m⁻² sr⁻¹ µm⁻¹.
- Per-pixel geolocation.
- L2: land surface temperature and the per-band ISAC transmittance τ and path radiance L_up.
- The Chino flight has neither L2 nor geolocation, so τ = 1, L_up = 0 and a site coordinate stand in.

### Spectroscopy: `gas/library.build_lwir_window`
- HITRAN line-by-line cross-sections for CH4, N2O, NH3, SO2, C2H6, C2H4 (newly fetched), H2O and O3, at 0.01 cm⁻¹.
- Five near-surface layers. The aircraft is low, and τ and L_up come from ISAC.
- SF6 is not available as HITRAN lines (cross-sections only), so it was dropped.

### Target: `gas/lwir.py`
A thin plume changes the signal by dx ≈ (τ·B(T_p) + L_up − x₀)·σ_b·α.
- The matched filter takes the offset form t = σ·(τB + L_up) − σ·μ.
- Each pixel's estimate is rescaled from its column's thermal contrast to its own.
- Pixels whose effective contrast |T_p − BT(surface-leaving radiance at 10.8 µm)| is below 1 K are masked as "cannot detect".
- The plume temperature T_p is the MERRA-2 2 m air temperature at the overpass:

  | Flight | T_p | Typical surface |
  |---|---:|---|
  | Salton Sea | 314 K | 310–345 K, so ΔT around −13 K |
  | Kern River | 311 K | 310–345 K, so ΔT around −13 K |
  | Chino | 300 K | |

### Two pitfalls found and fixed
1. **The ozone bands are linearly dependent.** In the L1 product, an exact ± combination of 18 bands at 9.5–9.8 µm has zero variance. With leave-one-out shrinkage near 1e-10, the whitened filter used that direction as noise-free. Injected NH3 and C2H4 showed SNRs over 1,000, and SO2's σ was 27× too small. Excluding 9.4–9.9 µm removes the degeneracy, after which leave-one-out and fixed 1e-4 shrinkage agree.
2. **The thermal background has heavy tails.** About 0.5% of pixels lie beyond 6σ, from surface emissivity structure, against about 2e-9 for Gaussian noise. Two changes cut the first segment from 9,800 to 1,400 candidates and its runtime from 700 s to 140 s:
   - candidates from all gas maps are merged into one per connected region, so each region is identified once;
   - thresholds are stricter: seed z > 8, grow above z > 3, at least 8 px.

Detector-edge columns (16 at each end) are also masked.

### Identification
- **Retrieve:** rank the library gases by whitened correlation with the plume's mean residual, and keep the top 3.
- **Fit:** a joint non-negative fit of those gases, H2O and O3 (nuisance terms) and a per-segment baseline, with BIC selection and the shifted-mask null of G2.
- **Names:** CH4, N2O, NH3, SO2 and C2H4 can be named. C2H6 is flag-only, because its only band in range sits at the 11.9 µm edge.
- **Confidence:** fitted on the injection draws (`configs/calibration/gas_hytes.json`).

## Results

### Identification on injected plumes

Single-gas plumes were injected into a Salton Sea segment (rows 2000–3875) and a Kern River segment (rows 6000–8000), 2 draws × 25 each. Peaks were set from each gas's measured background σ, for an SNR of about 2–40. The exact plume-layer transmittance was used.

| Plume SNR | n | Detected | Named correctly (of detected) |
|---|---:|---:|---:|
| < 5 | 42 | 0.05 | 0.50 |
| 5–10 | 16 | 0.94 | 0.87 |
| 10–20 | 20 | 0.70 | 0.93 |
| ≥ 20 | 22 | 0.82 | 1.00 |
| **≥ 5, all** | **58** | **0.81** | **0.936** |

- **By gas, at SNR ≥ 5:** N2O 1.00, SO2 1.00, CH4 0.93, C2H4 0.92, NH3 0.80.
- **Confusion:** no gas was named as another gas. The errors were "no gas named".
- **Calibration:** cross-fitted between the two flights, ECE is 0.11 and AUROC 0.83.

Background sensitivity, as the per-pixel 1σ in ppm·m:

| Gas | 1σ, ppm·m |
|---|---:|
| NH3 | about 3 |
| C2H4 | about 6 |
| N2O | about 54 |
| SO2 | about 97 |
| CH4 | about 134 |

CH4 is weakest because its 7.7 µm band sits where water vapour leaves τ at 0.07–0.3.

### Open set: a gas missing from the library

Plumes of C2H4, SO2 or NH3 were injected with that gas removed from the library, 20 per gas per flight.
- **Usually missed:** no matched filter is run for a gas the library lacks, so only 1 of 40 plumes per held-out gas was detected at all, through an overlapping gas's map.
- **Never confidently misnamed:** held-out C2H4 and SO2 went unnamed. Held-out NH3 was named C2H4, an overlapping band, but below the confidence threshold.

An unlisted gas is missed rather than misnamed. Catching unknown absorbers needs a generic anomaly detector next to the per-gas filters. That is follow-up work.

### Thermal contrast

NH3 plumes (20 per step) were injected into Salton Sea at plume temperatures relative to the median surface brightness temperature:

| ΔT (K) | −10 | −5 | −2 | −1.5 | −0.5 | +0.5 | +1.5 | +2 | +5 | +10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Detected | 18 | 20 | 10 | 3 | 3 | 4 | 9 | 7 | 18 | 18 |
| Named NH3 | 18 | 20 | 10 | 2 | 2 | 2 | 1 | 2 | 12 | 16 |

- **Near zero contrast, detection collapses.** The few detections at ±0.5 K come from parts of plumes over pixels with real local contrast. Surface temperature varies by several kelvin between pixels, and pixels within 1 K are masked so they never enter a candidate.
- **Warm plumes are harder to name.** Emission is named correctly 60–80% of the time, against 90–100% for cool plumes in absorption.

### Real flights and published detections

The expected gases come from the published HyTES case studies (Hulley et al. 2016, AMT 9, 2393; data/raw/MANIFEST.md): NH3 at the Salton Sea geothermal mud pots, CH4 at the Kern River oil field, and NH3 and CH4 at the Chino dairies.

| Flight | Candidates | Named | Reported (confidence ≥ 0.2) | Strongest coherent named plume |
|---|---:|---|---|---|
| Salton Sea | 1,428 | NH3 115, CH4 92, N2O 92, SO2 72, C2H4 33 | SO2 3, NH3 2 | **NH3**, 22 ppm·m, gas_z 99, coherence 0.86, confidence 0.99, at 33.218 N −115.600 W (the mud pots) ✓ |
| Chino dairies | 49 | SO2 6, CH4 5, N2O 5, C2H4 4, NH3 3 | none | **NH3** (top three by gas_z are all NH3: gas_z 3.7–8.9, up to 819 px), confidence ≤ 0.02 ✓ named, ✗ not reported |
| Kern River | 2,243 | SO2 417, CH4 313, N2O 245, NH3 175, C2H4 148 | SO2 14, CH4 7, N2O 5, NH3 4, C2H4 2 | N2O in the last 22 lines of the flight (likely an end-of-line artifact), then NH3 and SO2 ✗. The best CH4 plumes reach gas_z 23–26 with coherence 0.8 |

What this shows:
- **NH3 is right where it was published.** The strongest NH3 signal sits at the Salton Sea mud pots, and NH3 tops the Chino candidates.
- **Chino is below threshold** because the confidence model, trained on compact injected plumes, scores its large diffuse plumes low. This mirrors the G3 diffuse-plume finding.
- **Kern River's many SO2, NH3 and N2O calls are the open problem.** Thermal-infrared surface emissivity features can mimic gases:
  - sulfates and gypsum near 8.6 µm (SO2);
  - quartz between 8 and 9.5 µm.

  The USGS library here has no thermal-infrared mineral spectra, so the fit has no confuser terms for them, unlike the VSWIR. Real SO2 cannot be ruled out without ground truth.

## Exit criteria

| Criterion | Status |
|---|---|
| Right gas top-1 on published HyTES detections | **2 of 3.** Salton Sea NH3 and Chino NH3 (named, not reported); Kern River CH4 missed |
| Identification ≥ 80% on injected plumes at SNR ≥ 5 | **Pass:** 93.6% |
| Low-contrast mask suppresses detections at \|ΔT\| < 1 K | **Pass:** masked by construction; detections collapse near zero contrast |

## Not done in G4

- **Emission rates in the LWIR.** U_eff has no calibration for HyTES's 2–5 m pixels, and HRRR wind isn't archived before 2014-07-30. Column amounts (ppm·m) and plume outlines are reported. Rates would need a resolution-appropriate U_eff.
- **Thermal-infrared surface confusers.** These need emissivity spectra, such as the ECOSTRESS library's TIR files or an in-scene emissivity basis from HyTES L2.
- **HITRAN cross-section gases.** SF6, freons and aromatics are not fetched.
- **MCP and viewer integration for HyTES.** Results are written to `outputs/gas/hytes/<flight>/` (`specret detect-plumes-lwir`).
