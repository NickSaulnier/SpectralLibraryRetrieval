# Gas Phase G3 · Plume quantification and agent integration (EMIT)

**Date:** 2026-11-01
**Question:** can the tool turn identified plumes into emission rates with honest uncertainties, and can an agent ask "is anything leaking here, what is it, and how much?"
**Answer:**
- **The integration works.** `specret detect-plumes` writes plume records with rates, and the MCP tools, the viewer and `doctor` all cover gas.
- **The retrieval-to-mass chain is checked.** On injected plumes, IME is unbiased against the true mass in the mask, and the stated uncertainty covers 80% within 1σ and 97% within 2σ on held-out draws.
- **The rate exit criterion could not be tested.** No controlled-release data with EMIT overpasses is public. Against EMIT's own catalogue rates, ours are about 2.6× higher, which traces to a difference in method rather than an error (below).

## What was built

- **`io/wind.py`:** 10 m wind at a plume's location and the overpass time.
  - HRRR is used inside its US domain, interpolated between analysis hours, through herbie with no account.
  - MERRA-2 is used everywhere, through earthaccess with the Earthdata login.
  - Where both exist, their difference widens the uncertainty, and σ is at least 50% of the speed.
- **`gas/quantify.py`:** emission rate from IME.
  - IME = Σ α·m₁·A_px, with m₁ the gas mass per m² in 1 ppm·m at the surface.
  - Q = IME·U_eff/L, with L = √area and U_eff = 0.45·U10 + 0.67 (EMIT, from large-eddy simulations; Ferrari et al. 2025, arXiv 2506.01113).
  - The source is the most upwind of the plume's strong pixels.
- **Uncertainty budget** (relative terms, in quadrature):

  | Term | How it's measured |
  |---|---|
  | Background | IME spread of the same mask shifted to plume-free places |
  | Retrieval | 30%, measured on injected plumes |
  | Wind | The wind's σ |
  | Mask | Half the change in Q when the mask threshold moves ±0.5σ |

- **`gas/pipeline.build_plumes`:** turns named candidates at or above the confidence threshold into plumes. Duplicates across the CH4 and CO2 maps are merged, and each plume is quantified on its own gas's map.
- **Outputs** in `outputs/gas/<scene>/`:
  - `plume_mask.tif` and `confidence.tif`;
  - `plumes.geojson` (outlines plus attributes);
  - `plumes.csv`;
  - `candidates.csv` (every candidate and its verdict).
- **CLI:**
  - `specret detect-plumes <scene>` runs the whole pipeline (`--enhancement-only` keeps the G1 behaviour; `--wind U,V` overrides the wind);
  - `specret quantify <scene> --plume N [--wind U,V]` recomputes one rate.
- **MCP tools:**
  - `detect_plumes` (a background job), `list_plumes`, `explain_plume`, `quantify_plume` and `gas_signature`;
  - `scene_summary` gains a gas section, and `list_resources` lists gas scenes.
- **Viewer:** a "Gas plumes" panel.
  - Scene picker with a "Detect plumes" button and progress.
  - CH4 or CO2 enhancement map with plume outlines.
  - Plume table, where clicking a row marks the source and shows the fit, evidence and uncertainty budget.
- **`doctor`:** checks the HITRAN tables, the line-by-line cache, the L1B scenes analyzed and the gas confidence model.

Example, Permian 20230608T162322: 320 candidates give 46 plumes (42 CH4, 4 CO2). Plume 2 is CH4 with confidence 0.99, IME 197 kg and L 837 m.

| Wind | Q, kg/h | Breakdown |
|---|---:|---|
| HRRR 3.07 m/s (MERRA-2 0.38 m/s) | 1,736 ± 1,035 | wind 59%, background 8%, mask 1%, plus the retrieval term |
| User wind 1.41 ± 0.30 m/s | 1,106 ± 147 | |

The wind dominates the uncertainty, as the plan expected.

## Checks

### 1. IME on injected plumes

Compact CH4 plumes (peak 300–3000 ppm·m) were injected into the two no-plume scenes.
- **Calibration draws:** seeds 200–201, 80 plumes.
- **Held-out draws:** seeds 202–203, 80 plumes.

| | Calibration draws | Held-out draws |
|---|---:|---:|
| Plumes reported | 30 | 35 |
| IME ÷ true mass inside the mask (median) | 0.995 | 0.94 |
| Spread (sd of log ratio) | 0.36 | 0.25 |
| Mask mass ÷ total plume mass (median) | 0.56 | 0.66 |
| Error within 1σ / 2σ, background + mask only | 30% / 63% | — |
| Error within 1σ / 2σ, + 30% retrieval term | 67% / 90% | **80% / 97%** |

- **No selection bias:** IME is unbiased against the mass inside the mask. Thresholded masks don't inflate it, even though their pixels are selected on positive noise.
- **The masks miss tails:** they hold about 56–66% of the total plume mass. That truncation is what U_eff calibrations absorb, but ours can't be checked without LES plumes or releases.
- **The background term alone understates the error.** It misses saturation and the filter partly absorbing the plume, so a 30% retrieval term was added (`configs/plumes.yaml`). On held-out draws the coverage is slightly conservative.

### 2. Against EMIT CH4PLM rates

Ten of the catalogue's 49 plume complexes carry a rate. Nine of them match one of our plumes; complex 346 had no plume of ours.

| | Value |
|---|---:|
| Ours ÷ EMIT, same wind speed (geometric mean) | **2.65** (range 1.6–7.8) |
| Within 1σ / 2σ of each other (both σ) | 44% / 89% |

The difference is in the method, not the retrieval:
- **EMIT integrates near the source only.** Its rates imply Q = IME × U10 / fetch. Back-computing IME from its published Q, U10 and fetch length (about 1 km) gives 0.1–1.0× the mass of its own full plume complex. So EMIT integrates within roughly one fetch length of the source, while the IME/√area method integrates the whole mask. For complexes with several sources, or slow, accumulated plumes, the whole-mask IME is much larger.
- **At low wind, U_eff's 0.67 m/s offset dominates.** Complex 333 has U10 = 0.39 m/s, so U_eff is 0.87 m/s against EMIT's 0.39, a factor of 2.2 on its own. That is the 7.8× outlier.
- **The retrievals agree.** Our enhancement maps match EMIT's (G1), and our IME is unbiased on injections. The disagreement is the choice of rate estimator.

Which estimator is closer to the truth for EMIT can only be settled with controlled releases, the plan's exit criterion. The Stanford 2024–25 single-blind study includes EMIT (data/raw/MANIFEST.md) but has no public per-release table yet.

### 3. Large diffuse plumes (G2's open question)

CH4 plumes were injected long and wide (length 15–30 px, width 3–6 px, about 1000 px each), with peaks of 300–800 ppm·m. Seeds 300–301, 60 plumes.

| | Compact plumes (median 156 px) | Diffuse plumes (median 1016 px) |
|---|---:|---:|
| Detected as candidates | 50% | 32% |
| Named CH4, among detected | **97%** | **58%** |
| "No signal", among detected | 3% | 11% |
| Named a confuser or flag-only gas, among detected | 0% | 31% |

This supports the G2 explanation for the three missed weak Turkmenistan plumes. Averaged uniformly over a large mask, a diffuse plume's signal is diluted against large-scale surface texture. The fix is to weight pixels by enhancement when averaging the plume spectrum. That brings in a positive bias, because the weights come from the same noisy data, so the shifted-mask null and the confidence model would have to be recalibrated with it. It is left as follow-up work.

## Exit criteria

| Criterion | Status |
|---|---|
| Controlled-release rates unbiased within ±25%, about 68% within 1σ | **Not testable**: no public release data with EMIT overpasses. On injected plumes, the retrieval-to-mass chain is unbiased (0.94–0.995), with 80% within 1σ on held-out draws. U_eff itself is untested. |
| An agent can answer "is anything leaking here, what is it, and how much?" through MCP | **Met.** `detect_plumes` → `list_plumes` → `explain_plume` → `quantify_plume`, tested end to end on the Permian scene. |

## Caveats

- **Absolute rates rest on U_eff.** U_eff comes from a published LES calibration for EMIT, not our own. Our masks (hysteresis on a smoothed z map) differ from those LES masks, so the absolute rates could carry a bias of tens of percent that only releases can reveal.
- **Wind σ is a floor.** It is at least 50% of the speed, so most rates carry 30–60% uncertainty from wind alone.
- **CO2 rates use the same U_eff.** Power-plant plumes are larger and partly lofted, so CO2 rates are less certain than CH4 rates.
