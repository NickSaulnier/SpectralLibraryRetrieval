# Gas Plume Identification: Extension Plan

**Goal:** extend `specret` so it finds gas plumes in hyperspectral images, says which gas each plume is, and estimates how much gas it holds and how fast it's being emitted. Every answer carries a confidence and an "unknown absorber" option. The tool covers both VSWIR imagers (EMIT, AVIRIS-NG) and thermal LWIR imagers (HyTES first), and it's usable from the CLI, the viewer and the MCP tools, as the mineral pipeline is.

**What success means:**
- On EMIT scenes with published plumes, our methane enhancement map agrees closely with the official EMIT L2B CH4ENH product, and we find the plumes in EMIT's L2B CH4PLM catalogue.
- Emission rates fall within the stated uncertainty of controlled-release ground truth.
- In LWIR, the tool names the right gas for published HyTES detections.
- The tool says "no plume" or "unknown absorber" rather than reporting a gas over a bright carbonate outcrop or a hot roof.

Every results table includes a plain matched-filter row, for the same reason PLAN.md keeps a SAM row: it shows whether the extra machinery earns its keep.

## Decisions (2026-10-31)

| Question | Decision | Effect |
|---|---|---|
| Spectral range | **VSWIR and LWIR now** | PLAN.md's "thermal out of scope" boundary is lifted for gases only. Solid-material retrieval stays VSWIR. LWIR needs an emissive radiance model and new sensors and data (Phase G4). |
| Output depth | **Detect, identify and quantify** | Outputs: enhancement maps (ppm·m), gas identity with confidence, plume masks, integrated mass enhancement (IME) and emission rate with uncertainty. Quantification needs wind data (Phase G3). |
| Wind data | **No Copernicus account** | MERRA-2 through the existing Earthdata login is the default, with HRRR for US scenes. Both cover 10 m wind at the overpass time without a new account. |
| Ship rule | Same as the mineral tool | Real-scene agreement and controlled-release results decide what ships. Results on synthetic, injected plumes are a sanity check only. |

---

## 0. Why gases need a different pipeline

The mineral tool asks "which library spectrum does this pixel look like?" That question doesn't work for gas, for four reasons:

1. **Gases are detected in radiance, not reflectance.** Atmospheric correction (EMIT L2A, ISOFIT) treats extra methane as part of the atmosphere or leaves it as an unexplained residual, so the plume signal is partly removed from L2A. Gas work starts from **L1B at-sensor radiance**, which means new loaders.
2. **A plume is a small perturbation, not a material.** A strong methane plume changes SWIR radiance by a few percent. The surface, not the gas, dominates the spectrum. The standard approach is therefore **target detection**: a matched filter that whitens the pixel against the scene's own background statistics and projects it onto the gas's absorption signature. Nearest-neighbour library search doesn't apply.
3. **The library holds absorption coefficients, not reflectance spectra.** Gas signatures come from HITRAN line lists and absorption cross-sections (free sources only; the paid PNNL database is not used). They're turned into a sensor-specific **unit absorption spectrum** for each scene's viewing geometry, temperature and pressure. This is a per-scene target vector, not a fixed library row.
4. **Identification is a joint fit, not a ranking.** Overlapping absorbers (CH4 with water vapour and ethane near 2.3 µm; CO2 with water near 2.0 µm; many gases share the 8–12 µm LWIR window) have to be fitted together. That's structurally the same problem as Phase 4 unmixing: a sparse non-negative fit with model selection. The mineral library also returns as a source of **confusers**, because surface materials such as carbonates, hydrocarbon-bearing surfaces and plastics have absorption near 2.3 µm and are the main source of false methane plumes.

Reused as is or nearly so:
- `sensors/resample.py`, to convolve high-resolution gas spectra to sensor bands (it already handles Δλ weighting and the effective-width correction);
- the EMIT GLT orthorectification, the COG writer, the job runner, `Workspace`, the viewer shell and the MCP server;
- `unmix/fcls.py` and the pursuit/BIC design;
- `openset/calibrate.py`'s logistic combiner;
- the GPU batching pattern from `search/gpu_exact.py`;
- the mineral scene maps themselves, as a false-positive screen.

---

## 1. Physics

### 1.1 VSWIR (reflected sunlight, roughly 1.5–2.5 µm)

At-sensor radiance under a thin plume with column enhancement α (ppm·m):

L(λ) ≈ L₀(λ) · exp(−k(λ) · α · AMF) ≈ L₀(λ) · (1 − k(λ) · α · AMF)

- k is the gas absorption coefficient per ppm·m, from HITRAN convolved to the sensor.
- AMF = 1/cos θ_sun + 1/cos θ_view is the two-way air-mass factor, taken from the EMIT OBS geometry.

The **target signature** is t = μ ⊙ (−k · AMF), where μ is the background mean radiance. The **matched filter** estimate for a pixel x is:

α̂ = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t)

The background mean μ and covariance Σ are estimated **per detector column** (pushbroom sensors have column-dependent response), with shrinkage, and in a second pass with candidate plume pixels removed. Two optional refinements:
- albedo normalization, i.e. scaling t by the pixel's brightness;
- an iterative sparse variant in the style of mag1c, used as a baseline.

Diagnostic windows:
- **CH4:** 2200–2400 nm (main) and 1630–1700 nm.
- **CO2:** 1990–2090 nm and 1560–1620 nm.
- **Others:** N2O, NH3, H2S, C2H6 and CO have weak SWIR features. They are treated as **best-effort, flag-only** gases in VSWIR, which means the tool may report "absorption consistent with X" but never quantify them.

The unit absorption spectrum is computed by line-by-line integration over a standard atmosphere profile and a plume layer near the surface, at about 0.01 nm resolution, then resampled with the scene's own band centres and FWHM. No full RT engine (ISOFIT or MODTRAN) is needed for detection. A look-up table over AMF and water vapour, built once per sensor, keeps it fast.

### 1.2 LWIR (thermal emission, roughly 7.5–12 µm)

A plume layer at temperature T_p with transmittance τ_p = exp(−Σ_g σ_g c_g) sits over a background of radiance L_bg:

L = τ_a · [τ_p · L_bg + (1 − τ_p) · B(T_p)] + L_a↑

Linearized for an optically thin plume:

ΔL ≈ τ_a · (B(T_p) − L_bg) · Σ_g σ_g c_g

So the target for gas g is t_g = τ_a · (B(T_p) − μ) ⊙ σ_g. This differs from VSWIR in three ways:

1. **The signal depends on thermal contrast.** A plume cooler than the ground shows up in absorption, a warmer one in emission, and one at ground temperature is invisible. The detector reports ΔT = T_p − T_bg, plus a "low contrast, cannot detect" mask where |ΔT| is small.
2. **Plume temperature** is taken as near-surface air temperature, from weather data or a scene-supplied value. It's a key source of uncertainty in quantification.
3. **Atmospheric terms** (τ_a, L_a↑): HyTES flies low, so they're small. They're estimated in-scene, with ISAC-style blackbody-pixel fitting, or taken from the HyTES L2 products.

LWIR targets: SO2, NH3, CH4 (7.7 µm), N2O, and VOCs with HITRAN cross-sections (for example ethylene, propylene, benzene, ammonia derivatives, SF6, freons). This is where identification really is a **library problem**: tens to hundreds of candidate gases with overlapping bands.

### 1.3 Quantification (both ranges)

- **Plume mask:** threshold the enhancement map at k·σ_background, keep connected components above a minimum size, take the source as the upwind maximum, and apply morphological cleanup. The thresholds are fixed on the validation scenes.
- **IME:** Σ over plume pixels of α̂ · pixel area, converted from ppm·m to kg using the molar mass and air density.
- **Emission rate:** Q = U_eff · IME / L, where L = √(plume area). U_eff is a calibrated function of the 10 m wind speed (U_eff = a·U10 + b, after Varon et al. 2018), with separate coefficients per sensor resolution.
- **Wind:** 10 m wind at the overpass time, from one of these sources. The user can override it.
  - **MERRA-2** (default; global, hourly, ~50 km): the `M2T1NXSLV` collection's `U10M` and `V10M`, fetched through `earthaccess` with the existing Earthdata login.
  - **HRRR** (US scenes; hourly, 3 km): fetched through `herbie` from NOAA's public cloud bucket, no account needed. Preferred over MERRA-2 when the scene is inside its domain.
  - **ERA5, keyless:** Google's public ARCO-ERA5 copy (read with `xarray` from the public bucket) or the Open-Meteo historical API, used as a cross-check on the wind spread. Copernicus CDS is not used.
- **Uncertainty:** a quadrature sum of matched-filter noise (from the background σ), wind uncertainty (ensemble or reanalysis spread, at least ±50% by default) and mask sensitivity (re-run at ±1σ thresholds). Wind usually dominates. The tool says so in its output.

---

## 2. Code layout (additions)

```
configs/
  gases.yaml                  # gas registry: HITRAN id, molar mass, windows, range (vswir|lwir), quantify: bool
  sensors/hytes.yaml          # LWIR sensor; band centres from the HyTES product
  sensors/emit_l1b.yaml       # same bands as emit, radiance domain
  bad_bands.yaml              # + lwir sets (ozone 9.4-10.0 µm, water lines, detector edges)
  plumes.yaml                 # detection thresholds, min size, U_eff coefficients per sensor
src/specret/
  io/
    emit_l1b.py               # L1B RAD + OBS (geometry), same GLT orthorectify as emit.py
    avirisng.py               # AVIRIS-NG L1B radiance (ENVI, map grid) scattered to raw geometry via the GLT
    hytes.py                  # HyTES L1 radiance, L2 LST/emissivity
    emit_ghg.py               # EMIT L2B CH4ENH/CO2ENH rasters and CH4PLM/CO2PLM plume complexes (references)
    hitran.py                 # HAPI line-list fetch + cache -> data/raw/hitran/
    hitran_xsc.py             # HITRAN absorption cross-sections (VOCs, C2H6 VSWIR)
    wind.py                   # MERRA-2 / HRRR (+ keyless ERA5 cross-check) 10 m wind at a point and time
  gas/
    library.py                # HITRAN line-by-line cross-sections per layer -> data/derived/gas/lbl_<range>.npz
    detect.py                 # scene pipeline: windows, targets, screens, matched filter, outputs
    pipeline.py               # G2/G3 scene analysis: maps, candidates, identification, plume records
    tools.py                  # read-side helpers for MCP and the viewer (list, explain, requantify)
    lwir.py                   # G4 HyTES: thermal-contrast target, contrast mask, merge, identify, inject
    persistence.py            # G5 multi-overpass persistence of plume sources
    signatures.py             # unit absorption spectra per scene: VSWIR (AMF, H2O LUT), LWIR (thermal contrast)
    matched_filter.py         # column-wise robust mean/cov (torch), MF, ACE, two-pass plume exclusion
    identify.py               # joint multi-gas + confuser fit, BIC selection, calibrated confidence
    confusers.py              # surface-confuser signatures from the mineral library + mineral-map screen
    plumes.py                 # segmentation, source location, plume records
    quantify.py               # IME, U_eff, emission rate, uncertainty budget
    inject.py                 # synthetic plume injection into real radiance (benchmark + calibration)
    outputs.py                # COG layers + plumes.geojson / plumes.csv
  eval/
    ghg_compare.py            # vs EMIT L2B CH4ENH/CH4PLM; vs Carbon Mapper; vs controlled releases
    gas_benchmark.py          # injected-plume detection/identification/quantification metrics
```

Changes to existing code:
- **`sensors/sensor.py`:** add a `domain: reflectance | radiance` field and a `band_source: hytes`. Wavelengths stay in nm (7500–12000 nm is fine), with no wavenumber handling outside `gas/library.py`.
- **`workspace.py`:**
  - `SceneInfo.kind` gains `emit_l1b`, `avirisng_rad` and `hytes`;
  - `scenes()` discovers them, and pairs each L1B scene with its L2A sibling (same orbit and scene id) so a gas scene can find its mineral map;
  - new methods `detect_plumes`, `plume`, `explain_plume` and `quantify_plume`.
- **`scene/jobs.py`:** a generic `start_job(kind, ...)` so gas detection runs as a background job beside mineral mapping.
- **`project.py` (`setup`/`doctor`):** check for the gas library, HITRAN cache, wind access (the Earthdata login, plus whether HRRR can be reached) and L1B scenes, and report what's missing.
- **`cli.py`:** new commands `gas-library`, `detect-plumes <scene>`, `quantify <scene> --plume N [--wind u,v]`, `compare-ghg`, `inject-benchmark`.
- **`viewer`:**
  - a gas layer toggle (enhancement map with a diverging colour scale) and plume polygons;
  - clicking a plume shows the whitened residual against each candidate gas signature, the fitted gases, IME, Q ± σ, and the wind used;
  - a "mineral confuser" warning badge when the plume overlaps a carbonate or hydrocarbon map class.
- **MCP tools:**
  - `detect_plumes(scene_id, gases?, bbox?)`: an async job returning plume records;
  - `list_plumes(scene_id, min_q?)`;
  - `explain_plume(plume_id)`: fitted gases, per-window evidence, confusers considered, confidence;
  - `quantify_plume(plume_id, wind?)`: IME, Q, uncertainty budget;
  - `gas_signature(gas, sensor)`: the unit absorption spectrum, for the agent to reason with;
  - `scene_summary`, extended with a gas section.
- **Dependencies:** `hitran-api` (HAPI), `herbie-data` (HRRR, US only), `gcsfs` and `zarr` (keyless ERA5 cross-check), `scikit-image` (segmentation), `shapely` (plume polygons).

Outputs per scene go to `outputs/gas/<scene_id>/`:
- `ch4_enh.tif`, `co2_enh.tif`, … (ppm·m; LWIR gives column density times ΔT sign);
- `plume_mask.tif`, `gas_id.tif` and `confidence.tif`;
- `plumes.geojson` and `plumes.csv`, with one row per plume: gas, confidence, source lat/lon, IME, Q, σ_Q, wind, and confuser flags.

---

## 3. Phases

Effort assumes part-time work, as in PLAN.md. G1–G3 give a usable VSWIR methane and CO2 tool. G4 adds LWIR. G5 is hardening.

### Phase G0: Data and references (1 week)

**Status (2026-10-31):** mostly done; details in `data/raw/MANIFEST.md`.
- **EMIT:** seven scenes in `configs/gas_scenes.yaml`, fetched with `specret download emit-gas`: three CH4 plume scenes, two CO2 plume scenes, and Mountain Pass and Cuprite as no-plume scenes. All use L1B/L2A v001 and GHG v002.
- **HyTES:** Salton Sea, Kern River and Chino flights, fetched with `scripts/download_g0_hytes.py`. ORNL DAAC accepts the Earthdata login.
- **HITRAN:** line lists for 10 gases, fetched with `specret download hitran`.
- **Wind:** HRRR works with no account. MERRA-2 works through the Earthdata login (GES DISC approved 2026-11-01; Permian 2023-06-08 16:30 UTC reads 1.5 m/s against HRRR's 2.0 m/s). The keyless ERA5 cross-check is not verified yet.
- **PNNL:** not used. Nothing is bought; the LWIR library is HITRAN line lists plus HITRAN cross-sections.

Still open:
- HITRAN cross-section files (VOCs, C2H6 in VSWIR) are not fetched yet; they are needed by G4.
- No public per-release table with EMIT overpasses has been found yet.
- AVIRIS-NG radiance is deferred to G5 (open question 2).

- **EMIT:** download L1B RAD and OBS for 6–10 scenes with plumes in the EMIT L2B CH4PLM/CO2PLM catalogue, together with their L2B CH4ENH/CO2ENH and L2A RFL. Cover a spread of surfaces:
  - Permian Basin (oil and gas, bright soil);
  - Turkmenistan (super-emitters, desert);
  - a landfill;
  - a coal mine;
  - a power plant (CO2);
  - one bright carbonate or playa scene **with no plume**, to measure false positives.

  This adds `download emit --products L1B_RAD,L1B_OBS,L2B_CH4ENH,L2B_CH4PLM,...` to `io/download.py`. L1B RAD is about 2 GB per scene.
- **AVIRIS-NG radiance:** a few flights from the Carbon Mapper or JPL methane campaigns that have published plume lists.
- **Controlled releases:** the public blinded release datasets (Stanford's satellite single-blind release studies, 2022–2024, which include EMIT overpasses). Record the metered rates and the release times in `data/gas/releases.csv`.
- **HyTES:** sample flights with published gas detections, from the HyTES data portal. Candidates include CH4 from oil fields and dairies, NH3 from feedlots, and SO2 and NH3 from Salton Sea geothermal fumaroles. Verify availability first, because **this is the riskiest data item**.
- **Spectroscopy:**
  - HITRAN line lists through HAPI for CH4, CO2, H2O, N2O, NH3, H2S, CO, C2H6 and SO2;
  - HITRAN cross-sections for gases without line lists (VOCs, C2H6 in VSWIR).
- **Wind:** check that `earthaccess` can fetch MERRA-2 `M2T1NXSLV` for one scene's overpass hour, and that `herbie` can fetch HRRR for a US scene. No new account is needed.
- Update `data/raw/MANIFEST.md` with sources, DOIs and licences.

**Exit:** the scenes, references, spectroscopy and wind access are all on disk or working, with the manifest updated.

### Phase G1: VSWIR methane matched filter on EMIT (2 weekends)

**Status (2026-11-01): done.** All exit criteria pass; details in `results/gas_g1_ch4_mf.md`.
- **Agreement:** across 3 plume scenes, r over plumes is 0.93–0.97 and the Deming slope is 0.95–1.03.
- **Speed:** 11–14 s per scene.
- **Noise:** our background is about 30% lower than EMIT's, because of leave-one-out shrinkage on centred data, checked with injected plumes.
- **Deviations from the layout below:**
  - EMIT v002's wide band ranges (500–1340, 1500–1790, 1950–2450 nm) replace a CH4-only window;
  - the line-by-line cache is an npz per range in `data/derived/gas/`, not a parquet;
  - the pipeline lives in `gas/detect.py`, with the CLI commands `gas-library`, `detect-plumes` and `compare-ghg`.

- `io/emit_l1b.py`, `gas/library.py` (HITRAN only), `gas/signatures.py` (VSWIR) and `gas/matched_filter.py` (GPU, column-wise, two-pass).
- `detect-plumes` writes `ch4_enh.tif`.
- `eval/ghg_compare.py`: compare per pixel with EMIT L2B CH4ENH. Report correlation, slope and bias, and background σ.
- Tests:
  - the unit absorption spectrum is linear in α for small α;
  - the matched filter recovers a known injected α in a synthetic Gaussian background;
  - the column-wise statistics handle fill and bad columns.

**Exit:**
- Correlation with L2B CH4ENH r ≥ 0.85 over plume regions, with a slope within ±15%.
- A full EMIT scene takes under 1 min on the 4090.
- Any disagreement is written up in `results/gas_g1_ch4_mf.md`.

### Phase G2: Multi-gas identification and false-positive control, VSWIR (2–3 weeks)

**Status (2026-11-01): built; 2 of 3 exit criteria pass.** Details in `results/gas_g2_identification.md`.
- **False plumes:** down 97% on the no-plume scenes, from 309 to 8 (0.35–1.05 per 1000 km²).
- **Calibration:** ECE 0.027, AUROC 0.95.
- **Catalogue recall:** fails narrowly. It drops 6.1 points against the 5-point limit, losing 3 weak Turkmenistan plumes of 49, independent of the threshold.
- **Identification:** 88% of detected injected plumes get the right named gas, with no CH4/CO2 confusion.
- **CO2 maps vs EMIT CO2ENH:** r = 0.97–0.98. EMIT's single-pass product understates large plumes, which injected plumes confirm.
- **Deviations from the plan:**
  - only CH4 and CO2 are ever named; N2O, CO, NH3 and H2S are flag-only, and H2O is a nuisance term;
  - the L2B MIN carbonate screen is a recorded flag, not a filter, because Permian plumes sit on limestone;
  - the candidate's background covariance comes from shifted copies of its own mask;
  - candidates come from `gas/plumes.py` now, ahead of G3.

- **`gas/inject.py`:** inject synthetic plumes into real L1B radiance. Plume shapes are Gaussian-puff or LES-like masks, with known gas, α and footprint; the multiplicative Beer–Lambert term is applied at line-by-line resolution and then convolved. This is the benchmark's data, like `distort/` for minerals. Injection draws and calibration draws are kept separate.
- **`gas/confusers.py`:** build confuser signatures from the mineral library: carbonates, hydrocarbon-bearing materials, and plastics where splib07 has them. These are the surface spectra's own 2.3 µm structure, whitened the same way. Also read the paired mineral map: a plume pixel whose mineral top-1 is a confuser group is flagged.
- **`gas/identify.py`:** for each candidate plume, run a joint non-negative fit of the whitened residual against {CH4, CO2, H2O-excess, flag-only gases, confusers}, with greedy BIC selection and at most 3 terms. This reuses `fcls.nnls_fit` and the pursuit stopping rules. Output: the selected gases, their α, per-window evidence (how much each window contributes), and the confuser terms that were tried.
- **Confidence:** a logistic combiner (from `openset/calibrate.py`) over:
  - matched-filter SNR;
  - spatial coherence (plume-shaped versus speckle);
  - fit improvement versus confusers;
  - the window-consistency ratio (1.65 versus 2.3 µm for CH4).

  It's calibrated on injected plumes plus the no-plume scenes. "Unknown absorber" means the residual is significant but no gas term explains it.
- Add CO2: same pipeline, its own windows, and its own background σ (much higher in ppm·m).

**Metrics:**
- detection: probability of detection versus α at a fixed false-alarm rate per km²;
- identification: accuracy on injected multi-gas sets;
- false plumes per scene on the no-plume carbonate scene;
- agreement with the CH4PLM catalogue (recall, and precision against that catalogue).

**Exit:**
- The false-plume rate on the confuser scene drops by at least half against the plain matched filter, without losing more than 5 points of catalogue recall.
- Confidence is calibrated (ECE ≤ 0.1).

### Phase G3: Plume segmentation and quantification (2–3 weeks)

**Status (2026-11-01): built; the rate exit criterion is untestable without release data.** Details in `results/gas_g3_quantification.md`.
- **Wind:** `io/wind.py` reads HRRR in the US and MERRA-2 globally.
- **Quantification:** `gas/quantify.py` gives IME, U_eff = 0.45·U10 + 0.67 for EMIT (Ferrari et al. 2025), the upwind source, and Q with a four-term uncertainty budget.
- **Outputs:** plume records, `plumes.geojson`, `plumes.csv` and `candidates.csv`.
- **CLI:** `detect-plumes` runs the full pipeline, and `quantify` recomputes one plume's rate.
- **Integration:**
  - MCP: `detect_plumes`, `list_plumes`, `explain_plume`, `quantify_plume`, `gas_signature`, and a gas section in `scene_summary`;
  - viewer: a gas panel;
  - `doctor`: the gas checks.
- **Injected plumes:** IME is unbiased against the mass inside the mask, and the stated σ covers 80% within 1σ on held-out draws, after adding a measured 30% retrieval term.
- **Against EMIT catalogue rates:** ours are about 2.6× higher, from the estimator. EMIT integrates within one fetch length of the source with U10; we integrate the whole mask with U_eff/√area.
- **Diffuse plumes:** large ones are named CH4 only 58% of the time, against 97% for compact ones. That explains G2's three missed Turkmenistan plumes; pixel-weighted averaging is the follow-up.
- **Agent question:** the MCP criterion is met.

- `gas/plumes.py`, `io/wind.py` and `gas/quantify.py`. U_eff coefficients go into `configs/plumes.yaml`: start from published values for 30 m and 60 m pixels, and refit on injected LES plumes if any are available.
- Plume records, `plumes.geojson`, and the CLI `quantify` command.
- Integration:
  - viewer gas layers and a plume panel;
  - MCP `detect_plumes`, `list_plumes`, `explain_plume`, `quantify_plume` and `gas_signature`;
  - `scene_summary` gains a gas section;
  - `doctor` and `setup` cover the gas prerequisites.

**Metrics:**
- emission-rate error against controlled releases: bias, the fraction inside the 1σ and 2σ bands, and the detection limit;
- rate agreement with EMIT CH4PLM and Carbon Mapper published rates, where present.

**Exit:**
- Controlled-release estimates are unbiased within ±25% in aggregate, and about 68% fall inside the stated 1σ.
- An agent can answer "is anything leaking here, what is it, and how much?" through MCP.

### Phase G4: LWIR (HyTES) (3–5 weeks)

**Status (2026-11-01): built; 2 of 3 exit criteria pass, and the third is met on 2 of 3 flights.** Details in `results/gas_g4_lwir.md`.
- **Built:**
  - `io/hytes.py`;
  - the LWIR line-by-line cache (C2H4 added; SF6 is not available as lines);
  - `gas/lwir.py`, which has the thermal-contrast matched filter with per-pixel contrast correction, the low-contrast mask, candidates merged across gases, retrieve-then-fit identification, and thermal injection;
  - `eval/gas_lwir_benchmark.py`;
  - the `detect-plumes-lwir` command.
- **Injected plumes:** identification is 93.6% at SNR ≥ 5, with no cross-gas confusion.
- **Low-contrast mask:** works.
- **Published gases:**
  - Salton Sea NH3 at the mud pots: correct;
  - Chino NH3: named, but below the confidence threshold;
  - Kern River CH4: outranked by SO2, NH3 and N2O calls, probably surface emissivity with no TIR confuser terms available.
- **Fixed on the way:** linearly dependent ozone bands (9.4–9.9 µm excluded), and a heavy-tailed thermal background (stricter candidate thresholds).
- **Not done:** LWIR emission rates (no U_eff for HyTES resolution), TIR mineral confusers, cross-section gases, MCP and viewer for HyTES.

- **Data and sensors:** `io/hytes.py`, `configs/sensors/hytes.yaml`, and LWIR bad bands.
- **Library:** `gas/library.py` gains HITRAN cross-sections, resampled with the existing resampler at the HyTES band centres and FWHM.
- **Signatures:** `gas/signatures.py` gains the LWIR thermal-contrast target, using the HyTES L2 LST and emissivity for L_bg and weather-data air temperature for T_p. ΔT and low-contrast masks go to the outputs.
- **Detection:** the same matched filter (it's domain-agnostic once t is built), run once per gas in the registry.
- **Identification** is now a real library search over many gases. For each candidate plume:
  1. **Retrieve:** rank all library gases by whitened correlation with the plume's mean residual. This is the analogue of `gpu_exact`, and it's cheap.
  2. **Fit:** a joint non-negative fit over the top-N plus H2O and O3 with BIC, as in G2.
  3. **Score** with the calibrated combiner. Held-out gases are used for open-set calibration, mirroring Phase 4's held-out classes.
- **Quantification:** column density from α̂ / (B(T_p) − L_bg). Emission rate where wind is available, with the T_p uncertainty added to the budget.
- **Evaluation:**
  - injected plumes in real HyTES radiance across a ΔT sweep;
  - published HyTES detections (gas identity, plume location);
  - reported rates where they exist.

**Exit:**
- The right gas is top-1 on the published HyTES detections.
- Identification accuracy on injected plumes is ≥ 80% at a plume SNR ≥ 5.
- The low-contrast mask suppresses detections where |ΔT| < 1 K.

### Phase G5: Hardening (optional, 1–2 weeks)

**Status (2026-11-01): done for the items that need no new accounts.** Details in `results/gas_g5_hardening.md`.
- **ISOFIT and sRTMnet: not needed.** Against EMIT's MODTRAN-based product, our scale holds within ±2% over 0–1.65 km elevation and 1.5–2.1 g/cm² water.
- **AVIRIS-NG radiance:** `io/avirisng.py` scatters the map-grid radiance back into raw detector geometry through the GLT.
  - Against the published CH4 map: r = 0.88–0.90 with the published method's 2.1–2.5 µm window. Our wider default keeps a better SNR (5.1 against 3.7).
  - Injected recovery is 0.73. That gain, and the confidence model and U_eff, are not yet calibrated for this sensor.
  - Found on the way: the OBS file swaps sun zenith and azimuth.
- **Persistence:** `gas/persistence.py` and `gas-persistence`. On four Permian overpasses, 9 of the 57 covered plumes recur; that is too few to show a precision gain.
- **Cross-sectional flux:** added as a second rate estimate (open question 3). It comes out at 0.62× the IME rate on average, and 0.47× for plumes of 100 px or more.
- **Not done:** PRISMA and EnMAP (they need accounts).

- **AVIRIS-NG and other VSWIR sensors:** add PRISMA and EnMAP radiance loaders. The configs exist or are cheap to add.
- **Multi-overpass persistence:** the same source seen in several scenes raises confidence.
- **ISOFIT or sRTMnet signatures:** replace the line-by-line plus standard-atmosphere model only if G1–G3 residuals show a systematic bias with water vapour or elevation. This would also unblock PLAN.md §3.2.

---

## 4. Evaluation matrix

| Level | Data | Methods (rows) | Metrics |
|---|---|---|---|
| Injected, VSWIR | Real EMIT L1B + synthetic plumes | MF, MF+albedo, mag1c-style sparse, joint fit | Pd at fixed FAR, α bias/RMSE, ID accuracy |
| Real, EMIT | Plume scenes vs L2B CH4ENH / CH4PLM | MF, joint fit | Pixel r/slope, plume recall and precision vs catalogue |
| False positives | Confuser scene(s), no plume | MF, joint fit, + mineral screen | False plumes per 1000 km² |
| Quantification | Controlled releases; EMIT PLM / Carbon Mapper rates | IME + U_eff | Bias, 1σ/2σ coverage, detection limit |
| Injected, LWIR | Real HyTES + synthetic plumes, ΔT sweep | MF per gas, retrieve+fit | ID top-1/top-3, open-set AUROC |
| Real, LWIR | Published HyTES detections | Retrieve+fit | Gas ID, location agreement |

The L2B products and catalogues come from JPL's own matched filter. As with Tetracorder (PLAN.md Decision 7), agreement measures agreement with that algorithm, not with the truth. Controlled releases are the independent check.

---

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| HyTES data is sparse, hard to find, or lacks published gas labels | G4 has weak validation | Check availability in G0, before writing any LWIR code. Fall back on injected plumes and any published HyTES case studies, and keep G4 explicitly provisional |
| No PNNL (not buying it) | Fewer LWIR gases than PNNL offers | Use HITRAN cross-sections, which include many PNNL-derived spectra; report gases outside the library as unidentified |
| Wind error dominates the emission rate | Wide uncertainty bars | Report it honestly in the budget; allow user wind; prefer HRRR over MERRA-2 in the US; use the spread between sources as part of the wind uncertainty |
| Surface confusers (carbonate, hydrocarbons) create false CH4 plumes | Loss of trust | Joint fit with confuser terms plus the mineral-map screen (G2), and a dedicated no-plume test scene |
| L1B volume (~2 GB/scene) and memory | Slow or out-of-memory runs | Stream by row blocks, as the mineral mapper does; column statistics need only one pass over the cube |
| LWIR thermal contrast near zero | Silent misses | An explicit low-contrast mask and a "cannot detect" state in the outputs and MCP answers |
| Flag-only VSWIR gases over-reported | Misleading IDs | Never quantify them; require an LWIR or strong multi-window match before naming them with a confident tier |
| Controlled-release overpasses are few | Thin quantification statistics | Pool all public release studies; also report agreement with EMIT and Carbon Mapper rates as a secondary check |

---

## 6. Open questions

1. **Which HyTES flights:** pick them once the G0 availability check is done.
2. **AVIRIS-NG radiance in G1, or wait for G5?** The default is G5, keeping G1 EMIT-only, like Phase 3.
3. **Emission-rate method:** IME is the default. A cross-sectional flux method could be added in G5 as a second estimate for large plumes.
