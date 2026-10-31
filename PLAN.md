# Spectral Library Retrieval: Project Plan

**Goal:** a working tool that takes a measured spectrum (an EMIT or AVIRIS pixel, or a field spectrometer reading) and returns the most likely materials from a reference library. Each answer comes with a calibrated confidence, an "unknown" option, and abundances for mixed pixels, and the tool is usable from a viewer and from an LLM agent through MCP tools.

**What success means:** on real Cuprite scenes from both EMIT and AVIRIS, and on public field spectra, the tool beats a plain SAM baseline by a clear margin. It also says "unknown" instead of guessing when a material isn't in the library. Every results table includes the SAM row, because that's how you know the extra machinery is earning its keep.

**Local environment (checked 2026-10-31):** Windows 11, Python 3.11.9, `uv` 0.12.7, Docker 29.3, RTX 4090 (24 GB). The project folder is not a git repo yet.

## Decisions

| Question | Decision | Effect on the plan |
|---|---|---|
| Target outcome | **Working tool** | The usable path comes first: physics reranking, then scene mapping, viewer and MCP, then unmixing and open-set rejection. The learned encoder is a later, optional upgrade, adopted only if it beats the physics pipeline on real data. Paper-only work (sparse-unmixing and MESMA comparisons, literature review, full ablations) is dropped or optional. |
| Primary sensor | **Both EMIT and AVIRIS** | Both loaders and sensor configs are built in Phase 1, and every real-data evaluation reports both. AVIRIS covers AVIRIS-Classic Cuprite and AVIRIS-NG where it's available. |
| Field spectrometer data | **None owned: use public ASD data** | ASD binary (`.asd`) is the target format because public mineral field datasets exist in it (Section 3.4). A CSV/ASCII reader covers exported data too. Field spectra become a labelled real-world test set. |
| Radiative-transfer engine | **ISOFIT + sRTMnet (default, not yet confirmed)** | Used only for the atmospheric residual bank in Phase 5. htrdr was considered and set aside. |

---

## 0. Design decisions

1. **Library scale makes approximate search optional.** splib07 plus ECOSTRESS VSWIR comes to a few thousand spectra. Exact search against that is one GPU matrix multiply: 1.5M pixels × 10k library spectra × 285 bands takes under a second on the 4090.
   - **Scene-scale mapping uses exact batched GPU search** (`torch` matmul) against an in-memory library matrix, not Qdrant.
   - **Qdrant is the serving and filtering layer** for interactive queries, payload filters, multiple representations and the MCP tools. It's a derived index rebuilt from the canonical store, never the source of truth.
2. **Qdrant named vectors have a fixed length per collection.** EMIT (285 bands), AVIRIS-Classic (224) and AVIRIS-NG (~425) can't share a `sam` vector. Use **one collection per sensor** (`lib_emit`, `lib_avirisc`, `lib_avirisng`, and `lib_field` for field instruments), each with named vectors `sam`, `cr` (continuum-removed), `deriv` and `swir2` (see item 3). A learned embedding, if Phase 5 adds one, is the same length for every sensor and gets a shared collection, `lib_learned`.
3. **Partial coverage.** Some library spectra don't cover the full VSWIR range, and a vector database can't apply a per-point band mask. Full-coverage spectra go into the full vectors. Everything that covers 2.0–2.5 µm also gets a `swir2` vector: that window holds most of the diagnostic mineral features (Al-OH, Mg-OH, carbonate) and avoids the 1.4 and 1.9 µm water bands. Store `coverage_min_nm` and `coverage_max_nm` in the payload for filtering.
4. **Three fixes to the resampler from the original brief:**
   - Weight by Δλ (trapezoidal integration), because library wavelength grids aren't uniform.
   - Remove USGS deleted-value markers (`-1.23e34`) and NaNs before resampling. Mark a band invalid when most of its response function falls outside the library's valid range, rather than silently renormalizing.
   - Where the library's own resolution approaches the sensor's (parts of the Beckman SWIR, for example), convolve with an effective width of `sqrt(fwhm_sensor² − fwhm_lib²)`. splib07 ships per-instrument bandpass files for this.
5. **Residual queries in the unmixing loop.** A fit residual can be negative and isn't a reflectance spectrum. The unmixing loop therefore matches residuals by direct inner product against unit-length library spectra (non-negative orthogonal matching pursuit). Retrieval candidates seed only the first iteration.
6. **One distortion simulator, two jobs.** The physics-based simulator built in Phase 2 drives the synthetic benchmark, and later the encoder's training augmentation if Phase 5 goes ahead. The two must use separate random draws. Synthetic results are a sanity check only. Real scenes and field spectra decide whether a change ships.
7. **Tetracorder overlap.** The USGS Cuprite maps and the EMIT L2B mineral maps both come from Tetracorder, which uses the same splib library and expert-defined features. Agreement with them measures agreement with Tetracorder, not ground truth. Reusing Tetracorder's feature definitions for reranking pushes agreement up further. Report it as "agreement with Tetracorder". Public field spectra are an independent check (Section 3.4).

---

## 1. Repository layout

```
HyperspectralIndexing/
├── pyproject.toml              # uv-managed; package "specret"
├── docker-compose.yml          # qdrant
├── configs/
│   ├── sensors/                # emit.yaml, aviris_classic.yaml, aviris_ng.yaml, asd_fieldspec.yaml, prisma.yaml
│   ├── taxonomy.yaml           # mineral -> group -> class, curated (see 3.1)
│   ├── bad_bands.yaml          # shared water/edge masks per sensor (incl. field instruments)
│   └── features.yaml           # diagnostic absorption windows per material (Phase 2)
├── src/specret/
│   ├── schema.py               # Spectrum record + parquet schema
│   ├── io/                     # usgs_splib07.py, ecostress.py, emit.py, aviris.py, field.py
│   ├── sensors/                # response-function models, resample.py
│   ├── preprocess/             # masks.py, continuum.py, derivative.py, normalize.py, splice.py
│   ├── search/                 # gpu_exact.py, qdrant_store.py, fusion.py (RRF)
│   ├── rerank/                 # features.py, sff.py
│   ├── distort/                # mixing, photometric, noise, atmosphere bank, sensor jitter
│   ├── scene/                  # cluster.py, mapper.py, outputs.py
│   ├── unmix/                  # fcls.py, pursuit.py, model_selection.py
│   ├── openset/                # scores.py, calibrate.py
│   ├── learn/                  # Phase 5 only: tokenizer, models/, losses.py, train.py
│   ├── eval/                   # splits.py, metrics.py, benchmarks/
│   ├── viz/                    # match plots, feature highlighting, small web viewer
│   └── mcp/server.py           # FastMCP tools
├── notebooks/                  # 01_ingest_check, 02_cuprite_mvp, 03_field_check, ...
├── tests/
└── data/                       # gitignored
    ├── raw/                    # untouched downloads, one folder per source
    ├── field/                  # public field spectra + metadata sheet
    ├── canonical/              # library.parquet (native resolution, source of truth)
    ├── resampled/              # <sensor>/library.npz (matrix + band mask + ids)
    ├── scenes/                 # EMIT / AVIRIS-Classic / AVIRIS-NG
    └── refmaps/                # Tetracorder / L2BMIN references, co-registered
```

**Canonical record** (one row per library spectrum in `library.parquet`):
`spectrum_id, sample_id, source, source_record, material_name, mineral, group, class, chapter (USGS M/S/C/L/O/A/V), grain_size, purity_flag, purity_note, instrument, wavelength_nm[], reflectance[] (0–1), valid_mask[], coverage_min_nm, coverage_max_nm, native_fwhm_nm[]`

`sample_id` is the leakage-control key. Spectra of the same physical sample under different instruments or grain-size splits share a `sample_id`.

---

## 2. Phases

Effort estimates assume part-time work. Phases 1–3 produce a usable tool. Phase 4 makes it trustworthy on mixed and unfamiliar pixels. Phase 5 is an optional upgrade.

### Phase 0: Setup and data acquisition (2–4 days)

- `git init`, uv project, pre-commit with ruff, pytest, and `docker-compose.yml` for Qdrant. Tests use qdrant-client's local mode.
- **NASA Earthdata account.** You need to create this yourself. It's required for EMIT downloads through `earthaccess`.
- Downloads, each into its own `data/raw/<source>/`:
  - **USGS splib07** (ScienceBase data release): splib07a (measured), splib07b (oversampled), the AVIRIS-Classic convolved sets, the wavelength and bandpass files, and the HTML purity/description pages.
  - **ECOSTRESS library:** the VSWIR mineral, rock, soil and vegetation files. Reflectance there is stored in **percent**.
  - **EMIT L2A reflectance** granules over Cuprite (~37.53°N, 117.20°W), plus the matching **EMIT L2B mineral** granules. Pick a clear, low-water-vapour acquisition.
  - **AVIRIS-Classic Cuprite** (1995/1997 reflectance) and **AVIRIS-NG** Cuprite flights, if any exist.
  - The published USGS Tetracorder Cuprite maps, with a plan for georegistering them. This is the riskiest item, so start it first.
- **Field data:** download a public ASD dataset (Section 3.4) into `data/field/raw/`. Fill in `data/field/metadata.csv` with one row per spectrum: file, instrument, date, lat/lon if known, the material or sample if known, and notes.
- Defer RELAB and EcoSIS until there's a concrete need.

**Exit:** all raw data is on disk with a `data/raw/MANIFEST.md` listing source URL, version or DOI, date and licence. Qdrant is running and the field metadata sheet exists.

### Phase 1: Ingestion and SAM retrieval on both sensors and field data (2–3 weekends)

- `io/usgs_splib07.py`: parse the ASCII spectra and wavelength files, map the chapter codes, turn deleted-value markers into NaN, and parse purity notes into `purity_flag`.
- `io/ecostress.py`: parse the header block, convert percent to 0–1, and handle descending wavelengths.
- **Reflectance scale check:** reject or flag any spectrum whose median falls outside [0, 1.5] after normalization.
- **Deduplicate:** collapse identical arrays within a source. Across sources, flag near-duplicates (SAM < 0.5° plus a matching name) for review rather than deleting them.
- `sensors/resample.py` with the fixes from Decision 4. Read band centres and FWHM from each product rather than hardcoding them: EMIT from the L2A `sensor_band_parameters` group, AVIRIS from the ENVI header.
- **Resampler check:** resample splib07 to AVIRIS-Classic with this code and compare against USGS's own AVIRIS-convolved version. Target a median relative error below 1% and investigate per-band outliers. splib07 has no EMIT-convolved set, so this AVIRIS comparison is the check.
- **Scene loaders:**
  - `io/emit.py`: L2A reflectance, `good_wavelengths` and `reflectance_uncertainty`, orthorectified with the GLT.
  - `io/aviris.py`: ENVI reflectance cubes for AVIRIS-Classic and AVIRIS-NG via the `spectral` package, plus a scale check (older products store scaled integers). AVIRIS products often have no per-pixel uncertainty, so estimate noise from the scene, for example from differences between neighbouring pixels in uniform areas.
- `io/field.py`: see Section 3.4.
- `search/gpu_exact.py`: L2-normalize after masking, then cosine top-k. `qdrant_store.py` builds the `sam` vector for each sensor collection. A test checks that Qdrant and GPU-exact search return the same ranking.
- Notebooks:
  - `02_cuprite_mvp`: pick an EMIT or AVIRIS pixel and plot it against its top-5 matches. Check the known Cuprite zones: alunite, kaolinite, buddingtonite (the "Buddingtonite Bump"), calcite, muscovite, opal/chalcedony.
  - `03_field_check`: the same plot for a field spectrum.

**Exit:**
- The resampler check passes.
- Top-1 SAM matches the expected mineral group for most hand-picked pixels from the known Cuprite zones, on both EMIT and AVIRIS.
- Field spectra ingest cleanly.
- Failures are written down, because they motivate Phase 2.

### Phase 2: Physics reranking and a benchmark (2–3 weeks)

**2a. Taxonomy and splits (do this before tuning anything, then freeze it)**
- `configs/taxonomy.yaml`: hand-curate mineral → group → class for every splib07 and ECOSTRESS mineral name (kaolin group, alunite–jarosite, smectite, white mica, chlorite, carbonate, Fe-oxide/hydroxide, sulfate, silica, amphibole, ...). Expect a few hundred names. It's tedious, but the tool's group-level answers and its metrics both depend on it.
- `eval/splits.py`: hold out test samples by `sample_id` and hold out whole mineral classes for open-set testing. Save the split with a hash and never regenerate it.

**2b. Distortion simulator (`distort/`), kept lean for the tool**
- Linear areal mixtures with the dominant fraction ≥ τ.
- Brightness scaling, slight slope tilt, and a dark (shade) endmember.
- Wavelength-dependent noise: from EMIT `reflectance_uncertainty`, from the AVIRIS scene-based estimates, and from the field instrument's noise.
- Small band-centre shifts (±1 nm) and FWHM changes (±10%).
- Simple parametric atmospheric residuals: Gaussian-shaped errors at 0.94, 1.14, 1.4 and 1.9 µm.
- Named severity recipes: "clean", "noise", "mix-70", "atm-mild", "atm-strong", "all".

**2c. Physics representations and reranking**
- `preprocess/continuum.py`: a convex hull over the masked spectrum, plus local continuum removal around each feature.
- `preprocess/derivative.py`: a Savitzky–Golay derivative with respect to wavelength that handles irregular band spacing.
- `configs/features.yaml`: diagnostic windows (continuum endpoints and expected centre) for each material. Bootstrap from Tetracorder's open-source command files where possible, keeping the overlap from Decision 7 in mind.
- `rerank/sff.py`: for each candidate and feature, measure band centre, depth, width and asymmetry, and least-squares fit the scaled library feature to the query. Combine these into a score and **keep the per-feature breakdown**: it powers `explain_match` and the viewer.
- `search/fusion.py`: reciprocal rank fusion, Σ 1/(60 + rank), over the `sam`, `cr` and `swir2` rankings.
- Baselines in the same harness: SAM, Spectral Information Divergence (SID), Euclidean distance on the continuum-removed spectrum, and SFF on its own.

**Metrics:** Recall@1/5, MRR, hierarchical accuracy (exact / group / class), and a confusion matrix for the known hard pairs: kaolinite/dickite/halloysite, K/Na-alunite, muscovite/illite, calcite/dolomite, hematite/goethite. Report results on synthetic data, on Cuprite (EMIT and AVIRIS), and on labelled field spectra.

**Exit:** the default pipeline, RRF(sam, cr, swir2) followed by the SFF rerank, beats SAM on the hard pairs and on real data from both sensors. This becomes the tool's production retrieval.

### Phase 3: Scene mapping, viewer and MCP tools (2–3 weeks)

- `scene/mapper.py`: run exact GPU search on **every** pixel, since it's cheap (Decision 1). Run the SFF rerank only on cluster centroids (k-means or SLIC superpixels in a combined spectral and spatial space) and on low-confidence or boundary pixels.
- The same mapper works on EMIT and AVIRIS scenes through the sensor configs.
- Outputs as cloud-optimized GeoTIFF:
  - top-k material IDs and scores;
  - dominant mineral group;
  - a confidence layer (a provisional heuristic until Phase 4 calibrates it).
- `viz/`: overlay the query on its top matches, with diagnostic windows shaded and continuum-removed insets. Also build a small local web viewer (Panel, or FastAPI with Plotly): click a pixel on the scene's mineral map to see its spectrum and matches, or drop a field spectrum file onto it.
- `mcp/server.py` (FastMCP) tools:
  - `identify_spectrum(wavelengths, reflectance, sensor?, filters?)` returns ranked matches, scores and confidence.
  - `explain_match(query, spectrum_id)` returns the per-feature breakdown, for example "2.165/2.205 µm doublet, depth 0.18".
  - `identify_field_spectrum(path)` ingests a field file and calls `identify_spectrum`.
  - `map_scene(scene_id, bbox?)` runs as an async job, returning a job ID and later the output paths.
  - `scene_summary(scene_id, region)` returns the area fraction of each mineral group.
- Tools return structured JSON plus an optional plot path. They never report a material that isn't in the library.

**Exit:**
- A full EMIT scene and an AVIRIS Cuprite scene map end to end, and the map runs in minutes rather than hours.
- The viewer works.
- An agent can answer "what minerals are in this region, and why" through the MCP tools.

### Phase 4: Mixed pixels and unknown materials (2–4 weeks)

This phase is what makes the tool trustworthy on real pixels.

**Unmixing (`unmix/`)**
- FCLS: non-negative fractions, with an optional sum-to-one constraint via the weighted augmented-row trick, plus a shade endmember.
- `pursuit.py` runs the loop:
  1. Seed with the top-k candidates from the Phase 2 pipeline and fit.
  2. Compute the residual and pick the next spectrum by inner product with the residual (Decision 5).
  3. Refit and repeat.
  Each iteration keeps at most one spectrum per mineral, so near-duplicates don't pile up.
- **Stopping rule:** a χ² test of the residual against the noise estimate (EMIT uncertainty, AVIRIS scene noise, or field instrument noise), a BIC penalty for each added endmember, and a hard cap of 3–4 components.
- **Optional:** a quick comparison with sparse unmixing (SUnSAL) on Cuprite. Useful to know where the tool stands, but not required.

**Unknown-material detection (`openset/`)**
- Scores: top-1 similarity, the margin between the top two matches, the unmixing residual χ², and SFF fit quality.
- Calibrate on held-out *classes* (not just held-out samples) with a logistic combiner, or with conformal prediction to get set-valued answers such as "one of {kaolinite, dickite}" or "unknown".
- Replace the provisional Phase 3 confidence layer with the calibrated one, and add an unknown mask and abundance layers to the scene outputs.
- New MCP tool: `unmix_pixel(scene_id, x, y | lon, lat)` returns endmembers, fractions and the residual test result.

**Metrics:** component recall and abundance RMSE on synthetic mixtures; unknown-detection AUROC and accuracy at a fixed coverage; calibration error (ECE).

**Exit:** the tool reports "unknown" for held-out classes at a useful rate (target AUROC ≥ 0.85) without a large accuracy loss on known materials, and its confidence numbers are calibrated.

### Phase 5 (optional upgrade): learned sensor-agnostic encoder

Pursue this only if Phases 2–4 leave a clear gap, typically on heavily mixed or atmosphere-affected pixels, or on sensors without per-sensor tuning.

- **Model:** one token per valid band, `(λ, ρ, fwhm)`, with sinusoidal wavelength encoding, feeding a small transformer (about 1–5M parameters). One model then serves EMIT, both AVIRIS sensors and field instruments.
- **Training:** supervised contrastive loss with hierarchical positives, so same sample < same mineral < same group < different class in distance. Self-supervised pretraining on real EMIT and AVIRIS pixels by masked-band reconstruction helps against overfitting a small library.
- **Augmentation:** the Phase 2 simulator, upgraded with the radiative-transfer residual bank (Section 3.2).
- **Integration:** a `learned` vector added into RRF alongside the physics vectors. Keep the SFF rerank, which preserves explainability.
- **Ship rule:** adopt the encoder only if it improves real-scene and field-spectrum accuracy over the Phase 2 pipeline. Synthetic gains alone don't count.

---

## 3. Cross-cutting components

### 3.1 Taxonomy curation
This is manual work and easy to underestimate, so budget time for it in Phase 2a. Record each decision in `taxonomy.yaml` with a comment. The group-level answers the tool gives are only as good as this file.

### 3.2 Radiative-transfer atmospheric residual bank (Phase 5 only)
- **Engine:** ISOFIT with the sRTMnet emulator, by default. ISOFIT produces EMIT's L2A reflectance, and sRTMnet removes the need for a MODTRAN licence. Run it in WSL2 or a Linux container.
- **Procedure:**
  1. Simulate at-sensor radiance with the true atmospheric state.
  2. Invert with a perturbed state (wrong water vapour or aerosol amount, small calibration errors).
  3. Record the reflectance error.
  4. Fit it as a multiplicative-plus-additive residual spectrum.
  5. Store about 10⁴ residual pairs to sample from during training.
- **Check:** compare the bank's residual statistics with EMIT pixels over flat, well-characterized targets such as playas.

### 3.3 Reproducibility and tests
- Splits keyed by `sample_id`, hashed and frozen in Phase 2a.
- Config-driven experiments that log the git SHA, data manifest hash and split hash with each run. A simple JSON results log is enough for a tool; MLflow is optional.
- Tests:
  - the resampler against the USGS convolved spectra;
  - continuum removal on synthetic Gaussian absorptions, checking that it recovers the right centre and depth;
  - FCLS on known synthetic mixtures;
  - Qdrant ranking matches GPU-exact ranking;
  - field file parsing against a known-good reference export.

### 3.4 Field spectrometer data
- **Formats (`io/field.py`):** ASD binary `.asd` (a direct parser, including the white reference stored in version 6+ files) and CSV/ASCII exports in long or EcoSIS-style wide layout. SVC `.sig` and Spectral Evolution `.sed` can be added if a dataset needs them.
- **Public ASD datasets** (no field data of our own yet):
  - **Coso geothermal alteration minerals** (OpenEI): alunite, chalcedony, epidote, hematite, kaolinite and opal, with raw `.asd` plus ASCII copies. The closest match to the Cuprite mineral suite. Check its licence before redistributing anything.
  - **USGS Smith Creek Valley, Nevada** (DOI 10.5066/P9E2TSDF): ASD Indico `.asd` files of field samples, with CSV metadata. Public.
  - **USGS Black Rock Desert, Nevada, August 2022** (DOI 10.5066/P94QAHLC): ASD FieldSpec 4 Hi-Res NG over a calibration playa, CC0. Useful for the lab-to-image chain if an EMIT or AVIRIS overpass is close in time.
- **Validate the ASD reader** on one of these before trusting it: it has only been tested on synthetic files. The Coso ASCII copies give a direct check.
- **Preprocessing:**
  - Splice correction at the detector joins (about 1000 and 1800 nm on ASD instruments), in `preprocess/splice.py`.
  - Convert radiance or DN to reflectance using the white-reference panel if a file isn't already reflectance.
  - Mask the atmospheric water bands (about 1350–1450 nm, 1800–1950 nm, and >2400 nm when noisy).
  - Optionally average replicate readings.
- **Matching:** field spectra are near library resolution, so they get their own sensor config (`asd_fieldspec.yaml`, with FWHM of about 3 nm in the VNIR and 8–10 nm in the SWIR) and collection (`lib_field`).
- **Evaluation uses:**
  1. Field spectra with known materials are a labelled test set that's independent of Tetracorder, which is the strongest real-data check the tool has.
  2. If any were taken at Cuprite or inside an EMIT or AVIRIS footprint, compare each field reading's top match with the matching pixel's result. That tests the full lab-to-image chain.
  3. Their noise characteristics feed the field noise model in the distortion simulator.

### 3.5 Scope boundaries
- VSWIR only (0.38–2.5 µm). Thermal emissivity is out of scope.
- Linear areal mixing only.
- RELAB, EcoSIS and vegetation-specific work wait until after Phase 4.
- PRISMA is a configuration file only, unless a need comes up.

---

## 4. Evaluation matrix

| Level | Data | Methods (rows) | Metrics |
|---|---|---|---|
| Synthetic | Held-out samples × distortion recipes | SAM, SID, CR-Euclid, SFF, hybrid (+ learned, if Phase 5) | R@1/5, MRR, hierarchical accuracy |
| Hard pairs | Same data, filtered to the hard pairs | Same | Pairwise confusion |
| Real: EMIT | Cuprite EMIT vs. L2B mineral maps and Tetracorder | All retrieval rows | Per-mineral and group agreement |
| Real: AVIRIS | Cuprite AVIRIS-Classic and NG vs. Tetracorder maps | All retrieval rows | Per-mineral and group agreement |
| Field | Labelled public field spectra (Coso first) | All retrieval rows | Top-1/top-5 accuracy, group accuracy |
| Cross-sensor consistency | The same Cuprite ground seen by EMIT and AVIRIS | Hybrid | Group-level agreement between the two sensors' maps |
| Mixtures | Synthetic mixtures with known fractions | Hybrid pursuit (+ SUnSAL, optional) | Component recall, abundance RMSE |
| Unknown materials | Held-out classes | Each score and the combined score | AUROC, accuracy at coverage, ECE |

---

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| USGS Cuprite reference maps are hard to obtain or georegister | Weaker AVIRIS real-data numbers | Start the search in Phase 0. Fall back on EMIT L2B maps, which come on the EMIT grid, and lean on field spectra |
| Public field datasets are small, unlabelled, or far from any overpass | Field test set is weak | Prefer the Coso set (labelled minerals). Record what's known in the metadata sheet. Even unlabelled spectra exercise ingestion and splice correction |
| AVIRIS product formats and scaling vary between years | Wrong reflectance scale | Add a scale check to the loader and test against a known pixel |
| Tetracorder overlap inflates agreement | Results look better than they are | Report it as agreement with Tetracorder, and use field spectra as the independent check |
| Synthetic simulator is too clean | Choices tuned to the simulator don't carry over | Decide what ships from real-scene and field results, not synthetic numbers |
| Radiative-transfer tools on Windows | Lost setup time | Only needed in Phase 5. Use WSL2 or a Linux container |

---

## 6. Open questions

1. **Radiative-transfer engine:** ISOFIT with sRTMnet is the default. Only needed in Phase 5.
2. **Which public field dataset first:** Coso (labelled alteration minerals) is the default.

---

## 7. First-week checklist

- [x] `git init`, uv project skeleton, ruff, pytest, `docker-compose.yml` for Qdrant
- [x] Start the Qdrant server (`docker compose up -d`) and run `specret index` against it
- [x] Create the Earthdata account and test the `earthaccess` login
- [x] Download splib07a and the AVIRIS-2014-convolved set
- [ ] Download the ECOSTRESS VSWIR files (request form, by hand)
- [x] Find the EMIT L2A and L2B mineral granules over Cuprite and download one clear scene (2024-07-27); first comparison in `results/phase1_emit_cuprite.md`
- [ ] Find AVIRIS-Classic and AVIRIS-NG Cuprite reflectance, and start searching for the USGS Tetracorder maps
- [ ] Download a public ASD field dataset into `data/field/raw/` and fill in `metadata.csv`
- [x] Write the splib07 parser, `library.parquet`, and the resampler with its validation test (median relative error 0.004% against USGS's AVIRIS-2014 convolution)
