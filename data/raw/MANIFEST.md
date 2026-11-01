# Raw data manifest

One entry per source: where it came from, version, when it was fetched, and its licence.

## Downloaded

### USGS Spectral Library Version 7 (splib07)
- **Source:** ScienceBase item 586e8c88e4b0f5ce109fccae, USGS Data Series 1035 (https://doi.org/10.3133/ds1035), Kokaly et al. (2017)
- **Files:** `ASCIIdata_splib07a.zip` (21.8 MB, measured spectra), `ASCIIdata_splib07b_cvAVIRISc2014.zip` (4.3 MB, convolved to AVIRIS-Classic 2014 band passes)
- **Fetched:** 2026-10-31, with `specret download usgs`
- **Licence:** US Government public domain
- **Location:** `data/raw/usgs_splib07/` (zips) and `data/raw/usgs_splib07/extracted/`

### USGS splib07 HTML metadata (sample descriptions)
- **Source:** ScienceBase item 586e8d4de4b0f5ce109fccbb, `HTMLmetadata.zip` (1.35 GB, mostly photos)
- **Fetched:** 2026-10-31; only the .html/.htm/.pdf members were extracted (16 MB) and the zip was deleted
- **Licence:** US Government public domain
- **Location:** `data/raw/usgs_splib07_html/pages/`

### EMIT scenes for Phase 5b (pretraining and validation)
- **Source:** NASA LP DAAC (EMITL2ARFL, EMITL2BMIN), fetched with `scripts/download_phase5b.py`
- **Granules:** Pilbara `20240716T063940_2419804_012`, Kalahari `20230524T105835_2314407_002`, Sar Cheshmeh `20230626T070735_2317705_009` (RFL, MASK, RFLUNCERT); Mountain Pass `20240416T210255_2410714_015` (RFL, MASK, RFLUNCERT, L2B MIN)
- **Fetched:** 2026-10-31, ~15 GB (the uncertainty granules came from a since-fixed download filter)
- **Location:** `data/scenes/emit_extra/`; sampled pixel pools in `data/derived/pretrain/`

### EMIT scenes for gas plume work (PLAN_GAS.md, Phase G0)
- **Source:** NASA LP DAAC. L1B radiance and observation geometry (EMITL1BRAD v001), L2A reflectance and masks (EMITL2ARFL v001), L2B minerals (EMITL2BMIN v001), methane and CO2 enhancement with sensitivity and uncertainty (EMITL2BCH4ENH, EMITL2BCO2ENH v002), and plume complexes (EMITL2BCH4PLM, EMITL2BCO2PLM v002)
- **Scenes:** listed with site, role and products in `configs/gas_scenes.yaml`: two Permian Basin and one Turkmenistan CH4 scene, two CO2 power-plant scenes (Ningxia, Madhya Pradesh), and Mountain Pass and Cuprite as no-plume scenes
- **Chosen:** 2026-10-31, by ranking scenes on their number of CH4PLM/CO2PLM plume complexes in CMR metadata, then cloud cover and solar zenith
- **Fetch:** `specret download emit-gas` (`--dry-run` lists the URLs)
- **Plume metadata:** each `*PLMMETA*.json` (GeoJSON) holds, for the plume complex:
  - the outline and the maximum concentration (ppm·m) with its location;
  - for many plumes, an emission rate with uncertainty (kg/h), the wind speed and its source (HRRR), and the fetch length. Others give "NA".

  These rates are the G3 comparison set until controlled-release data is found.
- **Licence:** NASA open data, no restrictions
- **Location:** `data/scenes/emit_gas/<scene id>/`

### HyTES thermal-infrared flights (PLAN_GAS.md, Phase G0)
- **Source:** ORNL DAAC: L1 radiance (`HyTES_L1_Radiance_2013_2016_2476`, ENVI, 256 bands 7.45–12.0 µm), L1 geolocation (`HyTES_L1_Geolocation_2013_2016_2467`, CSV) and L2 surface temperature and emissivity (`HyTES_L2_Emissivity_LST_2460`, HDF5). Downloads work with the Earthdata login.
- **Flights:** Salton Sea 2014-07-06 (geothermal; NH3), Kern River oil field 2014-07-08 (CH4), Chino dairies 2013-04-26 (NH3, CH4; L1 only, no geolocation or L2). These are sites with published HyTES gas detections (Hulley et al. 2016, AMT 9, 2393).
- **Fetch:** `uv run python scripts/download_g0_hytes.py`, ~20 GB (fetched 2026-10-31; sizes match CMR)
- **L2 contents:** land surface temperature (lines × 512); emissivity (202 bands, raw and smoothed); and per-band ISAC transmittance and upwelling/downwelling sky radiance (256 bands). Those per-band terms are the τ_a and L_a↑ of the LWIR model.
- **Licence:** NASA open data, no restrictions
- **Location:** `data/scenes/hytes/<flight>/`

### HITRAN line lists
- **Source:** hitran.org through HAPI (`hitran-api`), no login. All isotopologues; 3950–6700 cm⁻¹ (VSWIR) and 750–1400 cm⁻¹ (LWIR), for the gases and ranges in `configs/gases.yaml`
- **Fetched:** 2026-10-31, with `specret download hitran`, ~180 MB. C2H6 has no lines in the VSWIR range; its 1.68 and 2.3 µm features need the HITRAN cross-section files
- **Licence:** free to use; cite Gordon et al. (2022), JQSRT 277, 107949, and HAPI, Kochanov et al. (2016), JQSRT 177, 15
- **Location:** `data/raw/hitran/` (HAPI `.data` and `.header` tables)

### Wind (for emission rates)
- **HRRR:** NOAA's public cloud bucket through `herbie`, no account. Tested 2026-10-31: 10 m wind at the Permian scene's overpass hour reads correctly
- **MERRA-2** (`M2T1NXSLV`, GES DISC): returns 403 Forbidden with the current Earthdata login until the account authorizes the GES DISC archive (see below)
- **Keyless ERA5** (Google ARCO zarr): not verified. Opening the full store took over 10 min and was stopped. Try the Open-Meteo historical API instead when the cross-check is built.
- Nothing is downloaded in bulk: wind is read per scene and hour when quantifying

## To fetch by hand

### ECOSTRESS spectral library
- **Source:** https://speclib.jpl.nasa.gov (download request form)
- **What:** VSWIR mineral, rock, soil and vegetation files. Reflectance is in percent.
- **Put in:** `data/raw/ecostress/`, then `specret ingest --usgs data/raw/usgs_splib07/extracted/ASCIIdata_splib07a --ecostress data/raw/ecostress`

### EMIT L2A reflectance and L2B mineral maps over Cuprite
- **Source:** NASA LP DAAC (short names EMITL2ARFL, EMITL2BMIN); needs an Earthdata login
- **Fetch:** `specret download emit --limit 2` (Cuprite box by default)
- **Put in:** `data/scenes/emit/`

### AVIRIS-Classic and AVIRIS-NG Cuprite reflectance
- **Source:** JPL AVIRIS and AVIRIS-NG data portals
- **Put in:** `data/scenes/aviris/` (ENVI .hdr + image)

### USGS Tetracorder Cuprite mineral maps
- **Source:** USGS Spectroscopy Lab publications (Clark, Swayze et al.); availability not yet confirmed
- **Put in:** `data/refmaps/`

### MERRA-2 access (GES DISC)
- **What:** in your Earthdata profile (urs.earthdata.nasa.gov → Applications → Authorized Apps), approve "NASA GESDISC DATA ARCHIVE". MERRA-2 reads then work with the existing login.

### PNNL gas-phase infrared database (LWIR gas library)
- **Source:** PNNL Northwest-Infrared (nwir.pnl.gov); historically a paid licence, with redistribution restricted
- **Status:** not fetched. Whether to buy it is open. The free fallback is the HITRAN cross-section database, which includes many PNNL-derived spectra
- **Put in:** `data/raw/pnnl/` (gitignored, never committed)

### Controlled methane releases with EMIT overpasses
- **Source:** Stanford one-year single-blind test (Reuland, Brandt et al., Nature 2026, doi 10.1038/s41586-026-11115-0), Casa Grande, Arizona, 2024–2025; EMIT is among the tested instruments
- **Status:** no public per-release table found yet (metered rate, time, location). The 2022 test (Sherwin et al., AMT 17, 765, 2024; Zenodo 10.5281/zenodo.10149992, CC-BY-4.0) did not include EMIT
- **Put in:** `data/gas/releases.csv`

### Public ASD field spectra
- **Coso geothermal alteration minerals** (OpenEI): alunite, chalcedony, epidote, hematite, kaolinite, opal; raw .asd plus ASCII. Check the licence.
- **USGS Smith Creek Valley, Nevada:** https://doi.org/10.5066/P9E2TSDF (ASD Indico .asd, public)
- **USGS Black Rock Desert, Nevada, 2022:** https://doi.org/10.5066/P94QAHLC (FieldSpec 4 Hi-Res NG, CC0)
- **Put in:** `data/field/raw/`, and add a row per spectrum to `data/field/metadata.csv`
