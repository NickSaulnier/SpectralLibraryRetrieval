# Raw data manifest

One entry per source: where it came from, version, when it was fetched, and its licence.

## Downloaded

### USGS Spectral Library Version 7 (splib07)
- **Source:** ScienceBase item 586e8c88e4b0f5ce109fccae, USGS Data Series 1035 (https://doi.org/10.3133/ds1035), Kokaly et al. (2017)
- **Files:** `ASCIIdata_splib07a.zip` (21.8 MB, measured spectra), `ASCIIdata_splib07b_cvAVIRISc2014.zip` (4.3 MB, convolved to AVIRIS-Classic 2014 band passes)
- **Fetched:** 2026-10-31, with `specret download usgs`
- **Licence:** US Government public domain
- **Location:** `data/raw/usgs_splib07/` (zips) and `data/raw/usgs_splib07/extracted/`

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

### Public ASD field spectra
- **Coso geothermal alteration minerals** (OpenEI): alunite, chalcedony, epidote, hematite, kaolinite, opal; raw .asd plus ASCII. Check the licence.
- **USGS Smith Creek Valley, Nevada:** https://doi.org/10.5066/P9E2TSDF (ASD Indico .asd, public)
- **USGS Black Rock Desert, Nevada, 2022:** https://doi.org/10.5066/P94QAHLC (FieldSpec 4 Hi-Res NG, CC0)
- **Put in:** `data/field/raw/`, and add a row per spectrum to `data/field/metadata.csv`
