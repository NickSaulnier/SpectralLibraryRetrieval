# Phase 3 · Scene mapping, viewer and MCP tools

**Date:** 2026-10-31
**Scene:** `EMIT_L2A_RFL_001_20240727T213005_2420914_007` (1,280 × 1,242 swath, 1,589,760 pixels).
**Retriever:** hybrid (sam + cr + cr_swir2 fusion, then feature-fit rerank, `sff_weight` 0.25). Library: minerals, mixtures, coatings and vegetation (chapters M, S, C, V).

## Scene mapping

```bash
uv run specret map-scene data/resampled/emit/library.npz \
  --scene data/scenes/emit/EMIT_L2A_RFL_001_20240727T213005_2420914_007.nc
```

- **118 s for the whole scene** on an RTX 4090 workstation. Every pixel gets the full hybrid pipeline.
- The plan's clustering step proved unnecessary. Three things made the full pipeline fast enough:
  - continuum removal compiled with numba: 3.5 s per 1.5M spectra, from ~6 min in pure Python;
  - rank fusion in numba: 1.5 s per 1.5M queries;
  - GPU exact search.
- **Outputs** go to `outputs/maps/<scene>/`:
  - Cloud-optimised GeoTIFFs in EPSG:4326 with overviews: `group.tif` (uint8 group codes with a colour table), `top1_row.tif`, `score.tif` and `margin.tif`.
  - `library.csv` and `groups.csv` legends.
  - `scene_map.npz`: top-5 rows and scores per pixel, in swath geometry, for fast lookups.
- **Whole-scene composition** (top-1 group): vegetation 37.1%, white mica 23.0%, calcite 8.6%, smectite 8.2%, epidote 4.7%, rock/soil 3.4%, other silicates 3.1%, sulfide 1.4%, alunite 1.1%, kaolin 1.1%. Within the Cuprite box: white mica 32%, smectite 10%, calcite 9%, alunite 7%.
- **Library choice matters.** With every chapter included, 10.9% of this desert scene matched man-made materials. Maps and tools therefore default to chapters M, S, C and V. The "sulfide" class (1.4%) is suspect: dark, featureless pixels such as shadow or basalt match dark sulfide spectra.

## Confidence is not yet informative

Top-1 scores are high almost everywhere: the 10th–90th percentile is 0.77–0.93, and no mapped pixel scores below 0.6. The ranking score says which candidate is best, not whether the best candidate is right. The tools label confidence "provisional". Calibrated rejection of materials missing from the library, and unmixing of mixed pixels, are Phase 4.

## MCP tools (`uv run specret mcp`, stdio)

| Tool | What it does |
|---|---|
| `list_resources` | Sensors with resampled libraries, available scenes and which are mapped |
| `identify_spectrum` | Ranked matches for a spectrum (resampled onto the sensor if needed), with per-window feature evidence |
| `identify_pixel` | The same for a scene pixel, by row/col or lat/lon |
| `explain_match` | Why a library spectrum does or does not fit: per-window centres, depths, shape r, purity grade, USGS sample note |
| `identify_field_spectrum` | A local .asd/.csv field file, with splice correction |
| `library_entry` | Metadata and sample note for one library spectrum |
| `map_scene` / `job_status` | Map a scene in a background thread |
| `scene_summary` | Group composition of a mapped scene, optionally inside a lat/lon box |

Example from an in-memory client: a Cuprite pixel (37.5355, −117.2165) identifies in ~2 s as `Alun_Na+Kaol+Hemat` (alunite group). The evidence is the Al-OH window, query 2175 nm against library 2170 nm with shape r 0.98. The answer also carries the USGS note: "Na alunite + kaolinite (mnr.) + hematite (tr.) … from XRD".

## Viewer (`uv run specret view`, http://127.0.0.1:8765)

- **The map:** the orthorectified group map with a legend of group fractions.
- **Clicking a pixel:** plots its spectrum against the top three matches, with the diagnostic windows shaded. It also lists the matches with score, purity grade and the two strongest feature comparisons.
- **Field spectra:** dropping an .asd or .csv file runs the same identification on it.
- **Checked in the browser pane:** map render, pixel click at Cuprite, and field upload (via the API).

## Not done

- **AVIRIS scene mapping:** the mapper and loaders support AVIRIS, but no AVIRIS Cuprite scene has been downloaded yet. The viewer's click-to-pixel lookup uses EMIT's geometry lookup table, so AVIRIS pixels can be reached only through the tools by row/col.
- **The colour table** spaces hues automatically. With ~30 groups, some neighbours look alike. A fixed palette for the main alteration minerals would read better.
