# SpectralLibraryRetrieval

Identify materials from a measured spectrum (an EMIT or AVIRIS pixel, or a field spectrometer
reading) by matching it against reference spectral libraries. The phased plan is in
[PLAN.md](PLAN.md). This repository covers Phases 0–4: ingestion and sensor resampling, physics-based
retrieval (continuum removal, rank fusion, feature-fit reranking) with a frozen benchmark,
scene-scale mapping, a local viewer and MCP tools, retrieve-and-unmix, and calibrated
confidence with unknown-material flagging. Results are in `results/`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for the Qdrant server, Docker.

```bash
uv sync --extra gpu        # CUDA PyTorch for GPU exact search; drop --extra gpu for CPU only
uv run pre-commit install
docker compose up -d       # Qdrant on localhost:6333
```

## Data

Raw data lives under `data/` and is not committed. `data/raw/MANIFEST.md` records where each
source came from.

```bash
uv run specret download usgs                       # splib07a + AVIRIS-2014 convolved set (~26 MB)
uv run specret download emit --limit 2             # EMIT L2A + L2B mineral granules over Cuprite
```

EMIT downloads need a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account. ECOSTRESS,
AVIRIS scenes and field datasets are fetched by hand; see the manifest.

## Pipeline

```bash
uv run specret ingest                               # -> data/canonical/library.parquet
uv run specret make-splits                          # once: frozen sample-level split in configs/splits/
uv run specret validate-resampler                   # compares against USGS's own convolution
uv run specret resample --sensor aviris_classic     # -> data/resampled/avirisc/library.npz
uv run specret resample --sensor emit --band-file data/scenes/emit/<EMIT_L2A_RFL_...>.nc
uv run specret index data/resampled/avirisc/library.npz      # load into Qdrant
uv run specret query data/resampled/emit/library.npz --emit <granule.nc> --lat 37.53 --lon -117.22
uv run specret query data/resampled/field_asd/library.npz --field <spectrum.asd>
uv run specret compare-l2bmin data/resampled/emit/library.npz --rfl <L2A RFL .nc> --min <L2B MIN .nc> --chapters M S C
uv run specret noise-curve --rfl <L2A RFL .nc> --uncert <L2A RFLUNCERT .nc>   # EMIT noise model
uv run specret benchmark data/resampled/emit/library.npz   # synthetic benchmark, frozen test split
```

Search vectors: `sam` (raw reflectance, the baseline), `swir2` (2.0–2.5 µm), `cr` and
`cr_swir2` (continuum-removed band depth). The default retriever, `hybrid`, fuses `sam`, `cr`
and `cr_swir2` and reranks the candidates by fitting diagnostic absorption features
(`configs/features.yaml`). Material groups come from `configs/taxonomy.yaml`. Results are in
`results/`.

Sensor definitions are in `configs/sensors/`. EMIT and AVIRIS-NG read their band centres from
the scene file, so the library is resampled once per sensor configuration.

## Maps, viewer and agent tools (Phase 3)

```bash
uv run specret map-scene data/resampled/emit/library.npz --scene <EMIT_L2A_RFL .nc>   # ~2 min per EMIT scene
uv run specret view                                  # http://127.0.0.1:8765
uv run specret mcp                                   # FastMCP server over stdio
```

To use the tools from Claude Code:

```bash
claude mcp add specret -- uv run --directory D:/Projects/HyperspectralIndexing specret mcp
```

Tools: `list_resources`, `identify_spectrum`, `identify_pixel`, `unmix_pixel`,
`unmix_spectrum`, `explain_match`, `identify_field_spectrum`, `library_entry`, `map_scene`,
`job_status`, `scene_summary`. Every answer carries a calibrated P(group is right) and a tier
(confident / uncertain / likely unknown). See `results/phase3_mapping.md` and
`results/phase4_unmix_openset.md`.

```bash
uv run specret calibrate data/resampled/emit/library.npz      # -> configs/calibration/emit.json
uv run specret eval-unmix data/resampled/emit/library.npz     # synthetic mixture benchmark
```

## Learned encoder (Phase 5, experimental)

```bash
uv run specret train-encoder --steps 4000      # ~5 min on a GPU -> models/encoder_v1.pt
uv run specret benchmark data/resampled/emit/library.npz --methods hybrid learned hybrid+learned
```

The encoder wins on the synthetic benchmark but not on the real EMIT scene, so it isn't part
of the default retriever. See `results/phase5_learned.md`.

## Layout

```
configs/           sensor band definitions and bad-band ranges
src/specret/
  io/              USGS splib07, ECOSTRESS, EMIT, AVIRIS and field (ASD/CSV) readers, downloads
  sensors/         sensor model, Gaussian band resampling, per-sensor library matrices
  preprocess/      masks, scale checks, ASD splice correction
  search/          GPU exact search, Qdrant collections, retrieval
  eval/            resampler check against USGS convolved spectra
notebooks/         jupytext notebooks (open the .py files in VS Code or Jupyter)
tests/             pytest suite; tests marked `data` skip when downloads are missing
```

## Tests

```bash
uv run pytest
uv run ruff check src tests notebooks
```
