# SpectralLibraryRetrieval

Identify materials from a measured spectrum (an EMIT or AVIRIS pixel, or a field spectrometer
reading) by matching it against reference spectral libraries. The phased plan is in
[PLAN.md](PLAN.md). This repository currently covers Phases 0–1: ingestion, sensor resampling,
and SAM-equivalent retrieval.

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
uv run specret validate-resampler                   # compares against USGS's own convolution
uv run specret resample --sensor aviris_classic     # -> data/resampled/avirisc/library.npz
uv run specret resample --sensor emit --band-file data/scenes/emit/<EMIT_L2A_RFL_...>.nc
uv run specret index data/resampled/avirisc/library.npz      # load into Qdrant
uv run specret query data/resampled/emit/library.npz --emit <granule.nc> --lat 37.53 --lon -117.22
uv run specret query data/resampled/field_asd/library.npz --field <spectrum.asd>
```

Sensor definitions are in `configs/sensors/`. EMIT and AVIRIS-NG read their band centres from
the scene file, so the library is resampled once per sensor configuration.

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
