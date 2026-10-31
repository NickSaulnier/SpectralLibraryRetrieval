# data/

Nothing here is committed except this file, `raw/MANIFEST.md` and `field/metadata.csv`.

| Folder | Contents | Rebuild with |
|---|---|---|
| `raw/<source>/` | Untouched downloads, one folder per source | `specret download ...` or by hand (see `raw/MANIFEST.md`) |
| `field/raw/` | Field spectrometer files (.asd, CSV) | by hand |
| `canonical/library.parquet` | Every library spectrum at native resolution (source of truth) | `specret ingest` |
| `resampled/<sensor>/library.npz` | Library resampled to one sensor's bands | `specret resample` |
| `scenes/` | EMIT, AVIRIS-Classic and AVIRIS-NG scenes | `specret download emit`, or by hand |
| `refmaps/` | Tetracorder / EMIT L2B reference maps | by hand |
| `qdrant/` | Qdrant server storage | `specret index` |
