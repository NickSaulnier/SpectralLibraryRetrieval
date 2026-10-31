# %% [markdown]
# # 02 · Cuprite MVP: SAM retrieval on a scene pixel
# Picks a pixel from an EMIT or AVIRIS Cuprite scene and plots it against its top matches.
#
# Before running:
# 1. `uv run specret ingest`
# 2. Get a scene: `uv run specret download emit` (EMIT, needs an Earthdata login), or place an
#    AVIRIS ENVI cube under data/scenes/aviris/.
# 3. Resample the library to that scene's bands:
#    `uv run specret resample --sensor emit --band-file <granule.nc>`
#    `uv run specret resample --sensor aviris_ng --band-file <cube.hdr>`

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from specret.schema import read_library
from specret.search.retrieve import Retriever, format_matches
from specret.sensors.library import SensorLibrary

SENSOR = "emit"  # or "avirisc" / "avirisng"
SCENE = sorted(Path("../data/scenes/emit").glob("EMIT_L2A_RFL_*.nc"))[0]  # edit as needed

canonical = read_library("../data/canonical/library.parquet")
lib = SensorLibrary.load(f"../data/resampled/{SENSOR}/library.npz")
retriever = Retriever(lib, canonical, vector="sam")

# %%
if SENSOR == "emit":
    from specret.io.emit import open_emit_l2a

    scene = open_emit_l2a(SCENE)
else:
    from specret.io.aviris import open_aviris

    scene = open_aviris(SCENE)
retriever.check_bands(np.asarray(scene.centers_nm))
scene.shape

# %% [markdown]
# ## Pick a pixel
# Known Cuprite targets to try: the Buddingtonite Bump, and the alunite and kaolinite zones
# of the east and west hydrothermal centres. Take coordinates from a published Cuprite
# mineral map; the default below is only a rough point in the district.

# %%
LAT, LON = 37.530, -117.220  # approximate; replace with a mapped target
row, col = scene.nearest_pixel(LAT, LON) if SENSOR == "emit" else (300, 300)
pixel = scene.pixel(row, col)
matches = retriever.query(pixel, k=5)[0]
print(format_matches(matches))

# %%
c = lib.sensor.centers_nm
good = lib.sensor.good_mask
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(c[good], (pixel / np.nanmax(pixel[good]))[good], "k.", ms=3, label="pixel (scaled)")
row_of = {sid: i for i, sid in enumerate(lib.spectrum_ids)}
for m in matches[:3]:
    s = lib.matrix[row_of[m["spectrum_id"]]]
    ax.plot(
        c[good],
        (s / np.nanmax(s[good]))[good],
        lw=1,
        label=f"{m['material_name']} ({m['sam_deg']:.1f}°)",
    )
ax.set(xlabel="Wavelength (nm)", ylabel="Scaled reflectance", xlim=(400, 2500))
ax.legend()

# %% [markdown]
# Record failures here: they are the motivation for continuum removal and feature fitting
# in Phase 2.
