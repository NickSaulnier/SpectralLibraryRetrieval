# %% [markdown]
# # 03 · Field spectrum check
# Loads a field spectrum (.asd or CSV), applies splice correction, and matches it against
# the library resampled to the field spectrometer's bands.
#
# Before running: `uv run specret resample --sensor asd_fieldspec`
# Public ASD data sources are listed in PLAN.md, section 3.4.

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from specret.io.field import read_field
from specret.preprocess.splice import splice_correct
from specret.schema import read_library
from specret.search.retrieve import Retriever, format_matches
from specret.sensors.library import SensorLibrary

FIELD_FILE = sorted(Path("../data/field/raw").glob("*.asd"))[0]  # edit as needed

lib = SensorLibrary.load("../data/resampled/field_asd/library.npz")
retriever = Retriever(lib, read_library("../data/canonical/library.parquet"))
spec = read_field(FIELD_FILE)[0]
print(spec.name, spec.data_type, "reference" if spec.reference is not None else "no reference")

# %%
refl = spec.reflectance
fixed = splice_correct(spec.wavelength_nm, refl)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(spec.wavelength_nm, refl, lw=0.8, label="as read")
ax.plot(spec.wavelength_nm, fixed, lw=0.8, label="splice corrected")
for j in (1000, 1800):
    ax.axvline(j, color="grey", lw=0.5, ls="--")
ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance")
ax.legend()

# %%
print(format_matches(retriever.query_field(spec.wavelength_nm, refl, k=5)))
