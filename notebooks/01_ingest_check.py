# %% [markdown]
# # 01 · Ingest check
# Sanity checks on the canonical library: counts, coverage, and the same sample measured
# by different instruments. Run `uv run specret ingest` first.
#
# This file is a jupytext "percent" notebook: open it in VS Code / Jupyter and run the cells,
# or convert with `uv run jupytext --to ipynb notebooks/01_ingest_check.py`.

# %%
import matplotlib.pyplot as plt

from specret.schema import read_library

df = read_library("../data/canonical/library.parquet")
print(len(df), "spectra")
df.groupby(["chapter", "measurement"]).size().unstack(fill_value=0)

# %%
df["instrument_family"] = df["instrument"].str.extract(r"^(ASDFR|ASDHR|ASDNG|BECK|NIC4|AVIRIS)")
df.groupby("instrument_family")[["coverage_min_nm", "coverage_max_nm"]].median()

# %% [markdown]
# ## One sample, several instruments
# Kaolinite CM9 was measured with the Beckman, ASD and Nicolet spectrometers. All three
# share `sample_id`, which is what keeps them on the same side of a train/test split.

# %%
sample = "usgs:CM9"
fig, ax = plt.subplots(figsize=(9, 4))
for _, r in df[df["sample_id"] == sample].iterrows():
    ax.plot(r["wavelength_nm"], r["reflectance"], label=f"{r['instrument']} {r['measurement']}")
ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance", title=sample, xlim=(350, 2600))
ax.legend()

# %% [markdown]
# ## Most common minerals

# %%
df[df["chapter"] == "M"]["mineral"].value_counts().head(25)
