"""Check our resampler against USGS's own convolved spectra.

splib07 ships each spectrum convolved to AVIRIS-Classic band passes. Resampling the native
spectra with our code and comparing against those files tests the band model, the coverage
handling and the resolution correction in one go.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from specret.io.usgs_splib07 import INSTRUMENTS
from specret.schema import SpectrumRecord
from specret.sensors.resample import resample
from specret.sensors.sensor import Sensor


@dataclass
class ResamplerReport:
    per_spectrum: pd.DataFrame  # one row per matched spectrum
    per_band: pd.DataFrame  # median relative error by band

    @property
    def median_rel_error(self) -> float:
        return float(self.per_spectrum["median_rel_err"].median())

    def summary(self) -> str:
        s = self.per_spectrum
        p95 = s["median_rel_err"].quantile(0.95)
        return (
            f"matched spectra: {len(s)}\n"
            f"median of per-spectrum median relative error: {s['median_rel_err'].median():.4%}\n"
            f"95th percentile of per-spectrum median error:  {p95:.4%}\n"
            f"median of per-spectrum max relative error:     {s['max_rel_err'].median():.4%}\n"
            f"by instrument (median of medians):\n"
            + s.groupby("instrument_family")["median_rel_err"]
            .median()
            .map(lambda v: f"{v:.4%}")
            .to_string()
        )


def compare(
    native: list[SpectrumRecord],
    convolved: list[SpectrumRecord],
    sensor: Sensor,
    use_library_fwhm: bool = True,
    floor: float = 0.01,
) -> ResamplerReport:
    """Relative error |ours - usgs| / max(|usgs|, floor) on the sensor's good bands."""
    by_key = {r.source_record: r for r in native}
    rows, band_errs = [], []
    for c in convolved:
        n = by_key.get(c.source_record)
        if n is None:
            continue
        ours = resample(
            n.wavelength_nm,
            n.reflectance,
            sensor.centers_nm,
            sensor.fwhm_nm,
            lib_fwhm=n.native_fwhm_nm if use_library_fwhm else None,
        )
        ref = c.reflectance
        ok = sensor.good_mask & np.isfinite(ours) & np.isfinite(ref)
        if ok.sum() < 10:
            continue
        err = np.full(sensor.n_bands, np.nan)
        err[ok] = np.abs(ours[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), floor)
        band_errs.append(err)
        rows.append(
            {
                "source_record": c.source_record,
                "instrument": n.instrument,
                "instrument_family": _family(n.instrument),
                "n_bands": int(ok.sum()),
                "median_rel_err": float(np.nanmedian(err)),
                "max_rel_err": float(np.nanmax(err)),
            }
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # masked bands are all-NaN
        band_median = np.nanmedian(np.vstack(band_errs), axis=0) if band_errs else np.nan
    per_band = pd.DataFrame({"center_nm": sensor.centers_nm, "median_rel_err": band_median})
    return ResamplerReport(pd.DataFrame(rows), per_band)


def _family(instrument: str | None) -> str:
    for name in INSTRUMENTS:
        if (instrument or "").startswith(name):
            return name
    return instrument or "unknown"
