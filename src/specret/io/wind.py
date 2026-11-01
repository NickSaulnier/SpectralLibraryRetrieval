"""10 m wind at a point and time, for emission rates (PLAN_GAS.md 1.3). No Copernicus account.

* HRRR (US, hourly, 3 km): NOAA's public bucket through herbie, no account. Analysis hours
  (f00) either side of the overpass are interpolated linearly in time.
* MERRA-2 (global, hourly, ~50 km): GES DISC M2T1NXSLV (U10M, V10M) through earthaccess with
  the Earthdata login; one daily file is cached per day in data/raw/merra2.

`wind_at` uses HRRR inside its domain and MERRA-2 elsewhere. Where both exist, their
difference adds to the uncertainty; either way the standard deviation is at least
`min_relative_sd` (configs/plumes.yaml) of the speed, because a grid-cell wind is not the wind
in the plume.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

HRRR_DIR = Path("data/raw/hrrr")
MERRA2_DIR = Path("data/raw/merra2")
HRRR_BOX = (21.0, 53.0, -134.0, -60.0)  # lat min, lat max, lon min, lon max (approximate)


@dataclass
class Wind:
    u: float  # m/s, eastward
    v: float  # m/s, northward
    source: str
    sd: float = float("nan")  # standard deviation of the speed, m/s

    @property
    def speed(self) -> float:
        return float(np.hypot(self.u, self.v))

    @property
    def direction_to_deg(self) -> float:
        """Direction the air moves towards, degrees clockwise from north."""
        return float(np.degrees(np.arctan2(self.u, self.v)) % 360)


def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def in_hrrr_domain(lat: float, lon: float) -> bool:
    a, b, c, d = HRRR_BOX
    return a <= lat <= b and c <= lon <= d


@lru_cache(maxsize=8)
def _hrrr_grid(hour: str, save_dir: str) -> tuple[np.ndarray, ...]:
    """(lat, lon 0-360, u10, v10) of one HRRR analysis hour, read once per process."""
    from herbie import Herbie

    with contextlib.redirect_stdout(io.StringIO()):  # herbie prints emoji the console can't
        h = Herbie(hour, model="hrrr", product="sfc", fxx=0, save_dir=save_dir, verbose=False)
        ds = h.xarray(r":[UV]GRD:10 m above ground", remove_grib=False)
    if isinstance(ds, list):
        import xarray as xr

        ds = xr.merge(ds, compat="override")
    return (ds.latitude.values, ds.longitude.values, ds["u10"].values, ds["v10"].values)


def _hrrr_hour(t: datetime, lat: float, lon: float, save_dir: Path) -> tuple[float, float]:
    glat, glon, u, v = _hrrr_grid(t.strftime("%Y-%m-%d %H:00"), str(save_dir))
    d = (glat - lat) ** 2 + ((glon - lon % 360) * np.cos(np.radians(lat))) ** 2
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(u[i, j]), float(v[i, j])


def wind_hrrr(lat: float, lon: float, t: datetime, save_dir: Path = HRRR_DIR) -> Wind:
    t = _utc(t)
    t0 = t.replace(minute=0, second=0, microsecond=0)
    w = (t - t0).total_seconds() / 3600
    u0, v0 = _hrrr_hour(t0, lat, lon, save_dir)
    if w < 1e-3:
        return Wind(u0, v0, "HRRR")
    u1, v1 = _hrrr_hour(t0 + timedelta(hours=1), lat, lon, save_dir)
    return Wind((1 - w) * u0 + w * u1, (1 - w) * v0 + w * v1, "HRRR")


def merra2_file(day: datetime, out_dir: Path = MERRA2_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = sorted(out_dir.glob(f"MERRA2_*.tavg1_2d_slv_Nx.{day:%Y%m%d}.nc4"))
    if hits:
        return hits[0]
    import earthaccess

    earthaccess.login(strategy="netrc")
    d = f"{day:%Y-%m-%d}"
    g = earthaccess.search_data(short_name="M2T1NXSLV", temporal=(d, d))
    g = [x for x in g if f".{day:%Y%m%d}." in x["umm"]["GranuleUR"]]
    if not g:
        raise RuntimeError(f"no MERRA-2 M2T1NXSLV granule for {d}")
    with contextlib.redirect_stdout(io.StringIO()):
        files = earthaccess.download(g[:1], str(out_dir))
    return Path(files[0])


@lru_cache(maxsize=4)
def _merra2_day(path: str):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return ds[["U10M", "V10M", "T2M"]].load()


def wind_merra2(lat: float, lon: float, t: datetime, out_dir: Path = MERRA2_DIR) -> Wind:
    t = _utc(t)
    ds = _merra2_day(str(merra2_file(t, out_dir)))
    p = ds[["U10M", "V10M"]].interp(
        lat=lat, lon=lon, time=np.datetime64(t.replace(tzinfo=None)), kwargs={"fill_value": None}
    )
    return Wind(float(p["U10M"]), float(p["V10M"]), "MERRA-2")


def wind_at(
    lat: float, lon: float, t: datetime, source: str = "auto", min_relative_sd: float = 0.5
) -> Wind:
    """Best available 10 m wind with an uncertainty. source: auto, hrrr or merra2."""
    winds = []
    if source in ("auto", "hrrr") and in_hrrr_domain(lat, lon):
        winds.append(wind_hrrr(lat, lon, t))
    if source in ("auto", "merra2") or not winds:
        try:
            winds.append(wind_merra2(lat, lon, t))
        except Exception:
            if not winds:
                raise
    best = winds[0]
    spread = abs(winds[0].speed - winds[1].speed) if len(winds) > 1 else 0.0
    best.sd = float(max(min_relative_sd * best.speed, spread))
    if len(winds) > 1:
        best.source = f"{best.source} (vs {winds[1].source} {winds[1].speed:.2f} m/s)"
    return best


def air_temperature(lat: float, lon: float, t: datetime, out_dir: Path = MERRA2_DIR) -> float:
    """MERRA-2 2 m air temperature (K) at a point and time."""
    t = _utc(t)
    ds = _merra2_day(str(merra2_file(t, out_dir)))
    p = ds["T2M"].interp(
        lat=lat, lon=lon, time=np.datetime64(t.replace(tzinfo=None)), kwargs={"fill_value": None}
    )
    return float(p)
