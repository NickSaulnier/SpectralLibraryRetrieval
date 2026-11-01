"""Download the Phase G0 HyTES flights (PLAN_GAS.md) from ORNL DAAC. Needs a saved Earthdata login.

Flights are over sites with published HyTES gas detections (Hulley et al. 2016, AMT 9, 2393):
L1 radiance (ENVI), L1 geolocation (CSV lat/lon/alt) and L2 land surface temperature and
emissivity (HDF5), which the LWIR model uses for the background radiance.

Run: uv run python scripts/download_g0_hytes.py [--dry-run]
"""

import sys
from pathlib import Path

import earthaccess

FLIGHTS = [
    "2014-07-06.194610.SaltonSea.Line2-Run1-Segment01",  # geothermal fumaroles: NH3, SO2?
    "2014-07-08.190512.KernRiverOil.Line2-Run1-Segment01",  # oil field: CH4
    "2013-04-26.220007.Dairy.Line1-Run1-Segment02",  # Chino dairies: NH3, CH4 (L1 only)
]
COLLECTIONS = [
    "HyTES_L1_Radiance_2013_2016_2476",
    "HyTES_L1_Geolocation_2013_2016_2467",
    "HyTES_L2_Emissivity_LST_2460",
]
OUT = Path("data/scenes/hytes")


def main() -> None:
    dry = "--dry-run" in sys.argv
    if not dry:
        earthaccess.login(strategy="netrc")
    for flight in FLIGHTS:
        granules = []
        for short in COLLECTIONS:
            found = earthaccess.search_data(short_name=short, granule_name=f"{flight}*")
            print(f"{flight}  {short}: {[round(g.size) for g in found]} MB")
            granules += found
        if not dry and granules:
            for p in earthaccess.download(granules, str(OUT / flight.split("-Run")[0])):
                print(p)


if __name__ == "__main__":
    main()
