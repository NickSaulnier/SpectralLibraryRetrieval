from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
USGS = REPO / "data" / "raw" / "usgs_splib07" / "extracted"
SPLIB07A = USGS / "ASCIIdata_splib07a"
AV14 = USGS / "ASCIIdata_splib07b_cvAVIRISc2014"


@pytest.fixture
def usgs_dirs():
    if not (SPLIB07A.is_dir() and AV14.is_dir()):
        pytest.skip("USGS splib07 data not downloaded (uv run specret download usgs)")
    return SPLIB07A, AV14
