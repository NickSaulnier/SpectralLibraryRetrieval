"""Download helpers for public data sources.

Only sources with stable direct links are automated here. ECOSTRESS (request form),
AVIRIS (data portals) and the USGS Cuprite maps are fetched by hand; see data/raw/MANIFEST.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# USGS Spectral Library v7, ScienceBase item 586e8c88e4b0f5ce109fccae (Data Series 1035).
SCIENCEBASE_FILE = (
    "https://www.sciencebase.gov/catalog/file/get/586e8c88e4b0f5ce109fccae?f=__disk__"
)
USGS_SETS = {
    "splib07a": "a7%2F4f%2F91%2Fa74f913e0b7d1b8123ad059e52506a02b75a2832",
    "splib07b": "61%2Fd1%2F3f%2F61d13f11fafae0e37aadf700eff1bde6e236165b",
    "splib07b_cvASD": "6f%2F81%2F62%2F6f8162500e6f3d250883a27b3def74cb77b927fc",
    "splib07b_cvAVIRISc1997": "e4%2F7d%2F6f%2Fe47d6f9fd757caea738bea3a605a17df48c3df77",
    "splib07b_cvAVIRISc2014": "b5%2Fa6%2F05%2Fb5a6051ad8a2c4c60569cb37a5790f4291bb6a8e",
}

CUPRITE_BBOX = (-117.30, 37.45, -117.10, 37.62)  # west, south, east, north


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing absolute paths and '..' components."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.replace("\\", "/").split("/")
            if name.startswith("/") or ".." in parts or ":" in name:
                raise ValueError(f"{zip_path.name}: unsafe member path {name!r}")
        zf.extractall(dest)


def download_usgs(sets: list[str], out_dir: Path) -> list[Path]:
    paths = []
    for s in sets:
        if s not in USGS_SETS:
            raise ValueError(f"unknown set {s!r}; choose from {sorted(USGS_SETS)}")
        z = fetch(SCIENCEBASE_FILE + USGS_SETS[s], out_dir / f"ASCIIdata_{s}.zip")
        safe_extract(z, out_dir / "extracted")
        paths.append(out_dir / "extracted" / f"ASCIIdata_{s}")
    return paths


def download_emit(
    bbox: tuple[float, float, float, float],
    temporal: tuple[str, str],
    out_dir: Path,
    products: tuple[str, ...] = ("EMITL2ARFL", "EMITL2BMIN"),
    limit: int | None = None,
) -> list[str]:
    """Download EMIT granules via earthaccess (prompts for Earthdata credentials if needed)."""
    import earthaccess

    from specret.io.emit import search_emit

    earthaccess.login()  # ~/.netrc, EARTHDATA_USERNAME/PASSWORD, or an interactive prompt
    results = search_emit(bbox, temporal, short_names=products)
    if limit is not None:
        results = results[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    return earthaccess.download(results, str(out_dir))
