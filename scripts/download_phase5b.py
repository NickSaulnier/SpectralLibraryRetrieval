"""Download the Phase 5b EMIT scenes (PLAN.md): reflectance + cloud mask for the pretraining
scenes, reflectance + L2B mineral map for the validation scene. Needs a saved Earthdata login.

Run: uv run python scripts/download_phase5b.py
"""

from pathlib import Path

import earthaccess

PRETRAIN = [
    "EMIT_L2A_RFL_001_20240716T063940_2419804_012",  # Pilbara
    "EMIT_L2A_RFL_001_20230524T105835_2314407_002",  # Kalahari
    "EMIT_L2A_RFL_001_20230626T070735_2317705_009",  # Sar Cheshmeh
]
VALIDATION = "EMIT_L2A_RFL_001_20240416T210255_2410714_015"  # Mountain Pass

OUT = Path("data/scenes/emit_extra")


def main() -> None:
    earthaccess.login(strategy="netrc")
    urls = []
    for ur in [*PRETRAIN, VALIDATION]:
        (g,) = earthaccess.search_data(short_name="EMITL2ARFL", granule_name=ur)
        wanted = ("_RFL_", "_MASK_") if ur in PRETRAIN else ("_RFL_",)
        names = {u: u.rsplit("/", 1)[-1] for u in g.data_links() if u.endswith(".nc")}
        urls += [u for u, n in names.items() if any(w in n for w in wanted)]
    stamp = "_".join(VALIDATION.split("_")[4:7])
    (m,) = earthaccess.search_data(
        short_name="EMITL2BMIN", granule_name=f"EMIT_L2B_MIN_001_{stamp}"
    )
    urls += [u for u in m.data_links() if u.rsplit("/", 1)[-1].startswith("EMIT_L2B_MIN_")]
    OUT.mkdir(parents=True, exist_ok=True)
    for path in earthaccess.download(urls, str(OUT)):
        print(path)


if __name__ == "__main__":
    main()
