"""Command-line entry point: `specret <command>` (or `uv run specret <command>`)."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CANONICAL = Path("data/canonical/library.parquet")
USGS_EXTRACTED = Path("data/raw/usgs_splib07/extracted")


def cmd_download(args: argparse.Namespace) -> None:
    from specret.io import download

    if args.source == "usgs":
        for p in download.download_usgs(args.sets, Path(args.out or "data/raw/usgs_splib07")):
            print(f"extracted {p}")
    elif args.source == "emit":
        bbox = tuple(args.bbox) if args.bbox else download.CUPRITE_BBOX
        files = download.download_emit(
            bbox,
            (args.start, args.end),
            Path(args.out or "data/scenes/emit"),
            products=tuple(args.products),
            limit=args.limit,
        )
        print(f"downloaded {len(files)} files")


def cmd_ingest(args: argparse.Namespace) -> None:
    from specret.io.ecostress import load_ecostress_dir
    from specret.io.usgs_splib07 import load_splib07_dir
    from specret.schema import write_library

    records = []
    for d in args.usgs or [str(USGS_EXTRACTED / "ASCIIdata_splib07a")]:
        recs = load_splib07_dir(d)
        print(f"{d}: {len(recs)} spectra")
        records.extend(recs)
    for d in args.ecostress or []:
        recs = load_ecostress_dir(d)
        print(f"{d}: {len(recs)} spectra")
        records.extend(recs)
    df = write_library(records, args.out)
    flagged = df["purity_flag"].notna().sum()
    print(f"wrote {len(df)} spectra to {args.out} ({flagged} flagged by the scale check)")


def cmd_resample(args: argparse.Namespace) -> None:
    from specret.schema import read_library
    from specret.sensors import load_sensor
    from specret.sensors.library import build_sensor_library

    sensor = load_sensor(args.sensor, band_file=args.band_file)
    df = read_library(args.library)
    if args.measurements:
        df = df[df["measurement"].isin(args.measurements)].reset_index(drop=True)
    lib = build_sensor_library(df, sensor)
    out = Path(args.out or f"data/resampled/{sensor.name}/library.npz")
    lib.save(out)
    print(
        f"{sensor.name}: {sensor.n_bands} bands ({sensor.n_good} good); "
        f"{len(df)} spectra, {int(lib.full_coverage.sum())} with full coverage, "
        f"{int(lib.swir2_coverage.sum())} with 2.0-2.5 um coverage -> {out}"
    )


def cmd_validate_resampler(args: argparse.Namespace) -> None:
    from specret.eval.resampler_check import compare
    from specret.io.usgs_splib07 import load_splib07_dir
    from specret.sensors import load_sensor

    native = load_splib07_dir(args.native)
    convolved = load_splib07_dir(args.convolved)
    report = compare(native, convolved, load_sensor(args.sensor))
    print(report.summary())
    if args.csv:
        report.per_spectrum.to_csv(args.csv, index=False)
    ok = report.median_rel_error < args.tolerance
    print(f"\n{'PASS' if ok else 'FAIL'}: target median relative error < {args.tolerance:.2%}")
    raise SystemExit(0 if ok else 1)


def cmd_index(args: argparse.Namespace) -> None:
    from specret.schema import read_library
    from specret.search.qdrant_store import QdrantStore, collection_name
    from specret.sensors.library import SensorLibrary

    lib = SensorLibrary.load(args.library_npz)
    with QdrantStore(url=args.qdrant_url, path=args.qdrant_path) as store:
        n = store.index_library(lib, read_library(args.library))
    print(f"indexed {n} points into {collection_name(lib.sensor.name)}")


def cmd_query(args: argparse.Namespace) -> None:
    import numpy as np

    from specret.schema import read_library
    from specret.search.retrieve import Retriever, format_matches
    from specret.sensors.library import SensorLibrary

    lib = SensorLibrary.load(args.library_npz)
    r = Retriever(lib, read_library(args.library), vector=args.vector)

    if args.field:
        from specret.io.field import read_field

        for spec in read_field(args.field):
            print(f"\n{spec.name}")
            print(format_matches(r.query_field(spec.wavelength_nm, spec.reflectance, args.k)))
        return

    if args.emit:
        from specret.io.emit import open_emit_l2a

        scene = open_emit_l2a(args.emit)
        if args.lat is not None:
            args.row, args.col = scene.nearest_pixel(args.lat, args.lon)
    elif args.aviris:
        from specret.io.aviris import open_aviris

        scene = open_aviris(args.aviris)
    else:
        raise SystemExit("give one of --field, --emit or --aviris")

    r.check_bands(np.asarray(scene.centers_nm))
    print(f"pixel row={args.row} col={args.col}")
    print(format_matches(r.query(scene.pixel(args.row, args.col), args.k)[0]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="specret", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download USGS splib07 sets or EMIT granules")
    d.add_argument("source", choices=["usgs", "emit"])
    d.add_argument("--sets", nargs="+", default=["splib07a", "splib07b_cvAVIRISc2014"])
    d.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    d.add_argument("--start", default="2022-08-01")
    d.add_argument("--end", default="2026-12-31")
    d.add_argument("--products", nargs="+", default=["EMITL2ARFL", "EMITL2BMIN"])
    d.add_argument("--limit", type=int)
    d.add_argument("--out")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("ingest", help="parse raw libraries into the canonical parquet")
    i.add_argument("--usgs", nargs="*", help="extracted ASCIIdata_<set> folders")
    i.add_argument("--ecostress", nargs="*", help="ECOSTRESS library folders")
    i.add_argument("--out", default=str(DEFAULT_CANONICAL))
    i.set_defaults(func=cmd_ingest)

    r = sub.add_parser("resample", help="resample the canonical library to a sensor")
    r.add_argument(
        "--sensor", required=True, help="config name (emit, aviris_classic, ...) or path"
    )
    r.add_argument("--band-file", help="EMIT granule or ENVI header supplying band centres")
    r.add_argument("--library", default=str(DEFAULT_CANONICAL))
    r.add_argument("--measurements", nargs="*", default=["AREF", "RREF", "RTGC"])
    r.add_argument("--out")
    r.set_defaults(func=cmd_resample)

    v = sub.add_parser("validate-resampler", help="compare against USGS AVIRIS-convolved spectra")
    v.add_argument("--native", default=str(USGS_EXTRACTED / "ASCIIdata_splib07a"))
    v.add_argument("--convolved", default=str(USGS_EXTRACTED / "ASCIIdata_splib07b_cvAVIRISc2014"))
    v.add_argument("--sensor", default="aviris_classic")
    v.add_argument("--tolerance", type=float, default=0.01)
    v.add_argument("--csv")
    v.set_defaults(func=cmd_validate_resampler)

    x = sub.add_parser("index", help="load a resampled library into Qdrant")
    x.add_argument("library_npz")
    x.add_argument("--library", default=str(DEFAULT_CANONICAL))
    x.add_argument("--qdrant-url", default="http://localhost:6333")
    x.add_argument("--qdrant-path", help="use an embedded on-disk store instead of a server")
    x.set_defaults(func=cmd_index)

    q = sub.add_parser("query", help="identify a field spectrum or a scene pixel")
    q.add_argument("library_npz")
    q.add_argument("--library", default=str(DEFAULT_CANONICAL))
    q.add_argument("--vector", default="sam", choices=["sam", "swir2"])
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--field", help=".asd or CSV file")
    q.add_argument("--emit", help="EMIT L2A granule")
    q.add_argument("--aviris", help="AVIRIS ENVI header")
    q.add_argument("--row", type=int)
    q.add_argument("--col", type=int)
    q.add_argument("--lat", type=float)
    q.add_argument("--lon", type=float)
    q.set_defaults(func=cmd_query)

    args = p.parse_args(argv)
    if getattr(args, "qdrant_path", None):
        args.qdrant_url = None
    args.func(args)


if __name__ == "__main__":
    main()
