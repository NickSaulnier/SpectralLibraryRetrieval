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
    df = write_library(records, args.out, classify=True)
    unclassified = int((df["group"] == "unclassified").sum())
    if unclassified:
        print(f"warning: {unclassified} spectra match no group in configs/taxonomy.yaml")
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


def cmd_compare_l2bmin(args: argparse.Namespace) -> None:
    import numpy as np

    from specret.eval.emit_l2bmin import compare_group2, load_l2bmin, summarize
    from specret.io.download import CUPRITE_BBOX
    from specret.io.emit import open_emit_l2a
    from specret.schema import read_library
    from specret.search.retrieve import Retriever
    from specret.sensors.library import SensorLibrary

    scene = open_emit_l2a(args.rfl)
    l2b = load_l2bmin(args.min)
    lib = SensorLibrary.load(args.library_npz)
    canonical = read_library(args.library)
    bbox = tuple(args.bbox) if args.bbox else CUPRITE_BBOX
    for vector in args.vectors:
        chapters = tuple(args.chapters) if args.chapters else None
        if vector in ("rrf", "hybrid"):
            from specret.search.hybrid import HybridRetriever

            r = HybridRetriever(lib, canonical, chapters=chapters, rerank=vector == "hybrid")
        else:
            r = Retriever(lib, canonical, vector=vector, chapters=chapters)
        r.check_bands(np.asarray(scene.centers_nm))
        df = compare_group2(scene, l2b, r, bbox, min_depth=args.min_depth, k=args.k)
        print(
            f"\n=== vector {vector}: {len(df)} pixels, top-1 group agreement "
            f"{df['top1_agree'].mean():.1%}, top-{args.k} {df['topk_agree'].mean():.1%}, "
            f"top-1 class agreement {df['top1_class_agree'].mean():.1%}"
        )
        print(summarize(df).round(3).to_string())
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            suffix = "_" + "".join(args.chapters) if args.chapters else ""
            df.to_parquet(Path(args.out) / f"l2bmin_compare_{vector}{suffix}.parquet")


def cmd_make_splits(args: argparse.Namespace) -> None:
    from specret.eval.splits import load_split_config, make_splits, write_splits
    from specret.schema import read_library

    cfg = load_split_config()
    df = read_library(args.library)
    if df["group"].isna().all():
        raise SystemExit("library has no groups: re-run `specret ingest`")
    splits = make_splits(df, cfg)
    path = write_splits(splits, cfg["version"], force=args.force)
    counts = splits.groupby("split").agg(
        spectra=("spectrum_id", "size"), samples=("sample_id", "nunique")
    )
    print(counts.to_string())
    print(f"wrote {path}")


def cmd_noise_curve(args: argparse.Namespace) -> None:
    import numpy as np

    from specret.distort.simulator import emit_noise_curve
    from specret.io.download import CUPRITE_BBOX
    from specret.io.emit import open_emit_l2a

    scene = open_emit_l2a(args.rfl)
    bbox = tuple(args.bbox) if args.bbox else CUPRITE_BBOX
    curve = emit_noise_curve(args.uncert, scene.lat, scene.lon, bbox)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, curve)
    good = scene.good_mask
    print(f"median uncertainty over good bands {np.nanmedian(curve[good]):.4f} -> {args.out}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    import pandas as pd

    from specret.distort.simulator import load_noise_curve
    from specret.eval.benchmark import run_benchmark
    from specret.eval.splits import load_splits
    from specret.schema import read_library
    from specret.sensors.library import SensorLibrary

    lib = SensorLibrary.load(args.library_npz)
    noise = load_noise_curve(args.noise, lib.sensor.n_bands)
    df = run_benchmark(
        lib, read_library(args.library), load_splits(), args.recipes, args.methods, noise, k=args.k
    )
    with pd.option_context("display.width", 200):
        for metric in ("group_top1", f"group_recall@{args.k}", "material_top1"):
            print(f"\n{metric} (n={df['n'].iloc[0]} test spectra)")
            print(
                df.pivot(index="recipe", columns="method", values=metric)
                .reindex(index=args.recipes, columns=args.methods)
                .round(3)
                .to_string()
            )
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)


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
    q.add_argument("--vector", default="sam", choices=["sam", "swir2", "cr", "cr_swir2"])
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--field", help=".asd or CSV file")
    q.add_argument("--emit", help="EMIT L2A granule")
    q.add_argument("--aviris", help="AVIRIS ENVI header")
    q.add_argument("--row", type=int)
    q.add_argument("--col", type=int)
    q.add_argument("--lat", type=float)
    q.add_argument("--lon", type=float)
    q.set_defaults(func=cmd_query)

    s = sub.add_parser("make-splits", help="create the frozen sample-level split (once)")
    s.add_argument("--library", default=str(DEFAULT_CANONICAL))
    s.add_argument("--force", action="store_true", help="overwrite an existing version")
    s.set_defaults(func=cmd_make_splits)

    n = sub.add_parser("noise-curve", help="per-band noise from an EMIT uncertainty granule")
    n.add_argument("--rfl", required=True, help="EMIT L2A RFL granule (for lat/lon)")
    n.add_argument("--uncert", required=True, help="matching EMIT L2A RFLUNCERT granule")
    n.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    n.add_argument("--out", default="data/derived/emit_noise.npy")
    n.set_defaults(func=cmd_noise_curve)

    b = sub.add_parser("benchmark", help="synthetic benchmark on the frozen test split")
    b.add_argument("library_npz")
    b.add_argument("--library", default=str(DEFAULT_CANONICAL))
    b.add_argument("--noise", default="data/derived/emit_noise.npy")
    b.add_argument(
        "--recipes",
        nargs="+",
        default=["clean", "photometric", "noise", "mix", "atm-mild", "atm-strong", "jitter", "all"],
    )
    b.add_argument(
        "--methods", nargs="+", default=["sam", "swir2", "cr", "cr_swir2", "rrf", "hybrid"]
    )
    b.add_argument("-k", type=int, default=5)
    b.add_argument("--csv")
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("compare-l2bmin", help="agreement with the EMIT L2B mineral map")
    c.add_argument("library_npz", help="library resampled to this scene's EMIT bands")
    c.add_argument("--rfl", required=True, help="EMIT L2A RFL granule")
    c.add_argument("--min", required=True, help="matching EMIT L2B MIN granule")
    c.add_argument("--library", default=str(DEFAULT_CANONICAL))
    c.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    c.add_argument("--min-depth", type=float, default=0.05)
    c.add_argument(
        "--vectors",
        nargs="+",
        default=["sam", "cr_swir2", "rrf", "hybrid"],
        help="sam, swir2, cr, cr_swir2, rrf (fusion of sam/cr/cr_swir2) or hybrid (rrf + SFF)",
    )
    c.add_argument("-k", type=int, default=5)
    c.add_argument("--chapters", nargs="+", help="restrict the library, e.g. M S C")
    c.add_argument("--out", help="folder for per-pixel results (parquet)")
    c.set_defaults(func=cmd_compare_l2bmin)

    args = p.parse_args(argv)
    if getattr(args, "qdrant_path", None):
        args.qdrant_url = None
    args.func(args)


if __name__ == "__main__":
    main()
