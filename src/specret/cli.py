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
    elif args.source == "hitran":
        from specret.io.hitran import fetch_lines

        for p in fetch_lines(args.gases, out_dir=Path(args.out or "data/raw/hitran")):
            print(f"{p}  {p.stat().st_size / 1e6:.1f} MB")
    elif args.source == "emit-gas":
        import yaml

        cfg = yaml.safe_load(Path(args.scene_list).read_text())
        scenes = cfg["scenes"]
        if args.scenes:
            scenes = [s for s in scenes if s["id"] in args.scenes]
        if args.dry_run:
            for s in scenes:
                for u in download.emit_scene_urls(s["id"], args.gas_products or s["products"]):
                    print(u)
            return
        out = Path(args.out or cfg["out_dir"])
        files = download.download_emit_scenes(scenes, out, args.gas_products)
        print(f"{len(files)} files in {out}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from specret.io.ecostress import load_ecostress_dir
    from specret.io.usgs_splib07 import load_splib07_dir
    from specret.schema import write_library

    notes = None
    html_dir = Path(args.usgs_html)
    if html_dir.is_dir():
        from specret.io.usgs_html import load_notes

        notes = load_notes(html_dir)
        print(f"{html_dir}: {len(notes)} sample notes")
    records = []
    for d in args.usgs or [str(USGS_EXTRACTED / "ASCIIdata_splib07a")]:
        recs = load_splib07_dir(d, notes=notes)
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
    flagged = df["quality_flag"].notna().sum()
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


FIELD_SUFFIXES = (".asd", ".csv", ".txt", ".tsv")


def _field_files(targets: list[str]) -> tuple[list[Path], list[str]]:
    """Split targets into field spectrum files (expanding folders and globs) and the rest."""
    import glob

    files, other = [], []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in FIELD_SUFFIXES)
        elif any(ch in t for ch in "*?["):
            matches = (Path(f) for f in glob.glob(t))
            found = sorted(f for f in matches if f.suffix.lower() in FIELD_SUFFIXES)
        elif p.suffix.lower() in FIELD_SUFFIXES:
            if not p.is_file():
                raise SystemExit(f"no such file: {t}")
            found = [p]
        else:
            other.append(t)
            continue
        if not found:
            raise SystemExit(f"no field spectra ({', '.join(FIELD_SUFFIXES)}) in {t}")
        files.extend(found)
    return files, other


def _print_identification(res: dict) -> None:
    s = res["summary"]
    head = res["source"]
    if res.get("name") and res["name"] != Path(res["source"]).stem:
        head += f" / {res['name']}"
    if "pixel" in res:
        px = res["pixel"]
        head += f"  pixel row {px['row']}, col {px['col']}"
        if "lat" in px:
            head += f" ({px['lat']:.5f}, {px['lon']:.5f})"
    print(f"\n{head}  [{res['sensor']} library]")
    if not res["matches"]:
        print(f"  no match: {s['reason']}")
        return
    p = s["p_group_correct"]
    print(f"  {s['verdict']}" + ("" if p is None else f", P(group right) {p:.2f}"))
    print(f"  best match: {s['best_material']}")
    if "unmixing" in res:
        u = res["unmixing"]
        if u is None:
            print("  unmixing: no fit")
        else:
            parts = " + ".join(
                f"{c['material_name']} {100 * c['fraction']:.0f}%" for c in u["components"]
            )
            print(f"  unmixed: {parts} (residual rms {u['residual_rms']:.4f})")
    print(f"\n  {'#':>2}  {'score':>5}  {'material':<36} {'group':<18} {'purity':<6} evidence")
    for i, m in enumerate(res["matches"], start=1):
        ev = " | ".join(
            f"{f['window']} {f['query_center_nm']:.0f}->{f['cand_center_nm']:.0f} nm "
            f"r {f['shape_r']:.2f}"
            for f in m.get("features", [])[:2]
        )
        print(
            f"  {i:>2}  {m['score']:5.2f}  {str(m['material_name'])[:36]:<36} "
            f"{str(m.get('group') or '-')[:18]:<18} {str(m.get('purity_flag') or '-'):<6} {ev}"
        )


def _identification_row(res: dict) -> dict:
    """One flat CSV row per identified spectrum."""
    s = res["summary"]
    row = {"source": res["source"], "name": res.get("name"), **res.get("pixel", {})}
    row.update(
        sensor=res["sensor"],
        verdict=s["verdict"],
        best_group=s.get("best_group"),
        best_material=s.get("best_material"),
        p_group_correct=s.get("p_group_correct"),
        confidence=s.get("confidence"),
    )
    for i, m in enumerate(res["matches"], start=1):
        row[f"match{i}_material"] = m["material_name"]
        row[f"match{i}_group"] = m.get("group")
        row[f"match{i}_score"] = round(m["score"], 4)
        row[f"match{i}_spectrum_id"] = m["spectrum_id"]
    u = res.get("unmixing")
    if u:
        row["unmixed"] = " + ".join(
            f"{c['material_name']} {c['fraction']:.2f}" for c in u["components"]
        )
    return row


def cmd_identify(args: argparse.Namespace) -> None:
    import json

    from specret.workspace import Workspace

    chapters = None if [c.lower() for c in args.chapters] == ["all"] else tuple(args.chapters)
    ws = Workspace(chapters=chapters)
    field, scenes = _field_files(args.target)
    try:
        if field and scenes:
            raise ValueError("give either field spectrum files or one scene, not both")
        if field:
            results = [
                {"source": str(f), **res}
                for f in field
                for res in ws.identify_field(f, args.k, unmix=args.unmix)
            ]
        else:
            if len(scenes) != 1:
                raise ValueError("give one scene (file path or id from data/scenes)")
            scene = ws.scene(scenes[0])
            row, col = ws.locate(scene, args.row, args.col, args.lat, args.lon)
            res = ws.identify(
                scene.pixel(row, col), ws.sensor_for_scene(scene), args.k, unmix=args.unmix
            )
            results = [{"source": scenes[0], "pixel": ws.pixel_info(scene, row, col), **res}]
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    if args.csv:
        import pandas as pd

        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([_identification_row(r) for r in results]).to_csv(args.csv, index=False)
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for res in results:
        _print_identification(res)
    if any(r["summary"].get("confidence") == "uncalibrated" for r in results):
        print("\nno confidence model for this sensor; run `specret calibrate` to add one")
    if args.csv:
        print(f"\nwrote {len(results)} rows to {args.csv}")


def cmd_doctor(args: argparse.Namespace) -> None:
    from specret.project import inspect, report

    items = inspect()
    print(f"specret doctor: {Path.cwd()}")
    print(report(items))
    raise SystemExit(0 if all(it.ok for it in items if it.level == "required") else 1)


def cmd_setup(args: argparse.Namespace) -> None:
    from specret.project import report, setup

    try:
        items = setup(main, map_scenes=args.map)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
    print("\n=== status")
    print(report(items))


def cmd_compare_l2bmin(args: argparse.Namespace) -> None:
    import numpy as np

    from specret.eval.benchmark import build_method
    from specret.eval.emit_l2bmin import compare_group2, load_l2bmin, summarize
    from specret.io.download import CUPRITE_BBOX
    from specret.io.emit import open_emit_l2a
    from specret.schema import read_library
    from specret.sensors.library import SensorLibrary

    scene = open_emit_l2a(args.rfl)
    l2b = load_l2bmin(args.min)
    lib = SensorLibrary.load(args.library_npz)
    canonical = read_library(args.library)
    bbox = tuple(args.bbox) if args.bbox else (None if args.whole_scene else CUPRITE_BBOX)
    for vector in args.vectors:
        chapters = tuple(args.chapters) if args.chapters else None
        purity = tuple(args.purity) if args.purity else None
        r = build_method(
            vector, lib, canonical, chapters=chapters, purity=purity, encoder_path=args.encoder
        )
        r.check_bands(np.asarray(scene.centers_nm))
        df = compare_group2(
            scene, l2b, r, bbox, min_depth=args.min_depth, k=args.k, max_pixels=args.max_pixels
        )
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


def cmd_gas_library(args: argparse.Namespace) -> None:
    from specret.gas.library import build_window, lbl_ranges, plume_config, window_path

    windows = plume_config()["matched_filter"]["windows"]
    for gas in args.gases:
        for lo, hi in lbl_ranges(windows[gas]):
            lbl = build_window(lo, hi, force=args.force)
            print(
                f"{gas} {lo:.0f}-{hi:.0f} nm: {lbl.nu_cm1.size} points, {sorted(lbl.sigma)} "
                f"-> {window_path(lo, hi)}"
            )


def cmd_detect_plumes(args: argparse.Namespace) -> None:
    import numpy as np

    from specret.gas.detect import detect, write_enhancement
    from specret.io.emit_l1b import find_l1b, open_emit_l1b

    for s in args.scenes:
        if s.startswith("ang"):  # AVIRIS-NG flight line
            from specret.io.avirisng import open_avirisng

            scene = open_avirisng(s)
        else:
            scene = open_emit_l1b(find_l1b(s))
        if args.enhancement_only:
            enh = detect(scene, args.gas)
            path = write_enhancement(scene, enh, Path(args.out))
            sig = np.nanmedian(enh.result.sigma)
            print(
                f"{scene.scene_id}: {path} ({enh.seconds:.0f} s, background sigma "
                f"{sig:.0f} ppm m, AMF {enh.conditions.amf:.2f}, H2O "
                f"{enh.conditions.h2o_gcm2:.2f} g/cm2)"
            )
            continue
        from specret.gas.detect import prepare
        from specret.gas.outputs import write_scene
        from specret.gas.pipeline import analyze, build_plumes

        prep = prepare(scene)
        enh, recs = analyze(prep)
        plumes = build_plumes(prep, enh, recs, wind=_wind_arg(args))
        folder = write_scene(prep, enh, recs, plumes, Path(args.out))
        print(f"{scene.scene_id}: {len(recs)} candidates, {len(plumes)} plumes -> {folder}")
        for pl in plumes[: args.show]:
            q = pl.quant
            rate = f"{q.q_kg_h:,.0f} +/- {q.sigma_q_kg_h:,.0f} kg/h" if q else "no wind"
            print(f"  #{pl.id} {pl.record.ident.gas} p={pl.confidence:.2f} {rate}")


def _wind_arg(args: argparse.Namespace):
    """A user wind from --wind U,V (m/s), or None to look it up."""
    if not getattr(args, "wind", None):
        return None
    from specret.io.wind import Wind

    u, v = (float(x) for x in args.wind.split(","))
    w = Wind(u, v, "user")
    w.sd = args.wind_sd if args.wind_sd is not None else 0.5 * w.speed
    return w


def cmd_quantify(args: argparse.Namespace) -> None:
    import pandas as pd

    from specret.gas.library import plume_config
    from specret.gas.quantify import requantify
    from specret.io.emit_l1b import find_l1b, scene_id

    sid = scene_id(find_l1b(args.scene))
    df = pd.read_csv(Path(args.out) / sid / "plumes.csv").set_index("plume_id")
    if args.plume not in df.index:
        raise SystemExit(f"no plume {args.plume} in {sid}; have {list(df.index)}")
    r = df.loc[args.plume]
    q = requantify(r.to_dict(), tuple(plume_config()["quantify"]["u_eff"]), _wind_arg(args))
    print(f"{sid} plume {args.plume}: {r['gas']}, confidence {r['confidence']:.2f}")
    print(f"  source {r['source_lat']:.5f}, {r['source_lon']:.5f}; {int(r['n_px'])} px")
    print(f"  IME {q['ime_kg']:.1f} kg, L {q['length_m']:.0f} m")
    print(
        f"  wind {q['u10']:.2f} +/- {q['u10_sd']:.2f} m/s ({q['wind_source']}), "
        f"U_eff {q['u_eff']:.2f} m/s"
    )
    budget = ", ".join(f"{k} {v:.0%}" for k, v in q["relative_uncertainty"].items())
    print(f"  Q = {q['q_kg_h']:,.0f} +/- {q['sigma_q_kg_h']:,.0f} kg/h  ({budget})")


def cmd_detect_plumes_lwir(args: argparse.Namespace) -> None:
    from specret.gas.lwir import analyze_flight, write_flight
    from specret.io.hytes import open_hytes

    for name in args.flights:
        f = open_hytes(name)
        rows = slice(args.rows[0], args.rows[1]) if args.rows else None
        df, maps = analyze_flight(f, args.plume_temperature, rows)
        folder = write_flight(f, df, maps, args.out)
        print(f"{f.name}: {len(df)} candidates -> {folder}")


def cmd_gas_persistence(args: argparse.Namespace) -> None:
    from specret.gas.persistence import load_scene, persistence
    from specret.io.emit_l1b import find_l1b, open_emit_l1b

    results = []
    for s in args.scenes:
        scene = open_emit_l1b(find_l1b(s))
        results.append(load_scene(Path(args.out) / scene.scene_id, scene))
    df = persistence(results, args.radius)
    path = Path(args.out) / "persistence.csv"
    df.to_csv(path, index=False)
    covered = df[df["overpasses_covering"] > 0]
    print(
        f"{len(df)} plumes in {len(results)} scenes; {len(covered)} covered by another "
        f"overpass, {int((covered['overpasses_seen'] > 0).sum())} seen again -> {path}"
    )


def cmd_compare_ghg(args: argparse.Namespace) -> None:
    import json

    from specret.eval.ghg_compare import compare
    from specret.io.emit_l1b import find_l1b, scene_id

    rows = []
    for s in args.scenes:
        l1b = find_l1b(s)
        sid = scene_id(l1b)
        ours = Path(args.out) / sid / f"{args.gas.lower()}_enh.tif"
        r = compare(ours, l1b.parent, sid, args.gas)
        rows.append(r)
        p, a = r["plume_pixels"], r["scene_pixels"]
        plume = (
            f"plumes n={p['n']} r={p['r']:.3f} slope={p['deming_slope']:.3f} "
            f"(OLS {p['slope']:.3f}) bias={p['mean_diff']:.0f}"
            if "r" in p
            else f"plumes n={p['n']}"
        )
        print(
            f"{sid}: {plume}; scene r={a['r']:.3f} slope={a['slope']:.3f}; background "
            f"sigma ours {r['background_sigma_ours']:.0f}, EMIT "
            f"{r['background_sigma_emit']:.0f} ppm m"
        )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=1))


def cmd_gas_benchmark(args: argparse.Namespace) -> None:
    import json

    from specret.eval.gas_benchmark import evaluate, run_all

    out = Path(args.out)
    if not args.evaluate_only:
        run_all(out, seeds=args.seeds, n_plumes=args.plumes)
    m = evaluate(out)
    truth = m.pop("truth")
    truth.to_parquet(out / "truth_found.parquet")
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    for k, v in m.items():
        if k not in ("model", "id_confusion") and not k.startswith("verdicts"):
            print(f"{k}: {v}")


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
        lib,
        read_library(args.library),
        load_splits(),
        args.recipes,
        args.methods,
        noise,
        k=args.k,
        purity=tuple(args.purity) if args.purity else None,
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


def cmd_map_scene(args: argparse.Namespace) -> None:
    from specret.scene.mapper import group_fractions, map_scene
    from specret.scene.outputs import write_scene_maps
    from specret.schema import read_library
    from specret.search.hybrid import HybridRetriever
    from specret.sensors.library import SensorLibrary
    from specret.workspace import Workspace, open_scene

    scene = open_scene(args.scene)
    canonical = read_library(args.library)
    chapters = tuple(args.chapters) if args.chapters else None
    r = HybridRetriever(
        SensorLibrary.load(args.library_npz),
        canonical,
        chapters=chapters,
        sff_weight=args.sff_weight,
    )
    # Calibrations are named after the library folder, as in the viewer and MCP tools.
    model = Workspace().confidence_model(Path(args.library_npz).parent.name)
    smap = map_scene(scene, r, canonical, k=args.k, confidence=model)
    out = Path(args.out or f"outputs/maps/{Path(args.scene).stem}")
    smap.save(out / "scene_map.npz")
    paths = write_scene_maps(smap, scene, canonical, out)
    n = int(smap.valid.sum())
    print(f"mapped {n:,} pixels in {smap.seconds:.0f} s -> {out}")
    print(", ".join(p.name for p in paths.values()))
    print(group_fractions(smap).head(12).round(3).to_string())


def cmd_calibrate(args: argparse.Namespace) -> None:
    from specret.distort.simulator import load_noise_curve
    from specret.eval.openset_eval import (
        build_table,
        coverage_at_threshold,
        cross_validate,
        fit_final,
        report,
    )
    from specret.eval.splits import load_splits
    from specret.schema import read_library
    from specret.sensors.library import SensorLibrary

    lib = SensorLibrary.load(args.library_npz)
    noise = load_noise_curve(args.noise, lib.sensor.n_bands)
    table, _ = build_table(lib, read_library(args.library), load_splits(), noise)
    cv = cross_validate(table)
    print(
        f"{lib.sensor.name}: {int(cv['known'].sum())} known and "
        f"{int((~cv['known']).sum())} unknown evaluation queries (cross-validated)"
    )
    print(report(cv).round(3).to_string())
    for th in (0.3, 0.5):
        print({k: round(v, 3) for k, v in coverage_at_threshold(cv, th).items()})
    out = Path(args.out or f"configs/calibration/{lib.sensor.name}.json")
    fit_final(table, threshold=args.threshold).save(out)
    print(f"wrote {out}")


def cmd_eval_unmix(args: argparse.Namespace) -> None:
    from specret.distort.simulator import load_noise_curve
    from specret.eval.benchmark import _subset
    from specret.eval.mixtures import evaluate, make_mixtures, summarize
    from specret.eval.splits import load_splits
    from specret.schema import read_library
    from specret.search.hybrid import HybridRetriever
    from specret.sensors.library import SensorLibrary
    from specret.unmix.pursuit import Unmixer

    lib = SensorLibrary.load(args.library_npz)
    canonical = read_library(args.library)
    splits = load_splits()
    noise = load_noise_curve(args.noise, lib.sensor.n_bands)
    ref_splits = ("train",) if args.split == "val" else ("train", "val")
    ref = _subset(lib, set(splits.loc[splits["split"].isin(ref_splits), "spectrum_id"]))
    queries = _subset(lib, set(splits.loc[splits["split"] == args.split, "spectrum_id"]))
    groups = canonical.set_index("spectrum_id").loc[queries.spectrum_ids, "group"].to_numpy()
    q, truth = make_mixtures(queries, groups, args.n, noise)
    df = evaluate(Unmixer(HybridRetriever(ref, canonical), canonical), q, truth, noise)
    print(summarize(df).round(3).to_string())
    for method, part in df.groupby("method"):
        r, p = part["recall"].mean(), part["precision"].mean()
        print(
            f"{method}: recall {r:.3f} precision {p:.3f} F1 {2 * r * p / (r + p):.3f} "
            f"abundance RMSE {part['abundance_rmse'].mean():.3f}"
        )


def cmd_sample_pixels(args: argparse.Namespace) -> None:
    from specret.learn.pixels import sample_pixels, save_pool

    px = sample_pixels(args.rfl, args.mask, args.n)
    save_pool(args.out, px, args.rfl)
    print(f"{len(px):,} clear pixels -> {args.out}")


def cmd_pretrain(args: argparse.Namespace) -> None:
    import json

    from specret.learn.pixels import load_pool
    from specret.learn.pretrain import pretrain

    pixels, centers, fwhm, good = load_pool(args.pools)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = pretrain(pixels, centers, fwhm, good, out, steps=args.steps, batch=args.batch)
    out.with_suffix(".json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"pretrained on {log['n_pixels']:,} pixels -> {out}")


def cmd_train_encoder(args: argparse.Namespace) -> None:
    import json

    import numpy as np

    from specret.eval.splits import load_splits
    from specret.learn.train import TrainConfig, train
    from specret.schema import read_library
    from specret.sensors.library import SensorLibrary

    libs = {n: SensorLibrary.load(f"data/resampled/{n}/library.npz") for n in args.sensors}
    noise = {}
    for n in args.sensors:
        path = Path(f"data/derived/{n}_noise.npy")
        if path.exists():
            noise[n] = np.load(path)
    real = None
    if args.real_pixels:
        from specret.learn.pixels import load_pool

        real = load_pool(args.real_pixels)[0]
    cfg = TrainConfig(
        sensors=tuple(args.sensors),
        steps=args.steps,
        batch=args.batch,
        init=args.init,
        real_pixels=real,
        real_mix=args.real_mix,
        aux_weight=args.aux_weight,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = train(libs, read_library(args.library), load_splits(), cfg, out, noise)
    out.with_suffix(".json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"best val mean {log['best_val_mean']:.3f} -> {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="specret", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "download", help="download USGS splib07 sets, EMIT granules or HITRAN line lists"
    )
    d.add_argument("source", choices=["usgs", "emit", "emit-gas", "hitran"])
    d.add_argument("--sets", nargs="+", default=["splib07a", "splib07b_cvAVIRISc2014"])
    d.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    d.add_argument("--start", default="2022-08-01")
    d.add_argument("--end", default="2026-12-31")
    d.add_argument("--products", nargs="+", default=["EMITL2ARFL", "EMITL2BMIN"])
    d.add_argument("--limit", type=int)
    d.add_argument("--out")
    d.add_argument("--scene-list", default="configs/gas_scenes.yaml", help="emit-gas: scene list")
    d.add_argument("--scenes", nargs="+", help="emit-gas: only these scene ids")
    d.add_argument("--gas-products", nargs="+", help="emit-gas: override each scene's product list")
    d.add_argument("--dry-run", action="store_true", help="emit-gas: print URLs only")
    d.add_argument("--gases", nargs="+", help="hitran: only these gases (configs/gases.yaml)")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("ingest", help="parse raw libraries into the canonical parquet")
    i.add_argument("--usgs", nargs="*", help="extracted ASCIIdata_<set> folders")
    i.add_argument("--ecostress", nargs="*", help="ECOSTRESS library folders")
    i.add_argument(
        "--usgs-html",
        default="data/raw/usgs_splib07_html/pages",
        help="extracted splib07 HTML pages, for sample notes (skipped if missing)",
    )
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

    idf = sub.add_parser(
        "identify",
        help="identify field spectra or a scene pixel (hybrid ranking, confidence, unmixing)",
        description="Identify materials in field spectrometer files or in one scene pixel. "
        "The library is picked to match the input's bands, as in the viewer and MCP tools.",
    )
    idf.add_argument(
        "target",
        nargs="+",
        help="field files (.asd, .csv, .txt, .tsv), folders or globs of them; "
        "or one scene (EMIT .nc / AVIRIS .hdr path, or a scene id from data/scenes)",
    )
    idf.add_argument("--row", type=int)
    idf.add_argument("--col", type=int)
    idf.add_argument("--lat", type=float, help="with --lon, the nearest EMIT pixel")
    idf.add_argument("--lon", type=float)
    idf.add_argument("-k", type=int, default=5, help="matches to list (default 5)")
    idf.add_argument("--unmix", action="store_true", help="also split into endmembers")
    idf.add_argument(
        "--chapters",
        nargs="+",
        default=["M", "S", "C", "V"],
        help="library chapters to search, or 'all' (default: minerals, mixtures, coatings, "
        "vegetation)",
    )
    idf.add_argument("--json", action="store_true", help="print the full results as JSON")
    idf.add_argument("--csv", help="also write one summary row per spectrum to this CSV")
    idf.set_defaults(func=cmd_identify)

    q = sub.add_parser("query", help="baseline SAM nearest neighbours (see `identify`)")
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
        "--methods",
        nargs="+",
        default=["sam", "sid", "swir2", "cr", "cr_euclid", "cr_swir2", "rrf", "hybrid"],
    )
    b.add_argument("-k", type=int, default=5)
    b.add_argument("--purity", nargs="+", help="keep USGS purity grades, e.g. a b u")
    b.add_argument("--csv")
    b.set_defaults(func=cmd_benchmark)

    vw = sub.add_parser("view", help="local web viewer for scene maps and field spectra")
    vw.add_argument("--host", default="127.0.0.1")
    vw.add_argument("--port", type=int, default=8765)
    vw.set_defaults(
        func=lambda a: __import__("specret.viewer.app").viewer.app.serve(a.host, a.port)
    )

    mc = sub.add_parser("mcp", help="run the FastMCP tool server (stdio)")
    mc.set_defaults(func=lambda _args: __import__("specret.mcp_server").mcp_server.main())

    cal = sub.add_parser("calibrate", help="fit the confidence / unknown-material model")
    cal.add_argument("library_npz")
    cal.add_argument("--library", default=str(DEFAULT_CANONICAL))
    cal.add_argument("--noise", default="data/derived/emit_noise.npy")
    cal.add_argument("--threshold", type=float, default=0.5)
    cal.add_argument("--out")
    cal.set_defaults(func=cmd_calibrate)

    eu = sub.add_parser("eval-unmix", help="synthetic mixture benchmark for unmixing")
    eu.add_argument("library_npz")
    eu.add_argument("--library", default=str(DEFAULT_CANONICAL))
    eu.add_argument("--noise", default="data/derived/emit_noise.npy")
    eu.add_argument("--split", default="test", choices=["val", "test"])
    eu.add_argument("-n", type=int, default=300)
    eu.set_defaults(func=cmd_eval_unmix)

    sp = sub.add_parser("sample-pixels", help="sample clear EMIT pixels for pretraining")
    sp.add_argument("--rfl", required=True)
    sp.add_argument("--mask", help="matching EMIT L2A MASK granule")
    sp.add_argument("-n", type=int, default=200_000)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_sample_pixels)

    pt = sub.add_parser("pretrain-encoder", help="masked-band pretraining on real pixels")
    pt.add_argument("pools", nargs="+", help="pixel pools from `sample-pixels`")
    pt.add_argument("--steps", type=int, default=20000)
    pt.add_argument("--batch", type=int, default=512)
    pt.add_argument("--out", default="models/pretrained_v1.pt")
    pt.set_defaults(func=cmd_pretrain)

    te = sub.add_parser("train-encoder", help="train the learned spectral encoder (Phase 5)")
    te.add_argument("--sensors", nargs="+", default=["emit", "avirisc"])
    te.add_argument("--steps", type=int, default=3000)
    te.add_argument("--batch", type=int, default=256)
    te.add_argument("--library", default=str(DEFAULT_CANONICAL))
    te.add_argument("--out", default="models/encoder_v1.pt")
    te.add_argument("--init", help="pretrained checkpoint to start from")
    te.add_argument("--real-pixels", nargs="+", help="pixel pools from `sample-pixels`")
    te.add_argument("--real-mix", type=float, default=0.0)
    te.add_argument("--aux-weight", type=float, default=0.0)
    te.set_defaults(func=cmd_train_encoder)

    ms = sub.add_parser("map-scene", help="map every pixel of a scene with the hybrid retriever")
    ms.add_argument("library_npz", help="library resampled to this scene's bands")
    ms.add_argument("--scene", required=True, help="EMIT L2A RFL .nc or AVIRIS .hdr")
    ms.add_argument("--library", default=str(DEFAULT_CANONICAL))
    ms.add_argument(
        "--chapters",
        nargs="+",
        default=["M", "S", "C", "V"],
        help="library chapters to search (default: minerals, mixtures, coatings, vegetation)",
    )
    ms.add_argument("--sff-weight", type=float, default=0.25)
    ms.add_argument("-k", type=int, default=5)
    ms.add_argument("--out", help="output folder (default outputs/maps/<scene>)")
    ms.set_defaults(func=cmd_map_scene)

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
        help="sam, sid, swir2, cr, cr_euclid, cr_swir2, rrf (fusion of sam/cr/cr_swir2) "
        "or hybrid (rrf + SFF)",
    )
    c.add_argument("-k", type=int, default=5)
    c.add_argument("--chapters", nargs="+", help="restrict the library, e.g. M S C")
    c.add_argument("--purity", nargs="+", help="keep USGS purity grades, e.g. a b u")
    c.add_argument("--whole-scene", action="store_true", help="use every pixel, not Cuprite")
    c.add_argument("--max-pixels", type=int, help="random sample of this many pixels")
    c.add_argument("--encoder", help="encoder checkpoint for learned methods")
    c.add_argument("--out", help="folder for per-pixel results (parquet)")
    c.set_defaults(func=cmd_compare_l2bmin)

    gl = sub.add_parser("gas-library", help="line-by-line gas cross-sections from HITRAN")
    gl.add_argument("--gases", nargs="+", default=["CH4"], help="matched-filter windows to build")
    gl.add_argument("--force", action="store_true", help="recompute even if cached")
    gl.set_defaults(func=cmd_gas_library)

    dp = sub.add_parser(
        "detect-plumes",
        help="find, identify and quantify gas plumes in EMIT L1B radiance",
        description="CH4 and CO2 matched filters, candidate plumes, joint gas + confuser "
        "identification, calibrated confidence and emission rates (wind from HRRR or MERRA-2). "
        "Writes enhancement maps, plume_mask.tif, confidence.tif, plumes.geojson, plumes.csv "
        "and candidates.csv to outputs/gas/<scene id>/. --enhancement-only writes one gas's "
        "enhancement map and stops.",
    )
    dp.add_argument(
        "scenes", nargs="+", help="EMIT scene ids, L1B RAD paths, or AVIRIS-NG line ids (ang...)"
    )
    dp.add_argument("--enhancement-only", action="store_true")
    dp.add_argument("--gas", default="CH4", choices=["CH4", "CO2"], help="with --enhancement-only")
    dp.add_argument("--wind", help="use this 10 m wind for every plume: U,V in m/s")
    dp.add_argument("--wind-sd", type=float, help="its standard deviation (default half the speed)")
    dp.add_argument("--show", type=int, default=10, help="plumes to print per scene")
    dp.add_argument("--out", default="outputs/gas")
    dp.set_defaults(func=cmd_detect_plumes)

    dl = sub.add_parser(
        "detect-plumes-lwir",
        help="find and identify gas plumes in HyTES thermal-infrared flights",
        description="LWIR matched filter per gas with the thermal-contrast target, candidates "
        "merged across gases, and library identification (retrieve, then joint fit). Plume "
        "temperature defaults to the MERRA-2 2 m air temperature. Writes candidates.csv, "
        "plumes.csv, plumes.geojson and enhancement.npz to outputs/gas/hytes/<flight>/.",
    )
    dl.add_argument("flights", nargs="+", help="HyTES flight names (or part), e.g. SaltonSea")
    dl.add_argument("--plume-temperature", type=float, help="plume temperature in K")
    dl.add_argument("--rows", nargs=2, type=int, metavar=("START", "STOP"))
    dl.add_argument("--out", default="outputs/gas/hytes")
    dl.set_defaults(func=cmd_detect_plumes_lwir)

    qp = sub.add_parser(
        "quantify",
        help="emission rate of one plume, optionally with your own wind",
        description="Recompute a plume's rate from plumes.csv (written by detect-plumes): "
        "Q = IME * U_eff / L, with U_eff = a * U10 + b (configs/plumes.yaml).",
    )
    qp.add_argument("scene", help="EMIT scene id")
    qp.add_argument("--plume", type=int, required=True)
    qp.add_argument("--wind", help="10 m wind U,V in m/s (default: the one detect-plumes used)")
    qp.add_argument("--wind-sd", type=float)
    qp.add_argument("--out", default="outputs/gas")
    qp.set_defaults(func=cmd_quantify)

    cg = sub.add_parser("compare-ghg", help="agreement with EMIT L2B CH4ENH/CO2ENH")
    cg.add_argument("scenes", nargs="+", help="scene ids already run through detect-plumes")
    cg.add_argument("--gas", default="CH4", choices=["CH4", "CO2"])
    cg.add_argument("--out", default="outputs/gas", help="where detect-plumes wrote its maps")
    cg.add_argument("--json", help="write the metrics to this file")
    cg.set_defaults(func=cmd_compare_ghg)

    gp = sub.add_parser(
        "gas-persistence",
        help="which plumes recur at the same place on other overpasses",
        description="For scenes already run through detect-plumes, count for each plume the "
        "other overpasses that imaged its source and those that saw the same gas within "
        "--radius metres. Writes outputs/gas/persistence.csv.",
    )
    gp.add_argument("scenes", nargs="+", help="EMIT scene ids")
    gp.add_argument("--radius", type=float, default=300.0, help="metres")
    gp.add_argument("--out", default="outputs/gas")
    gp.set_defaults(func=cmd_gas_persistence)

    gb = sub.add_parser(
        "gas-benchmark",
        help="G2 benchmark: injected plumes, false plumes, catalogue recall, calibration",
    )
    gb.add_argument("--out", default="outputs/gas/g2")
    gb.add_argument("--seeds", type=int, default=4, help="injection draws per no-plume scene")
    gb.add_argument("--plumes", type=int, default=25, help="plumes per draw")
    gb.add_argument("--evaluate-only", action="store_true", help="reuse saved runs")
    gb.set_defaults(func=cmd_gas_benchmark)

    dr = sub.add_parser("doctor", help="show what is set up and what to run next")
    dr.set_defaults(func=cmd_doctor)

    su = sub.add_parser(
        "setup",
        help="download, ingest, resample and calibrate whatever is missing",
        description="Run the missing pipeline steps in order: USGS download, ingest, sensor "
        "libraries, a library for each scene's bands, EMIT noise curve and calibration. "
        "Steps already done are skipped, so it is safe to re-run after adding scenes.",
    )
    su.add_argument("--map", action="store_true", help="also map unmapped scenes for the viewer")
    su.set_defaults(func=cmd_setup)

    args = p.parse_args(argv)
    if getattr(args, "qdrant_path", None):
        args.qdrant_url = None
    args.func(args)


if __name__ == "__main__":
    main()
