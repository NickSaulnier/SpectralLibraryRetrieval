"""Write a scene's gas results to outputs/gas/<scene id>/ (PLAN_GAS.md section 2).

* <gas>_enh.tif, <gas>_enh.json  matched-filter enhancement per gas (gas/detect.py)
* plume_mask.tif    int32 plume id on the GLT grid (0 = none)
* confidence.tif    float32 confidence of the plume at each pixel (nodata elsewhere)
* plumes.geojson    one feature per plume: outline polygon(s) plus the plumes.csv fields
* plumes.csv        one row per plume: gas, confidence, source, IME, Q, sigma_Q, wind, confusers
* candidates.csv    every candidate with its verdict and features, kept or not
* plume_spectra.npz each candidate's fit in log-radiance units (identify.FitSpectra), keyed
                    "<detection gas>_<candidate id>/<array>", plus the scene's gas signatures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine

from specret.gas.detect import Enhancement, Prepared, write_enhancement
from specret.gas.pipeline import CandidateRecord, Plume
from specret.io.emit_ghg import NODATA
from specret.scene.outputs import _write
from specret.workspace import jsonable


def plume_label_swath(shape, plumes: list[Plume]) -> np.ndarray:
    lab = np.zeros(shape, dtype=np.int32)
    for p in plumes:
        c = p.record.candidate
        lab[c.rows, c.cols] = np.where(lab[c.rows, c.cols] == 0, p.id, lab[c.rows, c.cols])
    return lab


def write_spectra(path: Path, fits) -> None:
    """Candidates' fit spectra, so the viewer can show the evidence without a rerun. `fits`
    is an iterable of (key, identify.FitSpectra or None); see `spectra_key`."""
    arrays: dict[str, np.ndarray] = {}
    for key, f in fits:
        if f is None:
            continue
        if "centers_nm" not in arrays:
            arrays["centers_nm"] = f.centers_nm.astype(np.float32)
            arrays["windows"] = np.asarray(f.windows, dtype=np.float32)
            arrays["signature_names"] = np.array(list(f.signatures))
            arrays["signatures"] = np.array(list(f.signatures.values()), dtype=np.float32)
        key += "/"
        arrays[key + "obs"] = f.obs.astype(np.float32)
        arrays[key + "null_sd"] = f.null_sd.astype(np.float32)
        arrays[key + "baseline"] = f.baseline.astype(np.float32)
        arrays[key + "term_names"] = np.array(list(f.terms), dtype=str)
        arrays[key + "terms"] = np.array(
            list(f.terms.values()) or np.zeros((0, f.obs.size)), dtype=np.float32
        )
    np.savez_compressed(path, **arrays)


def spectra_key(detection_gas: str, candidate_id: int, segment: int | None = None) -> str:
    """ "<gas>_<id>", or "<gas>_<segment>_<id>" for HyTES, whose ids restart per segment."""
    seg = "" if segment is None else f"{int(segment)}_"
    return f"{detection_gas}_{seg}{int(candidate_id)}"


def write_scene(
    prep: Prepared,
    enh: dict[str, Enhancement],
    records: list[CandidateRecord],
    plumes: list[Plume],
    out_dir: Path = Path("outputs/gas"),
) -> Path:
    from rasterio import features

    scene = prep.scene
    folder = out_dir / scene.scene_id
    folder.mkdir(parents=True, exist_ok=True)
    for e in enh.values():
        write_enhancement(scene, e, out_dir)
    transform = Affine.from_gdal(*scene.geotransform)
    lab = scene.orthorectify(plume_label_swath(scene.shape, plumes), fill=0).astype(np.int32)
    _write(folder / "plume_mask.tif", lab, transform, scene.crs_wkt, 0)
    conf = np.full(lab.shape, NODATA, dtype=np.float32)
    for p in plumes:
        conf[lab == p.id] = p.confidence
    _write(folder / "confidence.tif", conf, transform, scene.crs_wkt, NODATA)

    rows = [p.row() for p in plumes]
    pd.DataFrame(rows).to_csv(folder / "plumes.csv", index=False)
    pd.DataFrame([{"scene": scene.scene_id, **r.row()} for r in records]).to_csv(
        folder / "candidates.csv", index=False
    )
    write_spectra(
        folder / "plume_spectra.npz",
        ((spectra_key(r.detection_gas, r.candidate.id), r.ident.spectra) for r in records),
    )
    feats = []
    by_id = {r["plume_id"]: r for r in rows}
    for geom, v in features.shapes(lab, mask=lab > 0, transform=transform):
        feats.append({"type": "Feature", "geometry": geom, "properties": jsonable(by_id[int(v)])})
    gj = {"type": "FeatureCollection", "scene": scene.scene_id, "features": feats}
    (folder / "plumes.geojson").write_text(json.dumps(gj), encoding="utf-8")
    return folder
