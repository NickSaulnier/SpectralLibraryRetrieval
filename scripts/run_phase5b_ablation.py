"""Phase 5b ablations: fine-tune each variant, then score it on the Mountain Pass validation scene.

Cuprite is not touched here; it is run once, afterwards, for the selected variant only.

Run: uv run python scripts/run_phase5b_ablation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from specret.eval.benchmark import build_method
from specret.eval.emit_l2bmin import compare_group2, load_l2bmin
from specret.io.emit import open_emit_l2a
from specret.schema import read_library
from specret.sensors.library import SensorLibrary

POOLS = [f"data/derived/pretrain/{n}.npz" for n in ("pilbara", "kalahari", "sarcheshmeh")]
VARIANTS = {
    "v1b": [],  # fixed tokenizer, no pretraining (already trained)
    "pre": ["--init", "models/pretrained_v1.pt"],
    "pre_mix": ["--init", "models/pretrained_v1.pt", "--real-pixels", *POOLS, "--real-mix", "0.5"],
    "pre_mix_aux": [
        "--init",
        "models/pretrained_v1.pt",
        "--real-pixels",
        *POOLS,
        "--real-mix",
        "0.5",
        "--aux-weight",
        "0.5",
    ],
}
MP = "data/scenes/emit_extra/{}_001_20240416T210255_2410714_015.nc"


def main() -> None:
    lib = SensorLibrary.load("data/resampled/emit/library.npz")
    canonical = read_library("data/canonical/library.parquet")
    scene = open_emit_l2a(MP.format("EMIT_L2A_RFL"))
    l2b = load_l2bmin(MP.format("EMIT_L2B_MIN"))
    results = {}
    for name, extra in VARIANTS.items():
        ckpt = Path(f"models/encoder_{name}.pt")
        if not ckpt.exists():
            cmd = ["specret", "train-encoder", "--steps", "4000", "--out", str(ckpt), *extra]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        row = {}
        for method in ("learned", "hybrid+learned"):
            r = build_method(
                method, lib, canonical, chapters=("M", "S", "C"), encoder_path=str(ckpt)
            )
            df = compare_group2(scene, l2b, r, None, max_pixels=20000)
            row[method] = {
                "group_top1": float(df["top1_agree"].mean()),
                "class_top1": float(df["top1_class_agree"].mean()),
                "group_top5": float(df["topk_agree"].mean()),
                "by_group": df.groupby("l2b_group")["top1_agree"].mean().round(3).to_dict(),
            }
        val = (
            json.loads(ckpt.with_suffix(".json").read_text())["best_val_mean"]
            if ckpt.with_suffix(".json").exists()
            else np.nan
        )
        row["val_synthetic_mean"] = val
        results[name] = row
        print(
            name,
            json.dumps(
                {
                    k: v
                    if not isinstance(v, dict)
                    else {kk: vv for kk, vv in v.items() if kk != "by_group"}
                    for k, v in row.items()
                }
            ),
            flush=True,
        )
    Path("outputs/phase5b").mkdir(parents=True, exist_ok=True)
    Path("outputs/phase5b/ablation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
