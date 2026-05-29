"""Compute per-layer energy of DeepMA-decomposed Monash datasets.

For every ``<out_root>/<dataset>/{seasonal_kXXX.npz, trend.npz, meta.json}``
this script:

1. Streams each layer's NPZ (one ndarray per series) and accumulates
   ``Σ_t x_t^2`` across all series, plus the total point count.
2. Reports per-layer total energy, mean energy density (energy / n_points),
   and energy ratio (per-dataset normalized so all layers sum to 1).
3. Picks the **top-3** layers per dataset (ranked by energy ratio) and writes
   them into ``<out_csv>``.

Default outputs:
    /root/DeepMA/layer_energy_top3.csv           # top-3 per dataset (wide)
    /root/DeepMA/layer_energy_full.csv           # long form, all layers

Usage:
    python compute_layer_energy.py
    python compute_layer_energy.py --out_root /root/autodl-tmp/deepma_decomp_pow2
"""
import argparse
import csv
import json
import os
from typing import List, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it=None, **kw):
        return it if it is not None else range(0)


def layer_label(fname: str) -> str:
    """seasonal_k033.npz -> 'seasonal_k033'  ; trend.npz -> 'trend'."""
    stem = os.path.splitext(fname)[0]
    return stem


def accumulate_energy(npz_path: str) -> Tuple[float, int]:
    """Return (sum_of_squares, n_points) over every array in the NPZ."""
    sum_sq = 0.0
    n_pts = 0
    with np.load(npz_path) as data:
        for key in data.files:
            arr = data[key]
            # use float64 accumulator to avoid float32 overflow on long series
            sum_sq += float(np.square(arr, dtype=np.float64).sum())
            n_pts += int(arr.size)
    return sum_sq, n_pts


def process_dataset(ds_dir: str) -> List[dict]:
    """Compute per-layer stats for one dataset directory."""
    meta_path = os.path.join(ds_dir, "meta.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    name = meta["dataset"]
    layer_files = meta["layer_files"]

    rows = []
    layer_iter = tqdm(layer_files, desc=name, ncols=80, leave=False)
    for fname in layer_iter:
        fp = os.path.join(ds_dir, fname)
        if not os.path.exists(fp):
            continue
        sum_sq, n_pts = accumulate_energy(fp)
        rows.append({
            "dataset": name,
            "layer": layer_label(fname),
            "energy": sum_sq,
            "n_points": n_pts,
            "energy_density": sum_sq / n_pts if n_pts else 0.0,
        })

    if not rows:
        return rows
    total = sum(r["energy"] for r in rows) or 1.0
    for r in rows:
        r["energy_ratio"] = r["energy"] / total
    return rows


def write_full_csv(all_rows: List[dict], path: str) -> None:
    fieldnames = ["dataset", "layer", "energy", "energy_ratio",
                  "energy_density", "n_points"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})


def write_top3_csv(all_rows: List[dict], path: str) -> None:
    by_ds = {}
    for r in all_rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    fieldnames = [
        "dataset",
        "top1_layer", "top1_ratio",
        "top2_layer", "top2_ratio",
        "top3_layer", "top3_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ds in sorted(by_ds):
            rows = sorted(by_ds[ds], key=lambda r: r["energy_ratio"],
                          reverse=True)
            top = rows[:3]
            while len(top) < 3:
                top.append({"layer": "", "energy_ratio": 0.0})
            w.writerow({
                "dataset": ds,
                "top1_layer": top[0]["layer"],
                "top1_ratio": f"{top[0]['energy_ratio']:.6f}",
                "top2_layer": top[1]["layer"],
                "top2_ratio": f"{top[1]['energy_ratio']:.6f}",
                "top3_layer": top[2]["layer"],
                "top3_ratio": f"{top[2]['energy_ratio']:.6f}",
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/root/autodl-tmp/deepma_decomp_pow2",
                    help="Directory containing per-dataset sub-folders "
                         "produced by decompose_monash.py.")
    ap.add_argument("--top3_csv", default="/root/DeepMA/layer_energy_top3.csv")
    ap.add_argument("--full_csv", default="/root/DeepMA/layer_energy_full.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.out_root):
        raise SystemExit(f"out_root not found: {args.out_root}")

    sub = sorted(d for d in os.listdir(args.out_root)
                 if os.path.isdir(os.path.join(args.out_root, d)))
    print(f"Scanning {len(sub)} dataset sub-dirs under {args.out_root}")

    all_rows: List[dict] = []
    for d in tqdm(sub, desc="datasets", ncols=80):
        ds_dir = os.path.join(args.out_root, d)
        rows = process_dataset(ds_dir)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows produced -- meta.json missing in every dir?")

    write_full_csv(all_rows, args.full_csv)
    write_top3_csv(all_rows, args.top3_csv)
    print(f"  wrote per-layer rows : {args.full_csv} ({len(all_rows)} rows)")
    print(f"  wrote top-3 per ds   : {args.top3_csv}")


if __name__ == "__main__":
    main()
