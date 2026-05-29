"""Decompose every Monash .tsf into multi-scale MA components and save to disk.

Each TSF file is treated as a list of variable-length univariate series.
For each series we run :class:`DeepMA` with kernels
``[5, 13, 25, 49, 97, 125, 160, 200, 300]`` (9 seasonals + 1 trend = 10 layers)
and dump the layer-wise outputs to:

    <out_dir>/<dataset_stem>/
        seasonal_k005.npz   # contains s000000, s000001, ... (one ndarray per series)
        seasonal_k013.npz
        ...
        seasonal_k300.npz
        trend.npz
        meta.json

NPZ keys are zero-padded series indices (``s000123``); each value is a
1-D ``float32`` array sharing the original series length.

Usage
-----
    # All .tsf files under default dir, save to /root/autodl-tmp/deepma_decomp/
    python decompose_monash.py

    # Specific subset (use full Monash file names)
    python decompose_monash.py --monash_files \\
        australian_electricity_demand_dataset.tsf \\
        oikolab_weather_dataset.tsf

    # Custom output dir or kernels
    python decompose_monash.py --out_dir /tmp/decomp --kernels 5 13 25 49 97
"""
import argparse
import json
import os
import time
import types

# cgroup-aware thread setup (must run BEFORE numpy/torch import) ------------
def _detect_cpus() -> int:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
            if q != "max":
                n = int(int(q) / int(p))
                if n >= 1:
                    return n
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        if q > 0 and p > 0 and q // p >= 1:
            return q // p
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

_n_cpu = _detect_cpus()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_n_cpu))

import numpy as np
import torch
torch.set_num_threads(_n_cpu)

try:
    from tqdm.auto import tqdm
except ImportError:                          # graceful fallback
    def tqdm(it=None, **kw):
        return it if it is not None else range(0)

from deepma import DeepMA, parse_tsf
from deepma.config import MONASH_DIR


DEFAULT_KERNELS = [5, 13, 25, 49, 97, 125, 161, 201, 301]
DEFAULT_OUT_DIR = "/root/autodl-tmp/deepma_decomp"


def _sanitize_kernels(kernels):
    """DeepMA's symmetric padding requires odd kernels.

    Round each even kernel up to the next odd integer; warn on change.
    """
    out, changed = [], []
    for k in kernels:
        k = int(k)
        if k < 1:
            raise ValueError(f"kernel must be >= 1, got {k}")
        if k % 2 == 0:
            changed.append((k, k + 1))
            k += 1
        out.append(k)
    if changed:
        print("  [warn] rounded even kernels up to odd: "
              + ", ".join(f"{a}->{b}" for a, b in changed))
    return out


def decompose_one(model: DeepMA, series: np.ndarray):
    """Run DeepMA on a single 1-D series.

    Returns (seasonals, trend) where ``seasonals`` is a list of K float32
    arrays of shape (T,), and ``trend`` is float32 of shape (T,).
    """
    x = torch.from_numpy(series).float().view(1, -1, 1)        # (1, T, 1)
    with torch.no_grad():
        seasonals, trend = model(x)
    seasonals_np = [s.squeeze(0).squeeze(-1).numpy().astype(np.float32)
                    for s in seasonals]
    trend_np = trend.squeeze(0).squeeze(-1).numpy().astype(np.float32)
    return seasonals_np, trend_np


def process_tsf(tsf_path: str, out_root: str, kernels, window_kind: str = "rect"):
    name = os.path.splitext(os.path.basename(tsf_path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    series_list = parse_tsf(tsf_path)
    n_total = len(series_list)
    if n_total == 0:
        print(f"  [skip] {name}: 0 valid series in TSF")
        return

    cfg = types.SimpleNamespace(kernel_list=kernels, window_kind=window_kind)
    model = DeepMA(cfg).eval()

    K = len(kernels)
    # One dict per layer (K seasonals + 1 trend); keys = zero-padded series ids.
    layer_buffers = [dict() for _ in range(K + 1)]

    total_pts = 0
    n_saved = 0
    bar = tqdm(series_list, desc=name, unit="series",
               total=n_total, ncols=80, leave=False)
    for i, s in enumerate(bar):
        if s.size < 2:
            continue                                     # too short
        s = s.astype(np.float32, copy=False)
        seasonals_np, trend_np = decompose_one(model, s)
        key = f"s{i:06d}"
        for k in range(K):
            layer_buffers[k][key] = seasonals_np[k]
        layer_buffers[K][key] = trend_np
        total_pts += s.size
        n_saved += 1

    if n_saved == 0:
        print(f"  [skip] {name}: every series shorter than 2 points")
        return

    layer_files = []
    for k, kernel in enumerate(kernels):
        fname = f"seasonal_k{kernel:03d}.npz"
        np.savez_compressed(os.path.join(out_dir, fname), **layer_buffers[k])
        layer_files.append(fname)
    np.savez_compressed(os.path.join(out_dir, "trend.npz"), **layer_buffers[K])
    layer_files.append("trend.npz")

    meta = {
        "dataset": name,
        "tsf_path": tsf_path,
        "kernels": list(map(int, kernels)),
        "window_kind": window_kind,
        "n_layers": K + 1,
        "n_series_total": int(n_total),
        "n_series_saved": int(n_saved),
        "total_points_saved": int(total_pts),
        "dtype": "float32",
        "layer_files": layer_files,
        "key_format": "s{idx:06d}",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  [ok]   {name}: {n_saved}/{n_total} series, "
          f"{total_pts:,} pts, {K + 1} layers -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monash_dir", default=MONASH_DIR)
    ap.add_argument("--monash_files", nargs="+", default=None,
                    help="Subset of .tsf names to process; default = every "
                         ".tsf in --monash_dir.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--kernels", type=int, nargs="+", default=DEFAULT_KERNELS)
    ap.add_argument("--window_kind", default="rect",
                    choices=["rect", "hann", "hamming", "gauss", "tri"],
                    help="MA weighting; default 'rect' reproduces AvgPool. "
                         "Use 'hann' for low side-lobes / cleaner separation.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-decompose datasets even when meta.json exists.")
    args = ap.parse_args()

    args.kernels = _sanitize_kernels(args.kernels)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.monash_files:
        files = list(args.monash_files)
    else:
        files = sorted(f for f in os.listdir(args.monash_dir)
                       if f.endswith(".tsf"))

    print("=" * 64)
    print("DeepMA decomposition")
    print(f"  monash_dir  = {args.monash_dir}")
    print(f"  out_dir     = {args.out_dir}")
    print(f"  window_kind = {args.window_kind}")
    print(f"  kernels     = {args.kernels}  "
          f"({len(args.kernels)} seasonals + 1 trend)")
    print(f"  files       = {len(files)} total")
    print("=" * 64)

    t_start = time.time()
    for fn in files:
        fp = os.path.join(args.monash_dir, fn)
        if not os.path.exists(fp):
            print(f"  [skip] {fn} not found")
            continue
        stem = os.path.splitext(fn)[0]
        meta_path = os.path.join(args.out_dir, stem, "meta.json")
        if (not args.overwrite) and os.path.exists(meta_path):
            print(f"  [done] {stem}: meta.json exists (use --overwrite to redo)")
            continue
        t0 = time.time()
        try:
            process_tsf(fp, args.out_dir, args.kernels, window_kind=args.window_kind)
        except Exception as e:
            print(f"  [err]  {fn}: {type(e).__name__}: {e}")
            continue
        print(f"  [time] {fn}: {time.time() - t0:.1f}s")

    print("=" * 64)
    print(f"all done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
