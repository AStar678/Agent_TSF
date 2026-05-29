"""Train per-layer polynomial ridge on Monash decomposition (top-3 layers / dataset).

Pipeline
--------
1. Read ``layer_energy_top3.csv`` to know each dataset's top-3 layers.
2. For each dataset, walk through its original Monash .tsf, slide windows of
   length ``L_in + L_out``; per window compute mu / sd from the L_in context
   (per-window RevIN), then carve the *same window* out of the dataset's top-3
   layer NPZ files (already decomposed by DeepMA), apply the same RevIN
   normalization (trend layer subtracts mu, seasonals only divide by sd),
   and accumulate them into per-layer (X, Y) buffers.
3. Train one ridge regression per layer with a polynomial feature expansion
   ``[x, x**2, ..., x**p]`` (element-wise; ``p = --order``). Layers that no
   dataset selected as top-3 are simply skipped (no contribution at test time).
4. Test on ``--test_csv`` using the SAME inference procedure as benchmark.py:
   per-window RevIN -> DeepMA decomposition into K+1 layers -> per-layer
   polynomial expansion -> sum -> de-RevIN -> MSE / MAE.

Example
-------
    python train_top3_layers.py \
        --decomp_dir /root/autodl-tmp/deepma_decomp_pow2 \
        --top3_csv   /root/DeepMA/layer_energy_top3.csv \
        --test_csv   /root/dataset/ETT-small/ETTh1.csv \
        --L_in 96 --L_out 96 --order 2 --ridge 1e-4
"""
import argparse
import csv
import datetime as _dt
import json
import os
import time
import types

# --- multi-core CPU setup (mirror of benchmark.py) ------------------------
def _detect_cpus() -> int:
    try:
        with open("/sys/fs/cgroup/cpu.max") as _f:
            q, p = _f.read().split()
            if q != "max":
                n = int(int(q) / int(p))
                if n >= 1:
                    return n
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as _f:
            q = int(_f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as _f:
            p = int(_f.read())
        if q > 0 and p > 0:
            n = q // p
            if n >= 1:
                return n
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
import pandas as pd
import torch

torch.set_num_threads(_n_cpu)
try:
    torch.set_num_interop_threads(max(1, _n_cpu // 2))
except RuntimeError:
    pass

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it=None, **kw):
        return it if it is not None else range(0)

from deepma import (
    DeepMA, decompose_batched, fit_ridge,
    load_benchmark_split, revin,
)
from deepma.data import parse_tsf
from deepma.config import DEFAULT_RIDGE


# ---------------------------------------------------------------- helpers
def parse_top3(csv_path: str) -> dict:
    """Return {dataset_name: [top1_layer, top2_layer, top3_layer]}."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        layers = []
        for col in ("top1_layer", "top2_layer", "top3_layer"):
            v = row.get(col, "")
            if isinstance(v, str) and v:
                layers.append(v)
        out[str(row["dataset"])] = layers
    return out


def poly_expand(X: np.ndarray, p: int) -> np.ndarray:
    """X: (N, D) -> (N, D*p) with element-wise [X, X**2, ..., X**p]."""
    if p <= 1:
        return X.astype(np.float32, copy=False)
    parts = [X]
    cur = X
    for _ in range(2, p + 1):
        cur = cur * X
        parts.append(cur)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------- collect
def collect_layer_windows(
    decomp_dir, top3_map, L_in, L_out, stride,
    max_per_series, outlier_threshold, min_std,
    sample_frac, seed,
):
    """Return {layer_name: (X[N, L_in], Y[N, L_out])} in RevIN domain."""
    rng = np.random.default_rng(seed)
    W = L_in + L_out
    layer_X, layer_Y = {}, {}

    pbar = tqdm(sorted(top3_map.keys()), desc="datasets", ncols=80)
    for ds in pbar:
        ds_dir = os.path.join(decomp_dir, ds)
        meta_path = os.path.join(ds_dir, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        keep_layers = [L for L in top3_map[ds]
                       if os.path.exists(os.path.join(ds_dir, L + ".npz"))]
        if not keep_layers:
            continue

        tsf_path = meta.get("tsf_path", "")
        if not tsf_path or not os.path.exists(tsf_path):
            continue
        try:
            series_list = parse_tsf(tsf_path)
        except Exception as e:
            print(f"[warn] parse_tsf({ds}) failed: {e}")
            continue

        npzs = {L: np.load(os.path.join(ds_dir, L + ".npz"))
                for L in keep_layers}
        kept = 0
        try:
            for i, s in enumerate(series_list):
                s = np.asarray(s, dtype=np.float32)
                if s.size < W + 1:
                    continue
                key = f"s{i:06d}"
                comps = {}
                for L, npz in npzs.items():
                    if key in npz.files:
                        comps[L] = npz[key]
                if not comps:
                    continue

                starts = np.arange(0, s.size - W, stride, dtype=np.int64)
                if starts.size == 0:
                    continue
                if max_per_series and starts.size > max_per_series:
                    starts = rng.choice(starts, size=max_per_series, replace=False)
                if sample_frac < 1.0:
                    keep_n = max(1, int(round(starts.size * sample_frac)))
                    if keep_n < starts.size:
                        starts = rng.choice(starts, size=keep_n, replace=False)

                for st in starts:
                    x_raw = s[st:st + L_in]
                    mu = float(x_raw.mean())
                    sd = float(x_raw.std())
                    if sd < min_std:
                        continue
                    y_raw = s[st + L_in:st + W]
                    if np.abs((y_raw - mu) / sd).max() > outlier_threshold:
                        continue
                    inv_sd = 1.0 / sd
                    for L, comp in comps.items():
                        x_layer = comp[st:st + L_in]
                        y_layer = comp[st + L_in:st + W]
                        if L == "trend":
                            x_n = (x_layer - mu) * inv_sd
                            y_n = (y_layer - mu) * inv_sd
                        else:
                            x_n = x_layer * inv_sd
                            y_n = y_layer * inv_sd
                        layer_X.setdefault(L, []).append(x_n.astype(np.float32))
                        layer_Y.setdefault(L, []).append(y_n.astype(np.float32))
                    kept += 1
        finally:
            for npz in npzs.values():
                npz.close()
        pbar.set_postfix(ds=ds[:20], wins=kept)

    out = {}
    for L in layer_X:
        X = np.stack(layer_X[L]).astype(np.float32)
        Y = np.stack(layer_Y[L]).astype(np.float32)
        out[L] = (X, Y)
    return out


# ---------------------------------------------------------------- fit / predict
def fit_layer_polys(layer_data, order, ridge):
    Ws, bs = {}, {}
    for L, (X, Y) in tqdm(sorted(layer_data.items()),
                          desc="fit ridge", ncols=80, leave=False):
        Xp = poly_expand(X, order)
        W, b = fit_ridge(Xp, Y, lam=ridge)
        Ws[L] = W.astype(np.float32, copy=False)
        bs[L] = b.astype(np.float32, copy=False)
    return Ws, bs


def predict_top3(model, kernels, Ws, bs, order, X_rev, L_out):
    """X_rev: per-window RevIN'd test inputs (N, L_in). Returns (N, L_out)."""
    Xd = decompose_batched(model, X_rev, desc="test  decomp")
    layer_names = [f"seasonal_k{k:03d}" for k in kernels] + ["trend"]
    pred = np.zeros((X_rev.shape[0], L_out), dtype=np.float32)
    used = []
    for name, layer_x in zip(layer_names, Xd):
        if name not in Ws:
            continue
        Xp = poly_expand(layer_x.astype(np.float32, copy=False), order)
        pred += Xp @ Ws[name] + bs[name]
        used.append(name)
    return pred, used


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp_dir", default="/root/autodl-tmp/deepma_decomp_pow2")
    ap.add_argument("--top3_csv", default="/root/DeepMA/layer_energy_top3.csv")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--L_in", type=int, default=96)
    ap.add_argument("--L_out", type=int, default=96)
    ap.add_argument("--order", type=int, default=1,
                    help="Polynomial order p; features = [x, x^2, ..., x^p].")
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max_per_series", type=int, default=200)
    ap.add_argument("--outlier_threshold", type=float, default=10.0)
    ap.add_argument("--min_std", type=float, default=1e-6)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_csv", default="/root/DeepMA/results_top3.csv",
                    help="Append run config + metrics. Empty string disables.")
    args = ap.parse_args()

    if args.stride is None:
        args.stride = max(1, args.L_out // 2)
    if args.order < 1:
        ap.error("--order must be >= 1")
    if not (0.0 < args.frac <= 1.0):
        ap.error(f"--frac must be in (0, 1], got {args.frac}")

    top3_map = parse_top3(args.top3_csv)
    print(f"top3 datasets    : {len(top3_map)}")

    # infer kernels from any meta.json under decomp_dir
    kernels = None
    for d in sorted(os.listdir(args.decomp_dir)):
        mp = os.path.join(args.decomp_dir, d, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                kernels = json.load(f).get("kernels")
            if kernels:
                break
    if not kernels:
        ap.error(f"no kernels found in any meta.json under {args.decomp_dir}")
    print(f"kernels          : {kernels}")
    print(f"L_in={args.L_in}  L_out={args.L_out}  order={args.order}  ridge={args.ridge}")

    # -------- collect per-layer training windows
    t0 = time.time()
    layer_data = collect_layer_windows(
        args.decomp_dir, top3_map, args.L_in, args.L_out,
        args.stride, args.max_per_series, args.outlier_threshold,
        args.min_std, args.frac, args.seed,
    )
    elapsed = time.time() - t0
    if not layer_data:
        raise RuntimeError("collected zero layer windows; check decomp_dir / "
                           "L_in vs series lengths")
    total_per_layer = {L: X.shape[0] for L, (X, Y) in layer_data.items()}
    print(f"\ncollected layers : {len(layer_data)}   "
          f"({sum(total_per_layer.values())} total layer-windows in {elapsed:.1f}s)")
    for L in sorted(layer_data.keys()):
        X, Y = layer_data[L]
        print(f"  {L:>16}: X={X.shape}  Y={Y.shape}")
    train_windows = max(total_per_layer.values())

    # -------- fit ridge per layer (polynomial features)
    t0 = time.time()
    Ws, bs = fit_layer_polys(layer_data, args.order, args.ridge)
    print(f"fit ridge        : {time.time()-t0:.1f}s   "
          f"(layers fitted: {sorted(Ws.keys())})")

    # free training buffers before test
    del layer_data

    # -------- test (per-window RevIN, same inference protocol)
    t0 = time.time()
    X_te, Y_te = load_benchmark_split(
        args.test_csv, args.L_in, args.L_out, split="test"
    )
    print(f"test windows     : {X_te.shape[0]}  ({time.time()-t0:.1f}s)")

    cfg = types.SimpleNamespace(kernel_list=kernels)
    model = DeepMA(cfg).eval()

    Xr, _, mu_te, sd_te = revin(X_te)
    pred_r, used = predict_top3(model, kernels, Ws, bs, args.order, Xr, args.L_out)
    pred = pred_r * sd_te + mu_te
    mse = float(np.mean((pred - Y_te) ** 2))
    mae = float(np.mean(np.abs(pred - Y_te)))

    print("\n===== RESULT =====")
    print(f"  test={os.path.basename(args.test_csv)}  "
          f"L_in={args.L_in}  L_out={args.L_out}  order={args.order}")
    print(f"  layers used at test : {used}")
    print(f"  MSE = {mse:.4f}")
    print(f"  MAE = {mae:.4f}")

    # -------- log
    if args.results_csv:
        parent = os.path.dirname(args.results_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
        write_header = (not os.path.exists(args.results_csv)
                        or os.path.getsize(args.results_csv) == 0)
        row = {
            "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "deepma_top3_poly",
            "train_source": "monash_decomp_top3",
            "train": f"top3[{len(top3_map)}]",
            "test": args.test_csv,
            "L_in": args.L_in,
            "L_out": args.L_out,
            "order": args.order,
            "ridge": args.ridge,
            "kernels": " ".join(map(str, kernels)),
            "frac": args.frac,
            "seed": args.seed,
            "layers_used": " ".join(used),
            "train_windows": int(train_windows),
            "test_windows": int(X_te.shape[0]),
            "MSE": f"{mse:.6f}",
            "MAE": f"{mae:.6f}",
        }
        with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"  log appended -> {args.results_csv}")


if __name__ == "__main__":
    main()
