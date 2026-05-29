"""Train / test a chosen baseline on any benchmark CSV (Time-MoE protocol).

Supported models
----------------
* ``deepma``   multi-scale MA decomposition + per-layer ridge
* ``linear``   single ridge linear map  (X -> Y, no RevIN)
* ``rlinear``  per-window RevIN + single ridge + de-RevIN

Supported training sources
--------------------------
* ``csv``     the TRAIN segment of ``--train_csv`` (Time-MoE StandardScaler)
* ``monash``  a subset of Monash TSF files (per-window RevIN z-score)
* ``gift``    a subset of gift-eval HuggingFace datasets (per-window RevIN)

Evaluation always computes MSE/MAE in the StandardScaler domain of
``--test_csv`` (comparable across models / train sources).

Optionally dumps a PNG with N randomly chosen test samples (context + GT +
prediction) via ``--viz_samples N``.

Examples
--------
    # Self train-test on ETTh1 with DeepMA (auto-saves 5-sample viz)
    python benchmark.py --train_csv /root/dataset/ETT-small/ETTh1.csv \
        --model deepma --L_in 96 --L_out 96

    # Monash pretrain -> ETTh1 test, RLinear
    python benchmark.py --train_source monash --monash_preset etth_like \
        --test_csv /root/dataset/ETT-small/ETTh1.csv \
        --model rlinear --L_in 96 --L_out 96

    # Gift-eval pretrain -> ETTh1 test, DeepMA
    python benchmark.py --train_source gift --gift_preset mini \
        --test_csv /root/dataset/ETT-small/ETTh1.csv \
        --model deepma --L_in 96 --L_out 96
"""
import argparse
import csv
import datetime as _dt
import os
import time
import types

# --- multi-core CPU setup (must run BEFORE numpy/torch import) ------------
def _detect_cpus() -> int:
    """cgroup / sched-affinity aware CPU count (avoids over-subscription)."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as _f:
            q, p = _f.read().split()
            if q != "max":
                n = int(int(q) / int(p))
                if n >= 1:
                    return n
    except Exception:
        pass
    # cgroup v1
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
import torch

torch.set_num_threads(_n_cpu)
try:
    torch.set_num_interop_threads(max(1, _n_cpu // 2))
except RuntimeError:
    pass  # already set this process

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # graceful fallback

try:
    from tqdm.auto import tqdm
except ImportError:                          # graceful fallback
    def tqdm(it=None, **kw):
        return it if it is not None else range(0)

from deepma import (
    DeepMA,
    decompose_batched,
    fit_ridge,
    load_benchmark_split,
    build_monash_windows,
    PRESET_FILESETS,
    build_gift_windows,
    GIFT_PRESETS,
    list_gift_datasets,
    revin,
)
from deepma.config import DEFAULT_KERNELS, DEFAULT_RIDGE, MONASH_DIR, GIFT_DIR


# ---------------------------------------------------------------- model fits
def fit_deepma(X_tr, Y_tr, kernels, ridge):
    """Return a predict_fn(X) -> Y_hat closure."""
    cfg = types.SimpleNamespace(kernel_list=kernels)
    model = DeepMA(cfg)
    Xd = decompose_batched(model, X_tr, desc="train decomp X")
    Yd = decompose_batched(model, Y_tr, desc="train decomp Y")

    # Layer-wise ridge fits. Each fit_ridge already issues a multi-threaded
    # BLAS gemm, so a Python-level thread pool would just oversubscribe cores;
    # we keep the loop sequential and let BLAS use all threads per call.
    Ws, bs = [], []
    layer_iter = tqdm(list(zip(Xd, Yd)), desc="fit ridge",
                      ncols=80, leave=False)
    for Xi, Yi in layer_iter:
        W, b = fit_ridge(Xi, Yi, lam=ridge)
        Ws.append(W)
        bs.append(b)

    # 3) stack layer weights so a single batched matmul replaces K @-calls
    W_stack = np.stack(Ws, axis=0).astype(np.float32)   # (K, L_in, L_out)
    b_sum = np.stack(bs, axis=0).sum(axis=0).astype(np.float32)  # (L_out,)
    L_out = W_stack.shape[2]

    def predict_fn(X):
        Xd = decompose_batched(model, X, desc="test  decomp")
        Xd_stack = np.stack(Xd, axis=0)                  # (K, N, L_in)
        pred = np.einsum("knl,klo->no", Xd_stack, W_stack)
        pred += b_sum
        return pred.astype(np.float32, copy=False)
    return predict_fn


def fit_linear(X_tr, Y_tr, ridge):
    """Single ridge linear: X -> Y."""
    W, b = fit_ridge(X_tr, Y_tr, lam=ridge)

    def predict_fn(X):
        return (X @ W + b).astype(np.float32)
    return predict_fn


# ---------------------------------------------------------------- evaluation
def evaluate(predict_fn, X_te, Y_te, apply_revin):
    """Returns (mse, mae, pred). If apply_revin=True, the model lives in the
    per-window RevIN domain; we RevIN X_te, predict, de-RevIN before scoring.
    """
    if apply_revin:
        Xr, _, mu, sd = revin(X_te)
        pred_r = predict_fn(Xr)
        pred = pred_r * sd + mu
    else:
        pred = predict_fn(X_te)
    mse = float(np.mean((pred - Y_te) ** 2))
    mae = float(np.mean(np.abs(pred - Y_te)))
    return mse, mae, pred


# ---------------------------------------------------------------- results log
def _append_results_row(path: str, row: dict) -> None:
    """Append a single run record to a CSV; create with header if absent."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- visualize
def visualize_samples(X_te, Y_te, pred, n, out_path, seed, title):
    """Pick n random test windows and plot context + GT + prediction."""
    N = X_te.shape[0]
    n = min(n, N)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, size=n, replace=False))

    L_in = X_te.shape[1]
    L_out = Y_te.shape[1]
    t_ctx = np.arange(L_in)
    t_fut = np.arange(L_in, L_in + L_out)

    ncols = 2 if n >= 4 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.5 * ncols, 2.4 * nrows),
        squeeze=False,
    )
    for k, i in enumerate(idx):
        ax = axes[k // ncols][k % ncols]
        ax.plot(t_ctx, X_te[i], color="0.5", lw=0.8, label="context")
        ax.plot(t_fut, Y_te[i], color="C0", lw=1.3, label="GT")
        ax.plot(t_fut, pred[i], color="C3", lw=1.3, label="pred")
        ax.axvline(L_in - 0.5, color="k", lw=0.4, ls="--")
        sample_mse = float(np.mean((pred[i] - Y_te[i]) ** 2))
        ax.set_title(f"sample #{i}   MSE={sample_mse:.3f}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
    # hide unused cells
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    # ---- model ------------------------------------------------------------
    ap.add_argument("--model", choices=["deepma", "linear", "rlinear"],
                    default="deepma")
    # ---- source selection -------------------------------------------------
    ap.add_argument("--train_source", choices=["csv", "monash", "gift"],
                    default="csv")
    # ---- csv route --------------------------------------------------------
    ap.add_argument("--train_csv", type=str, default=None)
    # ---- monash route -----------------------------------------------------
    ap.add_argument("--monash_dir", type=str, default=MONASH_DIR)
    ap.add_argument("--monash_preset", type=str, default=None,
                    choices=sorted(PRESET_FILESETS.keys()))
    ap.add_argument("--monash_files", type=str, nargs="+", default=None)
    ap.add_argument("--monash_stride", type=int, default=None)
    ap.add_argument("--monash_max_per_series", type=int, default=200)
    # ---- gift-eval route --------------------------------------------------
    ap.add_argument("--gift_dir", type=str, default=GIFT_DIR,
                    help="Root dir containing HuggingFace gift-eval dataset subdirs.")
    ap.add_argument("--gift_preset", type=str, default=None,
                    choices=sorted(GIFT_PRESETS.keys()),
                    help="Named subset of gift-eval dataset dirs.")
    ap.add_argument("--gift_datasets", type=str, nargs="+", default=None,
                    help="Explicit gift-eval subdir names (overrides --gift_preset; "
                         "'all' means every subdir found under --gift_dir).")
    ap.add_argument("--gift_stride", type=int, default=None)
    ap.add_argument("--gift_max_per_series", type=int, default=200)
    # ---- shared -----------------------------------------------------------
    ap.add_argument("--test_csv", type=str, default=None)
    ap.add_argument("--L_in", type=int, default=96)
    ap.add_argument("--L_out", type=int, default=96)
    ap.add_argument("--kernels", type=int, nargs="+", default=DEFAULT_KERNELS)
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # ---- quick-test mode --------------------------------------------------
    ap.add_argument("--frac", type=float, default=1.0,
                    help="Fraction of training windows to keep (0, 1]. "
                         "csv: random sub-sample of (X_tr, Y_tr); "
                         "monash: per-series sub-sample so every .tsf still "
                         "contributes proportionally. Default 1.0 (all).")
    ap.add_argument("--quick", action="store_true",
                    help="Quick-test mode: shortcut for --frac 0.1.")
    # ---- visualization ----------------------------------------------------
    ap.add_argument("--viz_samples", type=int, default=5,
                    help="Randomly pick this many test windows and save a "
                         "PNG showing context + GT + prediction. "
                         "Default 5; set 0 to disable.")
    ap.add_argument("--viz_out", type=str, default=None,
                    help="Output PNG path. Default: "
                         "viz/<model>_<test>_L<in>_H<out>.png")
    ap.add_argument("--viz_seed", type=int, default=None,
                    help="Seed for sample selection (defaults to --seed).")
    # ---- results log ------------------------------------------------------
    ap.add_argument("--results_csv", type=str, default="results.csv",
                    help="Append run config + metrics to this CSV. "
                         "Set empty string to disable.")
    args = ap.parse_args()

    # --quick is a shortcut for --frac 0.1 (does not override an explicit --frac)
    if args.quick and args.frac == 1.0:
        args.frac = 0.1
    if not (0.0 < args.frac <= 1.0):
        ap.error(f"--frac must be in (0, 1], got {args.frac}")

    need_revin = (args.train_source in ("monash", "gift")) or (args.model == "rlinear")

    # ---------------- training windows ------------------------------------
    t0 = time.time()
    if args.train_source == "csv":
        if args.train_csv is None:
            ap.error("--train_csv is required when --train_source=csv")
        if args.test_csv is None:
            args.test_csv = args.train_csv
        train_name = os.path.basename(args.train_csv)
        X_tr, Y_tr = load_benchmark_split(
            args.train_csv, args.L_in, args.L_out, split="train"
        )
        if args.model == "rlinear":
            X_tr, Y_tr, _, _ = revin(X_tr, Y_tr)
        print(f"Train src : csv={args.train_csv}")
    elif args.train_source == "monash":
        if args.test_csv is None:
            ap.error("--test_csv is required when --train_source=monash")
        files = args.monash_files
        if files is None:
            files = PRESET_FILESETS[args.monash_preset or "etth_like"]
        train_name = f"monash[{len(files)}]"
        X_tr, Y_tr = build_monash_windows(
            data_dir=args.monash_dir,
            files=files,
            L_in=args.L_in,
            L_out=args.L_out,
            stride=args.monash_stride,
            max_windows_per_series=args.monash_max_per_series,
            seed=args.seed,
            verbose=True,
            sample_frac=args.frac,
        )
        print(f"Train src : monash dir={args.monash_dir}")
        print(f"Train src : monash files={files}")
    else:  # gift
        if args.test_csv is None:
            ap.error("--test_csv is required when --train_source=gift")
        gift_datasets = args.gift_datasets
        if gift_datasets is not None and len(gift_datasets) == 1 and \
                gift_datasets[0].lower() == "all":
            gift_datasets = list_gift_datasets(args.gift_dir)
        if gift_datasets is None:
            gift_datasets = GIFT_PRESETS[args.gift_preset or "mini"]
        train_name = f"gift[{len(gift_datasets)}]"
        X_tr, Y_tr = build_gift_windows(
            data_dir=args.gift_dir,
            datasets=gift_datasets,
            L_in=args.L_in,
            L_out=args.L_out,
            stride=args.gift_stride,
            max_windows_per_series=args.gift_max_per_series,
            seed=args.seed,
            verbose=True,
            sample_frac=args.frac,
        )
        print(f"Train src : gift dir={args.gift_dir}")
        print(f"Train src : gift datasets={gift_datasets}")

    test_name = os.path.basename(args.test_csv)
    print(f"Test  CSV : {args.test_csv}")
    print(f"Model     : {args.model}   need_revin_at_test={need_revin}")
    print(f"L_in={args.L_in}  L_out={args.L_out}  ridge={args.ridge}"
          + (f"  kernels={args.kernels}" if args.model == "deepma" else ""))
    print(f"  train windows = {X_tr.shape[0]}  ({time.time()-t0:.1f}s)"
          + (f"  [quick: frac={args.frac}]" if args.frac < 1.0 else ""))

    # csv route: random sub-sample of training windows after they are built.
    # (monash route already applied --frac per-series inside build_monash_windows.)
    if args.train_source == "csv" and args.frac < 1.0 and X_tr.shape[0] > 1:
        rng = np.random.default_rng(args.seed)
        keep = max(1, int(round(X_tr.shape[0] * args.frac)))
        idx = rng.choice(X_tr.shape[0], keep, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
        print(f"  csv sub-sampled (frac={args.frac}) -> {X_tr.shape[0]}")

    if args.max_train is not None and X_tr.shape[0] > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X_tr.shape[0], args.max_train, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
        print(f"  sub-sampled -> {X_tr.shape[0]}")

    # ---------------- fit --------------------------------------------------
    t0 = time.time()
    if args.model == "deepma":
        predict_fn = fit_deepma(X_tr, Y_tr, args.kernels, args.ridge)
    else:
        predict_fn = fit_linear(X_tr, Y_tr, args.ridge)
    print(f"  fit {time.time()-t0:.1f}s")

    # ---------------- test windows & evaluate -----------------------------
    t0 = time.time()
    X_te, Y_te = load_benchmark_split(
        args.test_csv, args.L_in, args.L_out, split="test"
    )
    print(f"  test  windows = {X_te.shape[0]}  ({time.time()-t0:.1f}s)")

    mse, mae, pred = evaluate(predict_fn, X_te, Y_te, apply_revin=need_revin)

    print("\n===== RESULT =====")
    print(f"  model={args.model}  train={train_name}  test={test_name}  "
          f"L_in={args.L_in}  L_out={args.L_out}")
    print(f"  MSE = {mse:.4f}")
    print(f"  MAE = {mae:.4f}")

    # ---------------- visualize -------------------------------------------
    viz_path = ""
    if args.viz_samples > 0:
        if args.viz_out is None:
            tag = os.path.splitext(test_name)[0]
            args.viz_out = (f"viz/{args.model}_{tag}_"
                            f"L{args.L_in}_H{args.L_out}.png")
        title = (f"{args.model.upper()}  train={train_name}  test={test_name}  "
                 f"L_in={args.L_in}  L_out={args.L_out}  "
                 f"MSE={mse:.3f}  MAE={mae:.3f}")
        viz_path = visualize_samples(
            X_te, Y_te, pred,
            n=args.viz_samples,
            out_path=args.viz_out,
            seed=(args.viz_seed if args.viz_seed is not None else args.seed),
            title=title,
        )
        print(f"  viz saved -> {viz_path}")

    # ---------------- log to results.csv ----------------------------------
    if args.results_csv:
        _append_results_row(args.results_csv, {
            "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "train_source": args.train_source,
            "train": (args.train_csv if args.train_source == "csv"
                      else train_name),
            "test": args.test_csv,
            "L_in": args.L_in,
            "L_out": args.L_out,
            "ridge": args.ridge,
            "kernels": (" ".join(map(str, args.kernels))
                        if args.model == "deepma" else ""),
            "max_train": args.max_train if args.max_train is not None else "",
            "frac": args.frac,
            "seed": args.seed,
            "train_windows": int(X_tr.shape[0]),
            "test_windows": int(X_te.shape[0]),
            "MSE": f"{mse:.6f}",
            "MAE": f"{mae:.6f}",
            "viz": viz_path,
        })
        print(f"  log appended -> {args.results_csv}")


if __name__ == "__main__":
    main()
