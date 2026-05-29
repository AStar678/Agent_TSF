"""Train a Linear model via closed-form OLS/ridge, then test on benchmark CSVs.

Supported models
----------------
* ``linear``   single ridge linear map  (X -> Y, no RevIN at test)
* ``rlinear``  per-window RevIN + ridge + de-RevIN at test

Supported training sources
--------------------------
* ``csv``     the TRAIN segment of ``--root_path/--data_path`` (Time-MoE StandardScaler)
* ``monash``  a subset of Monash TSF files (per-window RevIN z-score)

Evaluation always computes MSE/MAE in the StandardScaler domain of
the test CSV (comparable across models / train sources).

Examples
--------
    # Monash pretrain -> ETTh1 test, RLinear
    python run_ols.py --train_source monash --monash_preset etth_like \
        --root_path /root/dataset/ETT-small/ --data_path ETTh1.csv \
        --model rlinear --seq_len 96 --pred_len 96

    # Self train-test on ETTh1 with Linear
    python run_ols.py --train_source csv \
        --root_path /root/dataset/ETT-small/ --data_path ETTh1.csv \
        --model linear --seq_len 96 --pred_len 96
"""
import argparse
import csv
import datetime as _dt
import os
import time

import numpy as np

from monash_utils import (
    build_monash_windows,
    build_monash_windows_per_dataset,
    load_benchmark_split,
    fit_ridge,
    revin,
    PRESET_FILESETS,
    MONASH_DIR,
)


# ---------------------------------------------------------------- model fits
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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Train Linear (OLS/ridge, closed-form) on Monash or CSV, "
                    "test on benchmark CSV."
    )
    # ---- model ------------------------------------------------------------
    ap.add_argument("--model", choices=["linear", "rlinear"],
                    default="rlinear")
    # ---- source selection -------------------------------------------------
    ap.add_argument("--train_source", choices=["csv", "monash"], default="monash")
    # ---- data path (test target, same as TSLib convention) -----------------
    ap.add_argument("--root_path", type=str, default="./dataset/ETT-small/",
                    help="root path of the data file")
    ap.add_argument("--data_path", type=str, default="ETTh1.csv",
                    help="data file name")
    # ---- monash route -----------------------------------------------------
    ap.add_argument("--monash_dir", type=str, default=MONASH_DIR)
    ap.add_argument("--monash_preset", type=str, default=None,
                    choices=sorted(PRESET_FILESETS.keys()))
    ap.add_argument("--monash_files", type=str, nargs="+", default=None)
    ap.add_argument("--monash_stride", type=int, default=None)
    ap.add_argument("--monash_max_per_series", type=int, default=200)
    ap.add_argument("--monash_per_dataset", type=int, default=0,
                    help="If > 0, switch to per-dataset random sampling: "
                         "draw up to this many RevIN-filtered windows from "
                         "EACH .tsf file (overrides --monash_max_per_series).")
    # ---- shared -----------------------------------------------------------
    ap.add_argument("--seq_len", type=int, default=96,
                    help="input sequence length")
    ap.add_argument("--pred_len", type=int, default=96,
                    help="prediction sequence length")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # ---- results log ------------------------------------------------------
    ap.add_argument("--results_csv", type=str, default="results_ols.csv",
                    help="Append run config + metrics to this CSV. "
                         "Set empty string to disable.")
    args = ap.parse_args()

    # Build full CSV path from root_path + data_path
    test_csv = os.path.join(args.root_path, args.data_path)

    need_revin = (args.train_source == "monash") or (args.model == "rlinear")

    # ---------------- training windows ------------------------------------
    t0 = time.time()
    if args.train_source == "csv":
        train_name = os.path.basename(args.data_path)
        X_tr, Y_tr = load_benchmark_split(
            test_csv, args.seq_len, args.pred_len, split="train"
        )
        if args.model == "rlinear":
            X_tr, Y_tr, _, _ = revin(X_tr, Y_tr)
        print(f"Train src : csv={test_csv}")
    else:
        files = args.monash_files
        if files is None:
            files = PRESET_FILESETS[args.monash_preset or "etth_like"]
        train_name = f"monash[{len(files)}]"
        if args.monash_per_dataset and args.monash_per_dataset > 0:
            X_tr, Y_tr = build_monash_windows_per_dataset(
                data_dir=args.monash_dir,
                files=files,
                L_in=args.seq_len,
                L_out=args.pred_len,
                stride=args.monash_stride,
                max_windows_per_dataset=args.monash_per_dataset,
                seed=args.seed,
                verbose=True,
            )
            print(f"Train src : monash dir={args.monash_dir} "
                  f"(per-dataset sample = {args.monash_per_dataset})")
        else:
            X_tr, Y_tr = build_monash_windows(
                data_dir=args.monash_dir,
                files=files,
                L_in=args.seq_len,
                L_out=args.pred_len,
                stride=args.monash_stride,
                max_windows_per_series=args.monash_max_per_series,
                seed=args.seed,
                verbose=True,
            )
            print(f"Train src : monash dir={args.monash_dir}")
        print(f"Train src : monash files={files}")

    test_name = os.path.basename(test_csv)
    print(f"Test  CSV : {test_csv}")
    print(f"Model     : {args.model}   need_revin_at_test={need_revin}")
    print(f"seq_len={args.seq_len}  pred_len={args.pred_len}  ridge={args.ridge}")
    print(f"  train windows = {X_tr.shape[0]}  ({time.time()-t0:.1f}s)")

    if args.max_train is not None and X_tr.shape[0] > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X_tr.shape[0], args.max_train, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
        print(f"  sub-sampled -> {X_tr.shape[0]}")

    # ---------------- fit --------------------------------------------------
    t0 = time.time()
    predict_fn = fit_linear(X_tr, Y_tr, args.ridge)
    print(f"  fit {time.time()-t0:.1f}s")

    # ---------------- test windows & evaluate -----------------------------
    t0 = time.time()
    X_te, Y_te = load_benchmark_split(
        test_csv, args.seq_len, args.pred_len, split="test"
    )
    print(f"  test  windows = {X_te.shape[0]}  ({time.time()-t0:.1f}s)")

    mse, mae, pred = evaluate(predict_fn, X_te, Y_te, apply_revin=need_revin)

    print("\n===== RESULT =====")
    print(f"  model={args.model}  train={train_name}  test={test_name}  "
          f"seq_len={args.seq_len}  pred_len={args.pred_len}")
    print(f"  MSE = {mse:.4f}")
    print(f"  MAE = {mae:.4f}")

    # ---------------- log to results.csv ----------------------------------
    if args.results_csv:
        _append_results_row(args.results_csv, {
            "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "train_source": args.train_source,
            "train": (test_csv if args.train_source == "csv"
                      else f"monash[{len(files)}]"),
            "test": test_csv,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "ridge": args.ridge,
            "max_train": args.max_train if args.max_train is not None else "",
            "seed": args.seed,
            "train_windows": int(X_tr.shape[0]),
            "test_windows": int(X_te.shape[0]),
            "MSE": f"{mse:.6f}",
            "MAE": f"{mae:.6f}",
        })
        print(f"  log appended -> {args.results_csv}")


if __name__ == "__main__":
    main()
