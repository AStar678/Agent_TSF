"""In-domain / cross-domain evaluation within ETTh1 and ETTh2.

For each source in {ETTh1, ETTh2}:
  1. Build sliding (L_in -> L_out) windows on its TRAIN segment (its own StandardScaler).
  2. Apply per-window RevIN; decompose via DeepMA; fit per-layer ridge LS.
  3. Evaluate on {ETTh1, ETTh2} TEST segments with the same pipeline.

Metrics are reported in each dataset's own StandardScaler domain.
"""
import argparse
import os
import time
import types
import numpy as np

from deepma import (
    DeepMA,
    load_etth_train_windows, load_etth_test,
    decompose_batched, fit_ridge,
    revin, predict, save_model,
)
from deepma.config import (
    ETTH1_CSV, ETTH2_CSV, MODEL_DIR,
    DEFAULT_KERNELS, DEFAULT_L_IN, DEFAULT_L_OUT,
    DEFAULT_RIDGE, DEFAULT_OUTLIER_THR,
)


def fit_on_etth(csv_path, kernels, L_in, L_out, ridge, stride, outlier_thr):
    X, Y, _, _ = load_etth_train_windows(csv_path, L_in, L_out, stride=stride)
    X_rev, Y_rev, _, _ = revin(X, Y)
    keep = np.max(np.abs(Y_rev), axis=1) <= outlier_thr
    X_rev, Y_rev = X_rev[keep], Y_rev[keep]
    print(f"  train windows: {X.shape[0]} -> after RevIN+filter: {X_rev.shape[0]}")

    cfg = types.SimpleNamespace(kernel_list=kernels)
    model = DeepMA(cfg).eval()
    Xd = decompose_batched(model, X_rev)
    Yd = decompose_batched(model, Y_rev)

    Ws, bs = [], []
    total = np.zeros_like(Y_rev)
    for Xi, Yi in zip(Xd, Yd):
        W, b = fit_ridge(Xi, Yi, lam=ridge)
        Ws.append(W); bs.append(b)
        total += Xi @ W + b
    train_mse = float(np.mean((total - Y_rev) ** 2))
    train_mae = float(np.mean(np.abs(total - Y_rev)))
    print(f"  train (RevIN domain): MSE={train_mse:.4f}  MAE={train_mae:.4f}")
    return model, Ws, bs


def test_on_etth(csv_path, model, Ws, bs, L_in, L_out):
    X, Y, _, _ = load_etth_test(csv_path, L_in, L_out)
    X_rev, _, mu, sd = revin(X)
    Y_hat = predict(model, Ws, bs, X_rev) * sd + mu
    mse = float(np.mean((Y_hat - Y) ** 2))
    mae = float(np.mean(np.abs(Y_hat - Y)))
    return mse, mae, X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etth1", type=str, default=ETTH1_CSV)
    ap.add_argument("--etth2", type=str, default=ETTH2_CSV)
    ap.add_argument("--L_in", type=int, default=DEFAULT_L_IN)
    ap.add_argument("--L_out", type=int, default=DEFAULT_L_OUT)
    ap.add_argument("--kernels", type=int, nargs="+", default=DEFAULT_KERNELS)
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--outlier_thr", type=float, default=DEFAULT_OUTLIER_THR)
    ap.add_argument("--save_dir", type=str, default=MODEL_DIR)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    sources = [("ETTh1", args.etth1), ("ETTh2", args.etth2)]
    targets = sources
    results = {}
    for src_name, src_path in sources:
        print(f"\n========== Train on {src_name} ==========")
        t0 = time.time()
        model, Ws, bs = fit_on_etth(src_path, args.kernels, args.L_in, args.L_out,
                                     args.ridge, args.stride, args.outlier_thr)
        print(f"  fit time: {time.time()-t0:.1f}s")
        save_path = os.path.join(args.save_dir, f"deepma_linear_{src_name.lower()}.npz")
        save_model(save_path, args.kernels, args.L_in, args.L_out, Ws, bs)
        print(f"  saved weights -> {save_path}")

        results[src_name] = {}
        for tgt_name, tgt_path in targets:
            mse, mae, n = test_on_etth(tgt_path, model, Ws, bs, args.L_in, args.L_out)
            results[src_name][tgt_name] = (mse, mae, n)
            tag = "in-domain" if tgt_name == src_name else "cross-domain"
            print(f"  -> eval on {tgt_name:<6}({tag:<12}): n={n:<7} MSE={mse:.4f}  MAE={mae:.4f}")

    print(f"\n========== Summary (Time-MoE domain, {args.L_in} -> {args.L_out}) ==========")
    print(f"  {'Train\\Test':<14}" + "".join(f"{t:<22}" for t, _ in targets))
    print("  " + "-" * (14 + 22 * len(targets)))
    for src_name, _ in sources:
        row = f"  {src_name:<14}"
        for tgt_name, _ in targets:
            mse, mae, _ = results[src_name][tgt_name]
            row += f"MSE={mse:.4f} MAE={mae:.4f}  "
        print(row)


if __name__ == "__main__":
    main()
