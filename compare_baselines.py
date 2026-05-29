"""Compare three Monash-trained baselines on multi-horizon transfer.

Models (all trained on Monash, tested per Time-MoE protocol):
  1. Linear        : per-series z-score train, predict in StandardScaler domain (no RevIN).
  2. RLinear       : per-window RevIN + single L_in->H linear + de-RevIN.
  3. DeepMA-Linear : per-window RevIN + DeepMA decomp + layer-wise ridge; sum + de-RevIN.
"""
import argparse
import os
import time
import types
import numpy as np
import torch

from deepma import (
    DeepMA,
    build_monash_windows, build_monash_windows_series_zscore,
    load_etth_test,
    decompose_batched, fit_ridge,
    PRESET_FILESETS,
)
from deepma.config import (
    MONASH_DIR, ETTH1_CSV, ETTH2_CSV, WEATHER_CSV,
    DEFAULT_RIDGE, DEFAULT_MAX_PER_SERIES, DEFAULT_OUTLIER_THR,
)


def predict_linear_plain(W, b, X):
    return X @ W + b


def predict_rlinear(W, b, X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Y_n = ((X - mu) / sd) @ W + b
    return Y_n * sd + mu


def predict_deepma(model, Ws, bs, X, batch_size=4096):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (X - mu) / sd
    N = Xn.shape[0]
    Lout = Ws[0].shape[1]
    Y_hat_n = np.zeros((N, Lout), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            xt = torch.from_numpy(Xn[s:e]).float().unsqueeze(-1)
            seasonals, trend = model(xt)
            comps = [c.squeeze(-1).numpy() for c in seasonals] + [trend.squeeze(-1).numpy()]
            y = np.zeros((e - s, Lout), dtype=np.float32)
            for W, b, C in zip(Ws, bs, comps):
                y += C @ W + b
            Y_hat_n[s:e] = y
    return Y_hat_n * sd + mu


def metrics(Y_hat, Y):
    return float(np.mean((Y_hat - Y) ** 2)), float(np.mean(np.abs(Y_hat - Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monash_dir", type=str, default=MONASH_DIR)
    ap.add_argument("--preset", type=str, default="default",
                    choices=list(PRESET_FILESETS.keys()),
                    help="which Monash file preset to use")
    ap.add_argument("--files", type=str, nargs="+", default=None,
                    help="explicit Monash .tsf file names (overrides --preset)")
    ap.add_argument("--L_in", type=int, default=96)
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--kernels", type=int, nargs="+", default=[5, 13, 25, 49, 97])
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--max_per_series", type=int, default=DEFAULT_MAX_PER_SERIES)
    ap.add_argument("--outlier_thr", type=float, default=DEFAULT_OUTLIER_THR)
    ap.add_argument("--targets", type=str, nargs="+",
                    default=[ETTH1_CSV, ETTH2_CSV])
    args = ap.parse_args()

    files = args.files if args.files else PRESET_FILESETS[args.preset]
    print(f"Using Monash files ({len(files)}):")
    for f in files:
        print(f"  - {f}")

    target_names = [os.path.basename(p).replace(".csv", "") for p in args.targets]
    model_names = ["Linear", "RLinear", "DeepMA-Linear"]
    results = {H: {m: {} for m in model_names} for H in args.horizons}

    for H in args.horizons:
        print(f"\n========== Horizon {args.L_in} -> {H} ==========")
        t0 = time.time()
        X_rev, Y_rev = build_monash_windows(
            data_dir=args.monash_dir, files=files, L_in=args.L_in, L_out=H,
            max_windows_per_series=args.max_per_series,
            outlier_threshold=args.outlier_thr, verbose=False,
        )
        X_sz, Y_sz = build_monash_windows_series_zscore(
            data_dir=args.monash_dir, files=files, L_in=args.L_in, L_out=H,
            max_windows_per_series=args.max_per_series, verbose=False,
        )
        print(f"  data: revin={X_rev.shape[0]}  series-z-score={X_sz.shape[0]}  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        W_lin, b_lin = fit_ridge(X_sz, Y_sz, lam=args.ridge)
        print(f"  Linear        fit {time.time()-t0:.1f}s")

        t0 = time.time()
        W_rl, b_rl = fit_ridge(X_rev, Y_rev, lam=args.ridge)
        print(f"  RLinear       fit {time.time()-t0:.1f}s")

        t0 = time.time()
        cfg = types.SimpleNamespace(kernel_list=args.kernels)
        model = DeepMA(cfg).eval()
        Xd = decompose_batched(model, X_rev)
        Yd = decompose_batched(model, Y_rev)
        Ws_d, bs_d = [], []
        for Xi, Yi in zip(Xd, Yd):
            W, b = fit_ridge(Xi, Yi, lam=args.ridge)
            Ws_d.append(W); bs_d.append(b)
        print(f"  DeepMA-Linear fit {time.time()-t0:.1f}s")

        for name, path in zip(target_names, args.targets):
            if not os.path.exists(path):
                continue
            X, Y, _, _ = load_etth_test(path, args.L_in, H)
            results[H]["Linear"][name]        = metrics(predict_linear_plain(W_lin, b_lin, X), Y)
            results[H]["RLinear"][name]       = metrics(predict_rlinear(W_rl, b_rl, X), Y)
            results[H]["DeepMA-Linear"][name] = metrics(predict_deepma(model, Ws_d, bs_d, X), Y)
            print(f"  -> {name:<8} (n={X.shape[0]}): "
                  f"Linear MSE={results[H]['Linear'][name][0]:.4f}  "
                  f"RLinear MSE={results[H]['RLinear'][name][0]:.4f}  "
                  f"DeepMA MSE={results[H]['DeepMA-Linear'][name][0]:.4f}")

    for mi, mname in [(0, "MSE"), (1, "MAE")]:
        print(f"\n========== {mname}  (Monash -> X, L_in={args.L_in}) ==========")
        print(f"  {'Horizon':<8}{'Model':<16}" + "".join(f"{n:<12}" for n in target_names))
        print("  " + "-" * (8 + 16 + 12 * len(target_names)))
        for H in args.horizons:
            for m in model_names:
                row = f"  {H:<8}{m:<16}"
                for n in target_names:
                    row += f"{results[H][m][n][mi]:<12.4f}" if n in results[H][m] else f"{'-':<12}"
                print(row)
            print("  " + "." * (8 + 16 + 12 * len(target_names)))


if __name__ == "__main__":
    main()
