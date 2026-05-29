"""Train DeepMA + layer-wise ridge linear on Monash.

1. Build Monash (X, Y) windows (L_in -> L_out) with per-window RevIN z-score.
2. Decompose X and Y via DeepMA (K seasonals + trend).
3. Fit a ridge-regularized linear model independently per layer (with bias).
4. Save all weights to a single .npz file.

Identity ``sum_i x_i == x`` makes the per-layer MSE objective equivalent to the
DeepMA-Indep objective in the project notes.
"""
import argparse
import os
import time
import types
import numpy as np

from deepma import (
    DeepMA,
    build_monash_windows,
    decompose_batched,
    fit_ridge,
    save_model,
    PRESET_FILESETS,
)
from deepma.config import (
    MONASH_DIR, MODEL_DIR,
    DEFAULT_KERNELS, DEFAULT_L_IN, DEFAULT_L_OUT,
    DEFAULT_RIDGE, DEFAULT_MAX_PER_SERIES, DEFAULT_OUTLIER_THR,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monash_dir", type=str, default=MONASH_DIR)
    ap.add_argument("--preset", type=str, default="default",
                    choices=list(PRESET_FILESETS.keys()),
                    help="which Monash file preset to use")
    ap.add_argument("--files", type=str, nargs="+", default=None,
                    help="explicit Monash .tsf file names (overrides --preset)")
    ap.add_argument("--save_path", type=str, default=os.path.join(MODEL_DIR, "deepma_linear.npz"))
    ap.add_argument("--L_in", type=int, default=DEFAULT_L_IN)
    ap.add_argument("--L_out", type=int, default=DEFAULT_L_OUT)
    ap.add_argument("--kernels", type=int, nargs="+", default=DEFAULT_KERNELS)
    ap.add_argument("--max_per_series", type=int, default=DEFAULT_MAX_PER_SERIES)
    ap.add_argument("--outlier_thr", type=float, default=DEFAULT_OUTLIER_THR)
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    files = args.files if args.files else PRESET_FILESETS[args.preset]
    print(f"Using Monash files ({len(files)}):")
    for f in files:
        print(f"  - {f}")

    print("\n===== Loading Monash =====")
    t0 = time.time()
    X, Y = build_monash_windows(
        data_dir=args.monash_dir,
        files=files,
        L_in=args.L_in, L_out=args.L_out,
        max_windows_per_series=args.max_per_series,
        outlier_threshold=args.outlier_thr,
        seed=args.seed,
    )
    print(f"  elapsed: {time.time()-t0:.1f}s")
    if X.shape[0] == 0:
        raise RuntimeError("No training windows built. Check monash_dir.")

    print(f"\n===== DeepMA decomposition (kernels={args.kernels}) =====")
    cfg = types.SimpleNamespace(kernel_list=args.kernels)
    model = DeepMA(cfg)
    t0 = time.time()
    Xd = decompose_batched(model, X)
    Yd = decompose_batched(model, Y)
    print(f"  elapsed: {time.time()-t0:.1f}s  (layers={len(Xd)})")
    print(f"  max|sum(X_i)-X|={np.max(np.abs(sum(Xd) - X)):.3e}  "
          f"max|sum(Y_i)-Y|={np.max(np.abs(sum(Yd) - Y)):.3e}")

    print(f"\n===== Layer-wise ridge LS (lam={args.ridge}) =====")
    Ws, bs = [], []
    for i, (Xi, Yi) in enumerate(zip(Xd, Yd)):
        t0 = time.time()
        W, b = fit_ridge(Xi, Yi, lam=args.ridge)
        Ws.append(W); bs.append(b)
        pred = Xi @ W + b
        layer_mse = float(np.mean((pred - Yi) ** 2))
        layer_mae = float(np.mean(np.abs(pred - Yi)))
        tag = "Trend" if i == len(Xd) - 1 else f"S{i+1}(k={args.kernels[i]})"
        print(f"  [{tag:<12}] fit {time.time()-t0:.2f}s  MSE={layer_mse:.4f}  MAE={layer_mae:.4f}")

    pred_total = np.zeros_like(Y)
    for Xi, W, b in zip(Xd, Ws, bs):
        pred_total += Xi @ W + b
    full_mse = float(np.mean((pred_total - Y) ** 2))
    full_mae = float(np.mean(np.abs(pred_total - Y)))
    print(f"  [TOTAL       ] MSE={full_mse:.4f}  MAE={full_mae:.4f}")

    save_model(args.save_path, args.kernels, args.L_in, args.L_out, Ws, bs)
    print(f"\nSaved weights -> {args.save_path}")


if __name__ == "__main__":
    main()
