"""Evaluate a trained DeepMA+linear model on ETTh1/ETTh2 (Time-MoE protocol)."""
import argparse
import os
import numpy as np

from deepma import load_model, load_etth_test, revin, predict
from deepma.config import ETTH1_CSV, ETTH2_CSV, MODEL_DIR


def eval_one_csv(csv_path, model, Ws, bs, L_in, L_out):
    X, Y, _, _ = load_etth_test(csv_path, L_in, L_out)
    X_rev, _, mu, sd = revin(X)
    Y_hat_rev = predict(model, Ws, bs, X_rev)
    Y_hat = Y_hat_rev * sd + mu
    mse = float(np.mean((Y_hat - Y) ** 2))
    mae = float(np.mean(np.abs(Y_hat - Y)))
    return mse, mae, X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "deepma_linear.npz"))
    ap.add_argument("--etth1", type=str, default=ETTH1_CSV)
    ap.add_argument("--etth2", type=str, default=ETTH2_CSV)
    args = ap.parse_args()

    model, kernels, L_in, L_out, Ws, bs = load_model(args.model)
    print(f"Loaded model: kernels={kernels}  L_in={L_in}  L_out={L_out}  layers={len(Ws)}")

    print("\n===== Evaluate on ETTh1 / ETTh2 (Time-MoE protocol) =====")
    print(f"  {'Dataset':<10}{'#windows':<12}{'MSE':<12}{'MAE':<12}")
    print("  " + "-" * 44)
    for name, path in [("ETTh1", args.etth1), ("ETTh2", args.etth2)]:
        if not os.path.exists(path):
            print(f"  {name:<10}[missing csv: {path}]")
            continue
        mse, mae, n = eval_one_csv(path, model, Ws, bs, L_in, L_out)
        print(f"  {name:<10}{n:<12}{mse:<12.4f}{mae:<12.4f}")


if __name__ == "__main__":
    main()
