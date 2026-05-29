"""Visualize DeepMA layer-wise linear predictions on ETTh1.

Figures:
  1. vis_etth1_samples.png  : random windows, GT vs prediction.
  2. vis_etth1_layers.png   : one window's per-layer decomposition + contribution.
All plots are in the StandardScaler domain (Time-MoE convention).
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepma import load_model, load_etth_test, revin, predict
from deepma.config import ETTH1_CSV, MODEL_DIR, PROJECT_ROOT


def _decompose_one(model, x_rev):
    with torch.no_grad():
        xt = torch.from_numpy(x_rev).float().view(1, -1, 1)
        seasonals, trend = model(xt)
    return [c.squeeze().numpy() for c in seasonals] + [trend.squeeze().numpy()]


def plot_samples(X, Y, Y_hat, cols, L_in, L_out, n_show, save_path, seed=0):
    rng = np.random.default_rng(seed)
    C = len(cols)
    per_ch = X.shape[0] // C
    idxs = []
    n_per_ch = max(1, n_show // C)
    for c in range(C):
        choices = rng.choice(per_ch, size=min(n_per_ch, per_ch), replace=False)
        idxs.extend([c * per_ch + k for k in choices])
    idxs = idxs[:n_show]

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.2 * n_show), sharex=False)
    if n_show == 1:
        axes = [axes]
    t_ctx = np.arange(L_in)
    t_fut = np.arange(L_in, L_in + L_out)
    for ax, idx in zip(axes, idxs):
        ch = idx // per_ch
        ax.plot(t_ctx, X[idx], color="#888", label="context")
        ax.plot(t_fut, Y[idx], color="#1f77b4", label="ground truth")
        ax.plot(t_fut, Y_hat[idx], color="#d62728", linestyle="--", label="prediction")
        ax.axvline(L_in - 0.5, color="k", lw=0.6, alpha=0.4)
        mse = float(np.mean((Y_hat[idx] - Y[idx]) ** 2))
        ax.set_title(f"channel={cols[ch]}  idx={idx}  MSE={mse:.3f}", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time step")
    fig.suptitle(f"ETTh1 DeepMA-Indep  {L_in} -> {L_out}  predictions", y=1.002)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}")


def plot_layers(X, Y, mu, sd, model, Ws, bs, kernels, cols, L_in, L_out, idx, save_path):
    x, y = X[idx], Y[idx]
    x_rev = (x - mu[idx, 0]) / sd[idx, 0]
    comps = _decompose_one(model, x_rev)
    contribs_rev = [c @ W + b for c, W, b in zip(comps, Ws, bs)]
    y_hat = np.sum(contribs_rev, axis=0) * sd[idx, 0] + mu[idx, 0]

    n_layers = len(comps)
    ch = idx // (X.shape[0] // len(cols))

    fig, axes = plt.subplots(n_layers + 2, 1, figsize=(10, 1.5 * (n_layers + 2)), sharex=True)
    t_ctx = np.arange(L_in)
    t_fut = np.arange(L_in, L_in + L_out)

    axes[0].plot(t_ctx, x, color="k", label="context (scaled)")
    axes[0].plot(t_fut, y, color="#1f77b4", label="GT future")
    axes[0].plot(t_fut, y_hat, color="#d62728", ls="--", label="pred (sum)")
    axes[0].axvline(L_in - 0.5, color="k", lw=0.6, alpha=0.4)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(f"ETTh1  channel={cols[ch]}  window idx={idx}")

    for i in range(n_layers):
        ax = axes[i + 1]
        tag = "Trend" if i == n_layers - 1 else f"S{i+1}(k={kernels[i]})"
        ax.plot(t_ctx, comps[i], color="#555", label=f"{tag} of x (RevIN)")
        ax.plot(t_fut, contribs_rev[i], color="#d62728", label="layer pred (RevIN)")
        ax.axvline(L_in - 0.5, color="k", lw=0.6, alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[-1]
    ax.plot(t_fut, y - y_hat, color="#2ca02c", label="residual (GT - pred)")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("time step")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "deepma_linear.npz"))
    ap.add_argument("--etth1", type=str, default=ETTH1_CSV)
    ap.add_argument("--n_show", type=int, default=7)
    ap.add_argument("--layer_idx", type=int, default=-1,
                    help="window index for layer figure; -1 = median-MSE window")
    ap.add_argument("--out_dir", type=str, default=PROJECT_ROOT)
    args = ap.parse_args()

    model, kernels, L_in, L_out, Ws, bs = load_model(args.model)
    X, Y, _, cols = load_etth_test(args.etth1, L_in, L_out)
    X_rev, _, mu, sd = revin(X)
    Y_hat = predict(model, Ws, bs, X_rev) * sd + mu

    mses = np.mean((Y_hat - Y) ** 2, axis=1)
    if args.layer_idx < 0:
        order = np.argsort(mses)
        args.layer_idx = int(order[len(order) // 2])
    print(f"Global MSE={float(mses.mean()):.4f}  median-window idx={args.layer_idx}  "
          f"its MSE={float(mses[args.layer_idx]):.4f}")

    plot_samples(X, Y, Y_hat, cols, L_in, L_out, args.n_show,
                 os.path.join(args.out_dir, "vis_etth1_samples.png"))
    plot_layers(X, Y, mu, sd, model, Ws, bs, kernels, cols, L_in, L_out,
                args.layer_idx, os.path.join(args.out_dir, "vis_etth1_layers.png"))


if __name__ == "__main__":
    main()
