"""Sanity check of DeepMA multi-level decomposition on a synthetic signal."""
import os
import types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepma import DeepMA
from deepma.config import DEFAULT_KERNELS


def make_signal(L=512, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    high = 1.0 * np.sin(2 * np.pi * t / 8)
    mid = 1.5 * np.sin(2 * np.pi * t / 30)
    low = 2.0 * np.sin(2 * np.pi * t / 120)
    trend = 0.01 * t
    noise = 0.3 * rng.standard_normal(L)
    return high, mid, low, trend, noise, (high + mid + low + trend + noise)


def main():
    torch.manual_seed(0)
    L = 512
    kernel_list = DEFAULT_KERNELS
    high, mid, low, trend_gt, noise, x_np = make_signal(L=L)
    x = torch.tensor(x_np, dtype=torch.float32).view(1, L, 1)

    cfg = types.SimpleNamespace(kernel_list=kernel_list)
    model = DeepMA(cfg).eval()
    with torch.no_grad():
        seasonals, trend = model(x)

    total_e = float(x.var().item())
    print(f"Input var(x)={total_e:.4f}  kernels={kernel_list}")
    print("-" * 62)
    print(f"{'Level':<8}{'Kernel':<10}{'Shape':<18}{'Var(energy)':<14}{'Energy%':<10}")
    parts = []
    for i, (k, s) in enumerate(zip(kernel_list, seasonals)):
        e = float(s.var().item())
        parts.append((f"S{i+1}(k={k})", s.squeeze().numpy(), e))
        print(f"S{i+1:<7}{k:<10}{str(tuple(s.shape)):<18}{e:<14.4f}{100*e/total_e:<10.2f}")
    e_t = float(trend.var().item())
    parts.append(("Trend", trend.squeeze().numpy(), e_t))
    print(f"{'Trend':<8}{'-':<10}{str(tuple(trend.shape)):<18}{e_t:<14.4f}{100*e_t/total_e:<10.2f}")
    print("-" * 62)

    recon = sum(seasonals) + trend
    err = (recon - x).abs().max().item()
    print(f"Reconstruction max|err| = {err:.3e}  (should be ~0)")

    n_rows = 2 + len(kernel_list) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.6 * n_rows), sharex=True)
    axes[0].plot(x_np, color="k"); axes[0].set_ylabel("input")
    axes[1].plot(high, label="HF(gt)", alpha=.7)
    axes[1].plot(mid, label="MF(gt)", alpha=.7)
    axes[1].plot(low, label="LF(gt)", alpha=.7)
    axes[1].plot(trend_gt, label="Trend(gt)", alpha=.7)
    axes[1].legend(loc="upper right", fontsize=8); axes[1].set_ylabel("GT comps")
    for j, (name, arr, e) in enumerate(parts):
        ax = axes[2 + j]
        ax.plot(arr)
        ax.set_ylabel(f"{name}\nvar={e:.2f}")
    axes[-1].set_xlabel("t")
    fig.suptitle(f"DeepMA layer-wise decomposition (kernels={kernel_list})")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decomp.png")
    fig.savefig(out, dpi=120)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
