"""Synthetic multi-period sinusoid test for DeepMA decomposition.

Builds a 1-D signal as

    x(t) = sum_i  A_i * sin(2 * pi * t / T_i + phi_i)
         + trend_signal                # linear + very slow sine
         + small Gaussian noise

then decomposes it with DeepMA(kernels=[5,13,25,49,97,125,161,201,301]) and
quantifies how well each ground-truth component lands on its expected layer.

Outputs (under ``--out_dir``, default ``/root/DeepMA/viz/synthetic_decomp``):
  - synthetic_signal.png          decomposition vs ground-truth, layer by layer
  - synthetic_cosine.png          cosine-similarity heatmap (GT x layer)
  - synthetic_attribution.csv     per-layer energy attribution table

A DeepMA seasonal layer with kernel ``k_i`` mainly passes periods between
``k_{i-1}`` and ``k_i``; the trend layer collects everything slower than
``k_K``. Components are placed near layer mid-points to ease verification.
"""
import argparse
import csv
import os
import types

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepma import DeepMA


KERNELS = [5, 13, 25, 49, 97, 125, 161, 201, 301]


def _build_signal(N: int, rng: np.random.Generator):
    """Return (x, gt_components, gt_names, trend_signal, noise).

    Components are chosen so each one lives near the centre of one DeepMA
    seasonal band (k_prev, k_i]; the slow trend goes well beyond the
    largest kernel.
    """
    t = np.arange(N, dtype=np.float64)
    comps, names = [], []

    plan = [
        # period, amplitude, expected layer label
        (8,    1.0, "T= 8  (-> k=13)"),
        (20,   1.5, "T=20  (-> k=25)"),
        (36,   1.2, "T=36  (-> k=49)"),
        (72,   1.0, "T=72  (-> k=97)"),
        (144,  0.9, "T=144 (-> k=161)"),
        (240,  0.8, "T=240 (-> k=301)"),
    ]
    for T, A, name in plan:
        phi = float(rng.uniform(0.0, 2 * np.pi))
        comps.append(A * np.sin(2 * np.pi * t / T + phi))
        names.append(name)

    # slow trend: a linear ramp + a very slow sine (period 4*N) -> goes to trend layer
    trend = (0.002 * t) + 1.5 * np.sin(2 * np.pi * t / (4 * N))
    comps.append(trend)
    names.append("trend (linear + T=4N)")

    # high-frequency noise: targets the highest-frequency seasonal (k=5)
    noise = 0.10 * rng.standard_normal(N)
    comps.append(noise)
    names.append("noise (-> k=5)")

    x = np.zeros(N, dtype=np.float64)
    for c in comps:
        x += c
    return x.astype(np.float32), [c.astype(np.float32) for c in comps], names


def _decompose(x: np.ndarray, kernels, window_kind: str = "rect"):
    cfg = types.SimpleNamespace(kernel_list=kernels, window_kind=window_kind)
    model = DeepMA(cfg).eval()
    inp = torch.from_numpy(x).float().view(1, -1, 1)
    with torch.no_grad():
        seasonals, trend = model(inp)
    seasonals = [s.squeeze(0).squeeze(-1).numpy().astype(np.float32) for s in seasonals]
    trend_np = trend.squeeze(0).squeeze(-1).numpy().astype(np.float32)
    return seasonals, trend_np


def _cosine_matrix(gt: list, layer_outs: list) -> np.ndarray:
    """rows = GT components, cols = DeepMA layers; values in [-1, 1]."""
    M = np.zeros((len(gt), len(layer_outs)), dtype=np.float64)
    for i, g in enumerate(gt):
        gn = np.linalg.norm(g)
        for j, lo in enumerate(layer_outs):
            ln = np.linalg.norm(lo)
            if gn > 0 and ln > 0:
                M[i, j] = float(np.dot(g, lo) / (gn * ln))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4096, help="signal length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/root/DeepMA/viz/synthetic_decomp")
    ap.add_argument("--window_kind", default="rect",
                    choices=["rect", "hann", "hamming", "gauss", "tri"],
                    help="MA weighting (rect = original AvgPool).")
    ap.add_argument("--tag", default=None,
                    help="Suffix for output filenames; default = window_kind.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tag = args.tag or args.window_kind

    print("=" * 64)
    print("DeepMA synthetic multi-period decomposition test")
    print(f"  N           = {args.N}")
    print(f"  seed        = {args.seed}")
    print(f"  window_kind = {args.window_kind}")
    print(f"  kernels     = {KERNELS}")
    print("=" * 64)

    x, gt, gt_names = _build_signal(args.N, rng)
    seasonals, trend = _decompose(x, KERNELS, window_kind=args.window_kind)
    layer_outs = seasonals + [trend]
    layer_names = [f"k={k}" for k in KERNELS] + ["trend"]

    # ---- reconstruction sanity ----------------------------------------------
    recon = trend.copy()
    for s in seasonals:
        recon += s
    err = np.abs(recon - x)
    print(f"[recon] max|err|={err.max():.3e}   "
          f"rmse={np.sqrt((err.astype(np.float64) ** 2).mean()):.3e}   "
          f"signal RMS={np.sqrt((x.astype(np.float64) ** 2).mean()):.4g}")

    # ---- cosine matrix + dominant layer per component -----------------------
    M = _cosine_matrix(gt, layer_outs)
    print("\n[cosine similarity]   rows = GT component   cols = DeepMA layer")
    header = "                                  " + "  ".join(f"{n:>6s}" for n in layer_names)
    print(header)
    for i, name in enumerate(gt_names):
        row = "  ".join(f"{v:6.2f}" for v in M[i])
        best = int(np.argmax(np.abs(M[i])))
        print(f"  {name:<32s}  {row}     -> best={layer_names[best]} ({M[i, best]:+.2f})")

    # ---- attribution table (energy share of each layer, per GT comp) -------
    rows = []
    layer_energies = np.array([float((lo.astype(np.float64) ** 2).sum()) for lo in layer_outs])
    total_layer_energy = layer_energies.sum() or 1.0
    for j, ln in enumerate(layer_names):
        rows.append({
            "layer": ln,
            "energy_sum": layer_energies[j],
            "energy_ratio": layer_energies[j] / total_layer_energy,
            **{f"cos_with[{gt_names[i]}]": float(M[i, j]) for i in range(len(gt))},
        })
    csv_path = os.path.join(args.out_dir, f"synthetic_attribution_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[ok] attribution CSV  -> {csv_path}")

    # ---- figure 1: signal + GT + DeepMA layers ------------------------------
    fig, axes = plt.subplots(
        1 + len(gt) + len(layer_outs), 1,
        figsize=(13, 1.05 * (1 + len(gt) + len(layer_outs))), sharex=True,
    )
    t = np.arange(args.N)
    axes[0].plot(t, x, color="black", lw=0.7)
    axes[0].set_ylabel("signal", fontsize=9)
    axes[0].set_title(f"Synthetic signal & DeepMA decomposition  [window_kind={args.window_kind}]", fontsize=11)

    for i, (g, name) in enumerate(zip(gt, gt_names)):
        ax = axes[1 + i]
        ax.plot(t, g, color="tab:blue", lw=0.7)
        ax.set_ylabel(f"GT: {name}", fontsize=8)

    base = 1 + len(gt)
    for j, (lo, ln) in enumerate(zip(layer_outs, layer_names)):
        ax = axes[base + j]
        col = "crimson" if ln == "trend" else plt.cm.viridis(j / max(1, len(layer_outs) - 1))
        ax.plot(t, lo, color=col, lw=0.7)
        ax.set_ylabel(ln, fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    sig_path = os.path.join(args.out_dir, f"synthetic_signal_{tag}.png")
    fig.savefig(sig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] signal figure    -> {sig_path}")

    # ---- figure 2: cosine heat-map ------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * len(gt)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(gt_names)))
    ax.set_yticklabels(gt_names)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if abs(v) >= 0.05:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(f"GT component vs DeepMA layer (cosine)  [window_kind={args.window_kind}]")
    fig.tight_layout()
    heat_path = os.path.join(args.out_dir, f"synthetic_cosine_{tag}.png")
    fig.savefig(heat_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] heatmap figure   -> {heat_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
