"""Visualize DLinear (Time-Series-Library) checkpoint weights.

The DLinear model has two linear layers:
    Linear_Seasonal : nn.Linear(seq_len, pred_len)  # W shape (pred_len, seq_len)
    Linear_Trend    : nn.Linear(seq_len, pred_len)

We plot both weight matrices as heatmaps and both biases as line plots.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_CKPT = (
    "/root/Time-Series-Library-main/checkpoints/"
    "long_term_forecast_ETTh1_96_96_DLinear_ETTh1_ftM_sl1440_ll48_pl720_"
    "dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/"
    "checkpoint.pth"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="/root/DeepMA/dlinear_weights.png")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    W_s = sd["Linear_Seasonal.weight"].cpu().numpy()      # (pred_len, seq_len)
    b_s = sd["Linear_Seasonal.bias"].cpu().numpy()        # (pred_len,)
    W_t = sd["Linear_Trend.weight"].cpu().numpy()
    b_t = sd["Linear_Trend.bias"].cpu().numpy()
    pred_len, seq_len = W_s.shape
    init_val = 1.0 / seq_len
    print(f"seq_len={seq_len}  pred_len={pred_len}  init (1/seq_len)={init_val:.2e}")
    for name, W in [("Seasonal", W_s), ("Trend", W_t)]:
        print(f"  Linear_{name}.weight "
              f"min={W.min():+.4f} max={W.max():+.4f} "
              f"mean={W.mean():+.4f} std={W.std():.4f}")
    for name, b in [("Seasonal", b_s), ("Trend", b_t)]:
        print(f"  Linear_{name}.bias   "
              f"min={b.min():+.4f} max={b.max():+.4f} "
              f"mean={b.mean():+.4f} std={b.std():.4f}")

    # Symmetric colormap scale (shared between both W matrices for comparability)
    vmax = max(np.abs(W_s).max(), np.abs(W_t).max())

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # ---- heatmaps ---------------------------------------------------------
    for col, (name, W) in enumerate([("Seasonal", W_s), ("Trend", W_t)]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(W, aspect="auto", cmap="seismic",
                       vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_title(f"Linear_{name}.weight  shape={W.shape}")
        ax.set_xlabel("input index (seq_len)")
        ax.set_ylabel("output index (pred_len)")
        fig.colorbar(im, ax=ax, shrink=0.9)

    # ---- bias -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(b_s, color="C3", lw=0.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Linear_Seasonal.bias  shape=({pred_len},)")
    ax.set_xlabel("output index")
    ax.set_ylabel("bias value")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(b_t, color="C0", lw=0.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Linear_Trend.bias  shape=({pred_len},)")
    ax.set_xlabel("output index")
    ax.set_ylabel("bias value")

    fig.suptitle(
        f"DLinear weights  |  seq_len={seq_len}  pred_len={pred_len}  "
        f"init={init_val:.2e}  |  heatmap range=[-{vmax:.3f}, {vmax:.3f}]",
        fontsize=12,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
