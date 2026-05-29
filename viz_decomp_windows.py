"""Visualize one random window per random dataset, layer by layer.

For each chosen dataset:
  1. Pick a random series long enough to host the requested window.
  2. Pick a random start within that series.
  3. Plot the original signal (= sum of all 10 layers) on top, then each
     seasonal layer (sorted by ascending kernel), and finally the trend.

Output: one PNG per dataset under ``--out_dir``:
    decomp_window_<dataset>.png

Usage
-----
    python viz_decomp_windows.py
    python viz_decomp_windows.py --n_datasets 6 --window 1024 --seed 42
    python viz_decomp_windows.py --datasets us_births_dataset oikolab_weather_dataset
"""
import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepma_decomp_loader import load_meta, list_datasets


DEFAULT_ROOT = "/root/autodl-tmp/deepma_decomp"
DEFAULT_OUT = "/root/DeepMA/viz/decomp_windows"


def _pick_series(dataset_dir: str, min_len: int, rng: np.random.Generator,
                 max_probes: int = 200):
    """Return ``(key, series_length)`` for a random series with length >= min_len.

    Falls back to the longest series in the dataset if no probe satisfies the
    length requirement (rare; only for very short Monash datasets).
    """
    with np.load(os.path.join(dataset_dir, "trend.npz")) as z:
        keys = list(z.files)
        # random sampling first
        order = rng.permutation(len(keys))
        for j in order[:max_probes]:
            k = keys[int(j)]
            n = int(z[k].size)
            if n >= min_len:
                return k, n
        # fallback: scan everything for the longest one
        best_k, best_n = keys[0], int(z[keys[0]].size)
        for k in keys:
            n = int(z[k].size)
            if n > best_n:
                best_k, best_n = k, n
        return best_k, best_n


def _load_window(dataset_dir: str, layer_file: str, key: str,
                 start: int, length: int) -> np.ndarray:
    with np.load(os.path.join(dataset_dir, layer_file)) as z:
        arr = z[key]
        return arr[start:start + length].astype(np.float32)


def viz_one(dataset_dir: str, out_dir: str, window: int,
            rng: np.random.Generator) -> str:
    name = os.path.basename(dataset_dir.rstrip("/"))
    meta = load_meta(dataset_dir)
    kernels = meta["kernels"]                                    # 9 ints

    key, T = _pick_series(dataset_dir, window, rng)
    L = min(window, T)
    start = int(rng.integers(0, T - L + 1)) if T > L else 0

    seasonal_layers = [f"seasonal_k{k:03d}" for k in kernels]
    layer_data = {}
    for stem in seasonal_layers + ["trend"]:
        layer_data[stem] = _load_window(
            dataset_dir, stem + ".npz", key, start, L)

    # reconstructed original = trend + sum(seasonals)
    orig = layer_data["trend"].copy()
    for stem in seasonal_layers:
        orig += layer_data[stem]

    n_rows = 1 + len(kernels) + 1                                # 1+9+1=11
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.15 * n_rows),
                             sharex=True)

    t = np.arange(L)
    axes[0].plot(t, orig, color="black", lw=0.9)
    axes[0].set_ylabel("orig", fontsize=9)
    axes[0].set_title(
        f"{name}\nseries={key}  window=[{start}:{start + L}]  "
        f"series_len={T}  L={L}",
        fontsize=10,
    )
    for i, kernel in enumerate(kernels):
        stem = f"seasonal_k{kernel:03d}"
        axes[1 + i].plot(t, layer_data[stem], lw=0.7,
                         color=plt.cm.viridis(i / max(1, len(kernels) - 1)))
        axes[1 + i].set_ylabel(f"k={kernel}", fontsize=9)
    axes[-1].plot(t, layer_data["trend"], color="crimson", lw=1.0)
    axes[-1].set_ylabel("trend", fontsize=9)
    axes[-1].set_xlabel("time step (within window)")

    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"decomp_window_{name}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"  [ok] {name}: series={key}  start={start}  L={L}  -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--n_datasets", type=int, default=4,
                    help="Number of datasets to randomly sample (ignored "
                         "when --datasets is given).")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Explicit list of dataset stems; overrides "
                         "--n_datasets when provided.")
    ap.add_argument("--window", type=int, default=1024,
                    help="Window length (default 1024).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    all_ds = list_datasets(args.root)
    if args.datasets:
        chosen = [d for d in args.datasets if d in all_ds]
        missing = [d for d in args.datasets if d not in all_ds]
        if missing:
            print(f"[warn] dataset(s) not found, skipped: {missing}")
    else:
        n = min(args.n_datasets, len(all_ds))
        chosen = sorted(rng.choice(all_ds, size=n, replace=False).tolist())

    print("=" * 64)
    print("DeepMA decomposition window visualization")
    print(f"  root       = {args.root}")
    print(f"  out_dir    = {args.out_dir}")
    print(f"  window     = {args.window}")
    print(f"  seed       = {args.seed}")
    print(f"  datasets   = {chosen}")
    print("=" * 64)

    for d in chosen:
        viz_one(os.path.join(args.root, d), args.out_dir, args.window, rng)

    print(f"[done] all PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
