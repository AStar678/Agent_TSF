"""Sanity-check DeepMA decomposition: sum(layers) ?= original series.

Picks a random dataset (or one passed via --dataset), then a random series
(or one via --series_idx), reloads the source TSF, and reports element-wise
error between ``sum(seasonal_k*) + trend`` and the original signal.
"""
import argparse
import os

import numpy as np

from deepma import parse_tsf
from deepma_decomp_loader import load_meta, list_datasets


DEFAULT_ROOT = "/root/autodl-tmp/deepma_decomp"


def check_one(root: str, dataset: str, series_idx: int, n_extra: int = 0,
              rng: np.random.Generator = None):
    ds_dir = os.path.join(root, dataset)
    meta = load_meta(ds_dir)
    tsf_path = meta["tsf_path"]
    print("=" * 64)
    print(f"dataset    : {dataset}")
    print(f"tsf_path   : {tsf_path}")
    print(f"kernels    : {meta['kernels']}")
    print(f"n_layers   : {meta['n_layers']}  ({meta['n_layers'] - 1} seasonals + 1 trend)")
    print(f"n_series   : saved {meta['n_series_saved']} / total {meta['n_series_total']}")

    series_list = parse_tsf(tsf_path)
    n_total = len(series_list)
    if series_idx >= n_total:
        raise IndexError(f"series_idx={series_idx} out of range [0,{n_total})")

    indices = [series_idx]
    if n_extra > 0 and rng is not None:
        pool = [i for i in range(n_total) if i != series_idx]
        if pool:
            extra = rng.choice(pool, size=min(n_extra, len(pool)), replace=False)
            indices += [int(i) for i in extra]

    for idx in indices:
        key = f"s{idx:06d}"
        orig = np.asarray(series_list[idx], dtype=np.float32)

        recon = np.zeros_like(orig)
        for kernel in meta["kernels"]:
            with np.load(os.path.join(ds_dir, f"seasonal_k{kernel:03d}.npz")) as z:
                if key not in z.files:
                    print(f"  [skip] series idx={idx} ({key}) not in seasonal_k{kernel:03d}")
                    recon = None
                    break
                recon = recon + z[key]
        if recon is None:
            continue
        with np.load(os.path.join(ds_dir, "trend.npz")) as z:
            recon = recon + z[key]

        diff = recon - orig
        abs_diff = np.abs(diff)
        sig_rms = float(np.sqrt(np.mean(orig.astype(np.float64) ** 2)))
        err_rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        rel_rms = err_rms / max(sig_rms, 1e-30)

        print(f"\n  series idx = {idx}  key = {key}  len = {orig.size}")
        print(f"    orig    range : [{orig.min(): .6g}, {orig.max(): .6g}]   RMS = {sig_rms:.6g}")
        print(f"    |err|         : max = {abs_diff.max():.3e}   "
              f"mean = {abs_diff.mean():.3e}   "
              f"median = {np.median(abs_diff):.3e}")
        print(f"    rmse / rel    : rmse = {err_rms:.3e}   relative = {rel_rms:.3e}")
        # show worst-3 positions
        worst = np.argsort(-abs_diff)[:3]
        print(f"    worst 3 pts   : "
              + ", ".join(f"t={int(p)} orig={orig[p]:.6g} recon={recon[p]:.6g} "
                          f"err={diff[p]:+.3e}" for p in worst))

    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dataset", default=None,
                    help="Specific dataset stem; default = random.")
    ap.add_argument("--series_idx", type=int, default=None,
                    help="Specific series index; default = random.")
    ap.add_argument("--n_extra", type=int, default=2,
                    help="How many extra random series to also check "
                         "(default 2, in addition to the primary).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    all_ds = list_datasets(args.root)
    if not all_ds:
        raise SystemExit(f"no decomposed datasets under {args.root}")

    ds = args.dataset or str(rng.choice(all_ds))
    if ds not in all_ds:
        raise SystemExit(f"dataset {ds!r} not in {args.root}")

    meta = load_meta(os.path.join(args.root, ds))
    n_total = meta["n_series_total"]
    if args.series_idx is None:
        idx = int(rng.integers(0, max(1, n_total)))
    else:
        idx = args.series_idx

    check_one(args.root, ds, idx, n_extra=args.n_extra, rng=rng)


if __name__ == "__main__":
    main()
