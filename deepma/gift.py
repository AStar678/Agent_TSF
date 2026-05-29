"""Gift-eval HuggingFace dataset loader.

Each subdirectory under the gift-eval root (e.g. ``/root/gift-eval/Mini-GiftPretrain``)
is a HuggingFace ``datasets`` directory with features
``{item_id, start, freq, target, [past_feat_dynamic_real, ...]}``.
``target`` may be 1D (single univariate series) or 2D (multivariate,
shape ``(n_vars, T)``; each row is expanded into its own univariate series).

Windowing protocol mirrors :func:`deepma.data.build_monash_windows`:
per-window RevIN z-score; drop low-std and outlier windows; optional
per-series cap and global ``sample_frac`` sub-sampling.
"""
from __future__ import annotations

import os
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:                          # graceful fallback
    def tqdm(it=None, **kw):
        return it if it is not None else range(0)

from .config import GIFT_DIR, DEFAULT_MIN_STD


# ---------------------------------------------------------------- presets
# Default: every dataset currently shipped in Mini-GiftPretrain.
MINI_GIFT_DATASETS = [
    "BEIJING_SUBWAY_30MIN",
    "PEMS03",
    "australian_electricity_demand",
    "beijing_air_quality",
    "bitcoin_with_missing",
    "cdc_fluview_ilinet",
    "cif_2016_12",
]

# Hourly / energy / traffic subset most similar to ETTh.
ETTH_LIKE_GIFT_DATASETS = [
    "australian_electricity_demand",
    "PEMS03",
    "BEIJING_SUBWAY_30MIN",
    "beijing_air_quality",
]

GIFT_PRESETS = {
    "mini": MINI_GIFT_DATASETS,
    "etth_like": ETTH_LIKE_GIFT_DATASETS,
}


# ---------------------------------------------------------------- IO
def _is_hf_dataset_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "dataset_info.json"))
        and os.path.isfile(os.path.join(path, "state.json"))
    )


def list_gift_datasets(data_dir: str = GIFT_DIR):
    """Return sorted names of all HF dataset subdirs under ``data_dir``."""
    if not os.path.isdir(data_dir):
        return []
    names = []
    for d in sorted(os.listdir(data_dir)):
        if d.startswith(".") or d.startswith("_"):
            continue
        if _is_hf_dataset_dir(os.path.join(data_dir, d)):
            names.append(d)
    return names


def _example_to_series(target) -> list[np.ndarray]:
    """Expand one example's ``target`` field into a list of 1D float32 arrays.

    Handles 1D (single series) and 2D (multivariate ``(n_vars, T)``) targets,
    as well as ragged Python lists of lists.
    """
    arr = np.asarray(target)
    out: list[np.ndarray] = []
    if arr.ndim == 1:
        out.append(arr.astype(np.float32, copy=False))
    elif arr.ndim == 2:
        for row in arr:
            out.append(np.asarray(row, dtype=np.float32))
    elif arr.ndim == 0:
        # scalar-wrapped list-of-lists that failed to stack: iterate Python-side
        try:
            for row in target:
                sub = np.asarray(row, dtype=np.float32)
                if sub.ndim == 1:
                    out.append(sub)
        except Exception:
            pass
    return out


def _drop_nans(s: np.ndarray, min_len: int = 2) -> np.ndarray | None:
    """Drop NaN/Inf entries; return None when too few valid samples remain."""
    if not np.isfinite(s).all():
        s = s[np.isfinite(s)]
    if s.size < min_len:
        return None
    return s.astype(np.float32, copy=False)


def load_gift_series(
    data_dir: str = GIFT_DIR,
    datasets=None,
    verbose: bool = True,
):
    """Flatten all selected gift-eval datasets into a list[np.ndarray] of 1D series."""
    from datasets import load_from_disk  # local import -> keep package light

    if datasets is None:
        datasets = list_gift_datasets(data_dir) or MINI_GIFT_DATASETS

    all_series: list[np.ndarray] = []
    for name in datasets:
        path = os.path.join(data_dir, name)
        if not _is_hf_dataset_dir(path):
            if verbose:
                print(f"  [skip] {name}: not a HF dataset dir under {data_dir}")
            continue
        try:
            ds = load_from_disk(path)
        except Exception as e:
            if verbose:
                print(f"  [skip] {name}: {e}")
            continue

        before = len(all_series)
        for ex in ds:
            for s in _example_to_series(ex["target"]):
                s = _drop_nans(s)
                if s is not None:
                    all_series.append(s)
        added = len(all_series) - before
        total = sum(s.size for s in all_series[before:])
        if verbose:
            print(f"  [ok]   {name}: {len(ds)} examples -> "
                  f"{added} series, {total:,} points")

    if verbose:
        print(f"Total gift series: {len(all_series)}")
    return all_series


# ---------------------------------------------------------------- windows
def build_gift_windows(
    data_dir: str = GIFT_DIR,
    datasets=None,
    L_in: int = 96,
    L_out: int = 96,
    stride=None,
    max_windows_per_series: int = 200,
    outlier_threshold: float = 10.0,
    min_std: float = DEFAULT_MIN_STD,
    seed: int = 0,
    verbose: bool = True,
    sample_frac: float = 1.0,
):
    """(X, Y) windows with per-window RevIN z-score from gift-eval datasets.

    Exactly mirrors :func:`deepma.data.build_monash_windows` but reads from
    HuggingFace-format directories instead of Monash TSF files.
    """
    rng = np.random.default_rng(seed)
    series_list = load_gift_series(data_dir, datasets, verbose=verbose)
    W = L_in + L_out
    if stride is None:
        stride = max(1, L_out // 2)

    sample_frac = float(sample_frac)
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")

    iterator = series_list
    if verbose and len(series_list) > 1:
        iterator = tqdm(series_list, desc="gift-windows",
                        ncols=80, leave=False)

    Xs, Ys = [], []
    for s in iterator:
        if s.size < W + 1:
            continue
        starts = np.arange(0, s.size - W, stride, dtype=np.int64)
        if starts.size == 0:
            continue
        if starts.size > max_windows_per_series:
            starts = rng.choice(starts, size=max_windows_per_series, replace=False)
        if sample_frac < 1.0:
            keep = max(1, int(round(starts.size * sample_frac)))
            if keep < starts.size:
                starts = rng.choice(starts, size=keep, replace=False)
        for st in starts:
            win = s[st : st + W]
            x, y = win[:L_in], win[L_in:]
            mu, sd = x.mean(), x.std()
            if sd < min_std:
                continue
            x_n = (x - mu) / sd
            y_n = (y - mu) / sd
            if np.abs(y_n).max() > outlier_threshold:
                continue
            Xs.append(x_n.astype(np.float32))
            Ys.append(y_n.astype(np.float32))

    X = np.stack(Xs, axis=0) if Xs else np.zeros((0, L_in), np.float32)
    Y = np.stack(Ys, axis=0) if Ys else np.zeros((0, L_out), np.float32)
    if verbose:
        print(f"Built gift RevIN windows: X={X.shape}, Y={Y.shape}")
    if X.shape[0] == 0:
        raise RuntimeError(
            f"build_gift_windows produced 0 windows. "
            f"Check --gift_dir ({data_dir}) and --gift_datasets: {datasets}. "
            f"Available dirs: {list_gift_datasets(data_dir)}"
        )
    return X, Y
