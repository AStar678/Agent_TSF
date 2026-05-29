"""Time-MoE benchmark CSV loader.

Implements the split / normalization protocol:
  * For ``ETTh*``: fixed 12 / 4 / 4 month splits (hour granularity).
  * For ``ETTm*``: fixed 12 / 4 / 4 month splits (15-min granularity, x4).
  * For all other CSVs: 70 / 10 / 20 ratio.
  * StandardScaler is fit on the train segment only, then applied to the
    requested split (train / val / test).
  * Variables are treated channel-independently (each column after ``date``
    becomes its own univariate series).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


_SPLIT_IDX = {"train": 0, "val": 1, "test": 2}


def _get_borders(csv_path: str, context_length: int, n_rows: int):
    """Return (border1s, border2s) lists of length 3 following Time-MoE."""
    base = os.path.basename(csv_path).lower()
    if "etth" in base:
        b1s = [
            0,
            12 * 30 * 24 - context_length,
            12 * 30 * 24 + 4 * 30 * 24 - context_length,
        ]
        b2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
    elif "ettm" in base:
        b1s = [
            0,
            12 * 30 * 24 * 4 - context_length,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length,
        ]
        b2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
    else:
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_vali = n_rows - num_train - num_test
        b1s = [0, num_train - context_length, n_rows - num_test - context_length]
        b2s = [num_train, num_train + num_vali, n_rows]
    return b1s, b2s


def load_benchmark_split(
    csv_path: str,
    L_in: int,
    L_out: int,
    split: str = "train",
):
    """Build (X, Y) windows from a CSV following the Time-MoE protocol.

    Parameters
    ----------
    csv_path : str
        Path to the benchmark CSV (first column must be ``date``).
    L_in, L_out : int
        Context length and prediction length.
    split : {'train', 'val', 'test'}
        Which split to sample windows from. Scaler is always fit on 'train'.

    Returns
    -------
    X : np.ndarray (N, L_in) float32
    Y : np.ndarray (N, L_out) float32
    """
    if split not in _SPLIT_IDX:
        raise ValueError(f"split must be one of {list(_SPLIT_IDX)}, got {split!r}")

    df = pd.read_csv(csv_path)
    n = len(df)
    b1s, b2s = _get_borders(csv_path, L_in, n)

    cols = df.columns[1:]  # skip 'date'
    values = df[cols].values.astype(np.float32)  # (T, V)

    # Fit scaler on train segment only (Time-MoE protocol).
    train_data = values[b1s[0]:b2s[0]]
    scaler = StandardScaler().fit(train_data)

    sp = _SPLIT_IDX[split]
    if split == "train":
        seg = values[0:b2s[0]]
    else:
        seg = values[b1s[sp]:b2s[sp]]
    scaled = scaler.transform(seg).astype(np.float32)  # (T_seg, V)

    # Channel-independent: (V, T_seg)
    series = scaled.transpose(1, 0)

    window_length = L_in + L_out
    from numpy.lib.stride_tricks import sliding_window_view
    n_s = series.shape[1]
    if n_s < window_length:
        raise RuntimeError(
            f"No windows built from {csv_path} split={split} "
            f"with L_in={L_in}, L_out={L_out}. Try a smaller L_in/L_out."
        )
    W_view = sliding_window_view(series, window_shape=window_length, axis=1)
    flat = W_view.reshape(-1, window_length).copy().astype(np.float32, copy=False)
    X = np.ascontiguousarray(flat[:, :L_in])
    Y = np.ascontiguousarray(flat[:, L_in:])
    return X, Y
