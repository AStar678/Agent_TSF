"""Data loading utilities.

- TSF (Monash format) parser
- Monash window sampling (RevIN per-window OR per-series z-score)
- Benchmark CSV train/test loaders following the Time-MoE evaluation protocol.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import MONASH_DIR, DEFAULT_MIN_STD


# =============================================================== Monash IO
# Light-weight default subset that exists under /root/monash_tsf_all/data and
# mixes frequencies / domains (sum ~ 70 MB).
DEFAULT_MONASH_FILES = [
    "pedestrian_counts_dataset.tsf",
    "kdd_cup_2018_dataset_with_missing_values.tsf",
    "australian_electricity_demand_dataset.tsf",
    "solar_10_minutes_dataset.tsf",
    "oikolab_weather_dataset.tsf",
    "rideshare_dataset_with_missing_values.tsf",
    "nn5_daily_dataset_with_missing_values.tsf",
    "hospital_dataset.tsf",
    "fred_md_dataset.tsf",
    "saugeenday_dataset.tsf",
    "covid_deaths_dataset.tsf",
    "us_births_dataset.tsf",
]

# Hourly + energy/flow-like subset most similar to ETTh (hourly, 2-yr span,
# strong daily/weekly seasonality, continuous measurements).
ETTH_LIKE_MONASH_FILES = [
    "australian_electricity_demand_dataset.tsf",  # hourly electricity demand
    "traffic_hourly_dataset.tsf",                 # hourly traffic flow
    "kdd_cup_2018_dataset_with_missing_values.tsf",  # hourly air quality
    "oikolab_weather_dataset.tsf",                # hourly weather
    "pedestrian_counts_dataset.tsf",              # hourly pedestrian counts
]

# Full Monash TSF collection under /root/monash_tsf_all/data (32 files).
# Missing files are silently skipped by ``load_monash``.
ALL_MONASH_FILES = [
    "australian_electricity_demand_dataset.tsf",
    "bitcoin_dataset_with_missing_values.tsf",
    "car_parts_dataset_with_missing_values.tsf",
    "cif_2016_dataset.tsf",
    "covid_deaths_dataset.tsf",
    "fred_md_dataset.tsf",
    "hospital_dataset.tsf",
    "kaggle_web_traffic_dataset_with_missing_values.tsf",
    "kaggle_web_traffic_weekly_dataset.tsf",
    "kdd_cup_2018_dataset_with_missing_values.tsf",
    "london_smart_meters_dataset_with_missing_values.tsf",
    "nn5_daily_dataset_with_missing_values.tsf",
    "nn5_weekly_dataset.tsf",
    "oikolab_weather_dataset.tsf",
    "pedestrian_counts_dataset.tsf",
    "rideshare_dataset_with_missing_values.tsf",
    "saugeenday_dataset.tsf",
    "solar_10_minutes_dataset.tsf",
    "solar_4_seconds_dataset.tsf",
    "solar_weekly_dataset.tsf",
    "sunspot_dataset_with_missing_values.tsf",
    "temperature_rain_dataset_with_missing_values.tsf",
    "tourism_monthly_dataset.tsf",
    "tourism_quarterly_dataset.tsf",
    "tourism_yearly_dataset.tsf",
    "traffic_hourly_dataset.tsf",
    "traffic_weekly_dataset.tsf",
    "us_births_dataset.tsf",
    "vehicle_trips_dataset_with_missing_values.tsf",
    "weather_dataset.tsf",
    "wind_4_seconds_dataset.tsf",
    "wind_farms_minutely_dataset_with_missing_values.tsf",
]

PRESET_FILESETS = {
    "default": DEFAULT_MONASH_FILES,
    "etth_like": ETTH_LIKE_MONASH_FILES,
    "all": ALL_MONASH_FILES,
}


def parse_tsf(path):
    """Minimal Monash TSF parser. Returns list[np.ndarray(float32)] of series values.

    Each data line: ``series_name:start_ts:v1,v2,...``  ("?" means NaN).
    NaNs are dropped; series with <2 valid points are skipped.
    """
    series_list = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@data"):
                in_data = True
                continue
            if not in_data:
                continue
            vals = line.split(":")[-1]
            arr = np.fromiter(
                (float(v) if v and v != "?" else np.nan for v in vals.split(",")),
                dtype=np.float64,
            )
            arr = arr[~np.isnan(arr)]
            if arr.size >= 2:
                series_list.append(arr.astype(np.float32))
    return series_list


def load_monash(data_dir=MONASH_DIR, files=None, verbose=True):
    files = files or DEFAULT_MONASH_FILES
    all_series = []
    for fn in files:
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            if verbose:
                print(f"  [skip] {fn} not found")
            continue
        series = parse_tsf(fp)
        total = sum(s.size for s in series)
        if verbose:
            print(f"  [ok]   {fn}: {len(series)} series, {total:,} points")
        all_series.extend(series)
    if verbose:
        print(f"Total series: {len(all_series)}")
    return all_series


# =============================================================== Monash windows
def build_monash_windows(
    data_dir=MONASH_DIR,
    files=None,
    L_in=96,
    L_out=96,
    stride=None,
    max_windows_per_series=200,
    outlier_threshold=10.0,
    min_std=DEFAULT_MIN_STD,
    seed=0,
    verbose=True,
):
    """(X, Y) windows with RevIN per-window z-score.

    For each window: mean/std from X (context); apply to both X and Y.
    Drop windows where any ``|Y_norm| > outlier_threshold``.
    """
    rng = np.random.default_rng(seed)
    series_list = load_monash(data_dir, files, verbose=verbose)
    W = L_in + L_out
    if stride is None:
        stride = max(1, L_out // 2)

    Xs, Ys = [], []
    for s in series_list:
        if s.size < W + 1:
            continue
        starts = np.arange(0, s.size - W, stride, dtype=np.int64)
        if starts.size == 0:
            continue
        if starts.size > max_windows_per_series:
            starts = rng.choice(starts, size=max_windows_per_series, replace=False)
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
        print(f"Built RevIN windows: X={X.shape}, Y={Y.shape}")
    if X.shape[0] == 0:
        raise RuntimeError(
            f"build_monash_windows produced 0 windows. "
            f"Check --monash_dir ({data_dir}) and --monash_files: "
            f"{files}. Available files must end with the full Monash "
            f"name, e.g. 'australian_electricity_demand_dataset.tsf'."
        )
    return X, Y


def build_monash_windows_per_dataset(
    data_dir=MONASH_DIR,
    files=None,
    L_in=96,
    L_out=96,
    stride=None,
    max_windows_per_dataset=5000,
    outlier_threshold=10.0,
    min_std=DEFAULT_MIN_STD,
    seed=0,
    verbose=True,
):
    """Sample at most ``max_windows_per_dataset`` valid (X, Y) windows from
    EACH .tsf file in ``files``, with RevIN per-window z-score.

    Sampling procedure per dataset (file):
      1. Enumerate candidate (series_idx, start) pairs across all series
         (stride defaults to ``max(1, L_out//2)``); per-series candidates are
         capped at ``max_windows_per_dataset`` to bound memory.
      2. Randomly shuffle candidates and greedily take pairs until
         ``max_windows_per_dataset`` valid windows (passing std/outlier
         filters) are collected, or candidates are exhausted.
    """
    rng = np.random.default_rng(seed)
    files = files or DEFAULT_MONASH_FILES
    W = L_in + L_out
    if stride is None:
        stride = max(1, L_out // 2)

    Xs_all, Ys_all = [], []
    for fn in files:
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            if verbose:
                print(f"  [skip] {fn} not found")
            continue
        series_list = parse_tsf(fp)

        # 1) collect bounded candidate list (series_idx, start)
        cand = []
        for si, s in enumerate(series_list):
            if s.size < W + 1:
                continue
            starts = np.arange(0, s.size - W, stride, dtype=np.int64)
            if starts.size == 0:
                continue
            if starts.size > max_windows_per_dataset:
                starts = rng.choice(
                    starts, size=max_windows_per_dataset, replace=False
                )
            for st in starts:
                cand.append((si, int(st)))

        if not cand:
            if verbose:
                print(f"  [ok]   {fn}: 0 candidates (too short)")
            continue

        # 2) shuffle candidates and take target valid windows
        order = rng.permutation(len(cand))
        kept_X, kept_Y = [], []
        tried = 0
        for oi in order:
            if len(kept_X) >= max_windows_per_dataset:
                break
            si, st = cand[oi]
            s = series_list[si]
            win = s[st : st + W]
            x, y = win[:L_in], win[L_in:]
            mu, sd = x.mean(), x.std()
            tried += 1
            if sd < min_std:
                continue
            x_n = (x - mu) / sd
            y_n = (y - mu) / sd
            if np.abs(y_n).max() > outlier_threshold:
                continue
            kept_X.append(x_n.astype(np.float32))
            kept_Y.append(y_n.astype(np.float32))

        if verbose:
            print(f"  [ok]   {fn}: {len(cand)} candidates, "
                  f"tried {tried}, kept {len(kept_X)}")
        Xs_all.extend(kept_X)
        Ys_all.extend(kept_Y)

    X = np.stack(Xs_all, axis=0) if Xs_all else np.zeros((0, L_in), np.float32)
    Y = np.stack(Ys_all, axis=0) if Ys_all else np.zeros((0, L_out), np.float32)
    if verbose:
        print(f"Built per-dataset-sampled windows: X={X.shape}, Y={Y.shape}")
    return X, Y


def build_monash_windows_series_zscore(
    data_dir=MONASH_DIR,
    files=None,
    L_in=96,
    L_out=96,
    stride=None,
    max_windows_per_series=200,
    min_std=DEFAULT_MIN_STD,
    seed=0,
    verbose=True,
):
    """Same sampling as ``build_monash_windows`` but normalized per-SERIES
    (one global mean/std per series). Used as the plain-Linear baseline.
    """
    rng = np.random.default_rng(seed)
    series_list = load_monash(data_dir, files, verbose=verbose)
    W = L_in + L_out
    if stride is None:
        stride = max(1, L_out // 2)

    Xs, Ys = [], []
    for s in series_list:
        if s.size < W + 1:
            continue
        mu, sd = s.mean(), s.std()
        if sd < min_std:
            continue
        s_n = (s - mu) / sd
        starts = np.arange(0, s.size - W, stride, dtype=np.int64)
        if starts.size == 0:
            continue
        if starts.size > max_windows_per_series:
            starts = rng.choice(starts, size=max_windows_per_series, replace=False)
        for st in starts:
            win = s_n[st : st + W]
            Xs.append(win[:L_in].astype(np.float32))
            Ys.append(win[L_in:].astype(np.float32))

    X = np.stack(Xs, axis=0) if Xs else np.zeros((0, L_in), np.float32)
    Y = np.stack(Ys, axis=0) if Ys else np.zeros((0, L_out), np.float32)
    if verbose:
        print(f"Built series-zscore windows: X={X.shape}, Y={Y.shape}")
    return X, Y


# =============================================================== Benchmark CSV
def _etth_borders(context_length):
    b1 = [0, 12 * 30 * 24 - context_length, 12 * 30 * 24 + 4 * 30 * 24 - context_length]
    b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    return b1, b2


def _ettm_borders(context_length):
    b1 = [0, 12 * 30 * 24 * 4 - context_length, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length]
    b2 = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    return b1, b2


def _generic_borders(T, context_length):
    num_train = int(T * 0.7)
    num_test = int(T * 0.2)
    num_vali = T - num_train - num_test
    b1 = [0, num_train - context_length, T - num_test - context_length]
    b2 = [num_train, num_train + num_vali, T]
    return b1, b2


def _select_borders(csv_path, T, context_length):
    base = os.path.basename(csv_path).lower()
    if "etth" in base:
        return _etth_borders(context_length)
    if "ettm" in base:
        return _ettm_borders(context_length)
    return _generic_borders(T, context_length)


def load_etth_train_windows(csv_path, context_length=96, prediction_length=96, stride=1):
    """Sliding windows from the CSV TRAIN segment (all channels as univariate).

    StandardScaler is fit on the train segment only, per the Time-MoE protocol.
    """
    df = pd.read_csv(csv_path)
    cols = list(df.columns[1:])  # skip date column
    vals = df[cols].values.astype(np.float64)
    T = vals.shape[0]
    b1, b2 = _select_borders(csv_path, T, context_length)

    train_seg = vals[b1[0] : b2[0]]
    scaler = StandardScaler().fit(train_seg)
    scaled_train = scaler.transform(train_seg).astype(np.float32)

    W = context_length + prediction_length
    Xs, Ys = [], []
    T_train, C = scaled_train.shape
    for c in range(C):
        seq = scaled_train[:, c]
        for st in range(0, T_train - W + 1, stride):
            win = seq[st : st + W]
            Xs.append(win[:context_length])
            Ys.append(win[context_length:])
    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)
    return X, Y, scaler, cols


def load_etth_test(csv_path, context_length=96, prediction_length=96):
    """Sliding windows from the CSV TEST segment, Time-MoE convention.

    Returns
    -------
    X : (N, context_length)   standardized input
    Y : (N, prediction_length) standardized target
    scaler : fitted StandardScaler (column-wise)
    cols : column names
    """
    df = pd.read_csv(csv_path)
    cols = list(df.columns[1:])
    vals = df[cols].values.astype(np.float64)
    T = vals.shape[0]
    b1, b2 = _select_borders(csv_path, T, context_length)

    train_seg = vals[b1[0] : b2[0]]
    test_seg = vals[b1[2] : b2[2]]
    scaler = StandardScaler().fit(train_seg)
    scaled_test = scaler.transform(test_seg).astype(np.float32)

    W = context_length + prediction_length
    Xs, Ys = [], []
    T_test, C = scaled_test.shape
    for c in range(C):
        seq = scaled_test[:, c]
        for offset in range(W, T_test + 1):
            win = seq[offset - W : offset]
            Xs.append(win[:context_length])
            Ys.append(win[context_length:])
    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)
    return X, Y, scaler, cols
