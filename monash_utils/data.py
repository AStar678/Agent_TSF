"""Monash TSF data loading and window sampling utilities.

- TSF (Monash format) parser
- Monash window sampling with per-window RevIN z-score
- Per-dataset fixed-count sampling mode
"""
import os
import numpy as np

# =============================================================== Constants
MONASH_DIR = "/root/monash_tsf_all/data"
DEFAULT_MIN_STD = 1e-6

# Light-weight default subset (~70 MB).
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

# Hourly + energy/flow-like subset most similar to ETTh.
ETTH_LIKE_MONASH_FILES = [
    "australian_electricity_demand_dataset.tsf",
    "traffic_hourly_dataset.tsf",
    "kdd_cup_2018_dataset_with_missing_values.tsf",
    "oikolab_weather_dataset.tsf",
    "pedestrian_counts_dataset.tsf",
]

# Full Monash TSF collection (32 files).
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


# =============================================================== TSF Parser
def parse_tsf(path):
    """Minimal Monash TSF parser. Returns list[np.ndarray(float32)] of series.

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
    """Load all series from the given TSF files."""
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


# =============================================================== Window Sampling
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
    """Build (X, Y) windows with per-window RevIN z-score.

    For each window: mean/std from X (context); apply to both X and Y.
    Drop windows where any |Y_norm| > outlier_threshold.
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
            f"{files}."
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
    """Sample at most ``max_windows_per_dataset`` valid windows from EACH .tsf file,
    with per-window RevIN z-score.
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

        # Collect bounded candidate list (series_idx, start)
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

        # Shuffle and greedily collect valid windows
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
