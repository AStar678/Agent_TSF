"""DeepMA: layer-wise moving-average decomposition + closed-form ridge linear.

Public API
----------
- DeepMA (nn.Module)          : multi-level MA decomposition
- parse_tsf, load_monash      : Monash TSF IO
- build_monash_windows        : Monash sampling with RevIN
- load_gift_series            : gift-eval (HuggingFace) series IO
- build_gift_windows          : gift-eval sampling with RevIN
- load_etth_train_windows     : ETTh/benchmark CSV TRAIN sliding windows
- load_etth_test              : ETTh/benchmark CSV TEST sliding windows
- fit_ridge, decompose_batched: closed-form training primitives
- revin, predict              : inference helpers
- save_model, load_model      : weight IO
"""
from .model import DeepMA
from .data import (
    parse_tsf,
    load_monash,
    build_monash_windows,
    build_monash_windows_series_zscore,
    load_etth_train_windows,
    load_etth_test,
    DEFAULT_MONASH_FILES,
    ETTH_LIKE_MONASH_FILES,
    PRESET_FILESETS,
)
from .gift import (
    load_gift_series,
    build_gift_windows,
    list_gift_datasets,
    MINI_GIFT_DATASETS,
    ETTH_LIKE_GIFT_DATASETS,
    GIFT_PRESETS,
)
from .linalg import fit_ridge, decompose_batched
from .predict import revin, predict, save_model, load_model
from .benchmark import load_benchmark_split
from . import config

__all__ = [
    "DeepMA",
    "parse_tsf", "load_monash",
    "build_monash_windows", "build_monash_windows_series_zscore",
    "load_etth_train_windows", "load_etth_test",
    "DEFAULT_MONASH_FILES", "ETTH_LIKE_MONASH_FILES", "PRESET_FILESETS",
    "load_gift_series", "build_gift_windows", "list_gift_datasets",
    "MINI_GIFT_DATASETS", "ETTH_LIKE_GIFT_DATASETS", "GIFT_PRESETS",
    "fit_ridge", "decompose_batched",
    "revin", "predict", "save_model", "load_model",
    "load_benchmark_split",
    "config",
]
