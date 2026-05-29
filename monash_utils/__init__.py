"""Monash pre-training utilities for Time-Series-Library.

Public API
----------
- parse_tsf, load_monash       : Monash TSF IO
- build_monash_windows          : Monash sampling with per-window RevIN
- build_monash_windows_per_dataset : Per-dataset fixed-count sampling
- load_benchmark_split          : Time-MoE protocol CSV loader
- fit_ridge                     : Closed-form ridge regression
- revin                         : Per-window RevIN normalization
- PRESET_FILESETS               : Preset file lists
"""
from .data import (
    parse_tsf,
    load_monash,
    build_monash_windows,
    build_monash_windows_per_dataset,
    DEFAULT_MONASH_FILES,
    ETTH_LIKE_MONASH_FILES,
    ALL_MONASH_FILES,
    PRESET_FILESETS,
    MONASH_DIR,
)
from .linalg import fit_ridge, revin
from .benchmark import load_benchmark_split

__all__ = [
    "parse_tsf", "load_monash",
    "build_monash_windows", "build_monash_windows_per_dataset",
    "DEFAULT_MONASH_FILES", "ETTH_LIKE_MONASH_FILES", "ALL_MONASH_FILES",
    "PRESET_FILESETS", "MONASH_DIR",
    "fit_ridge", "revin",
    "load_benchmark_split",
]
