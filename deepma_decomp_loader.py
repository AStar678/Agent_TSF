"""Load a previously decomposed dataset (produced by ``decompose_monash.py``).

Example
-------
    from deepma_decomp_loader import load_dataset_decomp

    d = load_dataset_decomp(
        "/root/autodl-tmp/deepma_decomp/australian_electricity_demand_dataset"
    )
    d["meta"]                # dict loaded from meta.json
    d["seasonal_k005"]["s000000"]   # 1-D float32 array, same length as source
    d["trend"]["s000000"]
"""
import json
import os
from typing import Dict, List

import numpy as np


def load_meta(dataset_dir: str) -> dict:
    with open(os.path.join(dataset_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_layer(dataset_dir: str, layer_file: str) -> Dict[str, np.ndarray]:
    """Eager-load a single layer NPZ into ``{key: ndarray}``.

    For very large layers prefer :func:`iter_series` to stream series.
    """
    with np.load(os.path.join(dataset_dir, layer_file)) as z:
        return {k: z[k] for k in z.files}


def iter_series(dataset_dir: str, layer_file: str):
    """Yield ``(key, ndarray)`` pairs from an NPZ layer without fully loading it."""
    with np.load(os.path.join(dataset_dir, layer_file)) as z:
        for k in z.files:
            yield k, z[k]


def load_dataset_decomp(dataset_dir: str) -> dict:
    """Return ``{"meta": dict, "<layer_stem>": {key: ndarray}, ...}``.

    Layer stems are ``seasonal_k005``, ..., ``seasonal_k300``, ``trend``.
    Loads everything into memory; use :func:`iter_series` for large files.
    """
    meta = load_meta(dataset_dir)
    out = {"meta": meta}
    for fname in meta["layer_files"]:
        stem = os.path.splitext(fname)[0]
        out[stem] = load_layer(dataset_dir, fname)
    return out


def list_datasets(root_dir: str) -> List[str]:
    """Return stems of every dataset folder that already contains a meta.json."""
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, d, "meta.json"))
    )
