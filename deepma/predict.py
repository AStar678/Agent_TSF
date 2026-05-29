"""Inference helpers and weight IO."""
import types
import numpy as np
import torch

from .model import DeepMA
from .config import DEFAULT_MIN_STD


def revin(X, Y=None, min_std=DEFAULT_MIN_STD):
    """Per-window z-score using X's mean/std. Apply to Y with same stats if given."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < min_std, 1.0, sd)
    Xn = (X - mu) / sd
    Yn = (Y - mu) / sd if Y is not None else None
    return Xn, Yn, mu, sd


def predict(model: DeepMA, Ws, bs, X_rev: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Forward: decompose X_rev -> per-layer linear -> sum. Returns (N, L_out)."""
    N, _ = X_rev.shape
    L_out = Ws[0].shape[1]
    Y_hat = np.zeros((N, L_out), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            x = torch.from_numpy(X_rev[s:e]).float().unsqueeze(-1)   # (B, L, 1)
            seasonals, trend = model(x)
            comps = [c.squeeze(-1).numpy() for c in seasonals] + [trend.squeeze(-1).numpy()]
            y = np.zeros((e - s, L_out), dtype=np.float32)
            for W, b, C in zip(Ws, bs, comps):
                y += C @ W + b
            Y_hat[s:e] = y
    return Y_hat


def save_model(path, kernels, L_in, L_out, Ws, bs):
    data = {
        "kernels": np.asarray(kernels, dtype=np.int64),
        "L_in": np.int64(L_in),
        "L_out": np.int64(L_out),
        "n_layers": np.int64(len(Ws)),
    }
    for i, (W, b) in enumerate(zip(Ws, bs)):
        data[f"W_{i}"] = W
        data[f"b_{i}"] = b
    np.savez(path, **data)


def load_model(path):
    """Returns (model, kernels, L_in, L_out, Ws, bs)."""
    d = np.load(path)
    kernels = d["kernels"].tolist()
    L_in = int(d["L_in"])
    L_out = int(d["L_out"])
    n_layers = int(d["n_layers"])
    Ws = [d[f"W_{i}"] for i in range(n_layers)]
    bs = [d[f"b_{i}"] for i in range(n_layers)]
    cfg = types.SimpleNamespace(kernel_list=kernels)
    model = DeepMA(cfg).eval()
    return model, kernels, L_in, L_out, Ws, bs
