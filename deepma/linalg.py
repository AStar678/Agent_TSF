"""Closed-form training primitives."""
from typing import List
import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):  # graceful fallback
        return it

from .model import DeepMA


def decompose_batched(model: DeepMA, X: np.ndarray, batch_size: int = 4096,
                      desc: str = "decompose", show_progress: bool = True) -> List[np.ndarray]:
    """Run DeepMA on (N, L); return ``K+1`` arrays of shape (N, L): K seasonals + trend."""
    N, L = X.shape
    n_layers = len(model.kernel_list) + 1
    outs = [np.empty((N, L), dtype=np.float32) for _ in range(n_layers)]
    model.eval()
    iterator = range(0, N, batch_size)
    if show_progress and N > batch_size:
        iterator = tqdm(list(iterator), desc=desc, ncols=80, leave=False)
    with torch.no_grad():
        for s in iterator:
            e = min(s + batch_size, N)
            x = torch.from_numpy(X[s:e]).float().unsqueeze(-1)   # (B, L, 1)
            seasonals, trend = model(x)
            for i, comp in enumerate(seasonals):
                outs[i][s:e] = comp.squeeze(-1).numpy()
            outs[-1][s:e] = trend.squeeze(-1).numpy()
    return outs


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-4):
    """Closed-form ridge least squares with bias column (bias is NOT regularized).

    X: (N, D), Y: (N, P)  ->  W: (D, P), b: (P,)
    """
    N, D = X.shape
    A = np.concatenate([X, np.ones((N, 1), dtype=X.dtype)], axis=1)   # (N, D+1)
    reg = lam * np.eye(D + 1, dtype=np.float64)
    reg[-1, -1] = 0.0
    AtA = A.T.astype(np.float64) @ A.astype(np.float64) + reg
    AtY = A.T.astype(np.float64) @ Y.astype(np.float64)
    theta = np.linalg.solve(AtA, AtY)                                # (D+1, P)
    W = theta[:-1].astype(np.float32)
    b = theta[-1].astype(np.float32)
    return W, b
