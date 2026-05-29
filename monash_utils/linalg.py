"""Closed-form training primitives for linear models."""
import numpy as np


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-4):
    """Closed-form ridge least squares with bias column (bias is NOT regularized).

    X: (N, D), Y: (N, P)  ->  W: (D, P), b: (P,)
    """
    N, D = X.shape
    A = np.concatenate([X, np.ones((N, 1), dtype=X.dtype)], axis=1)  # (N, D+1)
    reg = lam * np.eye(D + 1, dtype=np.float64)
    reg[-1, -1] = 0.0
    AtA = A.T.astype(np.float64) @ A.astype(np.float64) + reg
    AtY = A.T.astype(np.float64) @ Y.astype(np.float64)
    theta = np.linalg.solve(AtA, AtY)  # (D+1, P)
    W = theta[:-1].astype(np.float32)
    b = theta[-1].astype(np.float32)
    return W, b


def revin(X, Y=None):
    """Per-window RevIN z-score normalization.

    Parameters
    ----------
    X : (N, L_in) input windows
    Y : (N, L_out) optional target windows

    Returns
    -------
    X_norm, Y_norm (or None), mu, sd
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    X_norm = (X - mu) / sd
    Y_norm = (Y - mu) / sd if Y is not None else None
    return X_norm, Y_norm, mu, sd
