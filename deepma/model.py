"""DeepMA: multi-level moving-average decomposition.

For a kernel list [k_1, ..., k_K] applied sequentially on the running trend:
    x_0  = input
    t_i  = MA(x_{i-1}, k_i)
    s_i  = x_{i-1} - t_i
    x_i  = t_i               (passed to next level)

Output: (seasonals=[s_1, ..., s_K], trend=t_K)
Identity: s_1 + s_2 + ... + s_K + t_K == input (exact up to fp error).

Reflection padding is symmetric via edge replication (front/back repeat).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_window(kernel_size: int, kind: str) -> torch.Tensor:
    """Return a length-``kernel_size`` 1-D weight that sums to 1.

    Supported ``kind`` values:
        "rect"  : flat (equivalent to AvgPool1d)
        "hann"  : Hann (raised cosine), low side-lobes
        "hamming" : Hamming
        "gauss" : Gaussian, sigma = (k-1)/6 -> ~3 sigma at edges
        "tri"   : triangular (Bartlett)
    """
    k = int(kernel_size)
    n = torch.arange(k, dtype=torch.float64)
    if kind == "rect":
        w = torch.ones(k, dtype=torch.float64)
    elif kind == "hann":
        w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / max(1, k - 1))
    elif kind == "hamming":
        w = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / max(1, k - 1))
    elif kind == "gauss":
        c = (k - 1) / 2.0
        sigma = max(1.0, (k - 1) / 6.0)
        w = torch.exp(-0.5 * ((n - c) / sigma) ** 2)
    elif kind == "tri":
        c = (k - 1) / 2.0
        w = 1.0 - torch.abs((n - c) / max(1.0, c))
    else:
        raise ValueError(f"unknown window kind: {kind!r}")
    w = w / w.sum()
    return w.float()


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, window_kind: str = "rect"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.kernel_size = kernel_size
        self.window_kind = window_kind
        if window_kind == "rect":
            # Keep the fast AvgPool path for backward compatibility.
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
            self.register_buffer("weight", None, persistent=False)
        else:
            self.avg = None
            w = _make_window(kernel_size, window_kind)        # (k,)
            # Conv1d weight: (out_channels=1, in_channels/groups=1, k); broadcast to C groups at runtime.
            self.register_buffer("weight", w.view(1, 1, kernel_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)         # (B, L+2*pad, C)
        x = x.permute(0, 2, 1)                        # (B, C, L+2*pad)
        if self.avg is not None:
            x = self.avg(x)                           # (B, C, L)
        else:
            C = x.shape[1]
            w = self.weight.to(dtype=x.dtype, device=x.device).expand(C, 1, self.kernel_size)
            x = F.conv1d(x, w, bias=None, stride=1, padding=0, groups=C)
        return x.permute(0, 2, 1)                     # (B, L, C)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int, window_kind: str = "rect"):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1, window_kind=window_kind)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DeepMA(nn.Module):
    """Layer-wise MA decomposition with progressively larger kernels.

    ``configs.window_kind`` (optional, default ``"rect"``) selects the MA
    weighting: "rect" reproduces the original AvgPool1d behaviour;
    "hann" / "hamming" / "gauss" / "tri" use weighted MA with much lower
    side-lobes, reducing cross-band leakage at the cost of slightly wider
    transition bands.
    """

    def __init__(self, configs):
        super().__init__()
        # configs.kernel_list: List[int]
        self.kernel_list = list(configs.kernel_list)
        self.window_kind = getattr(configs, "window_kind", "rect")
        self.blocks = nn.ModuleList([
            _SeriesDecomp(k, window_kind=self.window_kind)
            for k in self.kernel_list
        ])

    def forward(self, x: torch.Tensor):
        """x: (B, L, C) -> (seasonals: List[Tensor], trend: Tensor)."""
        seasonals = []
        for block in self.blocks:
            seasonal, trend = block(x)
            seasonals.append(seasonal)
            x = trend
        return seasonals, x  # x is the final trend
