"""Public Python API for torch_quickfps.

The core operator is implemented as a C++/CUDA extension and loaded via
``torch.ops.load_library`` in :mod:`torch_quickfps.__init__`.

We expose two user-facing functions:

* :func:`sample` – returns (points, indices) to match common FPS APIs.
* :func:`sample_idx` – returns indices only (no gather), for maximum speed.
* :func:`sample_baseline` – baseline vanilla FPS (CUDA only).
* :func:`sample_idx_baseline` – indices-only baseline vanilla FPS (CUDA only).

Both accept an optional ``mask`` where ``True`` indicates eligible points.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def sample(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
    return_points: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Bucket-based farthest point sampling.

    Args:
        x: Tensor with shape [..., N, C] (any batch dims).
        k: Number of samples (<= N).
        h: Bucket height parameter (controls bucket granularity).
        start_idx: Optional starting index to seed FPS.
        mask: Optional boolean/uint8 mask of shape [..., N] where True means
            the point is eligible to be sampled.
        low_d: Optional bucketing dimension for CUDA KD-tree bucketing. If None,
            bucketing uses the full feature dimension C. If set (e.g., 3 or 8),
            bucketing runs in a cheap Rademacher-projected space of size low_d
            while FPS distances are still computed in full C.
        return_points: If True (default), return ``(points, indices)``.
            If False, return indices only (equivalent to :func:`sample_idx`).

    Returns:
        If return_points=True: (sampled_points, sampled_indices)
        If return_points=False: sampled_indices
    """

    if return_points:
        return torch.ops.torch_quickfps.sample(x, k, h, start_idx, mask, low_d)
    return torch.ops.torch_quickfps.sample_idx(x, k, h, start_idx, mask, low_d)


def sample_idx(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
) -> torch.Tensor:
    """Bucket-based farthest point sampling (indices only).

    This avoids the internal gather and is typically faster when you only need
    indices (e.g., to gather features yourself).

    Returns:
        sampled_indices with shape [..., k]
    """
    return torch.ops.torch_quickfps.sample_idx(x, k, h, start_idx, mask, low_d)


def sample_baseline(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
    return_points: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Baseline vanilla FPS (CUDA only).

    Matches the signature of :func:`sample` for easy benchmarking.
    """

    if return_points:
        return torch.ops.torch_quickfps.sample_baseline(
            x, k, h, start_idx, mask, low_d
        )
    return torch.ops.torch_quickfps.sample_idx_baseline(
        x, k, h, start_idx, mask, low_d
    )


def sample_idx_baseline(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
) -> torch.Tensor:
    """Baseline vanilla FPS (indices only, CUDA only)."""
    return torch.ops.torch_quickfps.sample_idx_baseline(
        x, k, h, start_idx, mask, low_d
    )
