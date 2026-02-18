"""Hybrid Python/C++ registration for ABI-stable wheels.

This package builds an ABI-stable C++/CUDA extension that exposes only a small
set of *internal* dispatcher operators (implemented with the LibTorch Stable
ABI). We then define the *public* torch_quickfps operators in Python using
``CompositeExplicitAutograd`` so:

1) We keep a stable binary surface across PyTorch versions (2.10+).
2) Backward for the point-gather path is handled automatically by PyTorch.
3) We preserve the existing public API: ``torch.ops.torch_quickfps.sample`` etc.

The compiled extension must be loaded (via ``torch.ops.load_library``) *before*
this module is imported.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# NOTE:
# The compiled extension already registers the "torch_quickfps" namespace via
# a single C++ TORCH_LIBRARY/STABLE_TORCH_LIBRARY block (for the internal ops).
# In PyTorch, only one TORCH_LIBRARY("ns", ...) is allowed per namespace.
# To add *additional* operator definitions from Python, we must use a fragment.
_DEF = torch.library.Library("torch_quickfps", "FRAGMENT")

# Public schemas (match the original extension).
_DEF.define(
    "sample(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> (Tensor, Tensor)"
)
_DEF.define(
    "sample_idx(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> Tensor"
)
_DEF.define(
    "sample_baseline(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> (Tensor, Tensor)"
)
_DEF.define(
    "sample_idx_baseline(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> Tensor"
)


_IMPL = torch.library.Library("torch_quickfps", "IMPL", "CompositeExplicitAutograd")


def _bnorm_reshape(t: torch.Tensor) -> Tuple[torch.Size, torch.Tensor]:
    """Match the original C++ helper: flatten batch dims into one B dim."""
    if t.dim() > 2:
        old = t.shape
        return old, t.view(-1, t.size(-2), t.size(-1))
    if t.dim() == 2:
        old = t.shape
        return old, t.view(1, t.size(0), t.size(1))
    raise RuntimeError(f"x must have at least 2 dims, but got shape: {tuple(t.shape)}")


def _reshape_indices_back(old_shape: torch.Size, idx_bk: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of _bnorm_reshape for indices."""
    out_shape = list(old_shape[:-1])  # drop feature dim
    out_shape[-1] = k
    return idx_bk.view(out_shape).to(torch.long)


def _reshape_points_back(old_shape: torch.Size, pts_bkd: torch.Tensor, k: int) -> torch.Tensor:
    out_shape = list(old_shape)
    out_shape[-2] = k
    return pts_bkd.view(out_shape)


def _normalize_h(h: Optional[int]) -> int:
    hv = 5 if h is None else int(h)
    if hv < 1:
        hv = 1
    return hv


def _normalize_low_d(low_d: Optional[int]) -> int:
    return -1 if low_d is None else int(low_d)


def _prepare_mask_and_start(
    x_bnd: torch.Tensor,
    k: int,
    start_idx: Optional[int],
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (start_idx_t[B], invalid_mask_u8[B,N]) as the C++ extension did."""

    B, N, _ = x_bnd.shape

    # invalid_mask: uint8 with 1 meaning "not eligible".
    if mask is None:
        invalid = torch.zeros((B, N), device=x_bnd.device, dtype=torch.uint8)
        mask_b = None
    else:
        if mask.device != x_bnd.device:
            raise RuntimeError("mask must be on the same device as x")
        if mask.dtype not in (torch.bool, torch.uint8):
            raise RuntimeError(f"mask must have dtype bool or uint8, got {mask.dtype}")
        if mask.numel() != B * N:
            raise RuntimeError(
                f"mask must have shape (*, N) matching x's batch/point dims. Expected numel={B*N} but got numel={mask.numel()}"
            )

        mask_b = mask.to(torch.bool).contiguous().view(B, N)
        counts = mask_b.sum(dim=1)
        min_valid = int(counts.min().item()) if B > 0 else 0
        if min_valid < k:
            raise RuntimeError(
                f"mask has fewer than k valid points in at least one batch. min_valid={min_valid} k={k}"
            )
        invalid = (~mask_b).to(torch.uint8).contiguous()

    # Start idx per batch
    if start_idx is not None:
        s = int(start_idx)
        if not (0 <= s < N):
            raise RuntimeError(f"start_idx out of range: {s} for N={N}")
        if mask_b is not None:
            ok = bool(mask_b[:, s].all().item())
            if not ok:
                raise RuntimeError(f"mask disallows start_idx={s} in at least one batch")
        start_t = torch.full((B,), s, device=x_bnd.device, dtype=torch.int64)
    else:
        if mask_b is not None:
            # First valid per batch.
            start_t = mask_b.to(torch.int64).argmax(dim=1)
        else:
            start_t = torch.randint(0, N, (B,), device=x_bnd.device, dtype=torch.int64)

    return start_t, invalid


def _sample_idx(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
) -> torch.Tensor:
    old_shape, x_bnd = _bnorm_reshape(x)
    x_bnd = x_bnd.contiguous()
    k = int(k)
    if k < 1:
        raise RuntimeError(f"k must be >= 1, got {k}")
    if k > x_bnd.size(1):
        raise RuntimeError(f"k must be <= N. Got k={k} N={x_bnd.size(1)}")

    hv = _normalize_h(h)
    lowdv = _normalize_low_d(low_d)
    start_t, invalid = _prepare_mask_and_start(x_bnd, k, start_idx, mask)

    # Forward to the internal ABI-stable kernel.
    idx_bk = torch.ops.torch_quickfps._sample_idx_impl(
        x_bnd, k, hv, start_t, invalid, lowdv
    )
    return _reshape_indices_back(old_shape, idx_bk, k)


# NOTE: In PyTorch 2.10, Library.impl is not a decorator; it expects (op_name, fn).
_IMPL.impl("sample_idx", _sample_idx)


def _sample(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    old_shape, x_bnd = _bnorm_reshape(x)
    x_bnd = x_bnd.contiguous()
    k = int(k)
    hv = _normalize_h(h)
    lowdv = _normalize_low_d(low_d)
    start_t, invalid = _prepare_mask_and_start(x_bnd, k, start_idx, mask)
    idx_bk = torch.ops.torch_quickfps._sample_idx_impl(
        x_bnd, k, hv, start_t, invalid, lowdv
    )
    B, _, D = x_bnd.shape
    gathered = torch.gather(x_bnd, 1, idx_bk.view(B, k, 1).expand(B, k, D))
    pts = _reshape_points_back(old_shape, gathered, k)
    idx = _reshape_indices_back(old_shape, idx_bk, k)
    return pts, idx


_IMPL.impl("sample", _sample)


def _sample_idx_baseline(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
) -> torch.Tensor:
    # h/low_d are accepted for signature compatibility but ignored.
    old_shape, x_bnd = _bnorm_reshape(x)
    x_bnd = x_bnd.contiguous()
    k = int(k)
    start_t, invalid = _prepare_mask_and_start(x_bnd, k, start_idx, mask)
    idx_bk = torch.ops.torch_quickfps._sample_idx_baseline_impl(x_bnd, k, start_t, invalid)
    return _reshape_indices_back(old_shape, idx_bk, k)


_IMPL.impl("sample_idx_baseline", _sample_idx_baseline)


def _sample_baseline(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    low_d: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    old_shape, x_bnd = _bnorm_reshape(x)
    x_bnd = x_bnd.contiguous()
    k = int(k)
    start_t, invalid = _prepare_mask_and_start(x_bnd, k, start_idx, mask)
    idx_bk = torch.ops.torch_quickfps._sample_idx_baseline_impl(x_bnd, k, start_t, invalid)
    B, _, D = x_bnd.shape
    gathered = torch.gather(x_bnd, 1, idx_bk.view(B, k, 1).expand(B, k, D))
    pts = _reshape_points_back(old_shape, gathered, k)
    idx = _reshape_indices_back(old_shape, idx_bk, k)
    return pts, idx


_IMPL.impl("sample_baseline", _sample_baseline)


# Optional: FakeTensor support for torch.compile.
@torch.library.register_fake("torch_quickfps::sample_idx")
def _fake_sample_idx(x, k, h=None, start_idx=None, mask=None, low_d=None):
    # Output shape: x.shape[:-1] with last point dim replaced by k.
    out_shape = list(x.shape[:-1])
    out_shape[-1] = int(k)
    return torch.empty(out_shape, device=x.device, dtype=torch.long)


@torch.library.register_fake("torch_quickfps::sample")
def _fake_sample(x, k, h=None, start_idx=None, mask=None, low_d=None):
    k = int(k)
    pts_shape = list(x.shape)
    pts_shape[-2] = k
    idx_shape = list(x.shape[:-1])
    idx_shape[-1] = k
    return (
        torch.empty(pts_shape, device=x.device, dtype=x.dtype),
        torch.empty(idx_shape, device=x.device, dtype=torch.long),
    )


@torch.library.register_fake("torch_quickfps::sample_idx_baseline")
def _fake_sample_idx_baseline(x, k, h=None, start_idx=None, mask=None, low_d=None):
    out_shape = list(x.shape[:-1])
    out_shape[-1] = int(k)
    return torch.empty(out_shape, device=x.device, dtype=torch.long)


@torch.library.register_fake("torch_quickfps::sample_baseline")
def _fake_sample_baseline(x, k, h=None, start_idx=None, mask=None, low_d=None):
    k = int(k)
    pts_shape = list(x.shape)
    pts_shape[-2] = k
    idx_shape = list(x.shape[:-1])
    idx_shape[-1] = k
    return (
        torch.empty(pts_shape, device=x.device, dtype=x.dtype),
        torch.empty(idx_shape, device=x.device, dtype=torch.long),
    )
