"""Public API: `attention(q, k, v, causal=False, sm_scale=None)` with autograd."""

from __future__ import annotations

import math
from typing import Any

import torch

from tattn import reference


def triton_available(device: torch.device) -> bool:
    import os

    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"expected q, k, v of equal shape [batch, heads, seq, dim]; got {q.shape}, {k.shape}, {v.shape}"
        )
    d = q.shape[-1]
    if d not in (16, 32, 64, 128):
        raise ValueError(f"head dim must be 16, 32, 64 or 128, got {d}")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"q, k, v must share dtype float16 or float32, got {q.dtype}, {k.dtype}, {v.dtype}")


def _prec(t: torch.Tensor) -> str:
    return "ieee" if t.dtype == torch.float32 else "tf32"


def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    import triton

    from tattn.kernels import _fwd_kernel

    z, h, n, d = q.shape
    o = torch.empty_like(q)
    m = torch.empty((z, h, n), device=q.device, dtype=torch.float32)

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), z * h)

    _fwd_kernel[grid](
        q, k, v, o, m, scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        h, n, HEAD_DIM=d, CAUSAL=causal, DOT_PREC=_prec(q),
    )  # fmt: skip
    return o, m


def _backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, m: torch.Tensor, do: torch.Tensor,
    causal: bool, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    import triton

    from tattn.kernels import _bwd_dkdv_kernel, _bwd_dq_kernel

    z, h, n, d = q.shape
    q, k, v, do = (t.contiguous() for t in (q, k, v, do))
    delta = (do.float() * o.float()).sum(-1)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    stride = q.stride()

    def grid_n(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_N"]), z * h)

    def grid_m(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), z * h)

    _bwd_dkdv_kernel[grid_n](
        q, k, v, do, dk, dv, m, delta, scale, *stride, h, n, HEAD_DIM=d, CAUSAL=causal, DOT_PREC=_prec(q)
    )
    _bwd_dq_kernel[grid_m](
        q, k, v, do, dq, m, delta, scale, *stride, h, n, HEAD_DIM=d, CAUSAL=causal, DOT_PREC=_prec(q)
    )
    return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
    ) -> torch.Tensor:
        if triton_available(q.device):
            o, m = _forward(q, k, v, causal, scale)
            ctx.impl = "triton"
        else:
            o, m = reference.tiled_forward(q, k, v, causal, scale)
            ctx.impl = "reference"
        ctx.save_for_backward(q, k, v, o, m)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, o, m = ctx.saved_tensors
        if ctx.impl == "triton":
            dq, dk, dv = _backward(q, k, v, o, m, do, ctx.causal, ctx.scale)
        else:
            dq, dk, dv = reference.tiled_backward(q, k, v, o, m, do, ctx.causal, ctx.scale)
        return dq, dk, dv, None, None


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, sm_scale: float | None = None
) -> torch.Tensor:
    """Fused attention over [batch, heads, seq, dim] tensors. Uses the Triton kernels on CUDA (or
    under TRITON_INTERPRET=1), otherwise the tiled PyTorch reference of the same algorithm."""
    _check(q, k, v)
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    out: torch.Tensor = _Attention.apply(q, k, v, causal, scale)
    return out
