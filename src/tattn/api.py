"""Public API: `attention(q, k, v, causal=False, sm_scale=None)` with autograd."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from tattn import reference

_warned = False


def _warn_slow_triton(device: torch.device) -> None:
    """Triton >= 3.3 compiles tl.dot on sm75 (Turing) without tensor cores: about 10x slower."""
    global _warned
    if _warned or device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 5):
        return
    import triton

    major, minor = (int(x) for x in triton.__version__.split(".")[:2])
    if (major, minor) >= (3, 3):
        warnings.warn(
            f"Triton {triton.__version__} emits no tensor-core code on sm75; attention will be ~10x slower. "
            "Install triton==3.2.0 on Turing GPUs (see README).",
            RuntimeWarning,
            stacklevel=3,
        )
    _warned = True


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
    if not (q.device == k.device == v.device):
        raise ValueError(f"q, k, v must be on one device, got {q.device}, {k.device}, {v.device}")
    d = q.shape[-1]
    if d not in (16, 32, 64, 128):
        raise ValueError(f"head dim must be 16, 32, 64 or 128, got {d}")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"q, k, v must share dtype float16 or float32, got {q.dtype}, {k.dtype}, {v.dtype}")


def _bucket(n: int) -> int:
    """Autotune per power-of-two length bucket, not per exact length: a new sequence length
    shouldn't trigger a full re-tune inside a training step."""
    return 1 << max(0, (n - 1).bit_length())


def _prec(t: torch.Tensor) -> str:
    return "ieee" if t.dtype == torch.float32 else "tf32"


def _launch(kernel: Any, grid: Any, config: dict[str, int] | None, *args: Any, **kwargs: Any) -> None:
    """Run through the autotuner, or (config given) the raw kernel with that exact tile config, so
    tests can check every configuration the autotuner might pick."""
    if config is None:
        kernel[grid](*args, **kwargs)
    else:
        kernel.fn[lambda meta: grid({**meta, **config})](*args, **kwargs, **config)


def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float,
    config: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    import triton

    from tattn.kernels import _fwd_kernel

    z, h, n, d = q.shape
    o = torch.empty_like(q)
    m = torch.empty((z, h, n), device=q.device, dtype=torch.float32)

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), z * h)

    _launch(
        _fwd_kernel, grid, config,
        q, k, v, o, m, scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        h, n, _bucket(n), HEAD_DIM=d, CAUSAL=causal, DOT_PREC=_prec(q),
    )  # fmt: skip
    return o, m


def _backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, m: torch.Tensor, do: torch.Tensor,
    causal: bool, scale: float, config: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    import triton

    from tattn.kernels import _bwd_dkdv_kernel, _bwd_dq_kernel, _bwd_preprocess

    z, h, n, d = q.shape
    q, k, v, do, o = (t.contiguous() for t in (q, k, v, do, o))
    delta = torch.empty((z, h, n), device=q.device, dtype=torch.float32)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    stride = q.stride()

    def grid_n(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_N"]), z * h)

    def grid_m(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), z * h)

    _bwd_preprocess[(triton.cdiv(n, 16), z * h)](o, do, delta, *stride, h, n, HEAD_DIM=d, BLOCK_M=16)
    common = dict(HEAD_DIM=d, CAUSAL=causal, DOT_PREC=_prec(q))
    _launch(
        _bwd_dkdv_kernel,
        grid_n,
        config,
        q,
        k,
        v,
        do,
        dk,
        dv,
        m,
        delta,
        scale,
        *stride,
        h,
        n,
        _bucket(n),
        **common,
    )
    _launch(
        _bwd_dq_kernel, grid_m, config, q, k, v, do, dq, m, delta, scale, *stride, h, n, _bucket(n), **common
    )
    return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
    ) -> torch.Tensor:
        if triton_available(q.device):
            _warn_slow_triton(q.device)
            o, m = _forward(q, k, v, causal, scale)
            ctx.impl = "triton"
        else:
            o, m = reference.tiled_forward(q, k, v, causal, scale)
            ctx.impl = "reference"
        ctx.save_for_backward(q, k, v, o, m)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    @once_differentiable
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
