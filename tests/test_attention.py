"""Correctness against the definition. On Linux CI these run the Triton kernels under
TRITON_INTERPRET=1; elsewhere they run the tiled PyTorch reference of the same algorithm."""

import os

import pytest
import torch

from tattn import attention, triton_available
from tattn.reference import naive_attention, tiled_backward, tiled_forward

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHAPES = [(1, 1, 1, 16), (1, 2, 17, 16), (2, 1, 33, 32), (1, 1, 64, 64), (1, 2, 70, 16)]


def rand(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(DEV, dtype)


def tol(dtype: torch.dtype) -> dict[str, float]:
    return {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.float16 else {"atol": 2e-4, "rtol": 2e-4}


def dtypes() -> list[torch.dtype]:
    # fp16 matmuls on CPU are only reliable in recent torch; always test fp32, fp16 where it runs.
    return [torch.float32, torch.float16]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", dtypes())
def test_forward_matches_definition(
    shape: tuple[int, int, int, int], causal: bool, dtype: torch.dtype
) -> None:
    q, k, v = (rand(shape, dtype, s) for s in (1, 2, 3))
    got = attention(q, k, v, causal=causal)
    want = naive_attention(q.float(), k.float(), v.float(), causal=causal)
    assert got.dtype == dtype and got.shape == q.shape
    torch.testing.assert_close(got.float(), want, **tol(dtype))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", dtypes())
def test_gradients_match_autograd(shape: tuple[int, int, int, int], causal: bool, dtype: torch.dtype) -> None:
    q, k, v = (rand(shape, dtype, s).requires_grad_() for s in (4, 5, 6))
    do = rand(shape, dtype, 7)
    attention(q, k, v, causal=causal).backward(do)
    got = [t.grad.float() for t in (q, k, v)]  # type: ignore[union-attr]
    qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
    naive_attention(qr, kr, vr, causal=causal).backward(do.float())
    for g, w in zip(got, (qr.grad, kr.grad, vr.grad), strict=True):
        torch.testing.assert_close(g, w, **tol(dtype))


def test_custom_scale_and_noncontiguous_inputs() -> None:
    base = rand((1, 20, 2, 32), torch.float32, 9)  # [B, N, H, D] -> transpose to [B, H, N, D]
    q = base.transpose(1, 2)
    k = rand((1, 20, 2, 32), torch.float32, 10).transpose(1, 2)
    v = rand((1, 20, 2, 32), torch.float32, 11).transpose(1, 2)
    assert not q.is_contiguous()
    got = attention(q, k, v, sm_scale=0.3)
    torch.testing.assert_close(got, naive_attention(q, k, v, sm_scale=0.3), **tol(torch.float32))


def test_rejects_bad_inputs() -> None:
    x = torch.zeros(1, 1, 8, 16)
    with pytest.raises(ValueError, match="equal shape"):
        attention(x, x[..., :4, :], x)
    with pytest.raises(ValueError, match="head dim"):
        y = torch.zeros(1, 1, 8, 24)
        attention(y, y, y)
    with pytest.raises(ValueError, match="dtype"):
        attention(x, x.double(), x)
    with pytest.raises(ValueError, match="equal shape"):
        attention(x[0], x[0], x[0])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("blocks", [(16, 16), (16, 32), (32, 16), (8, 64)])
def test_tiled_reference_is_block_size_independent(causal: bool, blocks: tuple[int, int]) -> None:
    q, k, v, do = (rand((2, 2, 45, 16), torch.float32, s) for s in (12, 13, 14, 15))
    o, m = tiled_forward(q, k, v, causal, block_m=blocks[0], block_n=blocks[1])
    torch.testing.assert_close(o, naive_attention(q, k, v, causal=causal), atol=1e-5, rtol=1e-5)
    grads = tiled_backward(q, k, v, o, m, do, causal, block_m=blocks[0], block_n=blocks[1])
    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    naive_attention(qr, kr, vr, causal=causal).backward(do)
    for g, w in zip(grads, (qr.grad, kr.grad, vr.grad), strict=True):
        torch.testing.assert_close(g, w, atol=1e-5, rtol=1e-5)


def test_backend_selection() -> None:
    assert triton_available(torch.device("cpu")) == (
        os.environ.get("TRITON_INTERPRET") == "1" and _has_triton()
    )


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def test_backward_with_noncontiguous_inputs_and_grad() -> None:
    q, k, v, do = (rand((1, 21, 2, 16), torch.float32, s).transpose(1, 2) for s in (20, 21, 22, 23))
    qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
    attention(qs, ks, vs, causal=True).backward(do)
    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    naive_attention(qr, kr, vr, causal=True).backward(do)
    for g, w in zip((qs.grad, ks.grad, vs.grad), (qr.grad, kr.grad, vr.grad), strict=True):
        torch.testing.assert_close(g, w, **tol(torch.float32))


def test_head_dim_128() -> None:
    q, k, v = (rand((1, 1, 19, 128), torch.float16, s).requires_grad_() for s in (30, 31, 32))
    out = attention(q, k, v, causal=True)
    torch.testing.assert_close(
        out.float(), naive_attention(q.float(), k.float(), v.float(), causal=True), **tol(torch.float16)
    )
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad.float()).all()


def test_double_backward_is_refused_not_wrong() -> None:
    q, k, v = (rand((1, 1, 8, 16), torch.float32, s).requires_grad_() for s in (40, 41, 42))
    (dq,) = torch.autograd.grad(attention(q, k, v).sum(), q, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dq.sum(), k)


def test_rejects_mixed_devices() -> None:
    x = torch.zeros(1, 1, 8, 16)
    with pytest.raises(ValueError, match="one device"):
        attention(x, x.to("meta"), x)
